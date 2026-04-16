from __future__ import annotations

import itertools
import os
import random
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sb3_contrib import QRDQN
from sb3_contrib.qrdqn.policies import QRDQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path

# ─────────────────────────────── Constants ────────────────────────────────
ENV = "CartPole-v1"
SEED = 42
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

TOTAL_STEPS = 50_000
EVAL_INTERVAL = 5_000
EVAL_EPISODES = 10

DISTIL_EPOCHS = 20
OBS_BANK_STEPS = 120_000

PREF_EPISODES = 400
POS_THRESH = 450.0
NEG_THRESH = 50.0
MIN_NEG = 100  # guaranteed minimum number of negative clips

REWARD_UPDATE_EVERY = 4_000  # env steps
LAMBDA_BC = 0.1
BUFFER_SIZE = 50_000

REWARD_HIDDEN=32

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ───────────────────────── Utility helpers ────────────────────────────────

def fix_seed() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def py_int(x) -> int:
    """Convert numpy scalars to plain Python int (Gymnasium requirement)."""
    return int(x)

def random_cartpole_state(env: gym.Env) -> None:
    """State = (x, x_dot, theta, theta_dot)"""
    x      = np.random.uniform(-2.4, 2.4)
    x_dot  = np.random.uniform(-2.0, 2.0)
    theta  = np.random.uniform(-0.4, 0.4)          # büyük açı!
    th_dot = np.random.uniform(-3.0, 3.0)
    env.unwrapped.state = np.array([x, x_dot, theta, th_dot], dtype=np.float32)
    
def safe_features(policy: QRDQNPolicy, x: torch.Tensor) -> torch.Tensor:
    fe = getattr(policy, "features_extractor", None)
    return fe(x) if callable(fe) else x.view(x.shape[0], -1).to(policy.device)

def expected_q(policy: QRDQNPolicy, x: torch.Tensor) -> torch.Tensor:
    z = safe_features(policy, x)
    q = policy.quantile_net(z)
    q = q.view(-1, policy.action_space.n, policy.n_quantiles)
    return q.mean(2)  # [B, n_actions]

# ───────────────────────────── Reward Net ─────────────────────────────────
class RewardNet(nn.Module):
    """Tiny MLP: (obs, one‑hot(action)) → scalar reward."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = REWARD_HIDDEN):
        super().__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(obs_dim + n_actions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if act.dtype in (torch.int32, torch.int64):
            act = F.one_hot(act, num_classes=self.n_actions).float()
        x = torch.cat([obs.float(), act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)

# ─────────────────── Preference data structures ───────────────────────────
class Clip:
    def __init__(self, obs: np.ndarray, acts: np.ndarray):
        self.obs = obs
        self.acts = acts

    def to_tensor(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        o = torch.as_tensor(self.obs, dtype=torch.float32, device=device)
        a = torch.as_tensor(self.acts, dtype=torch.long, device=device)
        return o, a

class PrefDataset(Dataset):
    """Dataset of all (positive, negative) clip pairs."""

    def __init__(self, pos: List[Clip], neg: List[Clip]):
        super().__init__()
        self._pairs = list(itertools.product(pos, neg))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        return self._pairs[idx]

def collate_pref(batch):
    pos, neg = zip(*batch)
    return list(pos), list(neg)

# ──────────────────── Reward‑net training routine ─────────────────────────

def train_reward_net(net: RewardNet, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device):
    net.train()
    for pos_batch, neg_batch in loader:
        r_pos, r_neg = [], []
        for cp, cn in zip(pos_batch, neg_batch):
            o0, a0 = cp.to_tensor(device)
            o1, a1 = cn.to_tensor(device)
            r_pos.append(net(o0, a0).sum())
            r_neg.append(net(o1, a1).sum())
        loss = -torch.log(torch.sigmoid(torch.stack(r_pos) - torch.stack(r_neg))).mean()
        optim.zero_grad(); loss.backward(); optim.step()

# ──────────────── Modified QR‑DQN student with BC loss ────────────────────
class PrefQRDQN(QRDQN):
    """QR-DQN that can ① use a learned reward model and ② add BC auxiliary loss."""

    def __init__(self, *args, reward_net: RewardNet | None = None, bc_weight: float = 0.0,
                 good_transitions: List[Tuple[np.ndarray, int]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_net = reward_net
        self.bc_w = bc_weight
        self.good_transitions = good_transitions or []

    # new SB3 signature: (replay_buffer, actions, next_obs, rewards, dones, infos)
    def _store_transition(self, replay_buffer, buffer_actions, new_obs, rewards, dones, infos):  # noqa: N802
        # ① replace env reward with learned reward (vectorised across envs)
        if self.reward_net is not None:
            with torch.no_grad():
                obs_t = torch.as_tensor(self._last_obs, dtype=torch.float32, device=self.device)
                act_t = torch.as_tensor(buffer_actions, dtype=torch.long, device=self.device)
                rewards = self.reward_net(obs_t, act_t).cpu().numpy()
        # ② keep positive transitions for BC (assume scalar reward per env)
        if self.bc_w > 0:
            for o, a, r in zip(self._last_obs, buffer_actions, rewards):
                if r > 0.5:
                    self.good_transitions.append((o.copy(), int(a)))
        # ③ call parent implementation to actually add to replay buffer
        super()._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

    # after TD update we optionally add one auxiliary BC step
    def train(self, gradient_steps: int, batch_size: int):  # type: ignore[override]
        # standard TD-update(s)
        QRDQN.train(self, gradient_steps, batch_size)

        if self.bc_w == 0 or len(self.good_transitions) < batch_size:
            return
        idx = np.random.choice(len(self.good_transitions), batch_size, replace=False)
        obs_bc = torch.as_tensor(np.array([self.good_transitions[i][0] for i in idx]),
                                 dtype=torch.float32, device=self.device)
        act_bc = torch.as_tensor([self.good_transitions[i][1] for i in idx],
                                 dtype=torch.long, device=self.device)
        self.policy.optimizer.zero_grad()
        logp = F.log_softmax(expected_q(self.policy, obs_bc), dim=1)
        loss_bc = F.nll_loss(logp, act_bc) * self.bc_w
        loss_bc.backward(); self.policy.optimizer.step(); self._n_updates += 1

# ───────────────────────────── Eval callback ──────────────────────────────
class EvalHistory(BaseCallback):
    def __init__(self, env: gym.Env, tag: str, interval: int = EVAL_INTERVAL, episodes: int = EVAL_EPISODES):
        super().__init__(); self.env, self.tag, self.itv, self.eps = env, tag, interval, episodes
        self.hist: List[Tuple[int, float]] = []
    def _on_step(self) -> bool:
        if self.num_timesteps % self.itv: return True
        returns = []
        for _ in range(self.eps):
            obs, _ = self.env.reset(); done = False; R = 0.0
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = self.env.step(int(a)); R += r
                done = term or trunc
            returns.append(R)
        m = float(np.mean(returns))
        print(f"[{self.tag}] step {self.num_timesteps:6d} | mean {m:6.1f}")
        self.hist.append((self.num_timesteps, m)); return True
# ────────── Robust preference‑episode collection (w/ random fallback) ─────

def collect_preference_episodes(env_id: str, teacher: QRDQN, n_eps: int,
                                pos_th: float, neg_th: float, min_neg: int = MIN_NEG,
                                seed: int = SEED) -> Tuple[List[Clip], List[Clip]]:
    env = gym.make(env_id); rng = np.random.default_rng(seed)
    pos, neg = [], []

    def run_episode(action_fn) -> Tuple[float, Clip]:
        obs, _ = env.reset(seed=py_int(rng.integers(0, 1_000_000)))
        done, ret, o_buf, a_buf = False, 0.0, [], []
        while not done:
            act = action_fn(obs)
            o_buf.append(obs); a_buf.append(act)
            obs, r, term, trunc, _ = env.step(int(act)); ret += r; done = term or trunc
        return ret, Clip(np.array(o_buf), np.array(a_buf))

    # teacher episodes
    for _ in range(n_eps):
        R, clip = run_episode(lambda o: teacher.predict(o, deterministic=True)[0])
        if R >= pos_th:
            pos.append(clip)
        elif R <= neg_th:
            neg.append(clip)

    # ensure at least `min_neg` negatives
    if len(neg) < min_neg:
        print(f"⚠️  Only {len(neg)} negatives → adding random episodes ...")
        while len(neg) < min_neg:
            R, clip = run_episode(lambda _: env.action_space.sample())
            if R <= neg_th:
                neg.append(clip)
    return pos, neg

# ─────────────────────── Distillation helper ------------------------------

def distil_student(teacher: QRDQN, student: QRDQN, obs_bank: torch.Tensor,
                   epochs: int = DISTIL_EPOCHS, T: float = 0.1, lr: float = 1e-3):
    student.policy.set_training_mode(True)
    loader = DataLoader(obs_bank, batch_size=256, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(student.policy.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        losses = []
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                p_t = torch.softmax(expected_q(teacher.policy, batch) / T, dim=1)
            p_s = torch.log_softmax(expected_q(student.policy, batch) / T, dim=1)
            loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T ** 2)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"[Distil] epoch {ep}/{epochs} KL={np.mean(losses):.4f}")
    student.policy.set_training_mode(False)

# ─────────────────────────────── Main -------------------------------------

def run_experiment(
        results_dir      : str = "results",
        configs_override : List[Tuple[str,bool,bool,bool]] | None = None,
        lambda_bc        : float | None = None,
        seed             : int   | None = None,
        obs_bank_steps   : int   | None = None,   #  ← NEW
) -> None:
    global LAMBDA_BC, SEED, OBS_BANK_STEPS
    if lambda_bc      is not None: LAMBDA_BC   = lambda_bc
    if seed           is not None: SEED        = seed
    if obs_bank_steps is not None: OBS_BANK_STEPS = obs_bank_steps
    

    
    fix_seed()

    # 1) Teacher ----------------------------------------------------------------
    vec_env = make_vec_env(ENV, n_envs=1, seed=SEED)
    teacher_path = os.path.join(RESULTS_DIR, f"{ENV}_teacher_qrdqn_oracle.zip")
    if os.path.exists(teacher_path):
        teacher = QRDQN.load(teacher_path, device=DEVICE)
        print("✅ Teacher loaded from disk.")
    else:
        teacher = QRDQN("MlpPolicy", vec_env, learning_rate=1e-3, buffer_size=100_000,
                        learning_starts=5_000, batch_size=128, gamma=0.99,
                        train_freq=4, gradient_steps=1, target_update_interval=1_000,
                        exploration_fraction=0.12, exploration_final_eps=0.01,
                        policy_kwargs=dict(net_arch=[128, 128], n_quantiles=50),
                        seed=SEED, device=DEVICE, verbose=0)
        teacher.learn(200_000); teacher.save(teacher_path)

        # 4) Student builder --------------------------------------------------------
    base_vec = make_vec_env(ENV, n_envs=1, seed=SEED)

    def make_student(tag: str, use_pref: bool, use_bc: bool, use_distil: bool) -> PrefQRDQN:
        student = PrefQRDQN(
            "MlpPolicy", base_vec,
            learning_rate=2e-4, buffer_size=BUFFER_SIZE,
            learning_starts=2_000, batch_size=64, gamma=0.99,
            train_freq=4, gradient_steps=1, target_update_interval=500,
            exploration_fraction=0.01, exploration_final_eps=0.01,
            policy_kwargs=dict(net_arch=[32, 32], n_quantiles=25),
            seed=SEED, device=DEVICE, verbose=0,
            reward_net=reward_net if use_pref else None,
            bc_weight=LAMBDA_BC if use_bc else 0.0,
        )
        if use_distil:
            print(f"=== {tag}: distillation ===")
            distil_student(teacher, student, obs_bank)
        else:
            print(f"=== {tag}: scratch (no distillation) ===")
        return student

    default_cfg = [
        ("Distil",            False, False, True),   # baseline distillation only
        ("Distil+Pref",       True,  False, True),   # pref‑RL only
        ("Distil+Pref+BC",    True,  True,  True),   # pref‑RL + BC
        ("Pref",       True,  False, False),   # pref‑RL only
        ("Pref+BC",    True,  True,  False),   # pref‑RL + BC
        ("Scratch",           False, False, False),  # no distillation, learn from scratch
    ]

    
    configs = configs_override if configs_override is not None else default_cfg

    eval_env = gym.make(ENV)
    results: List[dict] = []

    for tag, pref_flag, bc_flag, distil_flag in configs:
        
        if pref_flag:
        
            # 2) Preference data ---------------------------------------------------------
            print("ℹ️ Collecting preference episodes ...")
            pos_clips, neg_clips = collect_preference_episodes(ENV, teacher, PREF_EPISODES,
                                                               POS_THRESH, NEG_THRESH)
            print(f"   collected {len(pos_clips)} positive & {len(neg_clips)} negative clips.")
    
            pref_loader = DataLoader(PrefDataset(pos_clips, neg_clips), batch_size=8,
                                     shuffle=True, drop_last=True, collate_fn=collate_pref)
    
            obs_dim = gym.make(ENV).observation_space.shape[0]
            n_actions = gym.make(ENV).action_space.n
            reward_net = RewardNet(obs_dim, n_actions).to(DEVICE)
            opt_r = torch.optim.Adam(reward_net.parameters(), lr=5e-4)
            
            print("ℹ️ Initial reward‑net fit (10 epochs)")
            for _ in range(10):
                train_reward_net(reward_net, pref_loader, opt_r, DEVICE)
                
                
        if distil_flag:
            # 3) Observation bank for distillation -------------------------------------
            obs_bank = []
            env_c = gym.make(ENV); obs, _ = env_c.reset(seed=SEED)
            for _ in range(OBS_BANK_STEPS):
                obs_bank.append(obs)
                mode=np.random.rand()
                
                if mode < 0.01:                       # 20% random-start
                    random_cartpole_state(env_c)
                    obs,_=env_c.reset()
                    act,_=teacher.predict(obs, deterministic=True)
                elif mode < 0.02:                     # 20% random action
                    act=env_c.action_space.sample()
                else:                                # 60% teacher normal
                    act,_=teacher.predict(obs, deterministic=True)
                    
                    
                #act, _ = teacher.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env_c.step(int(act))
                if term or trunc:
                    obs, _ = env_c.reset()
            obs_bank = torch.as_tensor(np.array(obs_bank), dtype=torch.float32)
        
        learner = make_student(tag, pref_flag, bc_flag, distil_flag)
        cb = EvalHistory(eval_env, tag)
        steps_done = 0
        while steps_done < TOTAL_STEPS:
            if pref_flag and steps_done > 0 and steps_done % REWARD_UPDATE_EVERY == 0:
                print(f"[RewardNet] fine‑tune at env step {steps_done}")
                train_reward_net(reward_net, pref_loader, opt_r, DEVICE)
            learner.learn(EVAL_INTERVAL, reset_num_timesteps=False, callback=cb)
            steps_done += EVAL_INTERVAL
        for s, r in cb.hist:
            results.append(dict(method=tag, step=s, reward=r))
    # 5) Save & plot ------------------------------------------------------------
    
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    ROOT       = Path(results_dir)
    ROOT.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(ROOT, "results_history.csv"); df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    for tag, _, _, _ in configs:
        sub = df[df.method == tag]
        plt.plot(sub.step, sub.reward, label=tag)
    plt.xlabel("Environment steps"); plt.ylabel("Mean episode reward")
    plt.title(f"{ENV} benchmark – QR‑DQN variants")
    plt.grid(True); plt.legend()
    png_path = os.path.join(ROOT, "benchmark_plot.png"); plt.savefig(png_path, dpi=150)

    print(f"\nSaved CSV  → {csv_path}\nSaved plot → {png_path}\n")

if __name__ == "__main__":
    run_experiment()
