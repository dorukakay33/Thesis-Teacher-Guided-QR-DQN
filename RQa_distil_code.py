import itertools, runpy, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# CONFIGURATION
MAIN_FILE   = "Main_Code.py"
ROOT        = Path("results/ablation_distil_T_50000_EE_10_SEED_50_GStep_5000")
ROOT.mkdir(parents=True, exist_ok=True)

SEEDS       = list(range(600, 650))         # 50 seeds
T_SET       = [0.1, 0.5, 1.0, 2.0]          # temperatures to compare
RUN_TRAINING = False

def make_distil_student_fn(T: float):
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    DEVICE        = ns["DEVICE"]
    expected_q    = ns["expected_q"]
    DISTIL_EPOCHS = ns["DISTIL_EPOCHS"]

    def distil_student(teacher, student, obs_bank,
                       epochs=DISTIL_EPOCHS, lr=1e-3):
        student.policy.set_training_mode(True)
        loader = DataLoader(obs_bank, batch_size=256,
                            shuffle=True, drop_last=True)
        opt = torch.optim.Adam(student.policy.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            losses = []
            for batch in loader:
                batch = batch.to(DEVICE)
                with torch.no_grad():
                    p_t = torch.softmax(expected_q(teacher.policy, batch) / T, 1)
                p_s = torch.log_softmax(expected_q(student.policy, batch) / T, 1)
                loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T ** 2)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            print(f"[Distil-T{T}] epoch {ep}/{epochs}  KL={np.mean(losses):.4f}")
        student.policy.set_training_mode(False)

    return distil_student

# TRAINING PHASE (optional)
if RUN_TRAINING:
    ns = runpy.run_path(MAIN_FILE)
    run_experiment = ns["run_experiment"]
    rows = []

    for sd in SEEDS:
        out = ROOT / "Scratch" / f"seed{sd}"
        out.mkdir(parents=True, exist_ok=True)
        run_experiment(results_dir=str(out),
                       configs_override=[("Scratch", False, False, False)],
                       seed=sd)
        df = pd.read_csv(out / "results_history.csv")
        df["method"] = "Scratch"; df["T"] = "Scratch"; df["seed"] = sd
        rows.append(df)

    for T, sd in itertools.product(T_SET, SEEDS):
        out = ROOT / f"Distil_T{T}" / f"seed{sd}"
        out.mkdir(parents=True, exist_ok=True)
        ns["distil_student"] = make_distil_student_fn(T)

        run_experiment(results_dir=str(out),
                       configs_override=[("Distil", False, False, True)],
                       seed=sd)
        df = pd.read_csv(out / "results_history.csv")
        df["method"] = f"Distil_T{T}"; df["T"] = T; df["seed"] = sd
        rows.append(df)

    pd.concat(rows, ignore_index=True).to_csv(ROOT / "all_runs.csv", index=False)

# LOAD RESULTS & PLOT
all_df = pd.read_csv(ROOT / "all_runs.csv")
agg = (all_df.groupby(["method", "step"])["reward"]
               .agg(["mean", "std"]).reset_index())

# Plot style
plt.rcParams.update({
    "font.size"       : 14,
    "font.weight"     : "bold",
    "axes.labelweight": "bold",
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 14,
    "legend.fontsize" : 12,
})

palette = plt.get_cmap("tab10")

def _plot_curve(ax, subdf, label, color):
    step = subdf["step"]
    mean = subdf["mean"]
    std = subdf["std"].clip(lower=0)  # avoid negative shading
    lower = np.maximum(mean - std, 0)
    upper = mean + std
    ax.plot(step, mean, label=label, color=color, lw=2)
    ax.fill_between(step, lower, upper, color=color, alpha=0.15)

def _finish(ax, filename):
    ax.set_ylim(0, 400)
    ax.set_xlabel("Environment Steps",  fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward",fontsize=14, fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    leg = ax.legend(loc="upper left")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# FULL HORIZON
fig, ax = plt.subplots(figsize=(9, 6))
scr = agg[agg.method == "Scratch"]
_plot_curve(ax, scr, "Scratch", palette(0))

for i, T in enumerate(T_SET, start=1):
    tag = f"Distil_T{T}"
    sub = agg[agg.method == tag]
    if not sub.empty:
        _plot_curve(ax, sub, f"Distil: T={T}", palette(i % 10))

_finish(ax, ROOT / "distil_T_vs_scratch.png")

# FIRST 30K
fig, ax = plt.subplots(figsize=(9, 6))
_plot_curve(ax, scr[scr.step <= 30_000], "Scratch", palette(0))

for i, T in enumerate(T_SET, start=1):
    tag = f"Distil_T{T}"
    sub = agg[(agg.method == tag) & (agg.step <= 30_000)]
    if not sub.empty:
        _plot_curve(ax, sub, f"Distil: T={T}", palette(i % 10))

_finish(ax, ROOT / "distil_T_vs_scratch_30k.png")

print(f"✓ Figures written to {ROOT}")