import itertools, runpy, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ─────────── USER SETTINGS ─────────────────────────────────────────────────
MAIN_FILE     = "Main_Code.py"
ROOT          = Path("results/ablation_pref_vs_scratch_V2_50000_EE_10_SEED_50_Gstep_5000")
ROOT.mkdir(parents=True, exist_ok=True)

SEEDS         = list(range(500, 550))          # 50 seeds
PARAM_PAIRS_TRAIN = [
    (4,   1),
    (20,  5),
    (100, 25),
    (200, 50),
    (400, 100),
]
PARAM_PAIRS_PLOT  = [
    (100, 25),
    (200, 50),
    (400, 100),
]
RUN_TRAINING  = False
# ────────────────────────────────────────────────────────────────────────────

# ═════════════════════ TRAINING PHASE (optional) ═══════════════════════════
if RUN_TRAINING:
    ns             = runpy.run_path(MAIN_FILE)
    run_experiment = ns["run_experiment"]

    rows = []

    for seed in SEEDS:
        run_dir = ROOT / "Scratch" / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_experiment(results_dir=str(run_dir),
                       configs_override=[("Scratch", False, False, False)],
                       seed=seed)
        df = pd.read_csv(run_dir / "results_history.csv")
        df["method"] = "Scratch"; df["seed"] = seed
        rows.append(df)

    for (pref_eps, min_neg), seed in itertools.product(PARAM_PAIRS_TRAIN, SEEDS):
        tag     = f"P{pref_eps}_M{min_neg}"
        run_dir = ROOT / tag / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        ns["PREF_EPISODES"] = pref_eps
        ns["MIN_NEG"]       = min_neg

        run_experiment(results_dir=str(run_dir),
                       configs_override=[("Pref", True, False, False)],
                       seed=seed)
        df = pd.read_csv(run_dir / "results_history.csv")
        df["method"] = tag; df["seed"] = seed
        rows.append(df)

    pd.concat(rows, ignore_index=True).to_csv(ROOT / "all_runs.csv", index=False)

# ═════════════════════ PLOTTING PHASE ══════════════════════════════════════
all_df = pd.read_csv(ROOT / "all_runs.csv")

keep_methods = ["Scratch"] + [f"P{m}_M{n}" for m, n in PARAM_PAIRS_PLOT]
all_df       = all_df[all_df.method.isin(keep_methods)]

agg = (all_df.groupby(["method", "step"])["reward"]
               .agg(["mean", "std"]).reset_index())

# ── global bold style ------------------------------------------------------
plt.rcParams.update({
    "font.size"       : 14,
    "font.weight"     : "bold",
    "axes.labelweight": "bold",
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 14,
    "legend.fontsize" : 12,
})

palette = plt.get_cmap("tab10")

def nice_label(method):
    if method == "Scratch":
        return r"Scratch"
    m, n = method.lstrip("P").split("_M")
    return rf"$M={m},\;n_{{\mathrm{{neg}}}}={n}$"

def _plot_curve(ax, subdf, label, color):
    step  = subdf["step"]
    mean  = subdf["mean"]
    std   = subdf["std"].clip(lower=0)
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

def _ordered_methods(df):
    def _key(m):
        if m == "Scratch": return (-1, -1)
        M, N = m.lstrip("P").split("_M")
        return (int(M), int(N))
    return sorted(df.method.unique(), key=_key)

# ---------------- FULL-HORIZON FIGURE --------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for i, m in enumerate(_ordered_methods(agg)):
    sub = agg[agg.method == m]
    _plot_curve(ax, sub, nice_label(m), palette(i))
_finish(ax, ROOT / "pref_vs_scratch.png")

# ---------------- FIRST 30K FIGURE -----------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
agg30 = agg[agg.step <= 30_000]
for i, m in enumerate(_ordered_methods(agg30)):
    sub = agg30[agg30.method == m]
    _plot_curve(ax, sub, nice_label(m), palette(i))
_finish(ax, ROOT / "pref_vs_scratch_30k.png")

print(f"✓ Figures written to {ROOT}")