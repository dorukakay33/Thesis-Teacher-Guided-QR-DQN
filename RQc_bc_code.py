import itertools, runpy, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ─────────── USER SETTINGS ─────────────────────────────────────────────────
MAIN_FILE    = "Main_Code.py"
ROOT         = Path("results/ablation_bc_weight_T_50000_EE_10_SEED_50_GStep_5000")
ROOT.mkdir(parents=True, exist_ok=True)

SEEDS        = list(range(500, 550))        # 50 seeds
BC_WEIGHTS   = [0.1, 0.5, 1.0]              # λBC values
RUN_TRAINING = False                        # flip to True to re-run training
# ────────────────────────────────────────────────────────────────────────────

if RUN_TRAINING:
    ns             = runpy.run_path(MAIN_FILE)
    run_experiment = ns["run_experiment"]
    rows = []

    for sd in SEEDS:
        out = ROOT / "Scratch" / f"seed{sd}"
        out.mkdir(parents=True, exist_ok=True)
        run_experiment(results_dir=str(out),
                       configs_override=[("Scratch", False, False, False)],
                       seed=sd,
                       lambda_bc=0.0)
        df = pd.read_csv(out / "results_history.csv")
        df["method"] = "Scratch"; df["lambda"] = 0.0; df["seed"] = sd
        rows.append(df)

    for lamb, sd in itertools.product(BC_WEIGHTS, SEEDS):
        tag = f"BC_{lamb}"
        out = ROOT / tag / f"seed{sd}"
        out.mkdir(parents=True, exist_ok=True)
        run_experiment(results_dir=str(out),
                       configs_override=[(tag, False, True, False)],
                       seed=sd,
                       lambda_bc=lamb)
        df = pd.read_csv(out / "results_history.csv")
        df["method"] = tag; df["lambda"] = lamb; df["seed"] = sd
        rows.append(df)

    pd.concat(rows, ignore_index=True).to_csv(ROOT / "all_runs.csv", index=False)

# ═════════════════════ PLOTTING PHASE ══════════════════════════════════════
all_df = pd.read_csv(ROOT / "all_runs.csv")
agg = (all_df.groupby(["method", "step"])["reward"]
               .agg(["mean", "std"]).reset_index())

plt.rcParams.update({
    "font.size"       : 14,
    "font.weight"     : "bold",
    "axes.labelweight": "bold",
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 14,
    "legend.fontsize" : 12,
})

palette   = plt.get_cmap("tab10")
label_map = {
    "Scratch" : r"Scratch ($\lambda_{\mathrm{BC}} = 0$)",
    "BC_0.1"  : r"BC: $\lambda_{\mathrm{BC}} = 0.1$",
    "BC_0.5"  : r"BC: $\lambda_{\mathrm{BC}} = 0.5$",
    "BC_1.0"  : r"BC: $\lambda_{\mathrm{BC}} = 1.0$",
}

def _plot_curve(ax, subdf, label, color):
    step  = subdf["step"]
    mean  = subdf["mean"]
    std   = subdf["std"].clip(lower=0)
    lower = np.maximum(mean - std, 0)
    upper = mean + std
    ax.plot(step, mean, label=label, color=color, lw=2)
    ax.fill_between(step, lower, upper, color=color, alpha=0.25)

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
    methods = sorted([m for m in df.method.unique() if m != "Scratch"],
                     key=lambda x: float(x.split("_")[1]))
    return ["Scratch"] + methods

# ---------------- FULL-HORIZON FIGURE --------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for i, m in enumerate(_ordered_methods(agg)):
    sub = agg[agg.method == m]
    _plot_curve(ax, sub, label_map.get(m, m), palette(i))
_finish(ax, ROOT / "bc_vs_scratch.png")

# ---------------- FIRST 30K FIGURE -----------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
agg30 = agg[agg.step <= 30_000]
for i, m in enumerate(_ordered_methods(agg30)):
    sub = agg30[agg30.method == m]
    _plot_curve(ax, sub, label_map.get(m, m), palette(i))
_finish(ax, ROOT / "bc_vs_scratch_30k.png")

print(f"✓ Figures written to {ROOT}")