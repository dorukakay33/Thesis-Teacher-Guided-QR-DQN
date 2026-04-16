import itertools
import runpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# General settings
MAIN_FILE = "Main_Code.py"
ROOT = Path("results/ablation_compare_methods_V2_50000_EE_10_SEED_50_Gstep_5000")
ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(500, 550))
ENABLE_TRAINING = False  # Only update plots if False

METHODS = {
    "Scratch": ("Scratch", False, False, False),
    "Distil": ("Distil", False, False, True),
    "Pref": ("Pref", True, False, False),
    "BC": ("BC", False, True, False),
    "Pref+BC": ("Pref+BC", True, True, False),
    "Distil+Pref": ("Distil+Pref", True, False, True),
    "Distil+BC": ("Distil+BC", False, True, True),
    "Distil+Pref+BC": ("Distil+Pref+BC", True, True, True),
}

# Train or read data
if ENABLE_TRAINING:
    ns = runpy.run_path(MAIN_FILE)
    run_experiment = ns["run_experiment"]
    rows = []

    for method_name, flags in METHODS.items():
        tag, use_pref, use_bc, use_distil = flags
        for sd in SEEDS:
            run_dir = ROOT / method_name / f"seed{sd}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_experiment(
                results_dir=str(run_dir),
                configs_override=[(tag, use_pref, use_bc, use_distil)],
                seed=sd
            )
            df = pd.read_csv(run_dir / "results_history.csv")
            df["method"] = method_name
            df["seed"] = sd
            rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(ROOT / "all_runs.csv", index=False)
else:
    all_df = pd.read_csv(ROOT / "all_runs.csv")

# Plot settings
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})

# Aggregate data
agg = (
    all_df.groupby(["method", "step"])["reward"]
    .agg(["mean", "std"])
    .reset_index()
)

palette = plt.get_cmap("tab10")
plot_order = ["Scratch"] + [m for m in METHODS if m != "Scratch"]
label_map = {m: m for m in METHODS}

def nice_plot(df, fname, step_filter=None):
    plt.figure(figsize=(9, 6))
    for i, m in enumerate(plot_order):
        sub = df[(df.method == m) & ((df.step <= step_filter) if step_filter else True)]
        c = palette(i)
        mean_rewards = sub["mean"]
        std_devs = np.maximum(sub["std"], 0)  # Ensure no negative std

        plt.plot(sub["step"], mean_rewards, lw=2, label=label_map[m], color=c)
        plt.fill_between(
            sub["step"],
            np.maximum(mean_rewards - std_devs, 0),
            mean_rewards + std_devs,
            alpha=0.15,
            color=c,
        )

    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episode Reward")
    plt.ylim(0, 400)
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(ROOT / fname, dpi=150)
    plt.close()

# Generate plots
nice_plot(agg, "methods_ablation.png")
nice_plot(agg, "methods_ablation_30k.png", step_filter=30000)

print(f"✓ Updated plots with shaded error saved in: {ROOT}")