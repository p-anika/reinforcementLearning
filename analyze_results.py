"""
analyze_results.py
------------------
Standalone analysis and plotting script for saved experiment results.

Run after training is complete — this script never touches the training code.
Decoupled from training so plots can be regenerated quickly without re-running
any experiments.

Usage
-----
    python analyze_results.py --results-dir results/2026-04-16_14-30-00

Outputs (saved to --results-dir by default, or --output-dir if specified):
    trust_curves.png    — Bayesian trust weight (w) over training steps
    loss_curves.png     — Policy gradient / actor loss over training
    summary_table.csv   — Success rates pivoted by agent × env × human × mode
    summary_table.txt   — Human-readable version of the same table

Each plot contains one subplot per run, labelled with agent/env/mode/human.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import glob
from scipy import stats

from training.evaluator import load_results


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from saved experiment results."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-dir",
        help="Path to a main run directory, e.g. results/2026-04-16_14-30-00"
    )
    group.add_argument(
        "--sensitivity-dir",
        help=(
            "Root directory containing sensitivity_w0/, sensitivity_N0/, and/or "
            "sensitivity_magnitude/ subdirectories (e.g. results/). "
            "Produces a sensitivity_analysis.png figure."
        )
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save plots (default: same as --results-dir / --sensitivity-dir)"
    )
    return parser.parse_args()


# ─── Success Rate Table ────────────────────────────────────────────────────────

def build_summary_table(df):
    """
    Build a pivot table of mean success rates across seeds.

    Rows: (agent, env_id, human)
    Columns: mode (sparse, naive, bayesian)
    Values: mean success_rate over all seeds
    """
    table = df.pivot_table(
        index=["agent", "env_id", "human"],
        columns="mode",
        values="success_rate",
        aggfunc="mean",
    )
    col_order = [c for c in ["sparse", "naive", "ema", "bayesian"] if c in table.columns]
    return table[col_order]


def save_summary_table(table, output_dir):
    """Save the summary table as both CSV and formatted text."""
    table.to_csv(os.path.join(output_dir, "summary_table.csv"))

    with open(os.path.join(output_dir, "summary_table.txt"), "w") as f:
        f.write("SUCCESS RATE (mean over seeds)\n")
        f.write("=" * 80 + "\n")
        f.write(table.to_string(float_format="{:.3f}".format))
        f.write("\n")

    print("\n─── SUCCESS RATE TABLE ───────────────────────────────────────────")
    print(table.to_string(float_format="{:.3f}".format))


# ─── Statistical Significance ──────────────────────────────────────────────────

def compute_significance(df):
    """
    Mann-Whitney U tests (non-parametric, appropriate for n=5 seeds) comparing
    BRF vs each baseline for every (env_id, human) pair.

    Returns DataFrame: env_id, human, comparison, p_value, significant
    """
    records = []
    for (env_id, human), grp in df.groupby(["env_id", "human"]):
        brf = grp[grp["mode"] == "bayesian"]["success_rate"].values
        for baseline in ["sparse", "naive", "ema"]:
            base_vals = grp[grp["mode"] == baseline]["success_rate"].values
            if len(brf) < 2 or len(base_vals) < 2:
                continue
            _, p = stats.mannwhitneyu(brf, base_vals, alternative="two-sided")
            records.append({
                "env_id":      env_id,
                "human":       human,
                "comparison":  f"BRF vs {baseline}",
                "p_value":     round(p, 4),
                "significant": p < 0.05,
            })
    return pd.DataFrame(records)


def print_significance_table(sig_df):
    if sig_df.empty:
        print("Not enough data for significance tests (need >=2 seeds per mode).")
        return
    n_sig = sig_df["significant"].sum()
    print(f"\n--- SIGNIFICANCE TABLE (Mann-Whitney U, two-sided) ---")
    print(f"  Significant at p<0.05: {n_sig} / {len(sig_df)} comparisons\n")
    sig_only = sig_df[sig_df["significant"]].copy()
    if not sig_only.empty:
        print(sig_only[["env_id", "human", "comparison", "p_value"]].to_string(index=False))
    else:
        print("  None reached p<0.05.")


# ─── Poster Visuals ────────────────────────────────────────────────────────


def save_poster_table(df, output_dir):
    """
    Creates a simplified, high-impact table for the Empty environment.
    This focuses on the 'working' case to explain the Bayesian advantage.
    """
    # Filter for the most representative agent and environment
    poster_df = df[(df["env_id"] == "MiniGrid-Empty-8x8-v0") & (df["agent"] == "PPO")]
    
    table = poster_df.pivot_table(
        index="human",
        columns="mode",
        values="success_rate",
        aggfunc="mean"
    )
    
    # Reorder and clean up names for the poster
    col_order = [c for c in ["sparse", "naive", "bayesian"] if c in table.columns]
    table = table[col_order]
    
    path_csv = os.path.join(output_dir, "poster_simplified_table.csv")
    table.to_csv(path_csv)
    
    print(f"Poster table saved  → {path_csv}")
    return table

def plot_poster_visuals(df, output_dir):
    """
    Generates the two key visuals for the center of the poster:
    1. The 'Robustness' Bar Chart (PPO on Empty Env)
    2. The 'Agent Reliability' Heatmap (Bayesian mode across all envs)
    """
    # Set global style for posters (large fonts, clean background)
    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. Bar Chart: Robustness to Noise
    # Focusing on PPO and the Empty environment where feedback is most critical
    plt.figure(figsize=(10, 6))
    subset = df[(df["env_id"] == "MiniGrid-Empty-8x8-v0") & (df["agent"] == "PPO")]
    
    # Standardize mode names and colors
    palette = {"sparse": "#95a5a6", "naive": "#e67e22", "bayesian": "#8e44ad"}
    
    ax = sns.barplot(
        data=subset, x="human", y="success_rate", hue="mode",
        palette=palette, errorbar="sd", capsize=.1
    )
    
    plt.title("PPO Success Rate: Resilience to Teacher Noise", pad=20)
    plt.ylabel("Success Rate (Mean ± SD)")
    plt.xlabel("Human Teacher Profile")
    plt.ylim(0, 1.1)
    plt.legend(title="Reward Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    path_bar = os.path.join(output_dir, "poster_bar_robustness.png")
    plt.savefig(path_bar, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Heatmap: Cross-Environment Performance
    # Showing where the Bayesian filter 'unlocks' new environments
    plt.figure(figsize=(10, 5))
    
    # Focus on Bayesian performance to show generalizability
    heat_data = df[df["mode"] == "bayesian"].pivot_table(
        index="human", 
        columns="env_id", 
        values="success_rate", 
        aggfunc="mean"
    )
    
    sns.heatmap(heat_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Success Rate'})
    plt.title("Bayesian Filter Performance Heatmap", pad=20)
    plt.xlabel("Environment ID")
    plt.ylabel("Human Teacher Profile")
    
    path_heat = os.path.join(output_dir, "poster_heatmap_bayesian.png")
    plt.savefig(path_heat, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Poster bar chart   → {path_bar}")
    print(f"Poster heatmap     → {path_heat}")


def plot_loss_curves_simplified(results_dir, output_dir):
    """
    Simplifies 100+ curves into one clear comparison plot.
    Groups by 'mode' and shows mean loss + standard deviation.
    """
   
    # 1. Load all .npy files into a structured list
    curves_dir = os.path.join(results_dir, "curves")
    loss_files = glob.glob(os.path.join(curves_dir, "*_loss.npy"))
    
    all_data = []
    for f in loss_files:
        mode = "bayesian" if "bayesian" in f else "naive" if "naive" in f else "sparse"
        data = np.load(f)
        # We normalize length because some runs might have slightly different update counts
        all_data.append({"mode": mode, "loss": data})

    # 2. Plotting
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    colors = {"sparse": "grey", "naive": "orange", "bayesian": "purple"}

    for mode in ["sparse", "naive", "bayesian"]:
        mode_losses = [d["loss"] for d in all_data if d["mode"] == mode]
        if not mode_losses: continue

        # Find minimum length to truncate for averaging
        min_len = min(len(l) for l in mode_losses)
        truncated_losses = np.array([l[:min_len] for l in mode_losses])
        
        mean_loss = np.mean(truncated_losses, axis=0)

        if np.isnan(mean_loss).all():
            print(f"Warning: All loss values for {mode} are NaN. Training likely failed.")
            continue

        std_loss = np.std(truncated_losses, axis=0)
        steps = np.arange(len(mean_loss))

        plt.plot(steps, mean_loss, label=f"{mode.capitalize()} (Mean)", color=colors[mode], lw=2)
        plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, 
                        color=colors[mode], alpha=0.2)

    plt.title("Training Convergence: Policy Loss by Reward Mode", fontsize=14)
    plt.xlabel("Gradient Updates", fontsize=12)
    plt.ylabel("Loss Magnitude", fontsize=12)
    plt.yscale("linear")
    plt.legend()
    
    path = os.path.join(output_dir, "simplified_loss_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Simplified loss plot saved → {path}")

# ─── Trust Score Curves ────────────────────────────────────────────────────────

def plot_trust_curves(curves_dir, output_dir):
    """
    Plot Bayesian trust weight (w) over training steps.

    Only bayesian-mode runs are plotted (sparse and naive have fixed w).
    The dashed line at w=0.5 marks the chance level (uniform Beta prior).
    """
    trust_files = sorted([
        f for f in os.listdir(curves_dir)
        if f.endswith("_trust.npy") and "bayesian" in f
    ])

    if not trust_files:
        print("No bayesian trust curve files found — skipping trust plot.")
        return

    n    = len(trust_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)

    for i, fname in enumerate(trust_files):
        ax    = axes[i // cols][i % cols]
        data  = np.load(os.path.join(curves_dir, fname))
        # Derive a readable title from the filename
        label = fname.replace("_trust.npy", "").replace("__", " | ").replace("_", " ")

        ax.plot(data, linewidth=0.7, alpha=0.9, color="purple")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (w=0.5)")
        ax.set_title(label, fontsize=7, pad=4)
        ax.set_xlabel("Training step", fontsize=8)
        ax.set_ylabel("Trust weight (w)", fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)

    # Hide any unused subplot panels
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Bayesian Trust Score Evolution During Training", fontsize=11, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "trust_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trust curves saved → {path}")


# ─── Policy Loss Curves ────────────────────────────────────────────────────────

def plot_loss_curves(curves_dir, output_dir):
    """
    Plot policy gradient loss (PPO) or TD loss (DQN) over training.

    Loss values are only recorded after gradient updates, so the x-axis
    represents the number of completed gradient steps, not environment steps.
    """
    loss_files = sorted([
        f for f in os.listdir(curves_dir)
        if f.endswith("_loss.npy")
    ])

    if not loss_files:
        print("No policy loss files found — skipping loss plot.")
        return

    n    = len(loss_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)

    for i, fname in enumerate(loss_files):
        ax    = axes[i // cols][i % cols]
        data  = np.load(os.path.join(curves_dir, fname))
        label = fname.replace("_loss.npy", "").replace("__", " | ").replace("_", " ")

        ax.plot(data, linewidth=0.7, alpha=0.9, color="steelblue")
        ax.set_title(label, fontsize=7, pad=4)
        ax.set_xlabel("Gradient update", fontsize=8)
        ax.set_ylabel("Policy loss", fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Policy Loss During Training", fontsize=11, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved  → {path}")


# ─── Per-Environment Bar Charts ────────────────────────────────────────────────

def plot_env_comparison(df, output_dir):
    """
    Bar chart comparing mode success rates per environment.

    One figure per environment; bars grouped by human scenario.
    Modes are colour-coded: sparse=grey, naive=orange, bayesian=purple.
    """
    mode_colors = {"sparse": "grey", "naive": "darkorange", "ema": "#2980b9", "bayesian": "purple"}
    envs = df["env_id"].unique()

    for env_id in envs:
        sub    = df[df["env_id"] == env_id]
        humans = sub["human"].unique()
        modes  = [m for m in ["sparse", "naive", "ema", "bayesian"] if m in sub["mode"].unique()]

        # Compute mean ± std over seeds for each (human, mode) pair
        agg = sub.groupby(["human", "mode"])["success_rate"].agg(["mean", "std"]).reset_index()

        x       = np.arange(len(humans))
        width   = 0.8 / len(modes)
        offsets = np.linspace(-(0.4 - width / 2), (0.4 - width / 2), len(modes))

        fig, ax = plt.subplots(figsize=(max(8, len(humans) * 1.5), 5))

        for j, mode in enumerate(modes):
            mode_data = agg[agg["mode"] == mode].set_index("human")
            means = [mode_data.loc[h, "mean"] if h in mode_data.index else 0 for h in humans]
            stds  = [mode_data.loc[h, "std"]  if h in mode_data.index else 0 for h in humans]

            ax.bar(
                x + offsets[j], means,
                width=width, label=mode,
                color=mode_colors.get(mode, "grey"),
                yerr=stds, capsize=3, alpha=0.85
            )

        ax.set_xticks(x)
        ax.set_xticklabels(humans, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Success rate (mean ± std over seeds)", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Success Rate by Mode — {env_id}", fontsize=10)
        ax.legend(title="Mode", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        safe_name = env_id.replace("/", "_").replace("-", "_")
        path = os.path.join(output_dir, f"env_comparison_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Env comparison    → {path}")


# ─── Agent Comparison ─────────────────────────────────────────────────────────

def plot_agent_comparison(df, output_dir):
    """
    Side-by-side bar chart comparing PPO vs DQN success rates,
    averaged over all environments and human scenarios.
    Bars are grouped by mode.
    """
    agents = df["agent"].unique()
    modes  = [m for m in ["sparse", "naive", "bayesian"] if m in df["mode"].unique()]

    agg = (
        df.groupby(["agent", "mode"])["success_rate"]
        .mean()
        .reset_index()
        .pivot(index="mode", columns="agent", values="success_rate")
    )

    mode_order = [m for m in ["sparse", "naive", "bayesian"] if m in agg.index]
    agg = agg.loc[mode_order]

    ax = agg.plot(kind="bar", figsize=(7, 4), colormap="Set2", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mode", fontsize=9)
    ax.set_ylabel("Mean success rate (over all envs & humans)", fontsize=9)
    ax.set_title("Agent Comparison: PPO vs DQN", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Agent", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(output_dir, "agent_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Agent comparison  → {path}")


# ─── BRF vs. Naive vs. Sparse Bar Chart ───────────────────────────────────────

def plot_brf_comparison(df, output_dir, agent="PPO", env_id="MiniGrid-Empty-8x8-v0"):
    """
    Grouped bar chart: BRF vs. Naive vs. Sparse success rate on one environment.
    Bars are grouped by human type; error bars show ± std across seeds.
    Human types are ordered from most to least reliable for readability.
    """
    sub = df[(df["env_id"] == env_id) & (df["agent"] == agent)].copy()
    if sub.empty:
        print(f"No data for {agent} / {env_id} — skipping BRF comparison.")
        return

    sub["mode"] = sub["mode"].replace({"bayesian": "BRF", "ema": "EMA"})

    human_order = [
        h for h in ["StochasticLow", "Realistic", "Fatigue",
                     "DistanceNoise", "StochasticHigh", "Adversarial"]
        if h in sub["human"].unique()
    ]

    mode_order = [m for m in ["BRF", "EMA", "naive", "sparse"] if m in sub["mode"].unique()]
    palette    = {"BRF": "#8e44ad", "EMA": "#2980b9", "naive": "#e67e22", "sparse": "#95a5a6"}

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=sub,
        x="human", y="success_rate", hue="mode",
        order=human_order, hue_order=mode_order,
        palette=palette, errorbar=("ci", 95),
        ax=ax,
    )

    ax.set_xlabel("Human Teacher Type", fontsize=12)
    ax.set_ylabel("Success Rate (mean ± 95% CI)", fontsize=12)
    ax.set_title(
        f"BRF vs. EMA vs. Naive vs. Sparse  —  {env_id}\n({agent})",
        fontsize=13,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(title="Reward Mode", fontsize=10, title_fontsize=10)
    ax.axhline(0, color="black", lw=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    safe_env = env_id.replace("/", "_").replace("-", "_")
    path = os.path.join(output_dir, f"brf_comparison_{agent}_{safe_env}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"BRF comparison chart -> {path}")


# ─── Trust Weight by Human Type ───────────────────────────────────────────────

def plot_trust_by_human(curves_dir, output_dir,
                        agent="PPO", env_id="MiniGrid-Empty-8x8-v0"):
    """
    Single-panel line plot of Bayesian trust weight w over training steps,
    one line per human type, averaged across seeds with ± std shading.

    Expected patterns:
      Adversarial   → collapses toward 0  (always wrong)
      StochasticHigh→ low / drifts down   (80% noise)
      DistanceNoise → moderate            (noise scales with distance)
      Fatigue       → decays over time    (noise grows with steps)
      Realistic     → moderately high     (mixed noise sources)
      StochasticLow → rises toward 1      (10% noise, mostly helpful)
    """
    from collections import defaultdict

    human_colors = {
        "Adversarial":    "#e74c3c",
        "StochasticHigh": "#e67e22",
        "DistanceNoise":  "#d4ac0d",
        "Fatigue":        "#3498db",
        "Realistic":      "#27ae60",
        "StochasticLow":  "#8e44ad",
    }

    prefix = f"{agent}__{env_id}__bayesian__"
    human_curves = defaultdict(list)

    for fname in sorted(os.listdir(curves_dir)):
        if not (fname.endswith("_trust.npy") and fname.startswith(prefix)):
            continue
        rest       = fname[len(prefix):]
        human_name = rest.split("__")[0]
        human_curves[human_name].append(
            np.load(os.path.join(curves_dir, fname))
        )

    if not human_curves:
        print(f"No trust curves found for {agent} / {env_id} in {curves_dir}")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    for human_name, curves in sorted(human_curves.items()):
        color   = human_colors.get(human_name, "gray")
        min_len = min(len(c) for c in curves)
        arr     = np.array([c[:min_len] for c in curves])
        mean    = arr.mean(axis=0)
        std     = arr.std(axis=0)
        steps   = np.arange(min_len)

        ax.plot(steps, mean, label=human_name, color=color, lw=2)
        ax.fill_between(
            steps,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color, alpha=0.12,
        )

    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Neutral prior (w = 0.5)")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Trust weight  w = α / (α + β)", fontsize=12)
    ax.set_title(
        f"Bayesian Trust Weight Evolution by Human Type\n({agent}  ·  {env_id})",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="center right")
    plt.tight_layout()

    path = os.path.join(output_dir, "trust_by_human.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trust-by-human plot -> {path}")


# ─── Sensitivity Analysis Figure ──────────────────────────────────────────────

def plot_sensitivity(sensitivity_root, output_dir):
    """
    Create a multi-panel sensitivity figure from run_sensitivity.py outputs.

    Expects one or more sensitivity_<param>/ subdirectories under sensitivity_root,
    each containing a results.csv produced by run_sensitivity.py.

    For each sweep axis one subplot is created:
      - Bayesian line: mean success rate across human types ± std shaded band
      - Sparse / naive dashed horizontal baselines
      - x-axis: swept parameter value  y-axis: mean success rate
    """
    sweep_dirs = sorted(glob.glob(os.path.join(sensitivity_root, "sensitivity_*")))
    if not sweep_dirs:
        print(f"No sensitivity_* directories found under {sensitivity_root}")
        return

    all_dfs = []
    for d in sweep_dirs:
        csv_path = os.path.join(d, "results.csv")
        if os.path.exists(csv_path):
            all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        print("Found sensitivity_* directories but none contain results.csv — skipping.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    sweep_params = sorted(df["sweep_param"].unique())
    n = len(sweep_params)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    param_labels = {
        "w0":        "Initial trust w₀  (α / (α+β))",
        "N0":        "Prior strength N₀  (α+β)",
        "magnitude": "Human feedback magnitude",
    }
    colors = {"bayesian": "purple", "sparse": "#95a5a6", "naive": "#e67e22"}

    for ax, param in zip(axes, sweep_params):
        sub = df[df["sweep_param"] == param].copy()
        sub["param_value"] = pd.to_numeric(sub["param_value"])

        # Bayesian: mean ± std across human types for each param value
        bayes = (
            sub[sub["mode"] == "bayesian"]
            .groupby("param_value")["success_rate"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("param_value")
        )
        ax.plot(
            bayes["param_value"], bayes["mean"],
            color=colors["bayesian"], lw=2, marker="o", label="Bayesian"
        )
        ax.fill_between(
            bayes["param_value"],
            (bayes["mean"] - bayes["std"]).clip(0),
            (bayes["mean"] + bayes["std"]).clip(0, 1),
            color=colors["bayesian"], alpha=0.15,
        )

        # Baselines: sparse and naive averaged across all param values and humans
        for mode in ["sparse", "naive"]:
            mode_mean = sub[sub["mode"] == mode]["success_rate"].mean()
            ax.axhline(
                mode_mean, color=colors[mode], linestyle="--", lw=1.5,
                label=f"{mode.capitalize()} baseline"
            )

        # Mark the default value with a vertical dotted line
        import config as cfg
        default_val = cfg.SENSITIVITY_DEFAULTS.get(param)
        if default_val is not None:
            ax.axvline(
                default_val, color="black", linestyle=":", lw=1,
                label=f"Default ({default_val})"
            )

        ax.set_xlabel(param_labels.get(param, param), fontsize=11)
        ax.set_ylabel("Mean success rate (± std across human types)", fontsize=10)
        ax.set_title(f"Sensitivity: {param_labels.get(param, param)}", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Bayesian RLHF Hyperparameter Sensitivity\n"
        "(PPO · MiniGrid-Empty-8x8-v0 · seed 42)",
        fontsize=13,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "sensitivity_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sensitivity figure → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Sensitivity mode ───────────────────────────────────────────────────────
    if args.sensitivity_dir:
        output_dir = args.output_dir or args.sensitivity_dir
        os.makedirs(output_dir, exist_ok=True)
        plot_sensitivity(args.sensitivity_dir, output_dir)
        return

    # ── Standard main-experiment mode ─────────────────────────────────────────
    results_dir = args.results_dir
    output_dir  = args.output_dir or results_dir

    os.makedirs(output_dir, exist_ok=True)

    # ── Load results DataFrame ─────────────────────────────────────────────────
    df = load_results(results_dir)
    print(f"Loaded {len(df)} records from {results_dir}/results.csv")

    # ── Summary table ──────────────────────────────────────────────────────────
    table = build_summary_table(df)
    save_summary_table(table, output_dir)

    # ── Curve plots ────────────────────────────────────────────────────────────
    curves_dir = os.path.join(results_dir, "curves")
    if os.path.isdir(curves_dir):
        plot_trust_curves(curves_dir, output_dir)
        plot_trust_by_human(curves_dir, output_dir)
        plot_loss_curves(curves_dir, output_dir)
    else:
        print("No curves/ directory found — skipping curve plots.")

    # ── Statistical significance ────────────────────────────────────────────────
    sig_df = compute_significance(df)
    print_significance_table(sig_df)
    if not sig_df.empty:
        sig_df.to_csv(os.path.join(output_dir, "significance_tests.csv"), index=False)

    # ── Bar charts ─────────────────────────────────────────────────────────────
    plot_env_comparison(df, output_dir)

    for env_id in df["env_id"].unique():
        plot_brf_comparison(df, output_dir, agent="PPO", env_id=env_id)

    save_poster_table(df, output_dir)
    plot_poster_visuals(df, output_dir)
    plot_loss_curves_simplified(results_dir, output_dir)
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
