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

from training.evaluator import load_results


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from saved experiment results."
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to a run directory, e.g. results/2026-04-16_14-30-00"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save plots (default: same as --results-dir)"
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
    # Reorder columns to put sparse first for easy comparison
    col_order = [c for c in ["sparse", "naive", "bayesian"] if c in table.columns]
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
    mode_colors = {"sparse": "grey", "naive": "darkorange", "bayesian": "purple"}
    envs = df["env_id"].unique()

    for env_id in envs:
        sub    = df[df["env_id"] == env_id]
        humans = sub["human"].unique()
        modes  = [m for m in ["sparse", "naive", "bayesian"] if m in sub["mode"].unique()]

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
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
        plot_loss_curves(curves_dir, output_dir)
    else:
        print("No curves/ directory found — skipping curve plots.")

    # ── Bar charts ─────────────────────────────────────────────────────────────
    plot_env_comparison(df, output_dir)

    if df["agent"].nunique() > 1:
        plot_agent_comparison(df, output_dir)
    else:
        print("Only one agent in results — skipping agent comparison plot.")

    save_poster_table(df, output_dir)
    plot_poster_visuals(df, output_dir)
    plot_loss_curves_simplified(results_dir, output_dir)
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
