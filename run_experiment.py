"""
run_experiment.py
-----------------
CLI entry point for running the full experiment matrix or any subset of it.

Usage
-----
Run everything defined in config.py:
    python run_experiment.py

Run a single agent/env/mode combination (fast smoke test):
    python run_experiment.py --agents PPO --envs MiniGrid-Empty-8x8-v0 \
        --modes bayesian --humans StochasticLow --seeds 42 --timesteps 500

Print the experiment matrix without running anything:
    python run_experiment.py --dry-run

All results are saved to results/<run-name>/ as CSV, JSON, and .npy curve files.
Use analyze_results.py to generate plots from saved results.

Experiment matrix
-----------------
The full matrix is the cartesian product of:
    agents × environments × modes × human_scenarios × seeds

With config.py defaults (2 agents × 3 envs × 3 modes × 6 humans × 3 seeds)
this is 324 runs. Use CLI flags to select any subset.
"""

import argparse
import itertools
import os
import time
from datetime import datetime

import numpy as np

import config
from env.wrappers import make_env, ResearchWrapper
from humans.teachers import get_human
from agents.factory import make_agent
from training.callbacks import ResearchLoggerCallback
from training.evaluator import evaluate_agent, save_results


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RLHF Bayesian trust benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Agents to test (default: all in config.AGENTS). E.g. --agents PPO DQN"
    )
    parser.add_argument(
        "--envs", nargs="+", default=None,
        help="Environment IDs (default: all in config.ENVIRONMENTS). "
             "E.g. --envs MiniGrid-Empty-8x8-v0"
    )
    parser.add_argument(
        "--modes", nargs="+", default=None,
        help="Reward modes (default: all in config.MODES). E.g. --modes sparse bayesian"
    )
    parser.add_argument(
        "--humans", nargs="+", default=None,
        help="Human scenario names from config.HUMANS (default: all). "
             "E.g. --humans StochasticLow Adversarial"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Random seeds (default: all in config.SEEDS). E.g. --seeds 42 123"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Training timesteps per run (default: config.TRAINING['total_timesteps'])"
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Name for this run's results directory (default: current timestamp)"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Root directory for results (default: results/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment matrix without running any training"
    )
    return parser.parse_args()


def build_matrix(args):
    """
    Build the full cartesian product of experiment configurations.
    CLI flags override config.py values when provided.
    """
    agents = args.agents or config.AGENTS
    envs   = args.envs   or list(config.ENVIRONMENTS.keys())
    modes  = args.modes  or config.MODES
    humans = args.humans or list(config.HUMANS.keys())
    seeds  = args.seeds  or config.SEEDS
    return list(itertools.product(agents, envs, modes, humans, seeds))


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    run_name  = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timesteps = args.timesteps or config.TRAINING["total_timesteps"]
    matrix    = build_matrix(args)

    # ── Dry run: print matrix and exit ────────────────────────────────────────
    if args.dry_run:
        print(f"Dry run — {len(matrix)} experiments would run with run_name='{run_name}':\n")
        for i, (agent, env, mode, human, seed) in enumerate(matrix, 1):
            print(f"  [{i:3d}] agent={agent:<4}  env={env:<28}  mode={mode:<9}  human={human:<14}  seed={seed}")
        return

    print(f"Starting run '{run_name}' — {len(matrix)} experiments, {timesteps} steps each\n")

    records = {}  # run_key → result dict
    curves  = {}  # run_key → {"trust": [...], "loss": [...]}

    for i, (agent_name, env_id, mode, human_name, seed) in enumerate(matrix, 1):
        print(f"[{i}/{len(matrix)}] {agent_name} | {env_id} | {mode} | {human_name} | seed={seed}")

        # ── Reproducibility ───────────────────────────────────────────────────
        np.random.seed(seed)

        # ── Build human teacher ───────────────────────────────────────────────
        class_name, kwargs = config.HUMANS[human_name]
        human = get_human(class_name, **kwargs)

        # ── Build wrapped training environment ────────────────────────────────
        base_env   = make_env(env_id)
        train_env  = ResearchWrapper(
            base_env, human, mode=mode, env_id=env_id,
            alpha_init=config.REWARD["alpha_init"],
            beta_init=config.REWARD["beta_init"],
        )

        model, train_env = make_agent(agent_name, train_env, seed)

        # ── Train ─────────────────────────────────────────────────────────────
        callback = ResearchLoggerCallback()
        t_start  = time.time()
        model.learn(total_timesteps=timesteps, callback=callback)
        train_seconds = round(time.time() - t_start, 2)

        # ── Evaluate ──────────────────────────────────────────────────────────
        # Create a fresh human instance to avoid any stateful carry-over
        eval_human = get_human(class_name, **kwargs)
        metrics = evaluate_agent(
            model, env_id, eval_human, mode,
            n_episodes=config.TRAINING["eval_episodes"],
            alpha_init=config.REWARD["alpha_init"],
            beta_init=config.REWARD["beta_init"],
        )

        # ── Record result ─────────────────────────────────────────────────────
        run_key = f"{agent_name}__{env_id}__{mode}__{human_name}__seed{seed}"
        records[run_key] = {
            "agent":         agent_name,
            "env_id":        env_id,
            "mode":          mode,
            "human":         human_name,
            "seed":          seed,
            "success_rate":  metrics["success_rate"],
            "mean_reward":   metrics["mean_reward"],
            "std_reward":    metrics["std_reward"],
            "train_seconds": train_seconds,
        }

        # Store time-series data for plotting
        curves[run_key] = {
            "trust": callback.trust_scores,
            "loss":  callback.policy_losses,
        }

        print(f"  → success={metrics['success_rate']:.2f}  "
              f"mean_reward={metrics['mean_reward']:.3f}  "
              f"time={train_seconds}s\n")

    # ── Save all results ──────────────────────────────────────────────────────
    record_list = list(records.values())
    save_results(record_list, curves, args.output_dir, run_name)

    # ── Print summary pivot table ─────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(record_list)
    print("\n─── SUCCESS RATE TABLE (all seeds) ───────────────────────────────")
    pivot = df.pivot_table(
        index=["agent", "env_id", "human"],
        columns="mode",
        values="success_rate",
        aggfunc="mean",
    )
    print(pivot.to_string())
    print(f"\nFull results at: {args.output_dir}/{run_name}/")


if __name__ == "__main__":
    main()
