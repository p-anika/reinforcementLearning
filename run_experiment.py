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

Checkpoint / Resume
-------------------
Results are appended to results/<run-name>/results.csv after each run.
If the script is interrupted, re-running with the same --run-name resumes
from where it left off — already-completed (env_id, mode, human, seed) tuples
are skipped automatically.

Experiment matrix
-----------------
The full matrix is the cartesian product of:
    agents x environments x modes x human_scenarios x seeds

With config.py defaults (1 agent x 3 envs x 4 modes x 6 humans x 5 seeds)
this is 360 runs. Use CLI flags to select any subset.
"""

import argparse
import itertools
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
from env.wrappers import make_env, ResearchWrapper
from humans.teachers import get_human
from agents.factory import make_agent
from training.callbacks import ResearchLoggerCallback
from training.evaluator import evaluate_agent, save_results


RESULT_COLUMNS = [
    "agent", "env_id", "mode", "human", "seed",
    "success_rate", "mean_reward", "std_reward", "train_seconds",
]


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RLHF Bayesian trust benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Agents to test (default: all in config.AGENTS). E.g. --agents PPO"
    )
    parser.add_argument(
        "--envs", nargs="+", default=None,
        help="Environment IDs (default: all in config.ENVIRONMENTS). "
             "E.g. --envs MiniGrid-Empty-8x8-v0"
    )
    parser.add_argument(
        "--modes", nargs="+", default=None,
        help="Reward modes (default: all in config.MODES). E.g. --modes sparse bayesian ema"
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
        "--run-name", default="main_run",
        help="Name for this run's results directory (default: main_run). "
             "Use the same name to resume an interrupted run."
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
    """Build the full cartesian product of experiment configurations."""
    agents = args.agents or config.AGENTS
    envs   = args.envs   or list(config.ENVIRONMENTS.keys())
    modes  = args.modes  or config.MODES
    humans = args.humans or list(config.HUMANS.keys())
    seeds  = args.seeds  or config.SEEDS
    return list(itertools.product(agents, envs, modes, humans, seeds))


# ─── Checkpoint helpers ────────────────────────────────────────────────────────

def load_done_keys(checkpoint_path):
    """Return set of (env_id, mode, human, seed) already recorded in CSV."""
    if not os.path.exists(checkpoint_path):
        return set()
    df = pd.read_csv(checkpoint_path)
    return set(zip(df["env_id"], df["mode"], df["human"], df["seed"]))


def init_checkpoint(checkpoint_path):
    """Write CSV header if file doesn't exist yet."""
    if not os.path.exists(checkpoint_path):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(checkpoint_path, index=False)


def append_result(checkpoint_path, record):
    """Append a single result row to the checkpoint CSV."""
    pd.DataFrame([record]).to_csv(checkpoint_path, mode="a", header=False, index=False)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    run_name  = args.run_name
    timesteps = args.timesteps or config.TRAINING["total_timesteps"]
    matrix    = build_matrix(args)

    out_dir         = os.path.join(args.output_dir, run_name)
    checkpoint_path = os.path.join(out_dir, "results.csv")

    # ── Dry run: print matrix and exit ────────────────────────────────────────
    if args.dry_run:
        print(f"Dry run -- {len(matrix)} experiments, run_name='{run_name}':\n")
        for i, (agent, env, mode, human, seed) in enumerate(matrix, 1):
            print(f"  [{i:3d}] agent={agent:<4}  env={env:<28}  mode={mode:<9}  human={human:<14}  seed={seed}")
        return

    os.makedirs(out_dir, exist_ok=True)
    init_checkpoint(checkpoint_path)
    done_keys = load_done_keys(checkpoint_path)

    skipped = sum(1 for _, env, mode, human, seed in matrix
                  if (env, mode, human, seed) in done_keys)
    remaining = len(matrix) - skipped
    print(f"Run '{run_name}': {len(matrix)} total, {skipped} already done, "
          f"{remaining} to run ({timesteps} steps each)\n")

    curves = {}  # run_key -> {"trust": [...], "loss": [...]}

    for i, (agent_name, env_id, mode, human_name, seed) in enumerate(matrix, 1):
        ckpt_key = (env_id, mode, human_name, seed)
        if ckpt_key in done_keys:
            print(f"[{i}/{len(matrix)}] [skip] {agent_name} | {env_id} | {mode} | {human_name} | seed={seed}")
            continue

        print(f"[{i}/{len(matrix)}] {agent_name} | {env_id} | {mode} | {human_name} | seed={seed}")

        np.random.seed(seed)

        class_name, kwargs = config.HUMANS[human_name]
        human = get_human(class_name, **kwargs)

        base_env  = make_env(env_id)
        train_env = ResearchWrapper(
            base_env, human, mode=mode, env_id=env_id,
            alpha_init=config.REWARD["alpha_init"],
            beta_init=config.REWARD["beta_init"],
            ema_alpha=config.REWARD["ema_alpha"],
            ema_w0=config.REWARD["ema_w0"],
        )

        model, train_env = make_agent(agent_name, train_env, seed)

        callback = ResearchLoggerCallback()
        t_start  = time.time()
        model.learn(total_timesteps=timesteps, callback=callback)
        train_seconds = round(time.time() - t_start, 2)

        eval_human = get_human(class_name, **kwargs)
        metrics = evaluate_agent(
            model, env_id, eval_human, mode,
            n_episodes=config.TRAINING["eval_episodes"],
            alpha_init=config.REWARD["alpha_init"],
            beta_init=config.REWARD["beta_init"],
        )

        record = {
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

        append_result(checkpoint_path, record)
        done_keys.add(ckpt_key)

        run_key = f"{agent_name}__{env_id}__{mode}__{human_name}__seed{seed}"
        curves[run_key] = {
            "trust": callback.trust_scores,
            "loss":  callback.policy_losses,
        }

        print(f"  -> success={metrics['success_rate']:.2f}  "
              f"mean_reward={metrics['mean_reward']:.3f}  "
              f"time={train_seconds}s\n")

    # ── Save trust/loss curves (append-friendly dict) ─────────────────────────
    if curves:
        import numpy as np_
        curves_dir = os.path.join(out_dir, "curves")
        os.makedirs(curves_dir, exist_ok=True)
        for run_key, data in curves.items():
            np_.save(os.path.join(curves_dir, f"{run_key}__trust.npy"),
                     np_.array(data["trust"]))
            np_.save(os.path.join(curves_dir, f"{run_key}__loss.npy"),
                     np_.array(data["loss"]))

    # ── Print summary pivot table ─────────────────────────────────────────────
    df = pd.read_csv(checkpoint_path)
    print("\n--- SUCCESS RATE TABLE (mean across seeds) ---")
    pivot = df.pivot_table(
        index=["agent", "env_id", "human"],
        columns="mode",
        values="success_rate",
        aggfunc="mean",
    )
    print(pivot.to_string())
    print(f"\nFull results at: {out_dir}/")


if __name__ == "__main__":
    main()
