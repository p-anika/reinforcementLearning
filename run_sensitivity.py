"""
run_sensitivity.py
------------------
One-at-a-time sensitivity analysis for the three Bayesian RLHF hyperparameters:

    w0        — initial trust weight (α/(α+β)); 0=skeptical, 0.5=neutral, 1=trusting
    N0        — prior strength (α+β); how many steps before evidence dominates the prior
    magnitude — maximum human feedback per step (teachers return ±0.1 by convention)

For each swept value the script trains all three modes (sparse, naive, bayesian)
across every human scenario. This means:
  - sparse/naive lines in the plot are flat (they don't use α/β), serving as sanity checks
  - magnitude sweep shows how all three modes shift as feedback scale changes

Usage
-----
    python run_sensitivity.py --sweep w0
    python run_sensitivity.py --sweep N0
    python run_sensitivity.py --sweep magnitude

    # Smoke test (1 human, fast):
    python run_sensitivity.py --sweep magnitude --humans StochasticLow --timesteps 500

    # Dry run (print matrix, no training):
    python run_sensitivity.py --sweep w0 --dry-run

Results are saved to results/sensitivity_<sweep>/  as results.csv and results.json.
Use analyze_results.py --sensitivity-dir results/ to generate the sensitivity figure.

Fixed setup
-----------
Agent      : PPO
Environment: MiniGrid-Empty-8x8-v0 (where Bayesian advantage is clearest)
Seed       : 42
"""

import argparse
import json
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
from training.evaluator import evaluate_agent

ENV_ID = "MiniGrid-Empty-8x8-v0"
AGENT  = "PPO"
SEED   = 42


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for Bayesian RLHF hyperparameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep", required=True, choices=["w0", "N0", "magnitude"],
        help="Which hyperparameter axis to sweep."
    )
    parser.add_argument(
        "--humans", nargs="+", default=None,
        help="Human scenarios to include (default: all in config.HUMANS)."
    )
    parser.add_argument(
        "--timesteps", type=int, default=config.TRAINING["total_timesteps"],
        help="Training timesteps per run (default: config.TRAINING['total_timesteps'])."
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Root directory for results (default: results/)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment matrix without running any training."
    )
    return parser.parse_args()


# ─── Parameter Resolution ─────────────────────────────────────────────────────

def resolve_params(sweep, param_val):
    """
    Return (alpha_init, beta_init, human_magnitude) for a given sweep axis and value.
    The two axes not being swept are fixed at their defaults.
    """
    d = config.SENSITIVITY_DEFAULTS
    if sweep == "w0":
        alpha, beta = config.prior_to_alpha_beta(param_val, d["N0"])
        magnitude   = d["magnitude"]
    elif sweep == "N0":
        alpha, beta = config.prior_to_alpha_beta(d["w0"], param_val)
        magnitude   = d["magnitude"]
    else:  # magnitude
        alpha, beta = config.prior_to_alpha_beta(d["w0"], d["N0"])
        magnitude   = param_val
    return alpha, beta, magnitude


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    sweep      = args.sweep
    humans     = args.humans or list(config.HUMANS.keys())
    timesteps  = args.timesteps
    param_vals = config.SENSITIVITY[sweep]
    modes      = config.MODES   # sparse, naive, bayesian

    # Build full matrix: param_val × mode × human
    matrix = [(pv, mode, h) for pv in param_vals for mode in modes for h in humans]

    out_dir = os.path.join(args.output_dir, f"sensitivity_{sweep}")

    if args.dry_run:
        print(f"Dry run -- {len(matrix)} experiments for sweep='{sweep}' -> {out_dir}\n")
        for i, (pv, mode, human) in enumerate(matrix, 1):
            print(f"  [{i:3d}] {sweep}={pv:<6}  mode={mode:<9}  human={human}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Sensitivity sweep: {sweep}  ({len(matrix)} runs, {timesteps} steps each)\n")

    records = []
    np.random.seed(SEED)

    for i, (param_val, mode, human_name) in enumerate(matrix, 1):
        alpha_init, beta_init, magnitude = resolve_params(sweep, param_val)
        print(
            f"[{i}/{len(matrix)}] {sweep}={param_val}  mode={mode:<9}  human={human_name}"
        )

        class_name, kwargs = config.HUMANS[human_name]
        human = get_human(class_name, **kwargs)

        base_env  = make_env(ENV_ID)
        train_env = ResearchWrapper(
            base_env, human, mode=mode, env_id=ENV_ID,
            alpha_init=alpha_init, beta_init=beta_init,
            human_magnitude=magnitude,
        )
        model, train_env = make_agent(AGENT, train_env, SEED)

        callback = ResearchLoggerCallback()
        t_start  = time.time()
        model.learn(total_timesteps=timesteps, callback=callback)
        train_seconds = round(time.time() - t_start, 2)

        eval_human = get_human(class_name, **kwargs)
        metrics = evaluate_agent(
            model, ENV_ID, eval_human, mode,
            n_episodes=config.TRAINING["eval_episodes"],
            alpha_init=alpha_init, beta_init=beta_init,
        )

        records.append({
            "sweep_param":   sweep,
            "param_value":   param_val,
            "mode":          mode,
            "human":         human_name,
            "success_rate":  metrics["success_rate"],
            "mean_reward":   metrics["mean_reward"],
            "std_reward":    metrics["std_reward"],
            "train_seconds": train_seconds,
        })

        print(f"  → success={metrics['success_rate']:.2f}  time={train_seconds}s\n")

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"Results saved to: {out_dir}/")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n─── BAYESIAN SUCCESS RATE BY PARAMETER VALUE ────────────────────────")
    bayes = df[df["mode"] == "bayesian"].groupby("param_value")["success_rate"].mean()
    print(bayes.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
