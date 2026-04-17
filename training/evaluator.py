"""
training/evaluator.py
---------------------
Post-training evaluation and results persistence.

evaluate_agent()  — run N deterministic episodes, return success metrics
save_results()    — write results CSV + JSON + curve .npy files to disk
load_results()    — read a saved results CSV back as a DataFrame

Success detection
-----------------
MiniGrid terminates the episode (terminated=True) when the agent reaches the
goal OR, in lava environments, when the agent steps on lava. We distinguish
these cases using info["r_env"] (the raw environment reward passed through
by ResearchWrapper.step), where r_env >= 1.0 indicates a genuine goal reach.
"""

import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper

from env.wrappers import ResearchWrapper, DiscreteToBoxWrapper


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_agent(model, env_id, human, mode, n_episodes, alpha_init, beta_init, agent_name="PPO"):
    """
    Run n_episodes deterministic rollouts with a trained model.

    Creates a fresh environment independent of the training environment
    to avoid any state leakage. Replicates all wrappers from training,
    including DiscreteToBoxWrapper for SAC.

    Success criterion: terminated=True AND info["r_env"] > 0.0
    (i.e. the agent reached the goal, not just timed out or died to lava).
    MiniGrid's goal reward is 1 - 0.9*(steps/max_steps), so it is never
    exactly 1.0 — but it is always > 0.0 when the goal is reached.

    Parameters
    ----------
    model       : trained SB3 model
    env_id      : str     environment ID
    human       : BaseHuman  teacher for evaluation (fresh instance recommended)
    mode        : str     'sparse' | 'naive' | 'bayesian'
    n_episodes  : int     number of evaluation episodes
    alpha_init  : float   Bayesian prior alpha
    beta_init   : float   Bayesian prior beta
    agent_name  : str     "PPO" or "SAC" — determines whether DiscreteToBoxWrapper is applied

    Returns
    -------
    dict with keys:
        success_rate : float  fraction of episodes that reached the goal
        mean_reward  : float  mean episode return (modified reward)
        std_reward   : float  std dev of episode returns
    """
    # Build evaluation environment with the same wrapper stack as training
    eval_env = FlatObsWrapper(gym.make(env_id, render_mode="rgb_array"))
    eval_env = ResearchWrapper(
        eval_env, human, mode=mode, env_id=env_id,
        alpha_init=alpha_init, beta_init=beta_init
    )
    # Apply action-space adapter for SAC (must match the training wrapper stack)
    if agent_name == "SAC":
        eval_env = DiscreteToBoxWrapper(eval_env)

    episode_rewards = []
    successes = 0

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward

            # Success: episode ended by reaching the goal (r_env > 0).
            # MiniGrid's goal reward is step-penalised: 1 - 0.9*(steps/max_steps),
            # so it is never exactly 1.0 — but it is always > 0.
            # Lava death gives r_env = 0. Timeout gives truncated=True, terminated=False.
            if terminated and info.get("r_env", 0.0) > 0.0:
                successes += 1

            done = terminated or truncated

        episode_rewards.append(ep_reward)

    return {
        "success_rate": successes / n_episodes,
        "mean_reward":  float(np.mean(episode_rewards)),
        "std_reward":   float(np.std(episode_rewards)),
    }


# ─── Results I/O ──────────────────────────────────────────────────────────────

def save_results(records, curves, output_dir, run_name):
    """
    Persist all experiment results to disk under <output_dir>/<run_name>/.

    Creates:
        results.csv   — flat table, one row per (agent, env, mode, human, seed)
        results.json  — same data in JSON for full programmatic access
        curves/       — per-run .npy arrays for trust scores and policy losses

    CSV schema:
        agent | env_id | mode | human | seed | success_rate |
        mean_reward | std_reward | train_seconds

    Parameters
    ----------
    records    : list[dict]  one dict per experiment run
    curves     : dict        run_key → {"trust": list[float], "loss": list[float|None]}
    output_dir : str         root directory for results (e.g. "results")
    run_name   : str         subdirectory name for this run (e.g. timestamp)
    """
    run_dir    = os.path.join(output_dir, run_name)
    curves_dir = os.path.join(run_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    # ── Flat CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    # ── JSON (full records list) ───────────────────────────────────────────────
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(records, f, indent=2)

    # ── Time-series curves as numpy arrays ────────────────────────────────────
    for key, data in curves.items():
        # Sanitise key for use as a filename
        safe_key = key.replace("/", "_").replace(" ", "_")

        # Trust scores are always present
        np.save(
            os.path.join(curves_dir, f"{safe_key}_trust.npy"),
            np.array(data["trust"], dtype=np.float32)
        )

        # Policy losses may have None gaps (between rollouts) — drop before saving
        loss_data = [v for v in data["loss"] if v is not None]
        if loss_data:
            np.save(
                os.path.join(curves_dir, f"{safe_key}_loss.npy"),
                np.array(loss_data, dtype=np.float32)
            )

    print(f"Results saved to: {run_dir}/")


def load_results(run_dir):
    """
    Load a previously saved results CSV as a DataFrame.

    Parameters
    ----------
    run_dir : str  full path to the run directory (e.g. "results/2026-04-16_14-30-00")

    Returns
    -------
    pd.DataFrame
    """
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results.csv found at: {path}")
    return pd.read_csv(path)
