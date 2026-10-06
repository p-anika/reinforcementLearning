"""
training/evaluator.py
---------------------
Post-training evaluation.

Success detection: MiniGrid ends an episode (terminated=True) when the agent
reaches the goal or, in lava environments, steps on lava. ResearchWrapper
passes the raw environment reward through info["r_env"]; r_env > 0 means the
goal was reached. The goal reward is 1 - 0.9 * steps / max_steps, so it is
never exactly 1.0 but is always positive. Lava deaths give r_env = 0, and
timeouts have terminated=False.
"""

import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper

from env.wrappers import ResearchWrapper


def evaluate_agent(model, env_id, human, mode, n_episodes, alpha_init, beta_init):
    """
    Run n_episodes deterministic rollouts with a trained model.

    A fresh environment with the same wrapper stack as training is built so
    no state leaks from the training environment.

    Returns
    -------
    dict with
        success_rate : fraction of episodes that reached the goal
        mean_reward  : mean episode return (modified reward)
        std_reward   : standard deviation of episode returns
    """
    eval_env = FlatObsWrapper(gym.make(env_id, render_mode="rgb_array"))
    eval_env = ResearchWrapper(
        eval_env, human, mode=mode, env_id=env_id,
        alpha_init=alpha_init, beta_init=beta_init,
    )

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

            if terminated and info.get("r_env", 0.0) > 0.0:
                successes += 1

            done = terminated or truncated

        episode_rewards.append(ep_reward)

    return {
        "success_rate": successes / n_episodes,
        "mean_reward":  float(np.mean(episode_rewards)),
        "std_reward":   float(np.std(episode_rewards)),
    }