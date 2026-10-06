"""
agents/factory.py
-----------------
Builds the Stable-Baselines3 agent used in the experiments.

Only PPO is used. BRF modifies rewards at collection time using the current
trust weight, which requires an on-policy learner (see the paper, Section 4.3).
"""

from stable_baselines3 import PPO

AGENT_REGISTRY = {"PPO": PPO}


def make_agent(agent_name, env, seed):
    """Return (untrained SB3 model, env) for agent_name ("PPO")."""
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Available: {list(AGENT_REGISTRY)}"
        )
    model = AGENT_REGISTRY[agent_name]("MlpPolicy", env, seed=seed, verbose=0)
    return model, env