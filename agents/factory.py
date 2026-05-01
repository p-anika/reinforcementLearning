"""
agents/factory.py
-----------------
Registry and factory for SB3 reinforcement learning agents.

Adding a new agent is a one-line change to AGENT_REGISTRY — no other
file needs to be modified.
"""

from stable_baselines3 import PPO, DQN


# ─── Agent Registry ───────────────────────────────────────────────────────────

# Maps string name → (SB3 class, extra constructor kwargs).
# DQN is off-policy and needs a replay buffer; PPO is on-policy and does not.
AGENT_REGISTRY = {
    "PPO": (PPO, {}),
    "DQN": (DQN, {"buffer_size": 100_000}),
}


# ─── Factory Function ─────────────────────────────────────────────────────────

def make_agent(agent_name, env, seed):
    """
    Instantiate an SB3 RL agent by name.

    DQN works natively with MiniGrid's Discrete(7) action space, so no
    action-space adapter is needed (unlike the previous SAC setup).

    Parameters
    ----------
    agent_name : str      "PPO" or "DQN" (must be in AGENT_REGISTRY)
    env        : gym.Env  training environment (ResearchWrapper on top)
    seed       : int      random seed for reproducibility

    Returns
    -------
    model : SB3 model (untrained)
    env   : gym.Env — the environment the model was created with (unchanged)

    Raises
    ------
    ValueError if agent_name is not in AGENT_REGISTRY.
    """
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Available: {list(AGENT_REGISTRY)}"
        )

    cls, extra_kwargs = AGENT_REGISTRY[agent_name]
    model = cls("MlpPolicy", env, seed=seed, verbose=0, **extra_kwargs)
    return model, env
