"""
agents/factory.py
-----------------
Registry and factory for SB3 reinforcement learning agents.

Adding a new agent is a one-line change to AGENT_REGISTRY — no other
file needs to be modified.

SAC compatibility note
----------------------
SB3's SAC requires a continuous Box action space. MiniGrid uses Discrete(7).
make_agent() detects this mismatch and automatically wraps the environment
with DiscreteToBoxWrapper (defined in env/wrappers.py) when SAC is requested.
The wrapper converts SAC's continuous output to a discrete action via argmax.

Returns
-------
make_agent() returns (model, env) so the caller always has a reference to the
(possibly re-wrapped) environment. Use the returned env for both training and
when constructing the evaluation environment for SAC.
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC

from env.wrappers import DiscreteToBoxWrapper


# ─── Agent Registry ───────────────────────────────────────────────────────────

# Maps string name → SB3 algorithm class
# Add new agents here; nothing else needs to change.
AGENT_REGISTRY = {
    "PPO": PPO,
    "SAC": SAC,
}


# ─── Factory Function ─────────────────────────────────────────────────────────

def make_agent(agent_name, env, seed):
    """
    Instantiate an SB3 RL agent by name.

    For SAC with a Discrete action space: wraps env with DiscreteToBoxWrapper
    before creating the model so SAC sees a compatible Box action space.

    Parameters
    ----------
    agent_name : str      "PPO" or "SAC" (must be in AGENT_REGISTRY)
    env        : gym.Env  training environment (ResearchWrapper on top)
    seed       : int      random seed for reproducibility

    Returns
    -------
    model : SB3 model (untrained)
    env   : gym.Env — the environment the model was created with
            (wrapped with DiscreteToBoxWrapper for SAC; unchanged for PPO)

    Raises
    ------
    ValueError if agent_name is not in AGENT_REGISTRY.
    """
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Available: {list(AGENT_REGISTRY)}"
        )

    # SAC requires a continuous Box action space; apply the adapter if needed
    if agent_name == "SAC" and isinstance(env.action_space, gym.spaces.Discrete):
        env = DiscreteToBoxWrapper(env)

    cls = AGENT_REGISTRY[agent_name]
    model = cls("MlpPolicy", env, seed=seed, verbose=0)
    return model, env
