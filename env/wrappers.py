"""
env/wrappers.py
---------------
Environment creation and reward-integration wrappers.

make_env()       — build a FlatObsWrapper-wrapped MiniGrid environment
ResearchWrapper  — integrate human feedback into the reward signal

The ResearchWrapper is the core of the research framework. It supports
three reward modes that are compared against each other:

    'sparse'   → Pure RL baseline (w = 0, ignore human entirely)
    'naive'    → Standard RLHF baseline (w = 1, fully trust human)
    'bayesian' → Novel method (w updated via Beta-Bernoulli each step)

Total reward equation: r_total = r_env + (w × r_human)
"""

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper

from reward_filter.bayesian import update_bayesian_trust, fresh_history
from env.potential import get_potential_fn


# ─── Environment Factory ───────────────────────────────────────────────────────

def make_env(env_id):
    """
    Create a MiniGrid environment ready for use with SB3.

    FlatObsWrapper converts the image-based observation (H×W×C tensor)
    to a flat 1D vector so MLP-based policies (PPO, SAC) can consume it
    without a CNN frontend.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    return env


# ─── Research Wrapper ──────────────────────────────────────────────────────────

class ResearchWrapper(gym.Wrapper):
    """
    Core research wrapper that integrates human feedback into the reward signal.

    On each step:
      1. The environment executes the action and returns r_env (sparse: 0 or 1).
      2. The potential function computes Δφ (objective progress signal).
      3. The human teacher returns r_human based on observed progress.
      4. The Bayesian belief (alpha, beta) is updated based on sign agreement
         between r_human and Δφ.
      5. Final reward: r_total = r_env + (w × r_human)

    Bayesian belief persists across episodes within one training run so that
    trust accumulates over the full training history, not just one episode.

    Parameters
    ----------
    env         : gym.Env   wrapped environment (FlatObsWrapper on top)
    human       : BaseHuman  the teacher model to query for feedback
    mode        : str        'sparse' | 'naive' | 'bayesian'
    env_id      : str        used to select the correct potential function
    alpha_init  : float      initial alpha for the Beta prior (default 1.0)
    beta_init   : float      initial beta  for the Beta prior (default 1.0)
    """

    def __init__(self, env, human, mode, env_id, alpha_init=1.0, beta_init=1.0):
        super().__init__(env)
        self.human      = human
        self.mode       = mode
        self.env_id     = env_id
        self.alpha_init = alpha_init
        self.beta_init  = beta_init

        # Select the environment-appropriate potential function
        self.potential_fn = get_potential_fn(env_id)

        # Bayesian belief state — persists across episodes within one training run
        self.history_data = fresh_history(alpha_init, beta_init)

        # State variables reset on each episode
        self.prev_phi            = 0.0
        self.current_w           = 1.0   # Current trust weight (exposed for callback)
        self.total_steps_elapsed = 0     # Cumulative step counter (used by fatigue models)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Compute initial potential from the reset environment state
        self.prev_phi = self.potential_fn.get_potential(self.env.unwrapped)

        # Set starting trust weight based on mode
        if self.mode == "sparse":
            self.current_w = 0.0          # Ignore human entirely
        elif self.mode == "naive":
            self.current_w = 1.0          # Trust human completely
        else:
            # Bayesian: derive w from the accumulated belief state
            alpha = self.history_data.get("alpha", self.alpha_init)
            beta  = self.history_data.get("beta",  self.beta_init)
            self.current_w = alpha / (alpha + beta)

        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        self.total_steps_elapsed += 1

        # Compute the objective progress signal (reality check)
        current_phi = self.potential_fn.get_potential(self.env.unwrapped)
        delta_phi   = current_phi - self.prev_phi

        # Get subjective feedback from the human teacher
        r_human = self.human.give_feedback(
            self.prev_phi, current_phi, self.total_steps_elapsed
        )

        # Update trust weight based on mode
        if self.mode == "bayesian":
            self.current_w, self.history_data = update_bayesian_trust(
                self.history_data, r_human, delta_phi
            )
        elif self.mode == "sparse":
            self.current_w = 0.0
        else:  # naive
            self.current_w = 1.0

        # Total reward: environment reward + trust-weighted human reward
        r_total = r_env + (self.current_w * r_human)

        # Pass raw env reward through info so evaluator can detect success reliably
        info["r_env"] = r_env

        self.prev_phi = current_phi
        return obs, r_total, terminated, truncated, info
