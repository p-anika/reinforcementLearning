"""
config.py
---------
Single source of truth for all hyperparameters and the experiment matrix.

To add a new environment: add an entry to ENVIRONMENTS.
To add a new agent: add it to AGENTS (and register it in agents/factory.py).
To add a new human model: add an entry to HUMANS.

All other files read from here; no magic numbers should appear elsewhere.
"""

# ─── Environments ─────────────────────────────────────────────────────────────
# Maps env_id → potential function class name (defined in env/potential.py).
# The environment reward is sparse: +1.0 at goal, 0.0 otherwise.
# Human reward is based on potential difference (see env/potential.py for logic).

ENVIRONMENTS = {
    "MiniGrid-Empty-8x8-v0":   "EmptyPotential",
    # Goal: reach green tile in an empty room.
    # Potential: −manhattan_distance(agent, goal)

    "MiniGrid-LavaGapS7-v0": "LavaGapPotential",
    # Goal: reach goal without stepping on lava strip.
    # Potential: −manhattan_distance(agent, goal) − lava_adjacency_penalty

    "MiniGrid-DoorKey-8x8-v0": "DoorKeyPotential",
    # Goal: pick up key, unlock door, reach goal.
    # Potential: staged — distance to key → door → goal
}

# ─── RL Agents ────────────────────────────────────────────────────────────────
# DQN is chosen over SAC because MiniGrid uses Discrete(7) actions — DQN
# handles discrete spaces natively, whereas SAC requires a continuous adapter.

AGENTS = ["PPO"]

# ─── Reward Modes ─────────────────────────────────────────────────────────────
# Four conditions compared in every experiment:
#   sparse   → pure RL baseline (ignores human)
#   naive    → standard RLHF (fully trusts human, w=1)
#   ema      → frequentist baseline: exponential moving average trust
#   bayesian → novel Bayesian trust-weighting (this paper's contribution)

MODES = ["sparse", "naive", "ema", "bayesian"]

# ─── Random Seeds ─────────────────────────────────────────────────────────────
# 5 seeds: community standard for reliable mean ± CI reporting.

SEEDS = [42, 123, 456, 789, 999]

# ─── Training & Evaluation ────────────────────────────────────────────────────

TRAINING = {
    "total_timesteps": 100_000,  # Training steps per agent per condition
    "eval_episodes":   50,       # Evaluation episodes after training
}

# ─── Reward Shaping ───────────────────────────────────────────────────────────

REWARD = {
    "human_magnitude": 0.1,   # Human feedback magnitude (±0.1 by convention)
    "alpha_init":      1.0,   # Beta prior: initial agreement count (uniform prior)
    "beta_init":       1.0,   # Beta prior: initial disagreement count (uniform prior)
    "ema_alpha":       0.01,  # EMA decay rate (α in w_t = (1-α)*w_{t-1} + α*agreement_t)
    "ema_w0":          0.5,   # EMA initial trust — neutral, matching Bayesian prior
}

# ─── Hyperparameter Sensitivity ───────────────────────────────────────────────
# One-at-a-time sweep grids for the three Bayesian-specific hyperparameters.
# Expressed in interpretable terms:
#   w0  = α / (α + β)  — initial trust weight (0 = skeptical, 0.5 = neutral, 1 = trusting)
#   N0  = α + β        — prior strength (how many steps before evidence dominates the prior)
#   magnitude          — maximum human feedback per step (teachers return ±0.1 by convention)

SENSITIVITY = {
    "w0":        [0.1, 0.3, 0.5, 0.7, 0.9],
    "N0":        [1,   2,   5,  10,  20  ],
    "magnitude": [0.02, 0.05, 0.1, 0.2, 0.5],
}

SENSITIVITY_DEFAULTS = {
    "w0":        0.5,    # neutral prior → alpha_init = beta_init = 1.0
    "N0":        2,      # weak prior  → α + β = 2
    "magnitude": 0.1,    # ±0.1 per step, 10% of terminal reward
}


def prior_to_alpha_beta(w0, N0):
    """
    Convert interpretable prior parameters to (alpha_init, beta_init).

    Parameters
    ----------
    w0 : float  initial trust weight in [0, 1]  (w0 = alpha / (alpha + beta))
    N0 : float  prior strength (N0 = alpha + beta)

    Returns
    -------
    (alpha_init, beta_init) : tuple[float, float]
    """
    return w0 * N0, (1.0 - w0) * N0

# ─── Human Teacher Scenarios ──────────────────────────────────────────────────
# Maps display name → (class_name, constructor_kwargs).
# class_name must match a key in humans/teachers.py::_HUMAN_CLASSES.

HUMANS = {
    "StochasticLow":  ("StochasticHuman",    {"noise_level": 0.1}),
    # Mostly helpful: flips advice 10% of the time

    "StochasticHigh": ("StochasticHuman",    {"noise_level": 0.8}),
    # Mostly unreliable: flips advice 80% of the time

    "Adversarial":    ("AdversarialHuman",   {}),
    # Always lies — worst-case attacker / maximally misinformed teacher

    "DistanceNoise":  ("DistanceNoiseHuman", {}),
    # Confusion scales with distance from goal

    "Fatigue":        ("FatigueHuman",       {"start_noise": 0.0, "max_noise": 0.8, "fatigue_rate": 50000}),
    # Noise grows linearly from 0% to 80% over 50,000 steps

    "Realistic":      ("RealisticHuman",     {"base_noise": 0.05}),
    # Combines distance confusion + fatigue + baseline noise (most realistic proxy)
}

# kjdhfjdf is this p
