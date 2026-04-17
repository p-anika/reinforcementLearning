"""
reward_filter/bayesian.py
-------------------------
Bayesian trust-update logic for the RLHF research framework.

This is the mathematical core of the project. It is completely isolated from
gymnasium, stable-baselines3, and all other project modules so it can be
tested and reasoned about independently.

Key equation:
    w = alpha / (alpha + beta)

where alpha counts agreement steps (human and environment agree) and
beta counts disagreement steps (human contradicts the environment).
"""

import numpy as np


def fresh_history(alpha_init=1.0, beta_init=1.0):
    """
    Return a new Bayesian belief dict initialised to the given Beta prior.

    Starting at alpha=beta=1.0 gives a uniform prior (w=0.5), meaning the
    agent neither trusts nor distrusts the human until evidence accumulates.
    """
    return {"alpha": float(alpha_init), "beta": float(beta_init)}


def update_bayesian_trust(history, rh, delta_phi):
    """
    Beta-Bernoulli conjugate prior update.

    Compares the sign of the human's feedback (rh) against the sign of the
    environment's potential change (delta_phi) to determine agreement.

        Agreement   (signs match)   → increment alpha  (build trust)
        Disagreement (signs differ) → increment beta   (reduce trust)

    We skip the update when delta_phi == 0 because a stationary state
    provides no information about the human's reliability.

    Parameters
    ----------
    history   : dict  {"alpha": float, "beta": float}
    rh        : float  human feedback signal (typically ±0.1)
    delta_phi : float  change in potential φ(s_{t+1}) − φ(s_t)

    Returns
    -------
    w       : float  trust weight in [0, 1] = alpha / (alpha + beta)
    history : dict   updated history (mutated in-place and returned)
    """
    # Only update when the environment actually changed state
    if delta_phi != 0:
        is_agreeing = (rh > 0 and delta_phi > 0) or (rh < 0 and delta_phi < 0)
        if is_agreeing:
            history["alpha"] += 1.0   # Human aligned with reality → trust
        else:
            history["beta"] += 1.0    # Human contradicted reality → distrust

    # Trust weight: expected value of the Beta(alpha, beta) distribution
    w = history["alpha"] / (history["alpha"] + history["beta"])
    return w, history
