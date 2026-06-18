"""
reward_filter/moving_average.py
--------------------------------
Exponential Moving Average (EMA) trust filter — frequentist baseline.

At each step where delta_phi != 0 the trust weight updates as:

    w_t = (1 - alpha) * w_{t-1}  +  alpha * agreement_t

where agreement_t = 1 if sign(r_human) == sign(delta_phi), else 0.

This is the implicit trust mechanism in TAMER-style corrective-advice
systems: no prior, all history treated equally once the window is long enough.
BRF is compared against this to validate that the Bayesian structure
(conjugate prior + exact posterior) contributes beyond online agreement tracking.
"""


def fresh_ema_history(w0=0.5):
    """Return initial EMA state dict. w0=0.5 is neutral (matching BRF prior)."""
    return {"w": w0}


def update_ema_trust(history, r_human, delta_phi, alpha=0.01):
    """
    Update EMA trust weight given new feedback and potential change.

    Parameters
    ----------
    history    : dict  {"w": current_trust}
    r_human    : float  human feedback signal
    delta_phi  : float  potential change (objective progress)
    alpha      : float  EMA decay rate (default 0.01)

    Returns
    -------
    (new_w, new_history) : float, dict
    """
    if delta_phi == 0:
        return history["w"], history
    agreement = 1.0 if (r_human > 0) == (delta_phi > 0) else 0.0
    w_new = (1.0 - alpha) * history["w"] + alpha * agreement
    return w_new, {"w": w_new}
