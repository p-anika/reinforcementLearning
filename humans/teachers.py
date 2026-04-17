"""
humans/teachers.py
------------------
Synthetic human teacher models for the RLHF research framework.

All classes inherit from BaseHuman and implement give_feedback(), which
returns a scalar reward signal (typically ±0.1) based on the agent's
observed progress. This module has no gymnasium or SB3 dependencies —
only numpy and abc.

Teacher hierarchy:
    BaseHuman (abstract)
    ├── StochasticHuman    — fixed-rate noise baseline
    ├── AdversarialHuman   — always lies (worst-case attacker)
    ├── DistanceNoiseHuman — confusion scales with distance from goal
    ├── FatigueHuman       — noise increases over training time
    └── RealisticHuman     — combines distance, fatigue, and base noise

Factory:
    get_human(class_name, **kwargs) — instantiate by string name
"""

import numpy as np
from abc import ABC, abstractmethod


# ─── Abstract Base ────────────────────────────────────────────────────────────

class BaseHuman(ABC):
    """
    Contract for all synthetic human teacher models.
    Every implementation must provide give_feedback() with this exact signature
    so ResearchWrapper can treat all humans uniformly.
    """

    @abstractmethod
    def give_feedback(self, prev_phi, current_phi, step_count):
        """
        Provide a scalar feedback signal based on the agent's progress.

        Parameters
        ----------
        prev_phi    : float  potential at the previous time step
        current_phi : float  potential at the current time step
        step_count  : int    total training steps elapsed (used by fatigue models)

        Returns
        -------
        float : feedback signal (positive = encouragement, negative = discouragement)
        """
        pass


# ─── Concrete Implementations ─────────────────────────────────────────────────

class StochasticHuman(BaseHuman):
    """
    The 'Classic' Noise Model — primary experimental baseline.

    Gives correct advice with probability (1 - noise_level), and
    flips the signal (lies) with probability noise_level.

    noise_level=0.1 → helpful teacher, occasional errors
    noise_level=0.8 → mostly unreliable teacher
    """

    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def give_feedback(self, prev_phi, current_phi, step_count):
        actual_is_good = current_phi > prev_phi
        true_advice = 0.1 if actual_is_good else -0.1
        # Flip the signal with probability noise_level
        if np.random.random() < self.noise_level:
            return -true_advice
        return true_advice


class AdversarialHuman(BaseHuman):
    """
    The 'Malicious' Model — security baseline.

    Always gives the opposite of helpful advice. Represents a worst-case
    attacker or a completely misinformed teacher. The Bayesian filter should
    converge w → 0 under sustained adversarial feedback.
    """

    def give_feedback(self, prev_phi, current_phi, step_count):
        actual_is_good = current_phi > prev_phi
        return -0.1 if actual_is_good else 0.1


class DistanceNoiseHuman(BaseHuman):
    """
    Ablation 1: Distance-Based Noise.

    Noise probability scales with |prev_phi| (a proxy for distance from goal).
    Logic: humans are worse at judging progress when the agent is far away.
    At max distance (~14 cells in an 8×8 grid), dist_factor ≈ 0.9.
    """

    def give_feedback(self, prev_phi, current_phi, step_count):
        delta_phi = current_phi - prev_phi
        # Normalize: max Manhattan distance in an 8×8 grid is ~14
        dist_factor = min(0.9, abs(prev_phi) / 15.0)
        if np.random.random() < dist_factor:
            return np.random.uniform(-0.1, 0.1)  # Random guess when confused
        return 0.1 if delta_phi > 0 else -0.1


class FatigueHuman(BaseHuman):
    """
    Ablation 2: Fatigue-Based Noise.

    Noise increases linearly from start_noise toward max_noise as
    training steps accumulate. Simulates a teacher becoming less attentive
    over long training runs.
    """

    def __init__(self, start_noise=0.0, max_noise=0.8, fatigue_rate=50000):
        self.start_noise = start_noise
        self.max_noise = max_noise
        self.fatigue_rate = fatigue_rate  # Steps to reach max_noise

    def give_feedback(self, prev_phi, current_phi, step_count):
        delta_phi = current_phi - prev_phi
        # Noise grows linearly from start_noise toward max_noise
        current_noise = min(self.max_noise, self.start_noise + (step_count / self.fatigue_rate))
        if np.random.random() < current_noise:
            return np.random.uniform(-0.1, 0.1)
        return 0.1 if delta_phi > 0 else -0.1


class RealisticHuman(BaseHuman):
    """
    The 'Real World' Proxy — multivariable noise model.

    Combines distance confusion, fatigue, and a small constant base noise.
    At high total noise, gives unreliable Normal(0, 0.05) signals rather
    than clean flips — more realistic than a binary correct/wrong model.

    This is intended to approximate real human annotators who degrade
    in multiple correlated ways simultaneously.
    """

    def __init__(self, base_noise=0.05):
        self.base_noise = base_noise

    def give_feedback(self, prev_phi, current_phi, step_count):
        delta_phi = current_phi - prev_phi
        # Accumulate all noise sources
        dist_factor    = abs(prev_phi) / 20.0
        fatigue_factor = step_count / 100000.0
        total_noise_prob = min(0.95, self.base_noise + dist_factor + fatigue_factor)

        if np.random.random() < total_noise_prob:
            # Unreliable signal — not a clean flip, just noise
            return np.random.normal(0, 0.05)
        return 0.1 if delta_phi > 0 else -0.1


# ─── Factory ──────────────────────────────────────────────────────────────────

_HUMAN_CLASSES = {
    "StochasticHuman":    StochasticHuman,
    "AdversarialHuman":   AdversarialHuman,
    "DistanceNoiseHuman": DistanceNoiseHuman,
    "FatigueHuman":       FatigueHuman,
    "RealisticHuman":     RealisticHuman,
}


def get_human(class_name, **kwargs):
    """
    Instantiate a human teacher by class name string.

    Used by run_experiment.py to build teachers from config.HUMANS entries
    without hard-coding if/elif chains.

    Example
    -------
    get_human("StochasticHuman", noise_level=0.8)
    """
    if class_name not in _HUMAN_CLASSES:
        raise ValueError(
            f"Unknown human class '{class_name}'. "
            f"Available: {list(_HUMAN_CLASSES)}"
        )
    return _HUMAN_CLASSES[class_name](**kwargs)
