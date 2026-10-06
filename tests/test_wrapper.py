"""Integration test: an adversarial teacher should drive BRF's trust weight down."""

import numpy as np
import pytest

pytest.importorskip("minigrid")

from env.wrappers import ResearchWrapper, make_env  # noqa: E402
from humans.teachers import AdversarialHuman  # noqa: E402

ENV_ID = "MiniGrid-Empty-8x8-v0"


def _run_random_steps(mode, n_steps=300):
    np.random.seed(0)
    env = ResearchWrapper(make_env(ENV_ID), AdversarialHuman(), mode=mode, env_id=ENV_ID)
    env.reset(seed=0)
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    return env


def test_bayesian_trust_drops_under_adversarial_teacher():
    assert _run_random_steps("bayesian").current_w < 0.2


def test_sparse_ignores_and_naive_fully_trusts_the_teacher():
    assert _run_random_steps("sparse", n_steps=50).current_w == 0.0
    assert _run_random_steps("naive", n_steps=50).current_w == 1.0