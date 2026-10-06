"""Unit tests for the trust filters in reward_filter/ and the prior helper in config.py."""

import pytest

from config import prior_to_alpha_beta
from reward_filter.bayesian import fresh_history, update_bayesian_trust
from reward_filter.moving_average import fresh_ema_history, update_ema_trust


def test_uniform_prior_gives_neutral_trust():
    history = fresh_history()
    w, _ = update_bayesian_trust(history, rh=0.1, delta_phi=0.0)
    assert w == pytest.approx(0.5)


def test_no_update_when_potential_is_unchanged():
    history = fresh_history()
    update_bayesian_trust(history, rh=0.1, delta_phi=0.0)
    assert history == {"alpha": 1.0, "beta": 1.0}


def test_agreement_raises_trust_and_disagreement_lowers_it():
    history = fresh_history()
    w_up, _ = update_bayesian_trust(history, rh=0.1, delta_phi=1.0)
    assert w_up == pytest.approx(2 / 3)  # Beta(2, 1)

    history = fresh_history()
    w_down, _ = update_bayesian_trust(history, rh=0.1, delta_phi=-1.0)
    assert w_down == pytest.approx(1 / 3)  # Beta(1, 2)


def test_negative_feedback_agrees_with_negative_progress():
    history = fresh_history()
    w, _ = update_bayesian_trust(history, rh=-0.1, delta_phi=-1.0)
    assert w == pytest.approx(2 / 3)


def test_always_wrong_teacher_drives_trust_toward_zero():
    history = fresh_history()
    for _ in range(100):
        w, history = update_bayesian_trust(history, rh=0.1, delta_phi=-1.0)
    assert w < 0.02


def test_always_right_teacher_drives_trust_toward_one():
    history = fresh_history()
    for _ in range(100):
        w, history = update_bayesian_trust(history, rh=0.1, delta_phi=1.0)
    assert w > 0.98


def test_trust_converges_to_teacher_reliability():
    history = fresh_history()
    for _ in range(900):
        update_bayesian_trust(history, rh=0.1, delta_phi=1.0)
    for _ in range(100):
        w, history = update_bayesian_trust(history, rh=0.1, delta_phi=-1.0)
    assert w == pytest.approx(0.9, abs=0.01)


def test_posterior_adapts_slowly_to_a_teacher_that_degrades():
    # Documents the stationarity limitation discussed in the paper:
    # after a long reliable phase, 200 consecutive wrong signals barely move w.
    history = fresh_history()
    for _ in range(1000):
        update_bayesian_trust(history, rh=0.1, delta_phi=1.0)
    for _ in range(200):
        w, history = update_bayesian_trust(history, rh=0.1, delta_phi=-1.0)
    assert w > 0.8


def test_ema_ignores_steps_with_no_progress():
    history = fresh_ema_history(w0=0.5)
    w, _ = update_ema_trust(history, r_human=0.1, delta_phi=0.0)
    assert w == 0.5


def test_ema_moves_toward_agreement_and_disagreement():
    w_up, _ = update_ema_trust(fresh_ema_history(0.5), 0.1, 1.0, alpha=0.01)
    w_down, _ = update_ema_trust(fresh_ema_history(0.5), 0.1, -1.0, alpha=0.01)
    assert w_up == pytest.approx(0.505)
    assert w_down == pytest.approx(0.495)


def test_prior_to_alpha_beta():
    assert prior_to_alpha_beta(0.5, 2) == pytest.approx((1.0, 1.0))
    assert prior_to_alpha_beta(0.9, 10) == pytest.approx((9.0, 1.0))