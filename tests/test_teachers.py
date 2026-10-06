"""Unit tests for the synthetic teachers in humans/teachers.py."""

import numpy as np
import pytest

import config
from humans.teachers import AdversarialHuman, StochasticHuman, get_human


def test_adversarial_teacher_always_contradicts_progress():
    teacher = AdversarialHuman()
    assert teacher.give_feedback(-5, -4, step_count=0) == -0.1  # progress -> discourages
    assert teacher.give_feedback(-4, -5, step_count=0) == 0.1   # regress  -> encourages


def test_stochastic_teacher_with_no_noise_is_always_correct():
    teacher = StochasticHuman(noise_level=0.0)
    assert all(teacher.give_feedback(-5, -4, 0) == 0.1 for _ in range(100))
    assert all(teacher.give_feedback(-4, -5, 0) == -0.1 for _ in range(100))


def test_stochastic_teacher_with_full_noise_always_flips():
    teacher = StochasticHuman(noise_level=1.0)
    assert all(teacher.give_feedback(-5, -4, 0) == -0.1 for _ in range(100))


def test_stochastic_teacher_flip_rate_matches_noise_level():
    np.random.seed(0)
    teacher = StochasticHuman(noise_level=0.1)
    flips = sum(teacher.give_feedback(-5, -4, 0) < 0 for _ in range(5000))
    assert flips / 5000 == pytest.approx(0.1, abs=0.02)


def test_every_teacher_in_config_can_be_built():
    for name, (class_name, kwargs) in config.HUMANS.items():
        assert get_human(class_name, **kwargs) is not None, name


def test_unknown_teacher_raises():
    with pytest.raises(ValueError):
        get_human("NotATeacher")