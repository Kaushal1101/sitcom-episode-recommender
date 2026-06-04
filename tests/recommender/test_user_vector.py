from __future__ import annotations

import math

import numpy as np
import pytest

from backend.models.session_state import SessionState
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS
from backend.recommender.user_vector import build_user_vector


def test_output_shape_and_dtype():
    state = SessionState()
    state.current_preferences.mood.humor = 0.5
    vec = build_user_vector(state)
    assert vec.shape == (17,)
    assert vec.dtype == np.float32


def test_zero_preferences_returns_zero_vector():
    state = SessionState()
    vec = build_user_vector(state)
    assert vec.shape == (17,)
    assert np.allclose(vec, 0.0)


@pytest.mark.parametrize(
    "field,position",
    [
        ("humor", 0),
        ("energy", 1),
        ("comfort", 2),
        ("sadness", 3),
    ],
)
def test_mood_dimension_lands_at_expected_position(field: str, position: int):
    state = SessionState()
    setattr(state.current_preferences.mood, field, 1.0)
    vec = build_user_vector(state)

    # The chosen mood position must be nonzero.
    assert vec[position] != 0.0

    # All other positions in the mood slice (0..3) must be zero.
    for i in range(4):
        if i != position:
            assert vec[i] == 0.0, f"mood slice position {i} should be 0 when only {field} is set"


@pytest.mark.parametrize("idx", range(len(TONE_DIMENSIONS)))
def test_tone_dimension_lands_at_expected_position(idx: int):
    tone_label = TONE_DIMENSIONS[idx]
    state = SessionState()
    state.current_preferences.tone_preferences[tone_label] = 1.0
    vec = build_user_vector(state)

    target = 4 + idx
    assert vec[target] != 0.0

    # All other tone positions (4..16, except target) must be zero.
    for i in range(4, 17):
        if i != target:
            assert vec[i] == 0.0, (
                f"tone slice position {i} should be 0 when only {tone_label} is set"
            )


def test_l2_norm_with_mixed_preferences():
    state = SessionState()
    state.current_preferences.mood.humor = 0.8
    state.current_preferences.mood.energy = 0.4
    state.current_preferences.mood.comfort = 0.2
    state.current_preferences.tone_preferences["lighthearted"] = 0.9
    state.current_preferences.tone_preferences["cringe"] = 0.5
    state.current_preferences.tone_preferences["wholesome"] = 0.3

    vec = build_user_vector(state)
    norm = float(np.linalg.norm(vec))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)
