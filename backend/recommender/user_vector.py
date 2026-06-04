from __future__ import annotations

import math

import numpy as np

from backend.models.session_state import SessionState
from backend.recommender.episode_feature_builder import (
    MOOD_DIMENSIONS,
    MOOD_WEIGHT,
    TONE_DIMENSIONS,
    TONE_WEIGHT,
)

# The user vector layout is [humor, energy, comfort, sadness, *TONE_DIMENSIONS] —
# the mood slice must match MOOD_DIMENSIONS in episode_feature_builder.py with the
# "_level" suffix stripped, in the same order. If MOOD_DIMENSIONS ever changes,
# build_user_vector must be updated to match.
_EXPECTED_MOOD_ORDER: tuple[str, ...] = (
    "humor_level",
    "energy_level",
    "comfort_level",
    "sadness_level",
)
assert tuple(MOOD_DIMENSIONS) == _EXPECTED_MOOD_ORDER, (
    "MOOD_DIMENSIONS ordering changed; update build_user_vector to match."
)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def build_user_vector(state: SessionState) -> np.ndarray:
    """
    Convert SessionState.current_preferences into a 17-dim float32 vector
    aligned with the episode mood vectors stored in ChromaDB.

    Layout:
      [0]    humor    (mood.humor)
      [1]    energy   (mood.energy)
      [2]    comfort  (mood.comfort)
      [3]    sadness  (mood.sadness)
      [4..16] tone    (TONE_DIMENSIONS order)

    Normalization mirrors build_features() in episode_feature_builder.py so that
    cosine similarity against episode vectors is well-defined:
      L2(mood) * sqrt(MOOD_WEIGHT)  ⊕  L2(tone) * sqrt(TONE_WEIGHT)  →  L2(.)

    A zero-preference state produces a zero vector (no normalization division).
    The session state is not mutated.
    """
    mood = state.current_preferences.mood
    mood_vec = np.array(
        [mood.humor, mood.energy, mood.comfort, mood.sadness],
        dtype=np.float32,
    )

    tone_prefs = state.current_preferences.tone_preferences
    tone_vec = np.array(
        [float(tone_prefs.get(label, 0.0)) for label in TONE_DIMENSIONS],
        dtype=np.float32,
    )

    mood_unit = _l2_normalize(mood_vec)
    tone_unit = _l2_normalize(tone_vec)

    combined = np.concatenate(
        [
            math.sqrt(MOOD_WEIGHT) * mood_unit,
            math.sqrt(TONE_WEIGHT) * tone_unit,
        ]
    )
    return _l2_normalize(combined).astype(np.float32)
