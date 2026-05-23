from __future__ import annotations

import json

import numpy as np
import pytest

from backend.recommender.episode_feature_builder import (
    MOOD_DIMENSIONS,
    TONE_DIMENSIONS,
    VECTOR_DIM,
    build_features,
)

_SENTINEL = object()


def make_row(
    humor: float | None,
    energy: float | None,
    comfort: float | None,
    sadness: float | None,
    tone: dict | None = None,
    tone_scores_str=_SENTINEL,
    episode_id: str = "test_s01_e01",
) -> dict:
    if tone_scores_str is _SENTINEL:
        tone_scores_str = json.dumps(tone) if tone is not None else None
    return {
        "episode_id": episode_id,
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": tone_scores_str,
    }


def full_tone_dict() -> dict:
    """Returns a tone dict with all 13 labels set to plausible non-zero scores."""
    scores = [0.82, 0.31, 0.74, 0.15, 0.44, 0.20, 0.55, 0.60, 0.35, 0.90, 0.18, 0.10, 0.42]
    return dict(zip(TONE_DIMENSIONS, scores))


def make_mood_enriched_dict(
    humor: float,
    energy: float,
    comfort: float,
    sadness: float,
    tone: dict,
    episode_id: str = "test_s01_e01",
) -> dict:
    """Build a mood_enriched.json-shaped dict for parity checks."""
    return {
        "episode_id": episode_id,
        "mood": {
            "humor_level": humor,
            "energy_level": energy,
            "comfort_level": comfort,
            "sadness_level": sadness,
            "tone": [label for label, score in tone.items() if score >= 0.4],
            "raw_scores": {
                "humor_level": {},
                "energy_level": {},
                "comfort_level": {},
                "sadness_level": {},
                "tone": tone,
            },
        },
    }


def make_row_from_mood_enriched(mood_enriched: dict) -> dict:
    mood = mood_enriched["mood"]
    tone = mood["raw_scores"]["tone"]
    return make_row(
        humor=mood["humor_level"],
        energy=mood["energy_level"],
        comfort=mood["comfort_level"],
        sadness=mood["sadness_level"],
        tone=tone,
        episode_id=mood_enriched["episode_id"],
    )


def test_vector_dim():
    row = make_row(humor=0.9, energy=0.8, comfort=0.5, sadness=0.1, tone=full_tone_dict())
    result = build_features(row)
    assert result is not None
    assert len(result.vector) == 17
    assert result.vector.dtype == np.float32


def test_vector_is_unit_length():
    row = make_row(humor=0.9, energy=0.8, comfort=0.5, sadness=0.1, tone=full_tone_dict())
    result = build_features(row)
    assert result is not None
    assert abs(np.linalg.norm(result.vector) - 1.0) < 1e-5


def test_mood_dims_at_indices_0_to_3():
    """Humor is dim 0, sadness is dim 3."""
    tone = {label: 0.0 for label in TONE_DIMENSIONS}
    row_high_humor = make_row(humor=1.0, energy=0.0, comfort=0.0, sadness=0.0, tone=tone)
    row_high_sadness = make_row(humor=0.0, energy=0.0, comfort=0.0, sadness=1.0, tone=tone)
    r1 = build_features(row_high_humor)
    r2 = build_features(row_high_sadness)
    assert r1 is not None
    assert r2 is not None
    assert r1.vector[0] > r1.vector[3]
    assert r2.vector[3] > r2.vector[0]


def test_tone_dims_at_indices_4_to_16():
    mood_zero = make_row(
        humor=0.0,
        energy=0.0,
        comfort=0.0,
        sadness=0.0,
        tone={"awkward": 1.0, **{label: 0.0 for label in TONE_DIMENSIONS if label != "awkward"}},
    )
    result = build_features(mood_zero)
    assert result is not None
    assert result.vector[4] > 0
    assert result.vector[0] == 0.0


def test_skipped_episode_returns_none():
    row = make_row(humor=None, energy=None, comfort=None, sadness=None, tone_scores_str=None)
    result = build_features(row)
    assert result is None


def test_missing_tone_labels_default_to_zero():
    """tone_scores JSON missing some labels — should not raise, missing dims = 0.0."""
    partial_tone = json.dumps({"awkward": 0.8, "chaotic": 0.6})
    row = make_row(
        humor=0.5, energy=0.5, comfort=0.5, sadness=0.5, tone_scores_str=partial_tone
    )
    result = build_features(row)
    assert result is not None
    for i, label in enumerate(TONE_DIMENSIONS):
        if label not in ("awkward", "chaotic"):
            assert result.tone_vec[i] == 0.0


def test_dimension_order_matches_vectorizer():
    """
    Regression guard: MOOD_DIMENSIONS and TONE_DIMENSIONS must stay in sync
    with backend.embedding.episode_vectorizer.
    """
    from backend.embedding.episode_vectorizer import (
        MOOD_DIMENSIONS as OLD_MOOD,
        TONE_DIMENSIONS as OLD_TONE,
    )

    assert MOOD_DIMENSIONS == OLD_MOOD
    assert TONE_DIMENSIONS == OLD_TONE
    assert VECTOR_DIM == len(OLD_MOOD) + len(OLD_TONE)


def test_vector_matches_existing_vectorizer():
    """
    Build a vector using the new feature builder and the old vectorizer from the same
    input data. Results should be identical.
    """
    from backend.embedding.episode_vectorizer import build_episode_vector

    tone_dict = full_tone_dict()
    mood_enriched = make_mood_enriched_dict(
        humor=0.9, energy=0.64, comfort=0.28, sadness=0.21, tone=tone_dict
    )
    row = make_row_from_mood_enriched(mood_enriched)

    old_result = build_episode_vector(mood_enriched)
    new_result = build_features(row)

    assert old_result is not None
    assert new_result is not None
    np.testing.assert_allclose(new_result.vector, old_result["episode_vec"], atol=1e-5)
