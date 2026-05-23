from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from backend.db.setup import setup_db
from backend.recommender.episode_feature_builder import (
    TONE_DIMENSIONS,
    build_features,
)
from backend.recommender.explanation_builder import explain_all
from backend.recommender.reranker import RankedCandidate


def make_ranked(
    episode_id: str = "ep_0",
    final_score: float = 0.80,
    similarity: float = 0.85,
    mood_alignment: float = 0.60,
    tone_alignment: float = 0.40,
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
) -> RankedCandidate:
    return RankedCandidate(
        episode_id=episode_id,
        final_score=final_score,
        similarity=similarity,
        mood_alignment=mood_alignment,
        tone_alignment=tone_alignment,
        series_slug=series_slug,
        season_number=season,
        episode_number=number,
        episode_title=title,
    )


def make_db_row(
    episode_id: str,
    humor: float = 0.5,
    energy: float = 0.5,
    comfort: float = 0.5,
    sadness: float = 0.5,
    tone: dict | None = None,
) -> dict:
    if tone is None:
        tone = {label: 0.5 for label in TONE_DIMENSIONS}
    return {
        "episode_id": episode_id,
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": json.dumps(tone),
    }


def make_db(tmp_path: Path, rows: list[dict]) -> Path:
    db_path = tmp_path / "app.sqlite3"
    setup_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for r in rows:
            conn.execute(
                "INSERT INTO episodes (episode_id, humor_level, energy_level, "
                "comfort_level, sadness_level, tone_scores) "
                "VALUES (:episode_id, :humor_level, :energy_level, "
                ":comfort_level, :sadness_level, :tone_scores)",
                r,
            )
        conn.commit()
    return db_path


def user_vec_from_mood(
    humor: float = 0.9,
    energy: float = 0.5,
    comfort: float = 0.5,
    sadness: float = 0.1,
    tone: dict | None = None,
) -> np.ndarray:
    """Build a realistic user_vector using the feature builder math."""
    if tone is None:
        tone = {label: 0.5 for label in TONE_DIMENSIONS}
    row = {
        "episode_id": "user",
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": json.dumps(tone),
    }
    return build_features(row).vector


def test_explain_all_count(tmp_path):
    ranked = [make_ranked(f"ep_{i}") for i in range(3)]
    rows = [make_db_row(f"ep_{i}", 0.8, 0.5, 0.6, 0.2) for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    assert len(results) == 3


def test_explain_episode_id_matches(tmp_path):
    ranked = [make_ranked("ep_0"), make_ranked("ep_1")]
    rows = [
        make_db_row("ep_0", 0.8, 0.5, 0.5, 0.1),
        make_db_row("ep_1", 0.2, 0.5, 0.5, 0.8),
    ]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    assert [r.episode_id for r in results] == ["ep_0", "ep_1"]


def test_explain_mood_match_detected(tmp_path):
    """High-humor user + high-humor episode → 'humor_level' in mood_matches."""
    user_vec = user_vec_from_mood(humor=0.95, energy=0.1, comfort=0.1, sadness=0.1)
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", humor=0.9, energy=0.1, comfort=0.1, sadness=0.1)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec, db_path)
    assert "humor_level" in results[0].mood_matches


def test_explain_tone_match_detected(tmp_path):
    """User + episode share a strong 'emotional' tone → 'emotional' in tone_matches."""
    tone = {label: 0.1 for label in TONE_DIMENSIONS}
    tone["emotional"] = 0.95
    user_vec = user_vec_from_mood(tone=tone)
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", 0.5, 0.5, 0.5, 0.5, tone=tone)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec, db_path)
    assert "emotional" in results[0].tone_matches


def test_explain_dominant_traits(tmp_path):
    """Episode with very high humor and low everything else → 'humor_level' dominates."""
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", humor=0.95, energy=0.1, comfort=0.1, sadness=0.1)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    assert "humor_level" in results[0].dominant_episode_traits


def test_explain_score_breakdown_keys(tmp_path):
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", 0.8, 0.5, 0.5, 0.2)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    bd = results[0].score_breakdown
    assert "similarity" in bd
    assert "mood_alignment" in bd
    assert "tone_alignment" in bd
    assert "final_score" in bd


def test_explain_empty_ranked(tmp_path):
    db_path = make_db(tmp_path, [])
    results = explain_all([], user_vec_from_mood(), db_path)
    assert results == []


def test_explain_missing_row_omitted(tmp_path):
    ranked = [make_ranked("ep_0"), make_ranked("ep_ghost")]
    rows = [make_db_row("ep_0", 0.8, 0.5, 0.5, 0.2)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    ids = [r.episode_id for r in results]
    assert "ep_0" in ids
    assert "ep_ghost" not in ids
