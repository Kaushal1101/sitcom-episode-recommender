from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from backend.db.setup import setup_db
from backend.recommender.engine import RecommendationResult, recommend
from backend.recommender.episode_feature_builder import (
    TONE_DIMENSIONS,
    build_features,
)
from backend.recommender.vector_indexer import index_all


def ep(
    episode_id: str = "show_s01_e01",
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
    humor: float | None = 0.9,
    energy: float | None = 0.8,
    comfort: float | None = 0.5,
    sadness: float | None = 0.1,
    tone: dict | None = None,
) -> dict:
    if tone is None:
        tone = {label: 0.5 for label in TONE_DIMENSIONS}
    return {
        "episode_id": episode_id,
        "series_slug": series_slug,
        "season_number": season,
        "episode_number": number,
        "episode_title": title,
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": json.dumps(tone),
    }


def make_db(tmp_path: Path, episodes: list[dict]) -> Path:
    """Create a fresh SQLite DB at tmp_path/app.sqlite3 with the given episode rows."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "app.sqlite3"
    setup_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for episode in episodes:
            conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, series_slug, season_number,
                    episode_number, episode_title,
                    humor_level, energy_level, comfort_level, sadness_level,
                    tone_scores
                )
                VALUES (
                    :episode_id, :series_slug, :season_number,
                    :episode_number, :episode_title,
                    :humor_level, :energy_level, :comfort_level, :sadness_level,
                    :tone_scores
                )
                """,
                episode,
            )
        conn.commit()
    return db_path


def make_indexed(tmp_path: Path, episodes: list[dict]) -> tuple[Path, Path]:
    """Create SQLite DB, index into Chroma, return (db_path, chroma_path)."""
    db_path = make_db(tmp_path, episodes)
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    return db_path, chroma_path


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


def test_engine_returns_result(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 6)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=3)
    assert isinstance(result, RecommendationResult)
    assert len(result.ranked) == 3
    assert len(result.explanations) == 3


def test_engine_top_k_respected(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 8)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=3)
    assert len(result.ranked) <= 3


def test_engine_excluded_ids(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 5)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(
        user_vec,
        db_path,
        chroma_path,
        top_k=10,
        excluded_ids={"show_s01_e01"},
    )
    ids = [r.episode_id for r in result.ranked]
    assert "show_s01_e01" not in ids


def test_engine_empty_collection(tmp_path):
    db_path = make_db(tmp_path, [])
    chroma_path = tmp_path / "chroma"
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=5)
    assert result.ranked == []
    assert result.explanations == []


def test_engine_ranked_and_explanations_aligned(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 5)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=4)
    ranked_ids = [r.episode_id for r in result.ranked]
    explanation_ids = [e.episode_id for e in result.explanations]
    assert ranked_ids == explanation_ids
