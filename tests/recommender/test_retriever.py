from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from backend.db.setup import setup_db
from backend.recommender.episode_feature_builder import (
    TONE_DIMENSIONS,
    VECTOR_DIM,
    build_features,
)
from backend.recommender.retriever import Candidate, retrieve
from backend.recommender.vector_indexer import index_all


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


def skipped_ep(episode_id: str = "show_s01_e99") -> dict:
    return {
        "episode_id": episode_id,
        "series_slug": "the_office",
        "season_number": 1,
        "episode_number": 99,
        "episode_title": "Skipped",
        "humor_level": None,
        "energy_level": None,
        "comfort_level": None,
        "sadness_level": None,
        "tone_scores": None,
    }


def make_indexed_chroma(tmp_path: Path, episodes: list[dict]) -> tuple[Path, Path]:
    """Create SQLite DB, index into Chroma, return (db_path, chroma_path)."""
    db_path = make_db(tmp_path / "db", episodes)
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    return db_path, chroma_path


def _unit_query_vector() -> np.ndarray:
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    return user_vec


def test_retrieve_returns_top_k(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 6)]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    results = retrieve(_unit_query_vector(), chroma_path, top_k=3)
    assert len(results) == 3


def test_retrieve_sorted_by_similarity(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 6)]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    results = retrieve(_unit_query_vector(), chroma_path, top_k=5)
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_retrieve_similarity_in_range(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 4)]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    results = retrieve(_unit_query_vector(), chroma_path, top_k=3)
    for r in results:
        assert 0.0 <= r.similarity <= 1.0


def test_retrieve_candidate_fields(tmp_path):
    episode = ep("show_s02_e04", season=2, number=4, title="The Fire")
    _, chroma_path = make_indexed_chroma(tmp_path, [episode])
    results = retrieve(_unit_query_vector(), chroma_path, top_k=1)
    assert len(results) == 1
    c = results[0]
    assert isinstance(c, Candidate)
    assert c.episode_id == "show_s02_e04"
    assert c.series_slug == "the_office"
    assert c.season_number == 2
    assert c.episode_number == 4
    assert c.episode_title == "The Fire"
    assert isinstance(c.similarity, float)


def test_retrieve_exact_match_ranks_first(tmp_path):
    """Querying with an episode's own vector should return it as the top result."""
    target = ep("show_s01_e01", humor=0.9, energy=0.8, comfort=0.5, sadness=0.1)
    others = [
        ep(f"show_s01_e0{i}", number=i, humor=0.1, energy=0.1) for i in range(2, 5)
    ]
    _, chroma_path = make_indexed_chroma(tmp_path, [target] + others)

    query_vec = build_features(target).vector

    results = retrieve(query_vec, chroma_path, top_k=4)
    assert results[0].episode_id == "show_s01_e01"
    assert results[0].similarity > 0.99


def test_retrieve_series_slug_filter(tmp_path):
    episodes = [
        ep("office_s01_e01", series_slug="the_office"),
        ep("office_s01_e02", series_slug="the_office", number=2),
        ep("friends_s01_e01", series_slug="friends"),
    ]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    results = retrieve(
        _unit_query_vector(), chroma_path, top_k=10, series_slug="the_office"
    )
    assert len(results) == 2
    assert all(r.series_slug == "the_office" for r in results)


def test_retrieve_top_k_larger_than_collection(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 4)]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    results = retrieve(_unit_query_vector(), chroma_path, top_k=100)
    assert len(results) == 3


def test_retrieve_empty_collection(tmp_path):
    chroma_path = tmp_path / "chroma"
    results = retrieve(_unit_query_vector(), chroma_path, top_k=10)
    assert results == []
