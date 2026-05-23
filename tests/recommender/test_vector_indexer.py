from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from backend.db.setup import setup_db
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS, build_features
from backend.recommender.vector_indexer import (
    COLLECTION_NAME,
    IndexResult,
    get_collection,
    index_all,
    index_episode,
)


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


def test_index_single_episode(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s01_e01")])
    chroma_path = tmp_path / "chroma"
    ok = index_episode("show_s01_e01", db_path, chroma_path)
    assert ok is True
    col = get_collection(chroma_path)
    assert col.count() == 1
    assert col.get(ids=["show_s01_e01"])["ids"] == ["show_s01_e01"]


def test_index_skipped_returns_false(tmp_path):
    db_path = make_db(tmp_path, [skipped_ep("show_s01_e99")])
    chroma_path = tmp_path / "chroma"
    ok = index_episode("show_s01_e99", db_path, chroma_path)
    assert ok is False
    col = get_collection(chroma_path)
    assert col.count() == 0


def test_index_all_count(tmp_path):
    episodes = [
        ep("show_s01_e01"),
        ep("show_s01_e02", number=2),
        skipped_ep("show_s01_e99"),
    ]
    db_path = make_db(tmp_path, episodes)
    chroma_path = tmp_path / "chroma"
    result = index_all(db_path, chroma_path)
    assert isinstance(result, IndexResult)
    assert result.indexed == 2
    assert result.skipped == 1
    assert result.total == 3
    assert get_collection(chroma_path).count() == 2


def test_metadata_fields(tmp_path):
    db_path = make_db(
        tmp_path,
        [ep("show_s02_e04", season=2, number=4, title="The Fire")],
    )
    chroma_path = tmp_path / "chroma"
    index_episode("show_s02_e04", db_path, chroma_path)
    col = get_collection(chroma_path)
    meta = col.get(ids=["show_s02_e04"], include=["metadatas"])["metadatas"][0]
    assert meta["series_slug"] == "the_office"
    assert meta["season_number"] == 2
    assert meta["episode_number"] == 4
    assert meta["episode_title"] == "The Fire"


def test_vector_matches_feature_builder(tmp_path):
    episode = ep("show_s01_e01")
    db_path = make_db(tmp_path, [episode])
    chroma_path = tmp_path / "chroma"
    index_episode("show_s01_e01", db_path, chroma_path)

    col = get_collection(chroma_path)
    stored = col.get(ids=["show_s01_e01"], include=["embeddings"])["embeddings"][0]

    expected = build_features(episode)
    assert expected is not None
    np.testing.assert_allclose(stored, expected.vector.tolist(), atol=1e-5)


def test_wipe_clears_existing(tmp_path):
    db_path = make_db(tmp_path / "db1", [ep("show_s01_e01"), ep("show_s01_e02", number=2)])
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    assert get_collection(chroma_path).count() == 2

    db_path2 = make_db(tmp_path / "db2", [ep("show_s01_e03", number=3)])
    index_all(db_path2, chroma_path, wipe=True)
    assert get_collection(chroma_path).count() == 1


def test_idempotent_upsert(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s01_e01")])
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    index_all(db_path, chroma_path)
    assert get_collection(chroma_path).count() == 1


def test_series_slug_filter(tmp_path):
    episodes = [
        ep("office_s01_e01", series_slug="the_office"),
        ep("friends_s01_e01", series_slug="friends"),
    ]
    db_path = make_db(tmp_path, episodes)
    chroma_path = tmp_path / "chroma"
    result = index_all(db_path, chroma_path, series_slug="the_office")
    assert result.indexed == 1
    assert result.total == 1
    col = get_collection(chroma_path)
    assert col.count() == 1
    assert col.get(ids=["office_s01_e01"])["ids"] == ["office_s01_e01"]


def test_index_episode_missing_raises(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s01_e01")])
    chroma_path = tmp_path / "chroma"
    with pytest.raises(ValueError):
        index_episode("does_not_exist", db_path, chroma_path)


def test_collection_uses_cosine_space(tmp_path):
    chroma_path = tmp_path / "chroma"
    col = get_collection(chroma_path)
    assert col.name == COLLECTION_NAME
    assert col.metadata.get("hnsw:space") == "cosine"
