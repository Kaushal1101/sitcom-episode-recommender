from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from backend.db.setup import setup_db
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS, build_features
from backend.recommender.retriever import Candidate
from backend.recommender.reranker import rerank


def make_candidate(
    episode_id: str = "show_s01_e01",
    similarity: float = 0.9,
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
) -> Candidate:
    return Candidate(
        episode_id=episode_id,
        similarity=similarity,
        series_slug=series_slug,
        season_number=season,
        episode_number=number,
        episode_title=title,
    )


def make_mood_row(
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


def test_rerank_preserves_all_non_excluded(tmp_path):
    candidates = [make_candidate(f"ep_{i}", similarity=0.9 - i * 0.1) for i in range(3)]
    rows = [make_mood_row(f"ep_{i}") for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path)
    assert len(results) == 3


def test_rerank_sorted_by_final_score(tmp_path):
    candidates = [make_candidate(f"ep_{i}", similarity=0.9 - i * 0.1) for i in range(4)]
    rows = [make_mood_row(f"ep_{i}") for i in range(4)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path)
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_mood_match_ranks_first(tmp_path):
    """
    Episode with exact same mood profile as user should outscore
    an episode with a very different mood, even if initial similarities differ.
    """
    user_vec = user_vec_from_mood(humor=0.9, energy=0.1, comfort=0.8, sadness=0.1)
    candidates = [
        make_candidate("high_match", similarity=0.85),
        make_candidate("low_match", similarity=0.90),
    ]
    rows = [
        make_mood_row("high_match", humor=0.9, energy=0.1, comfort=0.8, sadness=0.1),
        make_mood_row("low_match", humor=0.1, energy=0.9, comfort=0.1, sadness=0.9),
    ]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec, db_path)
    assert results[0].episode_id == "high_match"


def test_rerank_excluded_ids_removed(tmp_path):
    candidates = [make_candidate(f"ep_{i}") for i in range(3)]
    rows = [make_mood_row(f"ep_{i}") for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path, excluded_ids={"ep_1"})
    ids = [r.episode_id for r in results]
    assert "ep_1" not in ids
    assert len(results) == 2


def test_rerank_all_excluded_returns_empty(tmp_path):
    candidates = [make_candidate(f"ep_{i}") for i in range(3)]
    rows = [make_mood_row(f"ep_{i}") for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = rerank(
        candidates,
        user_vec_from_mood(),
        db_path,
        excluded_ids={"ep_0", "ep_1", "ep_2"},
    )
    assert results == []


def test_rerank_output_fields(tmp_path):
    c = make_candidate("ep_0", similarity=0.88, season=2, number=4, title="The Fire")
    db_path = make_db(tmp_path, [make_mood_row("ep_0")])
    results = rerank([c], user_vec_from_mood(), db_path)
    assert len(results) == 1
    r = results[0]
    assert r.episode_id == "ep_0"
    assert r.series_slug == "the_office"
    assert r.season_number == 2
    assert r.episode_number == 4
    assert r.episode_title == "The Fire"
    assert isinstance(r.final_score, float)
    assert isinstance(r.mood_alignment, float)
    assert isinstance(r.tone_alignment, float)
    assert r.similarity == 0.88


def test_rerank_empty_candidates(tmp_path):
    db_path = make_db(tmp_path, [])
    results = rerank([], user_vec_from_mood(), db_path)
    assert results == []


def test_rerank_score_formula(tmp_path):
    """Verify final_score = W_SIM * sim + W_MOOD * mood_aln + W_TONE * tone_aln."""
    from backend.recommender.reranker import W_MOOD, W_SIM, W_TONE

    c = make_candidate("ep_0", similarity=0.80)
    db_path = make_db(tmp_path, [make_mood_row("ep_0")])
    results = rerank([c], user_vec_from_mood(), db_path)
    r = results[0]
    expected = (
        W_SIM * r.similarity + W_MOOD * r.mood_alignment + W_TONE * r.tone_alignment
    )
    assert abs(r.final_score - expected) < 1e-5


def test_rerank_missing_sqlite_row_dropped(tmp_path):
    candidates = [
        make_candidate("ep_0"),
        make_candidate("ep_ghost"),
    ]
    db_path = make_db(tmp_path, [make_mood_row("ep_0")])
    results = rerank(candidates, user_vec_from_mood(), db_path)
    ids = [r.episode_id for r in results]
    assert "ep_0" in ids
    assert "ep_ghost" not in ids
