from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.db.setup import setup_db
from backend.models.session_state import ConversationMessage, SessionState
from backend.recommender.confidence import ConfidenceResult, TopCandidate
from backend.recommender.engine import RecommendationEngine
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS
from backend.recommender.reranker import RankedCandidate
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


def make_state() -> SessionState:
    """Build a SessionState with realistic nonzero mood + tone preferences."""
    state = SessionState()
    state.current_preferences.mood.humor = 0.9
    state.current_preferences.mood.energy = 0.5
    state.current_preferences.mood.comfort = 0.5
    state.current_preferences.mood.sadness = 0.1
    for label in TONE_DIMENSIONS:
        state.current_preferences.tone_preferences[label] = 0.5
    return state


@pytest.fixture
def indexed_engine(tmp_path):
    """RecommendationEngine bound to a DB + Chroma collection of 7 fake episodes."""
    episodes = [ep(f"show_s01_e{i:02d}", number=i) for i in range(1, 8)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    engine = RecommendationEngine(db_path, chroma_path)
    episode_ids = [e["episode_id"] for e in episodes]
    return engine, episode_ids


def test_get_candidates_returns_n_top_candidates(indexed_engine):
    engine, _ = indexed_engine
    state = make_state()
    candidates = engine.get_candidates(state, n=3)
    assert len(candidates) == 3
    assert all(isinstance(c, TopCandidate) for c in candidates)


def test_get_candidates_respects_excluded_episodes(indexed_engine):
    engine, episode_ids = indexed_engine
    state = make_state()
    state.hard_constraints.excluded_episodes = [episode_ids[0], episode_ids[1]]
    candidates = engine.get_candidates(state, n=10)
    returned_ids = {c.episode_id for c in candidates}
    assert episode_ids[0] not in returned_ids
    assert episode_ids[1] not in returned_ids


def test_get_candidates_respects_previously_recommended(indexed_engine):
    engine, episode_ids = indexed_engine
    state = make_state()
    state.recommendation_context.previously_recommended = [
        episode_ids[2],
        episode_ids[3],
    ]
    candidates = engine.get_candidates(state, n=10)
    returned_ids = {c.episode_id for c in candidates}
    assert episode_ids[2] not in returned_ids
    assert episode_ids[3] not in returned_ids


def test_get_recommendation_returns_ranked_and_confidence(indexed_engine):
    engine, _ = indexed_engine
    state = make_state()
    result = engine.get_recommendation(state)
    assert isinstance(result, tuple)
    assert len(result) == 2
    top, confidence = result
    assert isinstance(top, RankedCandidate)
    assert isinstance(confidence, ConfidenceResult)


def test_get_recommendation_raises_when_all_candidates_excluded(indexed_engine):
    engine, episode_ids = indexed_engine
    state = make_state()
    state.hard_constraints.excluded_episodes = list(episode_ids)
    with pytest.raises(ValueError, match="No candidates available after filtering."):
        engine.get_recommendation(state)


def test_questions_answered_matches_conversation_history(indexed_engine):
    engine, _ = indexed_engine
    state = make_state()
    state.conversation_history = [
        ConversationMessage(message="hello"),
        ConversationMessage(message="something funny"),
        ConversationMessage(message="not too long"),
    ]
    _, confidence = engine.get_recommendation(state)
    assert confidence.questions_answered == len(state.conversation_history)
