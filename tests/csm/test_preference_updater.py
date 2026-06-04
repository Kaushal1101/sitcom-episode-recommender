from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.csm.decision import InteractionDecision
from backend.csm.interaction_types import InteractionType
from backend.csm.preference_updater import (
    ITEM_STEP,
    YN_NEGATIVE,
    YN_POSITIVE,
    apply_answer,
)
from backend.models.session_state import SessionState
from backend.recommender.confidence import TopCandidate
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS

SERIES_SLUG = "the_office"


def make_state() -> SessionState:
    """Return a default SessionState with all preferences at 0.0."""
    return SessionState()


def make_decision(
    interaction_type: InteractionType,
    intent: str | None = None,
    options: list[str] | None = None,
    candidates: list[TopCandidate] | None = None,
) -> InteractionDecision:
    return InteractionDecision(
        interaction_type=interaction_type,
        question="question?",
        options=options,
        candidates=candidates,
        intent=intent,
    )


def make_candidate(
    episode_id: str = "ep_1",
    episode_title: str = "Test Episode",
    series_slug: str = SERIES_SLUG,
    season_number: int = 1,
    episode_number: int = 1,
    score: float = 0.8,
) -> TopCandidate:
    return TopCandidate(
        episode_id=episode_id,
        episode_title=episode_title,
        series_slug=series_slug,
        season_number=season_number,
        episode_number=episode_number,
        score=score,
    )


def _build_db(
    tmp_path: Path,
    rows: list[dict],
) -> Path:
    """Create a minimal episodes table in a temp SQLite DB.

    Each row dict must contain: episode_id, humor_level, energy_level,
    comfort_level, sadness_level, tone_scores.
    """
    db_path = tmp_path / "episodes.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE episodes (
                episode_id     TEXT PRIMARY KEY,
                humor_level    REAL,
                energy_level   REAL,
                comfort_level  REAL,
                sadness_level  REAL,
                tone_scores    TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO episodes
                (episode_id, humor_level, energy_level, comfort_level,
                 sadness_level, tone_scores)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["episode_id"],
                    r["humor_level"],
                    r["energy_level"],
                    r["comfort_level"],
                    r["sadness_level"],
                    r["tone_scores"],
                )
                for r in rows
            ],
        )
        conn.commit()
    return db_path


def _all_zero_tone_scores() -> str:
    return json.dumps({dim: 0.0 for dim in TONE_DIMENSIONS})


def test_attribute_yn_yes_sets_positive():
    state = make_state()
    decision = make_decision(
        InteractionType.ATTRIBUTE_YN, intent="humor", options=["Yes", "No"]
    )
    apply_answer(state, decision, "Yes")
    assert state.current_preferences.mood.humor == YN_POSITIVE


def test_attribute_yn_no_sets_negative():
    state = make_state()
    decision = make_decision(
        InteractionType.ATTRIBUTE_YN, intent="humor", options=["Yes", "No"]
    )
    apply_answer(state, decision, "No")
    assert state.current_preferences.mood.humor == YN_NEGATIVE


def test_attribute_yn_targets_tone_dimension():
    state = make_state()
    decision = make_decision(
        InteractionType.ATTRIBUTE_YN, intent="awkward", options=["Yes", "No"]
    )
    apply_answer(state, decision, "Yes")
    assert state.current_preferences.tone_preferences["awkward"] == YN_POSITIVE


def test_attribute_multi_sets_mapped_value():
    state = make_state()
    decision = make_decision(
        InteractionType.ATTRIBUTE_MULTI,
        intent="energy",
        options=["Fast-paced and chaotic", "Steady and engaging", "Slow and relaxed"],
    )
    apply_answer(state, decision, "Slow and relaxed")
    assert state.current_preferences.mood.energy == pytest.approx(-0.6)


def test_soft_critique_behaves_like_attribute_multi():
    state = make_state()
    decision = make_decision(
        InteractionType.SOFT_CRITIQUE,
        intent="comfort",
        options=[
            "Very cosy and safe",
            "A mix of comfort and tension",
            "Something edgier",
        ],
    )
    apply_answer(state, decision, "Something edgier")
    assert state.current_preferences.mood.comfort == pytest.approx(-0.6)


def _assert_all_prefs_zero(state: SessionState) -> None:
    mood = state.current_preferences.mood
    assert mood.humor == 0.0
    assert mood.energy == 0.0
    assert mood.comfort == 0.0
    assert mood.sadness == 0.0
    for dim in TONE_DIMENSIONS:
        assert state.current_preferences.tone_preferences[dim] == 0.0


def test_open_chat_does_not_change_preferences():
    state = make_state()
    decision = make_decision(InteractionType.OPEN_CHAT)
    apply_answer(state, decision, "something funny")
    _assert_all_prefs_zero(state)


def test_recommend_does_not_change_preferences():
    state = make_state()
    decision = make_decision(InteractionType.RECOMMEND)
    apply_answer(state, decision, "Yes please")
    _assert_all_prefs_zero(state)


def test_rating_does_not_change_preferences():
    state = make_state()
    decision = make_decision(
        InteractionType.RATING, options=["1", "2", "3", "4", "5"]
    )
    apply_answer(state, decision, "4")
    _assert_all_prefs_zero(state)


@pytest.mark.parametrize(
    "decision_factory,answer",
    [
        (lambda: make_decision(InteractionType.OPEN_CHAT), "open answer"),
        (
            lambda: make_decision(
                InteractionType.ATTRIBUTE_YN, intent="humor", options=["Yes", "No"]
            ),
            "Yes",
        ),
        (
            lambda: make_decision(
                InteractionType.ATTRIBUTE_MULTI,
                intent="humor",
                options=[
                    "Sharp and witty",
                    "Silly and slapstick",
                    "Dry and understated",
                    "Not really looking for laughs",
                ],
            ),
            "Sharp and witty",
        ),
        (lambda: make_decision(InteractionType.RECOMMEND), "Yes please"),
        (
            lambda: make_decision(
                InteractionType.RATING, options=["1", "2", "3", "4", "5"]
            ),
            "4",
        ),
    ],
)
def test_history_always_appended(decision_factory, answer):
    state = make_state()
    apply_answer(state, decision_factory(), answer)
    assert len(state.conversation_history) == 1
    assert state.conversation_history[0].message == answer


def test_item_yn_yes_nudges_toward_episode(tmp_path):
    candidate = make_candidate(episode_id="ep_humor")
    db_path = _build_db(
        tmp_path,
        [
            {
                "episode_id": "ep_humor",
                "humor_level": 1.0,
                "energy_level": 0.0,
                "comfort_level": 0.0,
                "sadness_level": 0.0,
                "tone_scores": _all_zero_tone_scores(),
            }
        ],
    )

    state = make_state()
    decision = make_decision(
        InteractionType.ITEM_YN,
        options=["Yes", "No"],
        candidates=[candidate],
    )
    apply_answer(state, decision, "Yes", db_path=db_path)

    assert state.current_preferences.mood.humor > 0.0
    assert state.current_preferences.mood.energy == 0.0
    assert state.current_preferences.mood.comfort == 0.0
    assert state.current_preferences.mood.sadness == 0.0
    for dim in TONE_DIMENSIONS:
        assert state.current_preferences.tone_preferences[dim] == 0.0


def test_item_yn_no_nudges_away_from_episode(tmp_path):
    candidate = make_candidate(episode_id="ep_humor")
    db_path = _build_db(
        tmp_path,
        [
            {
                "episode_id": "ep_humor",
                "humor_level": 1.0,
                "energy_level": 0.0,
                "comfort_level": 0.0,
                "sadness_level": 0.0,
                "tone_scores": _all_zero_tone_scores(),
            }
        ],
    )

    state = make_state()
    decision = make_decision(
        InteractionType.ITEM_YN,
        options=["Yes", "No"],
        candidates=[candidate],
    )
    apply_answer(state, decision, "No", db_path=db_path)

    assert state.current_preferences.mood.humor < 0.0


def test_item_multi_matches_correct_candidate(tmp_path):
    ep1 = make_candidate(
        episode_id="ep_1", episode_title="First", series_slug=SERIES_SLUG
    )
    ep2 = make_candidate(
        episode_id="ep_2", episode_title="Second", series_slug=SERIES_SLUG
    )
    db_path = _build_db(
        tmp_path,
        [
            {
                "episode_id": "ep_1",
                "humor_level": 1.0,
                "energy_level": 0.0,
                "comfort_level": 0.0,
                "sadness_level": 0.0,
                "tone_scores": _all_zero_tone_scores(),
            },
            {
                "episode_id": "ep_2",
                "humor_level": 0.0,
                "energy_level": 1.0,
                "comfort_level": 0.0,
                "sadness_level": 0.0,
                "tone_scores": _all_zero_tone_scores(),
            },
        ],
    )

    state = make_state()
    options = [
        f"{ep1.episode_title} ({ep1.series_slug})",
        f"{ep2.episode_title} ({ep2.series_slug})",
    ]
    decision = make_decision(
        InteractionType.ITEM_MULTI,
        options=options,
        candidates=[ep1, ep2],
    )
    apply_answer(
        state,
        decision,
        f"{ep2.episode_title} ({ep2.series_slug})",
        db_path=db_path,
    )

    assert state.current_preferences.mood.energy > 0.0
    assert state.current_preferences.mood.humor == 0.0


def test_item_update_clamps_to_valid_range(tmp_path):
    candidate = make_candidate(episode_id="ep_humor")
    db_path = _build_db(
        tmp_path,
        [
            {
                "episode_id": "ep_humor",
                "humor_level": 1.0,
                "energy_level": 0.0,
                "comfort_level": 0.0,
                "sadness_level": 0.0,
                "tone_scores": _all_zero_tone_scores(),
            }
        ],
    )

    state = make_state()
    state.current_preferences.mood.humor = 0.9
    decision = make_decision(
        InteractionType.ITEM_YN,
        options=["Yes", "No"],
        candidates=[candidate],
    )

    for _ in range(50):
        apply_answer(state, decision, "Yes", db_path=db_path)
        assert state.current_preferences.mood.humor <= 1.0
        assert state.current_preferences.mood.humor >= -1.0


def test_item_update_skipped_when_db_path_is_none():
    candidate = make_candidate(episode_id="ep_humor")
    state = make_state()
    decision = make_decision(
        InteractionType.ITEM_YN,
        options=["Yes", "No"],
        candidates=[candidate],
    )
    apply_answer(state, decision, "Yes", db_path=None)
    _assert_all_prefs_zero(state)
    assert len(state.conversation_history) == 1
    assert state.conversation_history[0].message == "Yes"


def test_item_yn_step_size_is_expected_proportion(tmp_path):
    candidate = make_candidate(episode_id="ep_humor")
    db_path = _build_db(
        tmp_path,
        [
            {
                "episode_id": "ep_humor",
                "humor_level": 1.0,
                "energy_level": 0.0,
                "comfort_level": 0.0,
                "sadness_level": 0.0,
                "tone_scores": _all_zero_tone_scores(),
            }
        ],
    )

    state = make_state()
    decision = make_decision(
        InteractionType.ITEM_YN,
        options=["Yes", "No"],
        candidates=[candidate],
    )
    apply_answer(state, decision, "Yes", db_path=db_path)

    assert state.current_preferences.mood.humor == pytest.approx(ITEM_STEP * 1.0)
