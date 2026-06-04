from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.csm.interaction_types import InteractionType
from backend.csm.strategy import ConversationStrategy
from backend.models.session_state import ConversationMessage, SessionState
from backend.recommender.confidence import TopCandidate
from backend.recommender.engine import RecommendationEngine

EPISODE_TITLE = "Diversity Day"
SERIES_SLUG = "the_office"


def make_mock_ranked(
    episode_id: str = "ep_1",
    episode_title: str = EPISODE_TITLE,
    series_slug: str = SERIES_SLUG,
    season_number: int = 1,
    episode_number: int = 2,
    final_score: float = 0.85,
) -> MagicMock:
    """A MagicMock standing in for a RankedCandidate."""
    return MagicMock(
        episode_id=episode_id,
        episode_title=episode_title,
        series_slug=series_slug,
        season_number=season_number,
        episode_number=episode_number,
        final_score=final_score,
    )


def make_top_candidate(index: int) -> TopCandidate:
    return TopCandidate(
        episode_id=f"ep_{index}",
        episode_title=f"Episode {index}",
        series_slug=SERIES_SLUG,
        season_number=1,
        episode_number=index,
        score=0.8 - 0.01 * index,
    )


def make_engine(
    confidence: float, num_candidates: int = 5
) -> tuple[MagicMock, MagicMock]:
    """Return (engine_mock, ranked_mock).

    `get_recommendation` returns (ranked_mock, confidence_mock).
    `get_candidates(state, n)` returns the first n TopCandidates from a
    pre-built pool sized `num_candidates`.
    """
    engine = MagicMock(spec=RecommendationEngine)
    ranked = make_mock_ranked()
    confidence_result = MagicMock(confidence=confidence)
    engine.get_recommendation.return_value = (ranked, confidence_result)

    pool = [make_top_candidate(i + 1) for i in range(num_candidates)]
    engine.get_candidates.side_effect = lambda state, n: pool[:n]
    return engine, ranked


def make_state_with_history(num_messages: int = 1) -> SessionState:
    state = SessionState()
    for i in range(num_messages):
        state.conversation_history.append(ConversationMessage(message=f"msg_{i}"))
    return state


@pytest.fixture
def state():
    return make_state_with_history()


def test_empty_history_returns_open_chat_without_calling_engine():
    engine, _ = make_engine(confidence=0.5)
    decision = ConversationStrategy(engine).next_action(SessionState())

    assert decision.interaction_type is InteractionType.OPEN_CHAT
    assert decision.options is None
    assert decision.candidates is None
    assert decision.intent is None
    engine.get_recommendation.assert_not_called()
    engine.get_candidates.assert_not_called()


def test_low_confidence_returns_attribute_multi(state):
    engine, _ = make_engine(confidence=0.2)
    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.ATTRIBUTE_MULTI
    assert decision.intent is not None
    assert decision.options is not None and len(decision.options) >= 2
    assert decision.candidates is None
    engine.get_recommendation.assert_called_once_with(state)


def test_mid_confidence_returns_attribute_yn(state):
    engine, _ = make_engine(confidence=0.5)
    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.ATTRIBUTE_YN
    assert decision.options == ["Yes", "No"]
    assert decision.intent is not None
    assert decision.candidates is None


def test_item_multi_pulls_three_candidates(state):
    engine, _ = make_engine(confidence=0.65)
    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.ITEM_MULTI
    assert decision.candidates is not None and len(decision.candidates) == 3
    assert decision.options is not None and len(decision.options) == 3
    assert decision.intent is None
    engine.get_candidates.assert_called_once_with(state, n=3)


def test_item_yn_pulls_one_candidate(state):
    engine, _ = make_engine(confidence=0.80)
    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.ITEM_YN
    assert decision.candidates is not None and len(decision.candidates) == 1
    assert decision.options == ["Yes", "No"]
    assert decision.intent is None
    engine.get_candidates.assert_called_once_with(state, n=1)


def test_high_confidence_returns_recommend_with_episode_title(state):
    engine, ranked = make_engine(confidence=0.95)
    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.RECOMMEND
    assert ranked.episode_title in decision.question
    assert decision.options is None
    assert decision.candidates is None
    assert decision.intent is None
    engine.get_recommendation.assert_called_once_with(state)


def test_accepted_previous_recommendation_returns_rating():
    engine, _ = make_engine(confidence=0.2)
    state = make_state_with_history()
    state.recommendation_context.previously_recommended.append("ep_prev")
    state.recommendation_context.accepted_recommendations.append("ep_prev")

    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.RATING
    assert decision.options == ["1", "2", "3", "4", "5"]
    assert decision.candidates is None
    assert decision.intent is None


def test_rejected_previous_recommendation_returns_soft_critique():
    engine, _ = make_engine(confidence=0.2)
    state = make_state_with_history()
    state.recommendation_context.previously_recommended.append("ep_prev")
    state.recommendation_context.rejected_recommendations.append("ep_prev")

    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.SOFT_CRITIQUE
    assert decision.question.startswith("That one didn't land.")
    assert decision.intent is not None
    assert decision.options is not None and len(decision.options) >= 2
    assert decision.candidates is None


def test_rating_check_short_circuits_before_engine_call():
    engine, _ = make_engine(confidence=0.95)
    state = make_state_with_history()
    state.recommendation_context.previously_recommended.append("ep_prev")
    state.recommendation_context.accepted_recommendations.append("ep_prev")

    decision = ConversationStrategy(engine).next_action(state)

    assert decision.interaction_type is InteractionType.RATING
    engine.get_recommendation.assert_not_called()
    engine.get_candidates.assert_not_called()
