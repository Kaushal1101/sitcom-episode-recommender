from __future__ import annotations

from pathlib import Path

from backend.models.session_state import SessionState
from backend.recommender.confidence import (
    ConfidenceResult,
    TopCandidate,
    compute_confidence,
)
from backend.recommender.reranker import RankedCandidate, rerank
from backend.recommender.retriever import retrieve
from backend.recommender.user_vector import build_user_vector

RETRIEVAL_MULTIPLIER: int = 5


def _ranked_to_top(r: RankedCandidate) -> TopCandidate:
    """Mirror of confidence._to_top_candidate; inlined to avoid importing a private name."""
    return TopCandidate(
        episode_id=r.episode_id,
        episode_title=r.episode_title,
        series_slug=r.series_slug,
        season_number=r.season_number,
        episode_number=r.episode_number,
        score=r.final_score,
    )


class RecommendationEngine:
    """
    Thin stateful facade over retrieve → rerank → compute_confidence.

    Configuration (DB and Chroma paths) is bound at construction time; each call
    accepts a SessionState describing the current user preferences, exclusion
    context, and conversation history.
    """

    def __init__(self, db_path: Path, chroma_path: Path) -> None:
        self.db_path = db_path
        self.chroma_path = chroma_path

    def _excluded_ids(self, state: SessionState) -> set[str]:
        ctx = state.recommendation_context
        return (
            set(state.hard_constraints.excluded_episodes)
            | set(ctx.previously_recommended)
            | set(ctx.rejected_recommendations)
            | set(ctx.accepted_recommendations)
        )

    def _retrieve_and_rerank(
        self, state: SessionState, top_k: int
    ) -> list[RankedCandidate]:
        user_vector = build_user_vector(state)
        candidates = retrieve(user_vector, self.chroma_path, top_k=top_k)
        return rerank(
            candidates,
            user_vector,
            self.db_path,
            excluded_ids=self._excluded_ids(state),
        )

    def get_candidates(self, state: SessionState, n: int) -> list[TopCandidate]:
        """Retrieve and rerank, returning up to n top candidates as TopCandidate objects."""
        ranked = self._retrieve_and_rerank(state, top_k=n * RETRIEVAL_MULTIPLIER)
        return [_ranked_to_top(r) for r in ranked[:n]]

    def get_recommendation(
        self, state: SessionState
    ) -> tuple[RankedCandidate, ConfidenceResult]:
        """
        Retrieve, rerank, and compute confidence for the leading candidate.

        Raises ValueError when no candidates remain after filtering.
        """
        ranked = self._retrieve_and_rerank(state, top_k=10 * RETRIEVAL_MULTIPLIER)
        if not ranked:
            raise ValueError("No candidates available after filtering.")
        confidence_result = compute_confidence(
            ranked,
            questions_answered=len(state.conversation_history),
        )
        return ranked[0], confidence_result
