from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.recommender.explanation_builder import MatchExplanation, explain_all
from backend.recommender.reranker import RankedCandidate, rerank
from backend.recommender.retriever import retrieve

RETRIEVAL_MULTIPLIER: int = 5


@dataclass
class RecommendationResult:
    ranked: list[RankedCandidate]
    explanations: list[MatchExplanation]


def recommend(
    user_vector: np.ndarray,
    db_path: Path,
    chroma_path: Path,
    top_k: int = 10,
    excluded_ids: set[str] | None = None,
    series_slug: str | None = None,
) -> RecommendationResult:
    """
    Full recommendation pipeline: retrieve → rerank → explain.

    Retrieves top_k * RETRIEVAL_MULTIPLIER candidates from Chroma,
    reranks them, trims to top_k, then builds explanations.
    """
    candidates = retrieve(
        user_vector,
        chroma_path,
        top_k=top_k * RETRIEVAL_MULTIPLIER,
        series_slug=series_slug,
    )
    ranked = rerank(candidates, user_vector, db_path, excluded_ids=excluded_ids)
    ranked = ranked[:top_k]
    explanations = explain_all(ranked, user_vector, db_path)
    return RecommendationResult(ranked=ranked, explanations=explanations)
