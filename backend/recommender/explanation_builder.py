from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.recommender.episode_feature_builder import (
    MOOD_DIMENSIONS,
    TONE_DIMENSIONS,
    load_mood_rows,
)
from backend.recommender.reranker import RankedCandidate

MOOD_THRESHOLD: float = 0.15
TONE_THRESHOLD: float = 0.10
TRAIT_THRESHOLD: float = 0.50
TOP_N_TRAITS: int = 3


@dataclass
class MatchExplanation:
    episode_id: str
    mood_matches: list[str]
    tone_matches: list[str]
    dominant_episode_traits: list[str]
    score_breakdown: dict[str, float] = field(default_factory=dict)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _row_to_vectors(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Parse a SQLite row into (ep_mood_vec, ep_tone_vec) as float32 arrays."""
    ep_mood = np.array(
        [
            row["humor_level"] or 0.0,
            row["energy_level"] or 0.0,
            row["comfort_level"] or 0.0,
            row["sadness_level"] or 0.0,
        ],
        dtype=np.float32,
    )
    tone_dict = json.loads(row["tone_scores"] or "{}")
    ep_tone = np.array(
        [tone_dict.get(label, 0.0) for label in TONE_DIMENSIONS],
        dtype=np.float32,
    )
    return ep_mood, ep_tone


def _build_from_row(
    ranked: RankedCandidate,
    row: dict,
    user_vector: np.ndarray,
    mood_threshold: float,
    tone_threshold: float,
    trait_threshold: float,
    top_n_traits: int,
) -> MatchExplanation:
    ep_mood_vec, ep_tone_vec = _row_to_vectors(row)

    user_mood = user_vector[0 : len(MOOD_DIMENSIONS)]
    user_tone = user_vector[len(MOOD_DIMENSIONS) :]

    ep_mood_unit = _l2_normalize(ep_mood_vec)
    ep_tone_unit = _l2_normalize(ep_tone_vec)

    mood_per_dim = user_mood * ep_mood_unit
    tone_per_dim = user_tone * ep_tone_unit

    mood_matches = [
        MOOD_DIMENSIONS[i]
        for i in range(len(MOOD_DIMENSIONS))
        if float(mood_per_dim[i]) > mood_threshold
    ]
    tone_matches = [
        TONE_DIMENSIONS[i]
        for i in range(len(TONE_DIMENSIONS))
        if float(tone_per_dim[i]) > tone_threshold
    ]

    mood_scored = [
        (MOOD_DIMENSIONS[i], float(ep_mood_vec[i])) for i in range(len(MOOD_DIMENSIONS))
    ]
    tone_scored = [
        (TONE_DIMENSIONS[i], float(ep_tone_vec[i])) for i in range(len(TONE_DIMENSIONS))
    ]
    all_scored = sorted(mood_scored + tone_scored, key=lambda x: x[1], reverse=True)
    dominant_episode_traits = [
        name for name, score in all_scored if score > trait_threshold
    ][:top_n_traits]

    score_breakdown = {
        "similarity": ranked.similarity,
        "mood_alignment": ranked.mood_alignment,
        "tone_alignment": ranked.tone_alignment,
        "final_score": ranked.final_score,
    }

    return MatchExplanation(
        episode_id=ranked.episode_id,
        mood_matches=mood_matches,
        tone_matches=tone_matches,
        dominant_episode_traits=dominant_episode_traits,
        score_breakdown=score_breakdown,
    )


def explain(
    ranked: RankedCandidate,
    user_vector: np.ndarray,
    db_path: Path,
    mood_threshold: float = MOOD_THRESHOLD,
    tone_threshold: float = TONE_THRESHOLD,
    trait_threshold: float = TRAIT_THRESHOLD,
    top_n_traits: int = TOP_N_TRAITS,
) -> MatchExplanation | None:
    """
    Build a MatchExplanation for a single RankedCandidate.
    Returns None if the episode is not found in SQLite.
    """
    rows = load_mood_rows([ranked.episode_id], db_path)
    row = rows.get(ranked.episode_id)
    if row is None:
        return None
    return _build_from_row(
        ranked,
        row,
        user_vector,
        mood_threshold,
        tone_threshold,
        trait_threshold,
        top_n_traits,
    )


def explain_all(
    ranked: list[RankedCandidate],
    user_vector: np.ndarray,
    db_path: Path,
    mood_threshold: float = MOOD_THRESHOLD,
    tone_threshold: float = TONE_THRESHOLD,
    trait_threshold: float = TRAIT_THRESHOLD,
    top_n_traits: int = TOP_N_TRAITS,
) -> list[MatchExplanation]:
    """
    Build explanations for a list of RankedCandidates in a single SQLite query.
    Episodes not found in SQLite are silently omitted.
    Order matches the input ranked list.
    """
    if not ranked:
        return []

    rows = load_mood_rows([r.episode_id for r in ranked], db_path)

    explanations: list[MatchExplanation] = []
    for r in ranked:
        row = rows.get(r.episode_id)
        if row is None:
            continue
        explanations.append(
            _build_from_row(
                r,
                row,
                user_vector,
                mood_threshold,
                tone_threshold,
                trait_threshold,
                top_n_traits,
            )
        )
    return explanations
