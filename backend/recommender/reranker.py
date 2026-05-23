from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.recommender.episode_feature_builder import TONE_DIMENSIONS, load_mood_rows
from backend.recommender.retriever import Candidate

W_SIM: float = 0.5
W_MOOD: float = 0.3
W_TONE: float = 0.2


@dataclass
class RankedCandidate:
    episode_id: str
    final_score: float
    similarity: float
    mood_alignment: float
    tone_alignment: float
    series_slug: str
    season_number: int
    episode_number: int
    episode_title: str


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _score(
    candidate: Candidate,
    row: dict,
    user_vector: np.ndarray,
) -> tuple[float, float, float]:
    """Returns (final_score, mood_alignment, tone_alignment)."""
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
    mood_alignment = float(np.dot(user_vector[0:4], _l2_normalize(ep_mood)))
    tone_alignment = float(np.dot(user_vector[4:17], _l2_normalize(ep_tone)))
    final_score = (
        W_SIM * candidate.similarity
        + W_MOOD * mood_alignment
        + W_TONE * tone_alignment
    )
    return final_score, mood_alignment, tone_alignment


def rerank(
    candidates: list[Candidate],
    user_vector: np.ndarray,
    db_path: Path,
    excluded_ids: set[str] | None = None,
) -> list[RankedCandidate]:
    """
    Score and rerank candidates using mood/tone alignment from SQLite.

    - Excluded episodes are removed entirely from output.
    - Candidates whose episode_id is not found in SQLite are silently dropped.
    - Output is sorted by final_score descending.
    """
    excluded = excluded_ids or set()
    active = [c for c in candidates if c.episode_id not in excluded]
    if not active:
        return []

    rows = load_mood_rows([c.episode_id for c in active], db_path)

    ranked: list[RankedCandidate] = []
    for c in active:
        row = rows.get(c.episode_id)
        if row is None:
            continue
        final_score, mood_aln, tone_aln = _score(c, row, user_vector)
        ranked.append(
            RankedCandidate(
                episode_id=c.episode_id,
                final_score=round(final_score, 6),
                similarity=c.similarity,
                mood_alignment=round(mood_aln, 6),
                tone_alignment=round(tone_aln, 6),
                series_slug=c.series_slug,
                season_number=c.season_number,
                episode_number=c.episode_number,
                episode_title=c.episode_title,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
