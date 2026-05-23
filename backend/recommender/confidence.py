from __future__ import annotations

from dataclasses import dataclass, field

from backend.recommender.reranker import RankedCandidate

MAX_EXPECTED_GAP: float = 0.15
COVERAGE_SATURATION_POINT: int = 5
TOP_SCORE_WEIGHT: float = 0.5
GAP_WEIGHT: float = 0.3
COVERAGE_WEIGHT: float = 0.2


@dataclass
class TopCandidate:
    episode_id: str
    episode_title: str
    series_slug: str
    season_number: int
    episode_number: int
    score: float


@dataclass
class ConfidenceResult:
    confidence: float
    top_score: float
    score_gap: float
    normalized_gap: float
    normalized_coverage: float
    questions_answered: int
    top_candidates: list[TopCandidate] = field(default_factory=list)


def _to_top_candidate(r: RankedCandidate) -> TopCandidate:
    return TopCandidate(
        episode_id=r.episode_id,
        episode_title=r.episode_title,
        series_slug=r.series_slug,
        season_number=r.season_number,
        episode_number=r.episode_number,
        score=r.final_score,
    )


def compute_confidence(
    ranked: list[RankedCandidate],
    questions_answered: int,
    n_gap_peers: int = 3,
    max_expected_gap: float = MAX_EXPECTED_GAP,
    coverage_saturation: int = COVERAGE_SATURATION_POINT,
    top_score_weight: float = TOP_SCORE_WEIGHT,
    gap_weight: float = GAP_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> ConfidenceResult:
    """
    Compute a deterministic confidence score for the top recommendation.

    Combines three signals:
      - top_score: reranker score of the leading candidate
      - score_gap: how far the leader sits ahead of the next n peers
      - coverage: how much the user has expressed (via questions answered)

    The output ConfidenceResult is the stable contract consumed by the CSM.
    """
    if not ranked:
        return ConfidenceResult(
            confidence=0.0,
            top_score=0.0,
            score_gap=0.0,
            normalized_gap=0.0,
            normalized_coverage=0.0,
            questions_answered=questions_answered,
            top_candidates=[],
        )

    top_score = float(ranked[0].final_score)

    if len(ranked) < 2:
        score_gap = 0.0
    else:
        peers = ranked[1 : 1 + n_gap_peers]
        peer_mean = sum(p.final_score for p in peers) / len(peers)
        score_gap = top_score - peer_mean

    normalized_gap = min(max(score_gap / max_expected_gap, 0.0), 1.0)
    normalized_coverage = min(questions_answered / coverage_saturation, 1.0)

    confidence = (
        top_score_weight * top_score
        + gap_weight * normalized_gap
        + coverage_weight * normalized_coverage
    )

    top_candidates = [_to_top_candidate(r) for r in ranked[: 1 + n_gap_peers]]

    return ConfidenceResult(
        confidence=float(confidence),
        top_score=top_score,
        score_gap=float(score_gap),
        normalized_gap=float(normalized_gap),
        normalized_coverage=float(normalized_coverage),
        questions_answered=questions_answered,
        top_candidates=top_candidates,
    )
