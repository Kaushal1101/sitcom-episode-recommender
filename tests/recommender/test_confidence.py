from __future__ import annotations

from backend.recommender.confidence import (
    COVERAGE_SATURATION_POINT,
    MAX_EXPECTED_GAP,
    ConfidenceResult,
    TopCandidate,
    compute_confidence,
)
from backend.recommender.reranker import RankedCandidate


def make_ranked(
    episode_id: str = "ep_0",
    final_score: float = 0.80,
    similarity: float = 0.80,
    mood_alignment: float = 0.50,
    tone_alignment: float = 0.50,
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
) -> RankedCandidate:
    return RankedCandidate(
        episode_id=episode_id,
        final_score=final_score,
        similarity=similarity,
        mood_alignment=mood_alignment,
        tone_alignment=tone_alignment,
        series_slug=series_slug,
        season_number=season,
        episode_number=number,
        episode_title=title,
    )


def make_ranked_list(scores: list[float]) -> list[RankedCandidate]:
    return [
        make_ranked(
            episode_id=f"ep_{i}",
            final_score=s,
            season=1,
            number=i + 1,
            title=f"Episode {i + 1}",
        )
        for i, s in enumerate(scores)
    ]


def test_empty_candidates_zero_confidence():
    result = compute_confidence([], questions_answered=3)
    assert isinstance(result, ConfidenceResult)
    assert result.confidence == 0.0
    assert result.top_score == 0.0
    assert result.score_gap == 0.0
    assert result.normalized_gap == 0.0
    assert result.normalized_coverage == 0.0
    assert result.questions_answered == 3
    assert result.top_candidates == []


def test_single_candidate_zero_gap():
    ranked = make_ranked_list([0.8])
    result = compute_confidence(ranked, questions_answered=2)

    assert result.score_gap == 0.0
    assert result.normalized_gap == 0.0
    assert result.top_score == 0.8
    assert result.normalized_coverage == 0.4
    expected = 0.5 * 0.8 + 0.3 * 0.0 + 0.2 * 0.4
    assert abs(result.confidence - expected) < 1e-9
    assert len(result.top_candidates) == 1
    assert isinstance(result.top_candidates[0], TopCandidate)
    assert result.top_candidates[0].score == 0.8


def test_fewer_peers_than_n_gap_peers():
    ranked = make_ranked_list([0.9, 0.7])
    result = compute_confidence(ranked, questions_answered=0, n_gap_peers=3)

    assert abs(result.score_gap - 0.2) < 1e-9
    assert result.normalized_gap == 1.0
    assert result.top_score == 0.9
    assert result.normalized_coverage == 0.0


def test_normal_multi_candidate_case():
    ranked = make_ranked_list([0.75, 0.60, 0.55, 0.50, 0.45])
    result = compute_confidence(ranked, questions_answered=2)

    assert 0.0 <= result.confidence <= 1.0
    assert result.top_score == 0.75
    expected_gap = 0.75 - (0.60 + 0.55 + 0.50) / 3
    assert abs(result.score_gap - expected_gap) < 1e-9
    assert 0.0 <= result.normalized_gap <= 1.0
    assert result.normalized_coverage == 0.4
    assert result.questions_answered == 2
    assert len(result.top_candidates) == 4
    assert [c.episode_id for c in result.top_candidates] == [
        "ep_0",
        "ep_1",
        "ep_2",
        "ep_3",
    ]


def test_gap_normalization_clipping():
    ranked = make_ranked_list([0.95, 0.20, 0.10, 0.05])
    result = compute_confidence(ranked, questions_answered=0)

    assert result.score_gap > MAX_EXPECTED_GAP
    assert result.normalized_gap == 1.0


def test_coverage_saturation_clamps_to_one():
    ranked = make_ranked_list([0.7, 0.5, 0.4, 0.3])
    result = compute_confidence(
        ranked,
        questions_answered=COVERAGE_SATURATION_POINT + 5,
    )

    assert result.normalized_coverage == 1.0
    assert result.questions_answered == COVERAGE_SATURATION_POINT + 5


def test_weight_override_changes_confidence():
    ranked = make_ranked_list([0.8, 0.6, 0.5, 0.4])
    default = compute_confidence(ranked, questions_answered=3)
    overridden = compute_confidence(
        ranked,
        questions_answered=3,
        top_score_weight=0.2,
        gap_weight=0.6,
        coverage_weight=0.2,
    )

    assert default.confidence != overridden.confidence
    expected = (
        0.2 * overridden.top_score
        + 0.6 * overridden.normalized_gap
        + 0.2 * overridden.normalized_coverage
    )
    assert abs(overridden.confidence - expected) < 1e-9


def test_worked_example_regression():
    ranked = make_ranked_list([0.86, 0.81, 0.79, 0.78])
    result = compute_confidence(ranked, questions_answered=3)
    assert abs(result.confidence - 0.684) < 0.001
