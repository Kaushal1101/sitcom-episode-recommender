from __future__ import annotations

import pytest

from backend.csm.interaction_types import InteractionType
from backend.csm.templates import TEMPLATES, render_question
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS

MOOD_DIMS = ("humor", "energy", "comfort", "sadness")


@pytest.mark.parametrize("dim", MOOD_DIMS)
@pytest.mark.parametrize(
    "interaction_type",
    [InteractionType.ATTRIBUTE_MULTI, InteractionType.ATTRIBUTE_YN],
)
def test_every_mood_dim_renders_question_and_options(dim, interaction_type):
    question, options = render_question(dim, interaction_type)
    assert isinstance(question, str) and question.strip()
    assert isinstance(options, list) and len(options) > 0
    assert all(isinstance(o, str) and o.strip() for o in options)


@pytest.mark.parametrize("dim", TONE_DIMENSIONS)
@pytest.mark.parametrize(
    "interaction_type",
    [InteractionType.ATTRIBUTE_MULTI, InteractionType.ATTRIBUTE_YN],
)
def test_every_tone_dim_renders_question_and_options(dim, interaction_type):
    question, options = render_question(dim, interaction_type)
    assert isinstance(question, str) and question.strip()
    assert isinstance(options, list) and len(options) > 0
    assert all(isinstance(o, str) and o.strip() for o in options)


@pytest.mark.parametrize("dim", list(MOOD_DIMS) + list(TONE_DIMENSIONS))
def test_attribute_yn_options_are_exactly_yes_no(dim):
    _, options = render_question(dim, InteractionType.ATTRIBUTE_YN)
    assert options == ["Yes", "No"]


@pytest.mark.parametrize("dim", list(MOOD_DIMS) + list(TONE_DIMENSIONS))
def test_attribute_multi_has_at_least_two_options(dim):
    _, options = render_question(dim, InteractionType.ATTRIBUTE_MULTI)
    assert len(options) >= 2


def test_unknown_intent_raises_key_error():
    with pytest.raises(KeyError):
        render_question("not_a_real_dimension", InteractionType.ATTRIBUTE_YN)


def test_returned_options_list_does_not_mutate_bank():
    _, options = render_question("humor", InteractionType.ATTRIBUTE_MULTI)
    options.append("MUTATED")
    _, fresh_options = render_question("humor", InteractionType.ATTRIBUTE_MULTI)
    assert "MUTATED" not in fresh_options
    assert (
        fresh_options
        == TEMPLATES["humor"][InteractionType.ATTRIBUTE_MULTI][1]
    )
