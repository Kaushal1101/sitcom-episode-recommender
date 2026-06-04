from __future__ import annotations

import pytest

from backend.csm.interaction_types import InteractionType
from backend.csm.router import route


@pytest.mark.parametrize("confidence", [0.0, 0.25, 0.50, 0.75, 0.90, 1.0])
def test_first_turn_always_open_chat(confidence):
    assert route(confidence, is_first_turn=True) is InteractionType.OPEN_CHAT


def test_zero_confidence_is_attribute_multi():
    assert route(0.0, is_first_turn=False) is InteractionType.ATTRIBUTE_MULTI


def test_just_below_attr_multi_threshold_is_attribute_multi():
    assert route(0.39, is_first_turn=False) is InteractionType.ATTRIBUTE_MULTI


def test_at_attr_multi_threshold_promotes_to_attribute_yn():
    assert route(0.40, is_first_turn=False) is InteractionType.ATTRIBUTE_YN


def test_just_below_attr_yn_threshold_is_attribute_yn():
    assert route(0.59, is_first_turn=False) is InteractionType.ATTRIBUTE_YN


def test_at_attr_yn_threshold_promotes_to_item_multi():
    assert route(0.60, is_first_turn=False) is InteractionType.ITEM_MULTI


def test_just_below_item_multi_threshold_is_item_multi():
    assert route(0.69, is_first_turn=False) is InteractionType.ITEM_MULTI


def test_at_item_multi_threshold_promotes_to_item_yn():
    assert route(0.70, is_first_turn=False) is InteractionType.ITEM_YN


def test_just_below_recommend_threshold_is_item_yn():
    assert route(0.89, is_first_turn=False) is InteractionType.ITEM_YN


def test_at_recommend_threshold_is_recommend():
    assert route(0.90, is_first_turn=False) is InteractionType.RECOMMEND


def test_full_confidence_is_recommend():
    assert route(1.0, is_first_turn=False) is InteractionType.RECOMMEND
