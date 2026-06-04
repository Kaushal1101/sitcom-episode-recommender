from __future__ import annotations

from backend.csm.interaction_types import InteractionType

THRESHOLD_ATTR_MULTI: float = 0.40
THRESHOLD_ATTR_YN: float = 0.60
THRESHOLD_ITEM_MULTI: float = 0.70
THRESHOLD_RECOMMEND: float = 0.90


def route(confidence: float, is_first_turn: bool) -> InteractionType:
    """Map a confidence score to an interaction type.

    The first turn is always OPEN_CHAT regardless of confidence. Otherwise the
    confidence falls into one of five bands; at-threshold values fall into the
    next (higher-confidence) band.
    """
    if is_first_turn:
        return InteractionType.OPEN_CHAT
    if confidence < THRESHOLD_ATTR_MULTI:
        return InteractionType.ATTRIBUTE_MULTI
    if confidence < THRESHOLD_ATTR_YN:
        return InteractionType.ATTRIBUTE_YN
    if confidence < THRESHOLD_ITEM_MULTI:
        return InteractionType.ITEM_MULTI
    if confidence < THRESHOLD_RECOMMEND:
        return InteractionType.ITEM_YN
    return InteractionType.RECOMMEND
