from __future__ import annotations

from enum import Enum


class InteractionType(str, Enum):
    """The kind of interaction the CSM has decided to surface this turn."""

    OPEN_CHAT = "open_chat"
    ATTRIBUTE_MULTI = "attribute_multi"
    ATTRIBUTE_YN = "attribute_yn"
    ITEM_MULTI = "item_multi"
    ITEM_YN = "item_yn"
    RECOMMEND = "recommend"
    SOFT_CRITIQUE = "soft_critique"
    RATING = "rating"
