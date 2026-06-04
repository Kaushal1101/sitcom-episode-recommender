from __future__ import annotations

from dataclasses import dataclass

from backend.csm.interaction_types import InteractionType
from backend.recommender.confidence import TopCandidate


@dataclass
class InteractionDecision:
    """The CSM's resolved decision for a single conversational turn.

    Field population by interaction_type:
      - options:    populated for ATTRIBUTE_MULTI, ATTRIBUTE_YN, ITEM_MULTI,
                    ITEM_YN, SOFT_CRITIQUE, RATING; None otherwise.
      - candidates: populated for ITEM_MULTI and ITEM_YN; None otherwise.
      - intent:     populated when the question targets a specific preference
                    dimension (ATTRIBUTE_*, SOFT_CRITIQUE); None for ITEM_*,
                    RECOMMEND, RATING, OPEN_CHAT.
      - episode_id: populated only for RECOMMEND decisions; None otherwise.
      - confidence: populated for confidence-routed decisions; None for
                    OPEN_CHAT, RATING, and SOFT_CRITIQUE.
    """

    interaction_type: InteractionType
    question: str
    options: list[str] | None
    candidates: list[TopCandidate] | None
    intent: str | None
    episode_id: str | None = None
    confidence: float | None = None
