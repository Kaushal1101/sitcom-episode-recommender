from __future__ import annotations

from backend.csm.decision import InteractionDecision
from backend.csm.interaction_types import InteractionType
from backend.csm.intent_selector import select_intent
from backend.csm.router import route
from backend.csm.templates import render_question
from backend.models.session_state import SessionState
from backend.recommender.engine import RecommendationEngine


class ConversationStrategy:
    """Main CSM entry point.

    Composes the router, intent selector, template bank, and recommendation
    engine into a single `next_action` call that returns the decision the UI
    layer should render this turn.
    """

    def __init__(self, engine: RecommendationEngine) -> None:
        self._engine = engine

    def next_action(self, state: SessionState) -> InteractionDecision:
        rec_ctx = state.recommendation_context
        last_recommended = (
            rec_ctx.previously_recommended[-1]
            if rec_ctx.previously_recommended
            else None
        )

        # Step 1: post-recommendation state takes precedence over confidence routing.
        if last_recommended is not None:
            if last_recommended in rec_ctx.accepted_recommendations:
                return InteractionDecision(
                    interaction_type=InteractionType.RATING,
                    question="How would you rate that episode? (1 = not for me, 5 = perfect)",
                    options=["1", "2", "3", "4", "5"],
                    candidates=None,
                    intent=None,
                )
            if last_recommended in rec_ctx.rejected_recommendations:
                intent = select_intent(state)
                question, options = render_question(
                    intent, InteractionType.ATTRIBUTE_MULTI
                )
                return InteractionDecision(
                    interaction_type=InteractionType.SOFT_CRITIQUE,
                    question="That one didn't land. " + question,
                    options=options,
                    candidates=None,
                    intent=intent,
                    episode_id=last_recommended,
                )

        # Step 2: first turn — no engine call needed.
        if len(state.conversation_history) == 0:
            return InteractionDecision(
                interaction_type=InteractionType.OPEN_CHAT,
                question="What kind of episode are you in the mood for?",
                options=None,
                candidates=None,
                intent=None,
            )

        # Step 3: get recommendation and route on confidence. Capture both
        # return values so the RECOMMEND branch can reuse `ranked` without a
        # second engine call.
        ranked, confidence_result = self._engine.get_recommendation(state)
        interaction_type = route(
            confidence_result.confidence, is_first_turn=False
        )

        # Step 4: build the decision for the routed interaction type.
        if interaction_type is InteractionType.ATTRIBUTE_MULTI:
            intent = select_intent(state)
            question, options = render_question(
                intent, InteractionType.ATTRIBUTE_MULTI
            )
            return InteractionDecision(
                interaction_type=interaction_type,
                question=question,
                options=options,
                candidates=None,
                intent=intent,
                confidence=confidence_result.confidence,
            )

        if interaction_type is InteractionType.ATTRIBUTE_YN:
            intent = select_intent(state)
            question, options = render_question(
                intent, InteractionType.ATTRIBUTE_YN
            )
            return InteractionDecision(
                interaction_type=interaction_type,
                question=question,
                options=options,
                candidates=None,
                intent=intent,
                confidence=confidence_result.confidence,
            )

        if interaction_type is InteractionType.ITEM_MULTI:
            candidates = self._engine.get_candidates(state, n=3)
            options = [
                f"{c.episode_title} ({c.series_slug})" for c in candidates
            ]
            return InteractionDecision(
                interaction_type=InteractionType.ITEM_MULTI,
                question="Which of these sounds most appealing right now?",
                options=options,
                candidates=candidates,
                intent=None,
                confidence=confidence_result.confidence,
            )

        if interaction_type is InteractionType.ITEM_YN:
            candidates = self._engine.get_candidates(state, n=1)
            candidate = candidates[0] if candidates else None
            question = (
                f"Does '{candidate.episode_title}' ({candidate.series_slug}) "
                "sound like something you'd watch?"
                if candidate
                else "No candidates available."
            )
            return InteractionDecision(
                interaction_type=InteractionType.ITEM_YN,
                question=question,
                options=["Yes", "No"],
                candidates=candidates if candidates else [],
                intent=None,
                confidence=confidence_result.confidence,
            )

        # interaction_type is InteractionType.RECOMMEND
        return InteractionDecision(
            interaction_type=InteractionType.RECOMMEND,
            question=(
                f"I recommend '{ranked.episode_title}' from {ranked.series_slug} "
                f"(S{ranked.season_number}E{ranked.episode_number})."
            ),
            options=None,
            candidates=None,
            intent=None,
            episode_id=ranked.episode_id,
            confidence=confidence_result.confidence,
        )
