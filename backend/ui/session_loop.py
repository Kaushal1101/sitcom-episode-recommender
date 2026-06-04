from __future__ import annotations

from datetime import datetime
from pathlib import Path

from backend.csm.decision import InteractionDecision
from backend.csm.interaction_types import InteractionType
from backend.csm.preference_updater import apply_answer
from backend.csm.strategy import ConversationStrategy
from backend.models.session_state import SessionState
from backend.recommender.engine import RecommendationEngine

_SEPARATOR = "─" * 40

_NUMBERED_TYPES = (
    InteractionType.ATTRIBUTE_MULTI,
    InteractionType.ITEM_MULTI,
    InteractionType.SOFT_CRITIQUE,
)
_YN_TYPES = (InteractionType.ATTRIBUTE_YN, InteractionType.ITEM_YN)


def _render(decision: InteractionDecision) -> None:
    print()
    interaction_type = decision.interaction_type

    if interaction_type is InteractionType.OPEN_CHAT:
        print(decision.question)
        return

    if interaction_type in _NUMBERED_TYPES:
        print(decision.question)
        for i, option in enumerate(decision.options or [], start=1):
            print(f"  {i}. {option}")
        return

    if interaction_type in _YN_TYPES:
        print(decision.question)
        print("  [yes / no]")
        return

    if interaction_type is InteractionType.RECOMMEND:
        print(_SEPARATOR)
        print("Recommendation:")
        print(decision.question)
        print(_SEPARATOR)
        print("Did you watch it? [yes / no]")
        return

    if interaction_type is InteractionType.RATING:
        print(decision.question)
        print("  [1 / 2 / 3 / 4 / 5]")
        return


def _collect_input(decision: InteractionDecision) -> str:
    interaction_type = decision.interaction_type

    while True:
        raw = input("> ").strip()
        if raw.lower() == "quit":
            return "quit"

        if interaction_type is InteractionType.OPEN_CHAT:
            if raw:
                return raw
            continue

        if interaction_type in _NUMBERED_TYPES:
            options = decision.options or []
            n = len(options)
            try:
                index = int(raw)
            except ValueError:
                print(f"  Please enter a number between 1 and {n}.")
                continue
            if 1 <= index <= n:
                return options[index - 1]
            print(f"  Please enter a number between 1 and {n}.")
            continue

        if interaction_type in _YN_TYPES or interaction_type is InteractionType.RECOMMEND:
            lowered = raw.lower()
            if lowered == "yes":
                return "Yes"
            if lowered == "no":
                return "No"
            print("  Please type 'yes' or 'no'.")
            continue

        if interaction_type is InteractionType.RATING:
            if raw in {"1", "2", "3", "4", "5"}:
                return raw
            print("  Please enter a number from 1 to 5.")
            continue

        return raw


def _update_state(
    state: SessionState,
    decision: InteractionDecision,
    answer: str,
    db_path: Path,
) -> None:
    apply_answer(state, decision, answer, db_path)

    if decision.interaction_type is InteractionType.RECOMMEND:
        episode_id = decision.episode_id
        if episode_id is not None:
            ctx = state.recommendation_context
            ctx.previously_recommended.append(episode_id)
            if answer == "Yes":
                ctx.accepted_recommendations.append(episode_id)
            elif answer == "No":
                ctx.rejected_recommendations.append(episode_id)

    state.timestamps.last_updated = datetime.utcnow()


def _print_farewell(decision: InteractionDecision, answer: str) -> None:
    if answer == "quit":
        print("See you next time.")
        return
    if decision.interaction_type is InteractionType.RATING:
        print("Thanks for the rating! Hope you enjoy it.")
        return
    print("Goodbye.")


def run(db_path: Path, chroma_path: Path) -> None:
    print("Sitcom Episode Recommender — type 'quit' to exit.")

    state = SessionState()
    engine = RecommendationEngine(db_path=db_path, chroma_path=chroma_path)
    strategy = ConversationStrategy(engine=engine)

    while True:
        try:
            decision = strategy.next_action(state)
        except ValueError:
            print("No episodes available. Try adjusting your preferences.")
            break

        _render(decision)
        answer = _collect_input(decision)

        if answer == "quit":
            _print_farewell(decision, answer)
            break

        _update_state(state, decision, answer, db_path)

        if decision.interaction_type is InteractionType.RATING:
            _print_farewell(decision, answer)
            break
