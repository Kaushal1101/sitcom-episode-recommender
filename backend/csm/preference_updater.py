from __future__ import annotations

import json
from pathlib import Path

from backend.csm.decision import InteractionDecision
from backend.csm.interaction_types import InteractionType
from backend.models.session_state import ConversationMessage, SessionState
from backend.recommender.episode_feature_builder import (
    MOOD_DIMENSIONS,
    TONE_DIMENSIONS,
    load_mood_rows,
)

YN_POSITIVE: float = 0.8
YN_NEGATIVE: float = -0.8
ITEM_STEP: float = 0.3
ATTRIBUTE_MULTI_STEP: float = 0.5

MOOD_PREF_NAMES: tuple[str, ...] = ("humor", "energy", "comfort", "sadness")

_MOOD_DB_TO_PREF: tuple[tuple[str, str], ...] = (
    ("humor_level", "humor"),
    ("energy_level", "energy"),
    ("comfort_level", "comfort"),
    ("sadness_level", "sadness"),
)


OPTION_VALUES: dict[str, dict[str, float]] = {
    "humor": {
        "Sharp and witty": 0.8,
        "Silly and slapstick": 0.6,
        "Dry and understated": 0.4,
        "Not really looking for laughs": -0.8,
    },
    "energy": {
        "Fast-paced and chaotic": 0.9,
        "Steady and engaging": 0.3,
        "Slow and relaxed": -0.6,
    },
    "comfort": {
        "Very cosy and safe": 0.9,
        "A mix of comfort and tension": 0.3,
        "Something edgier": -0.6,
    },
    "sadness": {
        "Light and easy": -0.6,
        "A little bittersweet": 0.4,
        "Something genuinely moving": 0.8,
    },
    "awkward": {
        "Love them": 0.9,
        "Fine in small doses": 0.3,
        "Please no": -0.9,
    },
    "chaotic": {
        "Maximum chaos": 0.9,
        "Some wild moments": 0.4,
        "Keep it grounded": -0.6,
    },
    "lighthearted": {
        "Completely breezy": 0.9,
        "Mostly light with some depth": 0.4,
        "More serious overall": -0.6,
    },
    "romantic": {
        "Heavy on the romance": 0.9,
        "A subplot is fine": 0.3,
        "Keep it minimal": -0.6,
    },
    "tense": {
        "Edge-of-your-seat stuff": 0.9,
        "Some stakes would be good": 0.4,
        "Low stakes please": -0.6,
    },
    "heartwarming": {
        "Bring on the warm fuzzies": 0.9,
        "A touching moment or two": 0.4,
        "Skip the sentimentality": -0.6,
    },
    "cringe": {
        "I live for it": 0.9,
        "A little goes a long way": 0.3,
        "Hard pass": -0.9,
    },
    "silly": {
        "Fully absurd": 0.9,
        "Playfully silly": 0.5,
        "Grounded and real": -0.6,
    },
    "dramatic": {
        "High drama": 0.9,
        "A bit of drama mixed in": 0.4,
        "Low drama": -0.6,
    },
    "emotional": {
        "Really hit me in the feels": 0.9,
        "Some emotion is fine": 0.3,
        "Keep it light": -0.6,
    },
    "wholesome": {
        "Pure wholesome goodness": 0.9,
        "Mostly wholesome": 0.5,
        "A bit of edge is fine": -0.3,
    },
    "dark": {
        "Bring on the dark themes": 0.9,
        "A touch of darkness is fine": 0.3,
        "Keep it bright": -0.8,
    },
    "bittersweet": {
        "Yes, that mix of happy and sad": 0.8,
        "Maybe a little": 0.3,
        "I want something clearer": -0.6,
    },
}


def _write_preference(state: SessionState, intent: str, value: float) -> None:
    if intent in MOOD_PREF_NAMES:
        setattr(state.current_preferences.mood, intent, value)
    else:
        state.current_preferences.tone_preferences[intent] = value


def _read_preference(state: SessionState, intent: str) -> float:
    if intent in MOOD_PREF_NAMES:
        return getattr(state.current_preferences.mood, intent)
    return state.current_preferences.tone_preferences.get(intent, 0.0)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _apply_attribute(
    state: SessionState,
    intent: str,
    interaction_type: InteractionType,
    answer: str,
) -> None:
    if interaction_type is InteractionType.ATTRIBUTE_YN:
        if answer == "Yes":
            _write_preference(state, intent, YN_POSITIVE)
        elif answer == "No":
            _write_preference(state, intent, YN_NEGATIVE)
        return

    intent_options = OPTION_VALUES.get(intent)
    if intent_options is None:
        return
    if answer not in intent_options:
        return
    target = intent_options[answer]
    current = _read_preference(state, intent)
    blended = _clamp(current + ATTRIBUTE_MULTI_STEP * (target - current))
    _write_preference(state, intent, blended)


def _apply_item(
    state: SessionState,
    decision: InteractionDecision,
    answer: str,
    db_path: Path,
) -> None:
    candidates = decision.candidates or []

    if decision.interaction_type is InteractionType.ITEM_MULTI:
        candidate = None
        for c in candidates:
            if answer == f"{c.episode_title} ({c.series_slug})":
                candidate = c
                break
        if candidate is None:
            return
        direction = 1
    elif decision.interaction_type is InteractionType.ITEM_YN:
        if not candidates:
            return
        candidate = candidates[0]
        if answer == "Yes":
            direction = 1
        elif answer == "No":
            direction = -1
        else:
            return
    else:
        return

    rows = load_mood_rows([candidate.episode_id], db_path)
    row = rows.get(candidate.episode_id)
    if row is None:
        return

    for db_col, pref_attr in _MOOD_DB_TO_PREF:
        ep_val = float(row[db_col] or 0.0)
        current = getattr(state.current_preferences.mood, pref_attr)
        new_val = _clamp(current + direction * ITEM_STEP * (ep_val - current))
        setattr(state.current_preferences.mood, pref_attr, new_val)

    tone_dict = json.loads(row["tone_scores"] or "{}")
    for dim in TONE_DIMENSIONS:
        ep_val = float(tone_dict.get(dim, 0.0))
        current = state.current_preferences.tone_preferences.get(dim, 0.0)
        new_val = _clamp(current + direction * ITEM_STEP * (ep_val - current))
        state.current_preferences.tone_preferences[dim] = new_val


def apply_answer(
    state: SessionState,
    decision: InteractionDecision,
    answer: str,
    db_path: Path | None = None,
) -> None:
    """Update session state in response to the user's answer for a given decision.

    Always appends `answer` to conversation history. Updates preferences based
    on the decision's interaction type — attribute answers set scalar targets,
    item answers nudge preferences toward (or away from) the chosen episode's
    mood/tone profile.
    """
    state.conversation_history.append(ConversationMessage(message=answer))

    interaction_type = decision.interaction_type

    if interaction_type is InteractionType.ATTRIBUTE_YN:
        _apply_attribute(
            state, decision.intent, InteractionType.ATTRIBUTE_YN, answer
        )
        return

    if interaction_type is InteractionType.ATTRIBUTE_MULTI:
        _apply_attribute(
            state, decision.intent, InteractionType.ATTRIBUTE_MULTI, answer
        )
        return

    if interaction_type is InteractionType.SOFT_CRITIQUE:
        _apply_attribute(
            state, decision.intent, InteractionType.SOFT_CRITIQUE, answer
        )
        return

    if interaction_type in (InteractionType.ITEM_MULTI, InteractionType.ITEM_YN):
        if db_path is None:
            return
        _apply_item(state, decision, answer, db_path)
        return
