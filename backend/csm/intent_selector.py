from __future__ import annotations

from backend.models.session_state import SessionState
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS


def select_intent(state: SessionState) -> str:
    """Return the name of the preference dimension to ask about next.

    Picks the dimension whose current value is closest to 0.0 (i.e. the most
    uncertain / unexpressed signal). Ties are broken deterministically by the
    fixed priority order: mood dimensions (humor, energy, comfort, sadness)
    first, then tone dimensions in TONE_DIMENSIONS order.
    """
    mood = state.current_preferences.mood
    tone = state.current_preferences.tone_preferences

    ordered: list[tuple[str, float]] = [
        ("humor", mood.humor),
        ("energy", mood.energy),
        ("comfort", mood.comfort),
        ("sadness", mood.sadness),
    ]
    for tone_name in TONE_DIMENSIONS:
        ordered.append((tone_name, tone.get(tone_name, 0.0)))

    best_name, best_distance = ordered[0][0], abs(ordered[0][1])
    for name, value in ordered[1:]:
        distance = abs(value)
        if distance < best_distance:
            best_name, best_distance = name, distance
    return best_name
