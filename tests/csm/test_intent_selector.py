from __future__ import annotations

from backend.csm.intent_selector import select_intent
from backend.models.session_state import SessionState
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS

MOOD_DIMS = ("humor", "energy", "comfort", "sadness")
VALID_DIMS = set(MOOD_DIMS) | set(TONE_DIMENSIONS)


def make_state(
    mood: dict[str, float] | None = None,
    tone: dict[str, float] | None = None,
) -> SessionState:
    """Build a SessionState seeded with explicit mood/tone values.

    Any dimension not present in `mood`/`tone` is left at the dataclass
    default of 0.0.
    """
    state = SessionState()
    if mood:
        for name, value in mood.items():
            setattr(state.current_preferences.mood, name, value)
    if tone:
        for name, value in tone.items():
            state.current_preferences.tone_preferences[name] = value
    return state


def test_all_zeros_returns_humor_by_priority():
    assert select_intent(make_state()) == "humor"


def test_only_humor_expressed_picks_next_mood_dim():
    state = make_state(mood={"humor": 0.8})
    assert select_intent(state) == "energy"


def test_all_mood_dims_expressed_picks_first_tone_in_order():
    state = make_state(
        mood={"humor": 0.9, "energy": 0.9, "comfort": 0.9, "sadness": 0.9}
    )
    assert select_intent(state) == "awkward"
    assert TONE_DIMENSIONS[0] == "awkward"


def test_lone_zero_tone_dim_is_selected():
    nonzero_tones = {label: 0.5 for label in TONE_DIMENSIONS}
    nonzero_tones["bittersweet"] = 0.0
    state = make_state(
        mood={"humor": 0.7, "energy": 0.4, "comfort": 0.6, "sadness": 0.3},
        tone=nonzero_tones,
    )
    assert select_intent(state) == "bittersweet"


def test_returned_intent_is_always_a_known_dimension_name():
    cases = [
        make_state(),
        make_state(mood={"humor": 0.8}),
        make_state(
            mood={"humor": 0.9, "energy": 0.9, "comfort": 0.9, "sadness": 0.9}
        ),
        make_state(
            mood={"humor": -0.2, "energy": 0.1, "comfort": -0.05, "sadness": 0.3},
            tone={label: 0.4 for label in TONE_DIMENSIONS},
        ),
    ]
    for state in cases:
        assert select_intent(state) in VALID_DIMS
