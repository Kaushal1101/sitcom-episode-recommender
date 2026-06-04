from __future__ import annotations

from backend.csm.interaction_types import InteractionType
from backend.recommender.episode_feature_builder import TONE_DIMENSIONS

TEMPLATES: dict[str, dict[InteractionType, tuple[str, list[str]]]] = {
    "humor": {
        InteractionType.ATTRIBUTE_MULTI: (
            "What kind of humour are you in the mood for?",
            [
                "Sharp and witty",
                "Silly and slapstick",
                "Dry and understated",
                "Not really looking for laughs",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something funny?",
            ["Yes", "No"],
        ),
    },
    "energy": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How energetic do you want the episode to feel?",
            ["Fast-paced and chaotic", "Steady and engaging", "Slow and relaxed"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Do you want something high-energy?",
            ["Yes", "No"],
        ),
    },
    "comfort": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How much comfort are you after?",
            ["Very cosy and safe", "A mix of comfort and tension", "Something edgier"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you looking for something warm and comforting?",
            ["Yes", "No"],
        ),
    },
    "sadness": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How emotionally heavy do you want it?",
            ["Light and easy", "A little bittersweet", "Something genuinely moving"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you open to something a bit sad or emotional?",
            ["Yes", "No"],
        ),
    },
    "awkward": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How do you feel about awkward cringe moments?",
            ["Love them", "Fine in small doses", "Please no"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you okay with awkward, cringe-y moments?",
            ["Yes", "No"],
        ),
    },
    "chaotic": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How much chaos and unpredictability do you want?",
            ["Maximum chaos", "Some wild moments", "Keep it grounded"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something chaotic and unpredictable?",
            ["Yes", "No"],
        ),
    },
    "lighthearted": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How lighthearted should the tone be?",
            [
                "Completely breezy",
                "Mostly light with some depth",
                "More serious overall",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Do you want something lighthearted and easy-going?",
            ["Yes", "No"],
        ),
    },
    "romantic": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How much romance do you want in the mix?",
            ["Heavy on the romance", "A subplot is fine", "Keep it minimal"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something romantic?",
            ["Yes", "No"],
        ),
    },
    "tense": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How much tension do you want?",
            [
                "Edge-of-your-seat stuff",
                "Some stakes would be good",
                "Low stakes please",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you okay with tense or stressful moments?",
            ["Yes", "No"],
        ),
    },
    "heartwarming": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How heartwarming do you want it to be?",
            [
                "Bring on the warm fuzzies",
                "A touching moment or two",
                "Skip the sentimentality",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something heartwarming?",
            ["Yes", "No"],
        ),
    },
    "cringe": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How do you feel about cringe comedy?",
            ["I live for it", "A little goes a long way", "Hard pass"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you okay with cringe comedy?",
            ["Yes", "No"],
        ),
    },
    "silly": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How silly do you want it to get?",
            ["Fully absurd", "Playfully silly", "Grounded and real"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something silly?",
            ["Yes", "No"],
        ),
    },
    "dramatic": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How dramatic should it be?",
            ["High drama", "A bit of drama mixed in", "Low drama"],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something dramatic?",
            ["Yes", "No"],
        ),
    },
    "emotional": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How emotionally intense do you want it?",
            [
                "Really hit me in the feels",
                "Some emotion is fine",
                "Keep it light",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Do you want something emotionally resonant?",
            ["Yes", "No"],
        ),
    },
    "wholesome": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How wholesome do you want the episode to be?",
            [
                "Pure wholesome goodness",
                "Mostly wholesome",
                "A bit of edge is fine",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you looking for something wholesome?",
            ["Yes", "No"],
        ),
    },
    "dark": {
        InteractionType.ATTRIBUTE_MULTI: (
            "How dark are you willing to go?",
            [
                "Bring on the dark themes",
                "A touch of darkness is fine",
                "Keep it bright",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you okay with dark or heavy themes?",
            ["Yes", "No"],
        ),
    },
    "bittersweet": {
        InteractionType.ATTRIBUTE_MULTI: (
            "Do you want something bittersweet?",
            [
                "Yes, that mix of happy and sad",
                "Maybe a little",
                "I want something clearer",
            ],
        ),
        InteractionType.ATTRIBUTE_YN: (
            "Are you in the mood for something bittersweet?",
            ["Yes", "No"],
        ),
    },
}


def render_question(
    intent: str, interaction_type: InteractionType
) -> tuple[str, list[str]]:
    """Look up a templated question for (intent, interaction_type).

    Returns (question_text, options). The options list is a fresh copy so the
    caller cannot mutate the underlying bank. Raises KeyError if either the
    intent or the interaction_type is not present in TEMPLATES.
    """
    intent_templates = TEMPLATES[intent]
    question_text, options = intent_templates[interaction_type]
    return question_text, list(options)
