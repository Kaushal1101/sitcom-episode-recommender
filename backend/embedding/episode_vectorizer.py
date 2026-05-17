from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MOOD_DIMENSIONS: list[str] = [
    "humor_level",
    "energy_level",
    "comfort_level",
    "sadness_level",
]

TONE_DIMENSIONS: list[str] = [
    "awkward",
    "chaotic",
    "lighthearted",
    "romantic",
    "tense",
    "heartwarming",
    "cringe",
    "silly",
    "dramatic",
    "emotional",
    "wholesome",
    "dark",
    "bittersweet",
]

MOOD_WEIGHT: float = 0.7
TONE_WEIGHT: float = 0.3


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def build_episode_vector(mood_enriched: dict) -> dict | None:
    """
    Build mood_vec, tone_vec, and combined episode_vec from a mood_enriched.json dict.
    Returns None if the episode was skipped (no narrative content).
    """
    if mood_enriched.get("skipped"):
        return None

    mood = mood_enriched["mood"]
    mood_vec = np.array([mood[dimension] for dimension in MOOD_DIMENSIONS], dtype=np.float32)
    tone_raw = mood["raw_scores"]["tone"]
    tone_vec = np.array([tone_raw[label] for label in TONE_DIMENSIONS], dtype=np.float32)

    mood_unit = _l2_normalize(mood_vec)
    tone_unit = _l2_normalize(tone_vec)

    combined = np.concatenate(
        [
            math.sqrt(MOOD_WEIGHT) * mood_unit,
            math.sqrt(TONE_WEIGHT) * tone_unit,
        ]
    )
    episode_vec = _l2_normalize(combined)

    return {
        "episode_id": mood_enriched["episode_id"],
        "mood_vec": mood_vec.tolist(),
        "tone_vec": tone_vec.tolist(),
        "episode_vec": episode_vec.tolist(),
        "mood_dimensions": MOOD_DIMENSIONS,
        "tone_dimensions": TONE_DIMENSIONS,
        "weights": {"mood": MOOD_WEIGHT, "tone": TONE_WEIGHT},
    }


def load_mood_enriched(episode_id: str, data_root: Path) -> dict:
    path = data_root / episode_id / "mood_enriched.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_mood_vector(result: dict, data_root: Path) -> None:
    path = data_root / result["episode_id"] / "mood_vector.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
