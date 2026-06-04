from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
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
VECTOR_DIM: int = len(MOOD_DIMENSIONS) + len(TONE_DIMENSIONS)


@dataclass
class EpisodeFeatures:
    episode_id: str
    vector: np.ndarray
    mood_vec: np.ndarray
    tone_vec: np.ndarray


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def build_features(row: dict) -> EpisodeFeatures | None:
    """
    Convert a SQLite row dict into EpisodeFeatures.

    Returns None if mood and tone data are both absent (skipped episodes).
    row must contain: episode_id, humor_level, energy_level, comfort_level,
    sadness_level, tone_scores (JSON string or None).
    """
    mood_values: list[float] = []
    for dimension in MOOD_DIMENSIONS:
        value = row.get(dimension)
        mood_values.append(float(value) if value is not None else 0.0)
    mood_vec = np.array(mood_values, dtype=np.float32)

    tone_scores_raw = row.get("tone_scores")
    if tone_scores_raw:
        tone_scores_dict = json.loads(tone_scores_raw)
    else:
        tone_scores_dict = {}
    tone_vec = np.array(
        [float(tone_scores_dict.get(label, 0.0)) for label in TONE_DIMENSIONS],
        dtype=np.float32,
    )

    if not np.any(mood_vec) and not np.any(tone_vec):
        return None

    mood_unit = _l2_normalize(mood_vec)
    tone_unit = _l2_normalize(tone_vec)

    combined = np.concatenate(
        [
            math.sqrt(MOOD_WEIGHT) * mood_unit,
            math.sqrt(TONE_WEIGHT) * tone_unit,
        ]
    )
    vector = _l2_normalize(combined).astype(np.float32)

    return EpisodeFeatures(
        episode_id=row["episode_id"],
        vector=vector,
        mood_vec=mood_vec,
        tone_vec=tone_vec,
    )


_SELECT_COLUMNS = (
    "episode_id, humor_level, energy_level, comfort_level, sadness_level, tone_scores"
)


def load_mood_rows(episode_ids: list[str], db_path: Path) -> dict[str, dict]:
    """Fetch mood columns for a list of episode_ids in one query. Returns {episode_id: row}."""
    if not episode_ids:
        return {}
    placeholders = ",".join("?" * len(episode_ids))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM episodes WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
    return {dict(r)["episode_id"]: dict(r) for r in rows}


def load_episode_row(episode_id: str, db_path: Path) -> dict | None:
    """Fetch a single episode row from SQLite. Returns None if not found."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM episodes WHERE episode_id = ?",
            (episode_id,),
        )
        result = cursor.fetchone()
    if result is None:
        return None
    return dict(result)


def load_all_rows(db_path: Path, series_slug: str | None = None) -> list[dict]:
    """
    Fetch all episode rows from SQLite.
    If series_slug is provided, filter to that show only.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if series_slug is None:
            cursor = conn.execute(f"SELECT {_SELECT_COLUMNS} FROM episodes")
        else:
            cursor = conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM episodes WHERE series_slug = ?",
                (series_slug,),
            )
        return [dict(row) for row in cursor.fetchall()]


def load_synopses(episode_ids: list[str], db_path: Path) -> dict[str, str]:
    """Fetch synopsis for a list of episode_ids. Returns {episode_id: synopsis}."""
    if not episode_ids:
        return {}
    placeholders = ",".join("?" * len(episode_ids))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT episode_id, synopsis FROM episodes "
            f"WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
    return {dict(r)["episode_id"]: (dict(r)["synopsis"] or "") for r in rows}
