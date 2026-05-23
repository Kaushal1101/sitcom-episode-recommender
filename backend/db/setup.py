from __future__ import annotations

import sqlite3
from pathlib import Path

_CREATE_EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id          TEXT PRIMARY KEY,
    series_slug         TEXT,
    series_title        TEXT,
    season_number       INTEGER,
    episode_number      INTEGER,
    episode_title       TEXT,
    air_date            TEXT,
    synopsis            TEXT,
    cold_open           TEXT,
    cast_main           TEXT,
    cast_supporting     TEXT,
    cast_recurring      TEXT,
    humor_level         REAL,
    energy_level        REAL,
    comfort_level       REAL,
    sadness_level       REAL,
    tone_labels         TEXT,
    tone_scores         TEXT,
    enriched_at         TEXT,
    enrichment_model    TEXT
)
"""


def setup_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_EPISODES_TABLE)
        conn.commit()
