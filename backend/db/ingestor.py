from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.db.setup import setup_db


def ingest_episode(episode_id: str, data_root: Path, db_path: Path) -> None:
    extracted_path = data_root / episode_id / "extracted_data.json"
    if not extracted_path.exists():
        raise FileNotFoundError(f"Missing extracted_data.json for {episode_id}: {extracted_path}")

    extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
    metadata = extracted.get("metadata") or {}
    narrative = extracted.get("narrative") or {}
    cast = narrative.get("cast") or {}

    row: dict[str, Any] = {
        "episode_id": episode_id,
        "series_slug": metadata.get("series_slug"),
        "series_title": metadata.get("series_title"),
        "season_number": metadata.get("season_number"),
        "episode_number": metadata.get("episode_number"),
        "episode_title": metadata.get("episode_title"),
        "air_date": metadata.get("air_date"),
        "synopsis": narrative.get("synopsis"),
        "cold_open": narrative.get("cold_open"),
        "cast_main": json.dumps(cast.get("main") or [], ensure_ascii=False),
        "cast_supporting": json.dumps(cast.get("supporting") or [], ensure_ascii=False),
        "cast_recurring": json.dumps(cast.get("recurring") or [], ensure_ascii=False),
        "humor_level": None,
        "energy_level": None,
        "comfort_level": None,
        "sadness_level": None,
        "tone_labels": None,
        "enriched_at": None,
        "enrichment_model": None,
    }

    mood_path = data_root / episode_id / "mood_enriched.json"
    if mood_path.exists():
        mood_enriched = json.loads(mood_path.read_text(encoding="utf-8"))
        row["enriched_at"] = mood_enriched.get("enriched_at")
        row["enrichment_model"] = mood_enriched.get("model_id")
        if not mood_enriched.get("skipped"):
            mood = mood_enriched.get("mood") or {}
            row["humor_level"] = mood.get("humor_level")
            row["energy_level"] = mood.get("energy_level")
            row["comfort_level"] = mood.get("comfort_level")
            row["sadness_level"] = mood.get("sadness_level")
            tone = mood.get("tone")
            if tone is not None:
                row["tone_labels"] = json.dumps(tone, ensure_ascii=False)

    setup_db(db_path)

    columns = ", ".join(row.keys())
    placeholders = ", ".join(f":{key}" for key in row.keys())
    sql = f"INSERT OR REPLACE INTO episodes ({columns}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, row)
        conn.commit()
