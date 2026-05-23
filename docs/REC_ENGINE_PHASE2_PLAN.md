# Recommendation Engine — Phase 2: Vector Indexer

## Goal

Build `backend/recommender/vector_indexer.py` — a module that reads all episode rows
from SQLite, builds their 17-dim vectors using Phase 1's `build_features`, and upserts
them into ChromaDB with metadata attached.

After Phase 2, ChromaDB will have all 199 indexable episodes (201 total minus 2 skipped)
stored as unit vectors, queryable by cosine similarity.

---

## Context

- **Phase 1 (done):** `episode_feature_builder.py` — converts a SQLite row dict into
  an `EpisodeFeatures` (17-dim unit vector + mood_vec + tone_vec). Returns `None` for
  skipped episodes.
- **Existing ChromaDB:** `data/chroma/`, collection `episode_mood_vectors`, currently
  has 22 entries (Season 2 only, built from the old embedding CLI). Phase 2 wipes and
  rebuilds this collection from SQLite as the authoritative source.
- **Existing chroma_store.py:** Lives in `backend/embedding/`. Do NOT import it from
  the recommender package. The vector_indexer uses `chromadb` directly.

---

## Architecture Decisions

- SQLite is the source of truth. The indexer reads from SQLite only — no file system
  access to `mood_enriched.json` or `mood_vector.json`.
- The indexer delegates all vector math to `build_features` from Phase 1.
- Chroma is a derived index. It can be fully rebuilt at any time from SQLite.
- Upsert semantics: re-running the indexer without `--wipe` is idempotent (updates
  existing entries, adds new ones).

---

## Collection

- **Name:** `episode_mood_vectors` (same collection the old embedding CLI used)
- **Distance metric:** cosine (`hnsw:space: cosine`)
- **ID:** `episode_id` string (e.g. `the_office_s02_e04`)
- **Embedding:** 17-dim float32 unit vector from `EpisodeFeatures.vector`
- **Metadata stored per entry:**

```python
{
    "series_slug":    str,   # e.g. "the_office"
    "season_number":  int,   # e.g. 2
    "episode_number": int,   # e.g. 4
    "episode_title":  str,   # e.g. "The Fire"
}
```

Keep metadata minimal. Full episode detail lives in SQLite and is fetched by episode_id.

---

## SQLite Query

The indexer fetches feature columns AND metadata columns in a single query:

```sql
SELECT episode_id, series_slug, season_number, episode_number, episode_title,
       humor_level, energy_level, comfort_level, sadness_level, tone_scores
FROM episodes
```

With optional `WHERE series_slug = ?` when filtering by show.

Do NOT reuse `load_all_rows` from Phase 1 — it only fetches feature columns.
The indexer does its own query to get metadata columns alongside feature columns.

---

## Module Structure

### Files to create

```
backend/recommender/
├── __init__.py          (already exists — do not modify)
├── episode_feature_builder.py   (Phase 1 — do not modify)
└── vector_indexer.py            (new)

tests/recommender/
├── __init__.py          (already exists — do not modify)
├── test_episode_feature_builder.py   (Phase 1 — do not modify)
└── test_vector_indexer.py            (new)
```

No `__main__.py` for the recommender package yet — CLI lives inside `vector_indexer.py`
via `if __name__ == "__main__"`, invoked as:
```bash
python -m backend.recommender.vector_indexer --all the_office [--wipe]
python -m backend.recommender.vector_indexer --episode-id the_office_s02_e01
```

---

## `vector_indexer.py` Public Interface

```python
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chromadb

from backend.recommender.episode_feature_builder import build_features

COLLECTION_NAME: str = "episode_mood_vectors"


@dataclass
class IndexResult:
    indexed: int   # episodes successfully upserted into Chroma
    skipped: int   # episodes where build_features returned None
    total: int     # total rows fetched from SQLite


def get_collection(chroma_path: Path) -> chromadb.Collection:
    """Return (or create) the episode_mood_vectors collection."""


def index_all(
    db_path: Path,
    chroma_path: Path,
    series_slug: str | None = None,
    wipe: bool = False,
) -> IndexResult:
    """
    Read all episodes from SQLite, build vectors, upsert into Chroma.
    If series_slug is provided, only index that show.
    If wipe=True, delete and recreate the collection before indexing.
    """


def index_episode(
    episode_id: str,
    db_path: Path,
    chroma_path: Path,
) -> bool:
    """
    Index a single episode by episode_id.
    Returns True if indexed, False if skipped (no mood data).
    Raises if episode_id not found in SQLite.
    """
```

---

## Implementation Notes

### Fetching rows

```python
def _fetch_rows(db_path: Path, series_slug: str | None) -> list[dict]:
    _COLUMNS = (
        "episode_id, series_slug, season_number, episode_number, episode_title, "
        "humor_level, energy_level, comfort_level, sadness_level, tone_scores"
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if series_slug is None:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM episodes").fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM episodes WHERE series_slug = ?",
                (series_slug,),
            ).fetchall()
    return [dict(r) for r in rows]
```

### Upsert loop

```python
features = build_features(row)
if features is None:
    skipped += 1
    continue

collection.upsert(
    ids=[features.episode_id],
    embeddings=[features.vector.tolist()],
    metadatas=[{
        "series_slug":    row["series_slug"],
        "season_number":  row["season_number"],
        "episode_number": row["episode_number"],
        "episode_title":  row["episode_title"] or "",
    }],
)
indexed += 1
```

### Wipe behaviour

```python
if wipe:
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection may not exist yet
```

### CLI (bottom of vector_indexer.py)

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", metavar="SERIES_SLUG")
    group.add_argument("--episode-id", metavar="EPISODE_ID")
    parser.add_argument("--wipe", action="store_true")
    args = parser.parse_args()

    DB_PATH = Path("data/app.sqlite3")
    CHROMA_PATH = Path("data/chroma")

    if args.all:
        result = index_all(DB_PATH, CHROMA_PATH, series_slug=args.all, wipe=args.wipe)
        print(f"Done: {result.indexed} indexed, {result.skipped} skipped, {result.total} total")
    else:
        ok = index_episode(args.episode_id, DB_PATH, CHROMA_PATH)
        print("indexed" if ok else "skipped (no mood data)")
```

---

## Tests

### File: `tests/recommender/test_vector_indexer.py`

Use `pytest` with `tmp_path` for both Chroma and SQLite — no dependency on real
`data/` files.

---

### Fixture: `make_db`

```python
def make_db(tmp_path: Path, episodes: list[dict]) -> Path:
    """
    Create a fresh SQLite DB at tmp_path/app.sqlite3 with the given episode rows.
    Each episode dict should have all columns the indexer fetches.
    """
    db_path = tmp_path / "app.sqlite3"
    # use setup_db to create schema, then insert rows
    from backend.db.setup import setup_db
    setup_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for ep in episodes:
            conn.execute("""
                INSERT INTO episodes (episode_id, series_slug, season_number,
                    episode_number, episode_title,
                    humor_level, energy_level, comfort_level, sadness_level, tone_scores)
                VALUES (:episode_id, :series_slug, :season_number,
                    :episode_number, :episode_title,
                    :humor_level, :energy_level, :comfort_level, :sadness_level, :tone_scores)
            """, ep)
        conn.commit()
    return db_path
```

---

### Episode factory

```python
import json

def ep(
    episode_id: str = "show_s01_e01",
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
    humor: float | None = 0.9,
    energy: float | None = 0.8,
    comfort: float | None = 0.5,
    sadness: float | None = 0.1,
    tone: dict | None = None,
) -> dict:
    if tone is None:
        tone = {label: 0.5 for label in TONE_DIMENSIONS}
    return {
        "episode_id": episode_id,
        "series_slug": series_slug,
        "season_number": season,
        "episode_number": number,
        "episode_title": title,
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": json.dumps(tone),
    }

def skipped_ep(episode_id: str = "show_s01_e99") -> dict:
    """Episode with no mood data — build_features returns None."""
    return {
        "episode_id": episode_id,
        "series_slug": "the_office",
        "season_number": 1,
        "episode_number": 99,
        "episode_title": "Skipped",
        "humor_level": None,
        "energy_level": None,
        "comfort_level": None,
        "sadness_level": None,
        "tone_scores": None,
    }
```

---

### Test 1 — Single episode indexed

```python
def test_index_single_episode(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s01_e01")])
    chroma_path = tmp_path / "chroma"
    ok = index_episode("show_s01_e01", db_path, chroma_path)
    assert ok is True
    col = get_collection(chroma_path)
    assert col.count() == 1
    assert col.get(ids=["show_s01_e01"])["ids"] == ["show_s01_e01"]
```

---

### Test 2 — Skipped episode not indexed

```python
def test_index_skipped_returns_false(tmp_path):
    db_path = make_db(tmp_path, [skipped_ep("show_s01_e99")])
    chroma_path = tmp_path / "chroma"
    ok = index_episode("show_s01_e99", db_path, chroma_path)
    assert ok is False
    col = get_collection(chroma_path)
    assert col.count() == 0
```

---

### Test 3 — index_all count and IndexResult

```python
def test_index_all_count(tmp_path):
    episodes = [ep("show_s01_e01"), ep("show_s01_e02"), skipped_ep("show_s01_e99")]
    db_path = make_db(tmp_path, episodes)
    chroma_path = tmp_path / "chroma"
    result = index_all(db_path, chroma_path)
    assert result.indexed == 2
    assert result.skipped == 1
    assert result.total == 3
    assert get_collection(chroma_path).count() == 2
```

---

### Test 4 — Metadata fields stored correctly

```python
def test_metadata_fields(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s02_e04", season=2, number=4, title="The Fire")])
    chroma_path = tmp_path / "chroma"
    index_episode("show_s02_e04", db_path, chroma_path)
    col = get_collection(chroma_path)
    meta = col.get(ids=["show_s02_e04"], include=["metadatas"])["metadatas"][0]
    assert meta["series_slug"] == "the_office"
    assert meta["season_number"] == 2
    assert meta["episode_number"] == 4
    assert meta["episode_title"] == "The Fire"
```

---

### Test 5 — Stored vector matches feature builder output

```python
def test_vector_matches_feature_builder(tmp_path):
    import numpy as np
    episode = ep("show_s01_e01")
    db_path = make_db(tmp_path, [episode])
    chroma_path = tmp_path / "chroma"
    index_episode("show_s01_e01", db_path, chroma_path)

    col = get_collection(chroma_path)
    stored = col.get(ids=["show_s01_e01"], include=["embeddings"])["embeddings"][0]

    from backend.recommender.episode_feature_builder import build_features
    expected = build_features(episode)
    assert expected is not None
    np.testing.assert_allclose(stored, expected.vector.tolist(), atol=1e-5)
```

---

### Test 6 — Wipe clears previous entries

```python
def test_wipe_clears_existing(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s01_e01"), ep("show_s01_e02")])
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    assert get_collection(chroma_path).count() == 2

    # rebuild DB with only one episode, wipe Chroma
    db_path2 = make_db(tmp_path / "db2", [ep("show_s01_e03")])
    index_all(db_path2, chroma_path, wipe=True)
    assert get_collection(chroma_path).count() == 1
```

---

### Test 7 — Idempotent upsert (no wipe)

```python
def test_idempotent_upsert(tmp_path):
    db_path = make_db(tmp_path, [ep("show_s01_e01")])
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    index_all(db_path, chroma_path)  # second run — no wipe
    assert get_collection(chroma_path).count() == 1
```

---

### Test 8 — series_slug filter

```python
def test_series_slug_filter(tmp_path):
    episodes = [
        ep("office_s01_e01", series_slug="the_office"),
        ep("friends_s01_e01", series_slug="friends"),
    ]
    db_path = make_db(tmp_path, episodes)
    chroma_path = tmp_path / "chroma"
    result = index_all(db_path, chroma_path, series_slug="the_office")
    assert result.indexed == 1
    assert result.total == 1
    col = get_collection(chroma_path)
    assert col.count() == 1
    assert col.get(ids=["office_s01_e01"])["ids"] == ["office_s01_e01"]
```

---

## Verification Commands

After implementation, rebuild the real Chroma index from scratch:

```bash
# Wipe and rebuild full index
python -m backend.recommender.vector_indexer --all the_office --wipe

# Expected output:
# Done: 199 indexed, 2 skipped, 201 total

# Sanity check: query similar episodes (reuse existing embedding CLI)
source .venv/bin/activate
python -m backend.embedding --similar-to the_office_s02_e04 --top-k 5
```

The `--similar-to` results from the existing embedding CLI should be identical before
and after Phase 2 (same vectors, same collection).

---

## Success Criteria

Phase 2 is complete when:

1. `python -m backend.recommender.vector_indexer --all the_office --wipe` reports
   `199 indexed, 2 skipped, 201 total`.
2. ChromaDB collection has exactly 199 entries.
3. All 8 tests pass.
4. Test 5 (vector matches feature builder) confirms no data loss in the upsert.
5. The existing `--similar-to` CLI still returns the same results as before Phase 2.
