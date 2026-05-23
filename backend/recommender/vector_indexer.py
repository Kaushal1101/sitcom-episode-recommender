from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chromadb

from backend.recommender.episode_feature_builder import build_features

COLLECTION_NAME: str = "episode_mood_vectors"

_COLUMNS = (
    "episode_id, series_slug, season_number, episode_number, episode_title, "
    "humor_level, energy_level, comfort_level, sadness_level, tone_scores"
)


@dataclass
class IndexResult:
    indexed: int
    skipped: int
    total: int


def get_collection(chroma_path: Path) -> chromadb.Collection:
    """Return (or create) the episode_mood_vectors collection."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _fetch_rows(db_path: Path, series_slug: str | None) -> list[dict]:
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


def _fetch_row(db_path: Path, episode_id: str) -> dict | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def _upsert(collection: chromadb.Collection, row: dict) -> bool:
    features = build_features(row)
    if features is None:
        return False
    collection.upsert(
        ids=[features.episode_id],
        embeddings=[features.vector.tolist()],
        metadatas=[
            {
                "series_slug": row["series_slug"],
                "season_number": row["season_number"],
                "episode_number": row["episode_number"],
                "episode_title": row["episode_title"] or "",
            }
        ],
    )
    return True


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
    client = chromadb.PersistentClient(path=str(chroma_path))
    if wipe:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    rows = _fetch_rows(db_path, series_slug)
    indexed = 0
    skipped = 0
    for row in rows:
        if _upsert(collection, row):
            indexed += 1
        else:
            skipped += 1
    return IndexResult(indexed=indexed, skipped=skipped, total=len(rows))


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
    row = _fetch_row(db_path, episode_id)
    if row is None:
        raise ValueError(f"episode_id {episode_id!r} not found in {db_path}")
    collection = get_collection(chroma_path)
    return _upsert(collection, row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index episode feature vectors into ChromaDB from SQLite",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", metavar="SERIES_SLUG")
    group.add_argument("--episode-id", metavar="EPISODE_ID")
    parser.add_argument("--wipe", action="store_true")
    args = parser.parse_args()

    DB_PATH = Path("data/app.sqlite3")
    CHROMA_PATH = Path("data/chroma")

    if args.all:
        result = index_all(DB_PATH, CHROMA_PATH, series_slug=args.all, wipe=args.wipe)
        print(
            f"Done: {result.indexed} indexed, {result.skipped} skipped, "
            f"{result.total} total"
        )
    else:
        ok = index_episode(args.episode_id, DB_PATH, CHROMA_PATH)
        print("indexed" if ok else "skipped (no mood data)")
