from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.recommender.vector_indexer import get_collection

DEFAULT_TOP_K: int = 50


@dataclass
class Candidate:
    episode_id: str
    similarity: float
    series_slug: str
    season_number: int
    episode_number: int
    episode_title: str


def _build_where(
    series_slug: str | None,
    season_number: int | None,
) -> dict | None:
    conditions: list[dict] = []
    if series_slug is not None:
        conditions.append({"series_slug": series_slug})
    if season_number is not None:
        conditions.append({"season_number": season_number})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(
    user_vector: np.ndarray,
    chroma_path: Path,
    top_k: int = DEFAULT_TOP_K,
    series_slug: str | None = None,
    season_number: int | None = None,
) -> list[Candidate]:
    """
    Query ChromaDB with user_vector and return up to top_k nearest candidates.

    Results are sorted by similarity descending (most similar first).
    Returns an empty list if the collection is empty or no results match the filters.
    """
    collection = get_collection(chroma_path)
    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)
    where = _build_where(series_slug, season_number)

    query_kwargs: dict = dict(
        query_embeddings=[user_vector.tolist()],
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    candidates: list[Candidate] = []
    for episode_id, distance, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
        strict=True,
    ):
        candidates.append(
            Candidate(
                episode_id=episode_id,
                similarity=round(1.0 - float(distance), 6),
                series_slug=meta["series_slug"],
                season_number=int(meta["season_number"]),
                episode_number=int(meta["episode_number"]),
                episode_title=meta["episode_title"],
            )
        )
    return candidates
