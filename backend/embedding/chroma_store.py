from __future__ import annotations

from typing import Any

import chromadb

COLLECTION_NAME = "episode_mood_vectors"
CHROMA_PATH = "data/chroma"


def get_client(chroma_path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=chroma_path)


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_episode(
    collection: chromadb.Collection,
    vector_result: dict[str, Any],
    *,
    series_slug: str,
    season_number: int,
    episode_number: int,
) -> None:
    collection.upsert(
        ids=[vector_result["episode_id"]],
        embeddings=[vector_result["episode_vec"]],
        metadatas=[
            {
                "series_slug": series_slug,
                "season_number": season_number,
                "episode_number": episode_number,
            }
        ],
    )


def query_similar(
    collection: chromadb.Collection,
    episode_vec: list[float],
    *,
    top_k: int = 5,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    n_results = top_k + (1 if exclude_id else 0)
    results = collection.query(
        query_embeddings=[episode_vec],
        n_results=n_results,
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    matches: list[dict[str, Any]] = []
    for episode_id, distance, metadata in zip(ids, distances, metadatas, strict=True):
        if episode_id == exclude_id:
            continue
        matches.append(
            {
                "episode_id": episode_id,
                "similarity": round(1 - distance, 4),
                **metadata,
            }
        )
    return matches[:top_k]
