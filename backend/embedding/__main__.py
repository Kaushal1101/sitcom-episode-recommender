"""Plain Python CLI for episode mood/tone vectors — no LangGraph."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.cli_utils import data_root, list_episode_ids, parse_episode_id, repo_root, resolve_series
from backend.embedding.chroma_store import (
    CHROMA_PATH,
    get_client,
    get_collection,
    query_similar,
    upsert_episode,
)
from backend.embedding.episode_vectorizer import (
    build_episode_vector,
    load_mood_enriched,
    write_mood_vector,
)


def _chroma_path() -> str:
    return str(repo_root() / CHROMA_PATH)


def _vectorize_episode(
    episode_id: str,
    *,
    raw_data_root: Path,
    collection,
    skip_existing: bool,
) -> str:
    out_path = raw_data_root / episode_id / "mood_vector.json"
    if skip_existing and out_path.exists():
        print(f"skip existing {episode_id}", file=sys.stderr)
        return "existing"

    print(f"vectorizing {episode_id} ...", file=sys.stderr)
    mood_enriched = load_mood_enriched(episode_id, raw_data_root)
    vector_result = build_episode_vector(mood_enriched)
    if vector_result is None:
        print(f"  skipped (no mood data for {episode_id})", file=sys.stderr)
        return "skipped"

    write_mood_vector(vector_result, raw_data_root)
    series_slug, season_number, episode_number = parse_episode_id(episode_id)
    upsert_episode(
        collection,
        vector_result,
        series_slug=series_slug,
        season_number=season_number,
        episode_number=episode_number,
    )
    print(f"  upserted {episode_id}", file=sys.stderr)
    return "vectorized"


def _run_similar_to(episode_id: str, top_k: int) -> int:
    vector_path = data_root() / episode_id / "mood_vector.json"
    if not vector_path.exists():
        raise SystemExit(
            f"Missing {vector_path}. Run vectorization for {episode_id!r} first."
        )

    vector_data = json.loads(vector_path.read_text(encoding="utf-8"))
    collection = get_collection(get_client(_chroma_path()))
    matches = query_similar(
        collection,
        vector_data["episode_vec"],
        top_k=top_k,
        exclude_id=episode_id,
    )

    print(f"Nearest neighbours to {episode_id}:")
    for rank, match in enumerate(matches, start=1):
        print(
            f"  {rank}. {match['episode_id']} "
            f"(similarity={match['similarity']}, S{match['season_number']}E{match['episode_number']})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build episode mood/tone vectors and ChromaDB index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-id", metavar="ID", help="Vectorize one episode")
    group.add_argument(
        "--season",
        nargs=2,
        metavar=("SLUG", "N"),
        help="Vectorize all enriched episodes in a season",
    )
    group.add_argument("--all", metavar="SLUG", help="Vectorize all enriched episodes for a series")
    group.add_argument(
        "--similar-to",
        metavar="ID",
        help="Print top-k nearest neighbours for an episode",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes that already have mood_vector.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar episodes to return (default: 5)",
    )
    args = parser.parse_args(argv)

    if args.similar_to:
        return _run_similar_to(args.similar_to, args.top_k)

    if args.episode_id:
        episode_ids = [args.episode_id]
    elif args.season:
        series_slug = args.season[0]
        season_number = int(args.season[1])
        resolve_series(series_slug)
        episode_ids = list_episode_ids(series_slug, season_number=season_number, required_file="mood_enriched.json")
    else:
        series_slug = args.all
        resolve_series(series_slug)
        episode_ids = list_episode_ids(series_slug, required_file="mood_enriched.json")

    if not episode_ids:
        raise SystemExit("No enriched episodes found to vectorize.")

    raw_data_root = data_root()
    collection = get_collection(get_client(_chroma_path()))

    vectorized = 0
    skipped = 0
    existing = 0
    failed = 0

    for episode_id in episode_ids:
        try:
            status = _vectorize_episode(
                episode_id,
                raw_data_root=raw_data_root,
                collection=collection,
                skip_existing=args.skip_existing,
            )
            if status == "vectorized":
                vectorized += 1
            elif status == "skipped":
                skipped += 1
            else:
                existing += 1
        except Exception as exc:
            failed += 1
            print(f"  FAILED {episode_id}: {exc}", file=sys.stderr)

    print(
        f"Done: {vectorized} vectorized, {skipped} skipped, {existing} existing, {failed} failed",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
