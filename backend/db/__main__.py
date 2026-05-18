"""Plain Python CLI for SQLite episode ingestion — no LangGraph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.cli_utils import data_root, list_episode_ids, parse_episode_id, repo_root, resolve_series
from backend.db.ingestor import ingest_episode


def _db_path() -> Path:
    return repo_root() / "data" / "app.sqlite3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest episode JSON into SQLite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-id", metavar="ID", help="Ingest one episode")
    group.add_argument(
        "--season",
        nargs=2,
        metavar=("SLUG", "N"),
        help="Ingest all episodes in a season",
    )
    group.add_argument("--all", metavar="SLUG", help="Ingest all scraped episodes for a series")
    args = parser.parse_args(argv)

    if args.episode_id:
        resolve_series(parse_episode_id(args.episode_id)[0])
        episode_ids = [args.episode_id]
    elif args.season:
        series_slug = args.season[0]
        season_number = int(args.season[1])
        resolve_series(series_slug)
        episode_ids = list_episode_ids(series_slug, season_number=season_number)
    else:
        series_slug = args.all
        resolve_series(series_slug)
        episode_ids = list_episode_ids(series_slug)

    if not episode_ids:
        raise SystemExit("No episodes found to ingest.")

    raw_data_root = data_root()
    db_path = _db_path()

    ingested = 0
    failed = 0

    for episode_id in episode_ids:
        print(f"ingesting {episode_id} ...", file=sys.stderr)
        try:
            ingest_episode(episode_id, raw_data_root, db_path)
            ingested += 1
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    print(f"Done: {ingested} ingested, {failed} failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
