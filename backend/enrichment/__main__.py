"""Plain Python CLI for mood enrichment — no LangGraph."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from backend.enrichment.mood_tagger import enrich_episode, load_classifier
from backend.enrichment.text_formatter import format_for_zero_shot
from backend.scraping.series_config import THE_OFFICE, SeriesConfig

SERIES_REGISTRY: dict[str, SeriesConfig] = {
    "the_office": THE_OFFICE,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    return _repo_root() / "data" / "raw"


def _resolve_series(slug: str) -> SeriesConfig:
    config = SERIES_REGISTRY.get(slug)
    if config is None:
        known = ", ".join(sorted(SERIES_REGISTRY))
        raise SystemExit(f"Unknown series {slug!r}. Known: {known}")
    return config


def _episode_id_pattern(series_slug: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(series_slug)}_s\d{{2}}_e\d{{2}}$")


def _list_episode_ids(series_slug: str, *, season_number: int | None = None) -> list[str]:
    data_root = _data_root()
    if not data_root.is_dir():
        return []

    pattern = _episode_id_pattern(series_slug)
    episode_ids: list[str] = []
    for entry in data_root.iterdir():
        if not entry.is_dir():
            continue
        episode_id = entry.name
        if not pattern.match(episode_id):
            continue
        if season_number is not None:
            season_token = f"_s{season_number:02d}_"
            if season_token not in episode_id:
                continue
        if (entry / "extracted_data.json").exists():
            episode_ids.append(episode_id)

    return sorted(episode_ids)


def _show_text(episode_id: str) -> int:
    extracted_path = _data_root() / episode_id / "extracted_data.json"
    if not extracted_path.exists():
        raise SystemExit(f"Missing {extracted_path}")
    episode_data = json.loads(extracted_path.read_text(encoding="utf-8"))
    formatted = format_for_zero_shot(episode_data)
    if formatted is None:
        print("(no narrative content — enrichment would be skipped)", file=sys.stderr)
        return 1
    print(formatted)
    print(f"\n--- {len(formatted)} chars ---", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich scraped episodes with mood tags")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-id", metavar="ID", help="Enrich one episode")
    group.add_argument(
        "--season",
        nargs=2,
        metavar=("SLUG", "N"),
        help="Enrich all episodes in a season",
    )
    group.add_argument("--all", metavar="SLUG", help="Enrich all scraped episodes for a series")
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print formatted input text and exit (no model)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes that already have mood_enriched.json",
    )
    args = parser.parse_args(argv)

    if args.show_text:
        if not args.episode_id:
            raise SystemExit("--show-text requires --episode-id")
        return _show_text(args.episode_id)

    if args.episode_id:
        series_slug, _, _ = _parse_episode_id(args.episode_id)
        episode_ids = [args.episode_id]
    elif args.season:
        series_slug = args.season[0]
        season_number = int(args.season[1])
        _resolve_series(series_slug)
        episode_ids = _list_episode_ids(series_slug, season_number=season_number)
    else:
        series_slug = args.all
        _resolve_series(series_slug)
        episode_ids = _list_episode_ids(series_slug)

    if not episode_ids:
        raise SystemExit("No episodes found to enrich.")

    data_root = _data_root()
    classifier = load_classifier()

    enriched = 0
    skipped = 0
    failed = 0
    timings: list[float] = []

    for episode_id in episode_ids:
        out_path = data_root / episode_id / "mood_enriched.json"
        if args.skip_existing and out_path.exists():
            print(f"skip existing {episode_id}", file=sys.stderr)
            continue

        print(f"enriching {episode_id} ...", file=sys.stderr)
        started = time.perf_counter()
        try:
            result = enrich_episode(episode_id, data_root, classifier)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            if result.get("skipped"):
                skipped += 1
                print(f"  skipped ({result.get('reason')}) in {elapsed:.1f}s", file=sys.stderr)
            else:
                enriched += 1
                mood = result.get("mood", {})
                print(
                    f"  humor={mood.get('humor_level')} tone={mood.get('tone')} in {elapsed:.1f}s",
                    file=sys.stderr,
                )
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    avg_s = sum(timings) / len(timings) if timings else 0.0
    print(
        f"Done: {enriched} enriched, {skipped} skipped, {failed} failed "
        f"(avg {avg_s:.1f}s/episode on this run)",
        file=sys.stderr,
    )
    return 1 if failed else 0


def _parse_episode_id(episode_id: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"(.+)_s(\d+)_e(\d+)", episode_id)
    if not match:
        raise SystemExit(
            f"Invalid episode_id {episode_id!r}. Expected {{series_slug}}_s{{season:02d}}_e{{episode:02d}}"
        )
    return match.group(1), int(match.group(2)), int(match.group(3))


if __name__ == "__main__":
    raise SystemExit(main())
