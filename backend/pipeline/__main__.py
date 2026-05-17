"""Plain Python CLI for wiki scraping — no LangGraph."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from backend.scraping.episode_ref import make_episode_id
from backend.scraping.runner import run_scrape_for_ref
from backend.scraping.season_discoverer import SeasonDiscoverer
from backend.scraping.series_config import THE_OFFICE, SeriesConfig

SERIES_REGISTRY: dict[str, SeriesConfig] = {
    "the_office": THE_OFFICE,
}

THE_OFFICE_SEASON_RANGE = range(1, 10)


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


def _parse_episode_id(episode_id: str) -> tuple[str, int, int]:
    m = re.fullmatch(r"(.+)_s(\d+)_e(\d+)", episode_id)
    if not m:
        raise SystemExit(
            f"Invalid episode_id {episode_id!r}. Expected format: {{series_slug}}_s{{season:02d}}_e{{episode:02d}}"
        )
    return m.group(1), int(m.group(2)), int(m.group(3))


def _discover_refs(series_config: SeriesConfig, season_number: int) -> list:
    return SeasonDiscoverer(series_config).discover(season_number)


def _ref_for_episode_id(series_config: SeriesConfig, episode_id: str):
    series_slug, season, episode = _parse_episode_id(episode_id)
    if series_slug != series_config.series_slug:
        raise SystemExit(
            f"Episode {episode_id!r} belongs to {series_slug!r}, not {series_config.series_slug!r}"
        )
    refs = _discover_refs(series_config, season)
    for ref in refs:
        if episode_id in ref.all_episode_ids():
            return ref
    raise SystemExit(
        f"No episode ref found for {episode_id!r} in season {season}. "
        "Check season discovery or COMBINED_OVERRIDES."
    )


def _dedupe_refs(refs: list) -> list:
    """One scrape per unique wiki page (primary episode_number wins)."""
    by_page: dict[str, object] = {}
    for ref in refs:
        key = ref.wiki_page_title
        existing = by_page.get(key)
        if existing is None or ref.episode_number < existing.episode_number:
            by_page[key] = ref
    return sorted(by_page.values(), key=lambda r: r.episode_number)


def _scrape_refs(
    refs: list,
    series_config: SeriesConfig,
    *,
    skip_scrape: bool,
) -> tuple[list[dict], list[dict]]:
    data_root = _data_root()
    unique_refs = _dedupe_refs(refs)
    results: list[dict] = []
    errors: list[dict] = []

    for ref in unique_refs:
        target_ids = ref.all_episode_ids()
        if skip_scrape:
            print(f"would scrape {', '.join(target_ids)} ({ref.wiki_page_title})", file=sys.stderr)
            results.append({"episode_ids": list(target_ids), "wiki_page": ref.wiki_page_title, "skipped": True})
            continue

        for ep_id in target_ids:
            print(f"scraping {ep_id} ...", file=sys.stderr)
        try:
            summary = run_scrape_for_ref(ref, series_config, data_root)
            results.append(summary)
        except Exception as exc:
            print(f"FAILED {ref.episode_id}: {exc}", file=sys.stderr)
            errors.append({"episode_id": ref.episode_id, "error": str(exc)})

    return results, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape sitcom episode data from Fandom wikis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-id", metavar="ID", help="Scrape one episode by id")
    group.add_argument(
        "--season",
        nargs=2,
        metavar=("SLUG", "N"),
        help="Discover and scrape all episodes in a season",
    )
    group.add_argument(
        "--all-seasons",
        metavar="SLUG",
        help="Discover and scrape all seasons for a series",
    )
    parser.add_argument("--skip-scrape", action="store_true", help="Dry run: print targets only")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Print summary as JSON")
    args = parser.parse_args(argv)

    if args.episode_id:
        series_slug, season, _ = _parse_episode_id(args.episode_id)
        series_config = _resolve_series(series_slug)
        ref = _ref_for_episode_id(series_config, args.episode_id)
        refs = [ref]
    elif args.season:
        series_config = _resolve_series(args.season[0])
        season_number = int(args.season[1])
        refs = _discover_refs(series_config, season_number)
    else:
        series_config = _resolve_series(args.all_seasons)
        refs = []
        for season_number in THE_OFFICE_SEASON_RANGE:
            refs.extend(_discover_refs(series_config, season_number))

    results, errors = _scrape_refs(refs, series_config, skip_scrape=args.skip_scrape)

    summary = {
        "series_slug": series_config.series_slug,
        "refs_discovered": len(refs),
        "scrapes_attempted": len(results) + len(errors),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }

    if args.json_out:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Done: {summary['succeeded']} succeeded, {summary['failed']} failed "
            f"({summary['refs_discovered']} episodes discovered)",
            file=sys.stderr,
        )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
