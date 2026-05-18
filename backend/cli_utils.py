from __future__ import annotations

import re
from pathlib import Path

from backend.scraping.series_config import THE_OFFICE, SeriesConfig

SERIES_REGISTRY: dict[str, SeriesConfig] = {
    "the_office": THE_OFFICE,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return repo_root() / "data" / "raw"


def resolve_series(slug: str) -> SeriesConfig:
    config = SERIES_REGISTRY.get(slug)
    if config is None:
        known = ", ".join(sorted(SERIES_REGISTRY))
        raise SystemExit(f"Unknown series {slug!r}. Known: {known}")
    return config


def parse_episode_id(episode_id: str) -> tuple[str, int, int]:
    m = re.fullmatch(r"(.+)_s(\d+)_e(\d+)", episode_id)
    if not m:
        raise SystemExit(
            f"Invalid episode_id {episode_id!r}. Expected {{series_slug}}_s{{season:02d}}_e{{episode:02d}}"
        )
    return m.group(1), int(m.group(2)), int(m.group(3))


def episode_id_pattern(series_slug: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(series_slug)}_s\d{{2}}_e\d{{2}}$")


def list_episode_ids(
    series_slug: str,
    *,
    season_number: int | None = None,
    required_file: str = "extracted_data.json",
) -> list[str]:
    root = data_root()
    if not root.is_dir():
        return []

    pattern = episode_id_pattern(series_slug)
    episode_ids: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        episode_id = entry.name
        if not pattern.match(episode_id):
            continue
        if season_number is not None and f"_s{season_number:02d}_" not in episode_id:
            continue
        if (entry / required_file).exists():
            episode_ids.append(episode_id)

    return sorted(episode_ids)
