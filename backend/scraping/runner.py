"""Scrape one episode to data/raw/{episode_id}/ (Fandom wiki)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from backend.scraping.episode_ref import EpisodeRef, make_episode_id
from backend.scraping.extracted_data import (
    build_extracted_data,
    narrative_plot_text,
    write_extracted_data,
)
from backend.scraping.providers.fandom_wiki import FandomWikiProvider
from backend.scraping.series_config import SeriesConfig


def run_scrape_for_ref(
    ref: EpisodeRef, series_config: SeriesConfig, data_root: Path
) -> dict[str, Any]:
    """
    Fetch wiki article, write raw caches + extracted_data.json under ``data_root / ref.episode_id``.

    For combined episodes (one wiki page, multiple broadcast episodes), fetches once and
    writes one directory per episode number.
    """
    provider = FandomWikiProvider(series_config, data_root=data_root)
    result = provider.fetch_episode(ref)

    primary_out_dir = data_root / ref.episode_id
    episode_outputs: list[dict[str, Any]] = []

    for ep_num in ref.all_episode_numbers():
        ep_id = make_episode_id(ref.series_slug, ref.season_number, ep_num)
        out_dir = data_root / ep_id
        if out_dir != primary_out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("wiki_parse_api.json", "wiki_article_from_api.html", "wiki_extracted.json"):
                src = primary_out_dir / filename
                if src.exists():
                    shutil.copy2(src, out_dir / filename)
            copied_extracted = out_dir / "wiki_extracted.json"
            if copied_extracted.exists():
                payload = json.loads(copied_extracted.read_text(encoding="utf-8"))
                payload["episode_id"] = ep_id
                copied_extracted.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        extracted = build_extracted_data(
            ref,
            result,
            series_config,
            season_number=ref.season_number,
            episode_number=ep_num,
            episode_id=ep_id,
        )
        extracted_path = out_dir / "extracted_data.json"
        write_extracted_data(extracted_path, extracted)
        episode_outputs.append(
            {
                "episode_id": ep_id,
                "output_dir": str(out_dir),
                "extracted_data": str(extracted_path),
            }
        )

    plot = narrative_plot_text(result.sections)
    trivia = result.sections.get("Trivia", "")

    files_written: list[str] = [
        str(primary_out_dir / "wiki_parse_api.json"),
        str(primary_out_dir / "wiki_article_from_api.html"),
        str(primary_out_dir / "wiki_extracted.json"),
    ]
    for item in episode_outputs:
        files_written.append(str(Path(item["output_dir"]) / "extracted_data.json"))

    return {
        "episode_id": ref.episode_id,
        "episode_ids_written": [item["episode_id"] for item in episode_outputs],
        "wiki_page": result.page_title,
        "output_dir": str(primary_out_dir),
        "episode_outputs": episode_outputs,
        "files_written": files_written,
        "api_url": result.api_url,
        "evidence": {
            "narrative_plot_char_count": len(plot),
            "narrative_trivia_char_count": len(trivia),
            "fetch_count": 1,
        },
        "section_ids": list(result.sections.keys()),
    }
