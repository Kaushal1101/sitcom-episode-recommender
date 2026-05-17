from __future__ import annotations

import logging
from urllib.parse import unquote, urlencode

from bs4 import BeautifulSoup

from backend.scraping.episode_ref import COMBINED_OVERRIDES, EpisodeRef
from backend.scraping.http_client import fetch_get
from backend.scraping.series_config import SeriesConfig

logger = logging.getLogger(__name__)


def _mw_api_parse_url(origin: str, page_title: str) -> str:
    base = origin.rstrip("/") + "/api.php"
    q = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "formatversion": "2",
        "format": "json",
    }
    return f"{base}?{urlencode(q)}"


def _wiki_title_from_href(href: str) -> str | None:
    if not href.startswith("/wiki/"):
        return None
    title = unquote(href.split("/wiki/", 1)[1].split("#")[0])
    if title.startswith("File:") or title.startswith("Category:"):
        return None
    return title


def _parse_episode_numbers(raw: str) -> tuple[int, tuple[int, ...]] | None:
    """Parse episode number cell: '4', '1/2', or '18/19'."""
    raw = raw.strip().rstrip("‡").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split("/") if p.strip()]
    nums: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        nums.append(int(part))
    if not nums:
        return None
    return nums[0], tuple(nums[1:])


class SeasonDiscoverer:
    def __init__(self, series_config: SeriesConfig) -> None:
        self.series_config = series_config

    def discover(self, season_number: int) -> list[EpisodeRef]:
        html = self._fetch_season_html(season_number)
        discovered = self._parse_episode_refs(html, season_number)
        return self._apply_combined_overrides(discovered, season_number)

    def _season_page_title(self, season_number: int) -> str:
        return self.series_config.season_page_template.format(n=season_number)

    def _fetch_season_html(self, season_number: int) -> str:
        page_title = self._season_page_title(season_number)
        api_url = _mw_api_parse_url(self.series_config.wiki_origin, page_title)
        resp = fetch_get(api_url)
        if resp.status_code == 404:
            raise ValueError(f"Season page not found (404): {api_url}")
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise ValueError(f"Wiki API error for {api_url}: {payload['error']}")
        parse_block = payload.get("parse")
        if not parse_block:
            raise ValueError(f"No parse block in API response for {api_url}")
        html = parse_block.get("text", "")
        if not html:
            raise ValueError(f"Empty HTML in API response for {api_url}")
        return html

    def _parse_episode_refs(self, html: str, season_number: int) -> list[EpisodeRef]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="episodelist")
        if table is None:
            page_title = self._season_page_title(season_number)
            api_url = _mw_api_parse_url(self.series_config.wiki_origin, page_title)
            raise ValueError(
                f"Episode table (table.episodelist) not found on season page: {api_url}"
            )

        refs: list[EpisodeRef] = []
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 5:
                continue

            title_cell = cells[1]
            link = title_cell.find("a", href=True)
            if link is None:
                logger.warning("Skipping row: no episode title link in season %s", season_number)
                continue

            wiki_page_title = _wiki_title_from_href(link["href"])
            if wiki_page_title is None:
                logger.warning(
                    "Skipping row: unparseable wiki href %r in season %s",
                    link["href"],
                    season_number,
                )
                continue

            ep_parsed = _parse_episode_numbers(cells[2].get_text())
            if ep_parsed is None:
                logger.warning(
                    "Skipping row: unparseable episode number %r in season %s",
                    cells[2].get_text(strip=True),
                    season_number,
                )
                continue

            episode_number, linked = ep_parsed
            refs.append(
                EpisodeRef(
                    series_slug=self.series_config.series_slug,
                    season_number=season_number,
                    episode_number=episode_number,
                    wiki_origin=self.series_config.wiki_origin,
                    wiki_page_title=wiki_page_title,
                    linked_episode_numbers=linked,
                )
            )

        if not refs:
            page_title = self._season_page_title(season_number)
            api_url = _mw_api_parse_url(self.series_config.wiki_origin, page_title)
            raise ValueError(f"No episodes parsed from season page: {api_url}")

        return refs

    def _apply_combined_overrides(
        self, discovered: list[EpisodeRef], season_number: int
    ) -> list[EpisodeRef]:
        overridden: list[EpisodeRef] = []
        seen_page_titles: set[str] = set()
        for ref in discovered:
            key = (self.series_config.series_slug, season_number, ref.episode_number)
            if key in COMBINED_OVERRIDES:
                override = COMBINED_OVERRIDES[key]
                if override.wiki_page_title not in seen_page_titles:
                    overridden.append(override)
                    seen_page_titles.add(override.wiki_page_title)
            else:
                overridden.append(ref)
        return overridden
