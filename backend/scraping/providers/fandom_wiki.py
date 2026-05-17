from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from bs4.element import Tag

from backend.scraping.episode_ref import EpisodeRef
from backend.scraping.http_client import fetch_get
from backend.scraping.series_config import SeriesConfig


@dataclass
class WikiArticleResult:
    episode_id: str
    wiki_origin: str
    page_title: str
    api_url: str
    raw_parse_json: dict[str, Any]
    article_html: str
    sections: dict[str, str] = field(default_factory=dict)
    infobox_text: str | None = None
    lede_text: str | None = None


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


def _section_text_after_h2(h2: Tag) -> str:
    chunks: list[str] = []
    for sib in h2.find_next_siblings():
        if getattr(sib, "name", None) == "h2":
            break
        if isinstance(sib, Tag):
            t = sib.get_text("\n", strip=True)
            if t:
                chunks.append(t)
    return "\n\n".join(chunks).strip()


def _extract_infobox_and_lede(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    root = soup.select_one(".mw-parser-output")
    if not root:
        return None, None
    infobox = root.select_one("aside.portable-infobox") or root.select_one("aside")
    infobox_text = infobox.get_text("\n", strip=True) if infobox else None
    lede: str | None = None
    parent_p = infobox.find_parent("p") if infobox else None
    if parent_p is not None and infobox is not None:
        infobox.extract()
        lede = parent_p.get_text(" ", strip=True) or None
    return infobox_text, lede


def _parse_sections_from_html(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output")
    if not root:
        return {}

    sections: dict[str, str] = {}
    for h2 in root.find_all("h2"):
        span = h2.find("span", class_="mw-headline")
        if not span or not span.get("id"):
            continue
        key = span["id"]
        sections[key] = _section_text_after_h2(h2)
    return sections


def fetch_fandom_article(
    ref: EpisodeRef,
    *,
    series_config: SeriesConfig,
    data_root: Path,
) -> WikiArticleResult:
    """
    Load episode article from Fandom via MediaWiki parse API (same content as the wiki page;
    avoids Cloudflare browser challenges on direct /wiki/ HTML fetches).
    """
    _ = series_config  # reserved for future per-series fetch options
    api_url = _mw_api_parse_url(ref.wiki_origin, ref.wiki_page_title)
    resp = fetch_get(api_url)
    if resp.status_code == 404:
        raise FileNotFoundError(f"Wiki API 404 for page={ref.wiki_page_title!r}")
    if resp.status_code == 403:
        raise PermissionError("Wiki API returned 403 Forbidden")
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise ValueError(f"Wiki API error: {payload['error']}")

    parse_block = payload.get("parse") or {}
    title = parse_block.get("title", ref.wiki_page_title)
    full_html = parse_block.get("text", "")
    soup = BeautifulSoup(full_html, "lxml")
    infobox_text, lede_text = _extract_infobox_and_lede(soup)
    sections = _parse_sections_from_html(full_html)

    out_dir = data_root / ref.episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "wiki_parse_api.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_path = out_dir / "wiki_article_from_api.html"
    html_path.write_text(full_html, encoding="utf-8")

    extracted = {
        "episode_id": ref.episode_id,
        "wiki_origin": ref.wiki_origin,
        "page_title": title,
        "api_url": api_url,
        "infobox_text": infobox_text,
        "lede_text": lede_text,
        "sections": sections,
    }
    extracted_path = out_dir / "wiki_extracted.json"
    extracted_path.write_text(json.dumps(extracted, indent=2, ensure_ascii=False), encoding="utf-8")

    return WikiArticleResult(
        episode_id=ref.episode_id,
        wiki_origin=ref.wiki_origin,
        page_title=title,
        api_url=api_url,
        raw_parse_json=payload,
        article_html=full_html,
        sections=sections,
        infobox_text=infobox_text,
        lede_text=lede_text,
    )


class FandomWikiProvider:
    def __init__(self, series_config: SeriesConfig, data_root: Path | None = None) -> None:
        self.series_config = series_config
        root = data_root
        if root is None:
            root = Path(__file__).resolve().parents[3] / "data" / "raw"
        self.data_root = root

    def fetch_episode(self, ref: EpisodeRef) -> WikiArticleResult:
        return fetch_fandom_article(ref, series_config=self.series_config, data_root=self.data_root)
