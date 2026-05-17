from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from bs4 import BeautifulSoup
from bs4.element import Tag

from backend.scraping.episode_ref import EpisodeRef
from backend.scraping.providers.fandom_wiki import WikiArticleResult
from backend.scraping.series_config import SeriesConfig

COLD_OPEN_IDS = ("Cold_open", "Cold_Open")
SUMMARY_IDS = ("Summary", "Synopsis")
CAST_BUCKETS = ("main", "supporting", "recurring", "other")


def narrative_plot_text(sections: Mapping[str, str]) -> str:
    """Fandom wikis use Cold_open + Summary instead of a heading named Plot."""
    cold = _first_non_empty(sections, COLD_OPEN_IDS) or ""
    summary = _first_non_empty(sections, SUMMARY_IDS) or ""
    parts = [p for p in (cold, summary) if p]
    return "\n\n".join(parts).strip()


def _first_non_empty(sections: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        text = (sections.get(k) or "").strip()
        if text:
            return text
    return None


def _split_cast_lines(cast_text: str) -> list[str]:
    lines = [line.strip() for line in cast_text.splitlines()]
    return [line for line in lines if line and line not in {"[", "]"}]


def _bucket_for_heading(line: str) -> str | None:
    normalized = re.sub(r"\[[^\]]*\]", "", line.lower()).strip().rstrip(":")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in {"main cast", "main"}:
        return "main"
    if normalized in {"supporting cast", "supporting", "guest cast", "guest"}:
        return "supporting"
    if normalized in {"recurring cast", "recurring"}:
        return "recurring"
    return None


def _looks_like_name_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if not stripped:
        return False
    if lower == "as" or lower.startswith("as "):
        return False
    if " as " in lower:
        return False
    if stripped.startswith("(") and stripped.endswith(")"):
        return False
    if any(ch in stripped for ch in ":;[]{}|"):
        return False

    tokens = re.split(r"\s+", stripped)
    if not 1 <= len(tokens) <= 5:
        return False

    for token in tokens:
        cleaned = token.strip(".,")
        if not cleaned:
            return False
        if re.match(r"^[A-Za-z][A-Za-z'.-]*$", cleaned) is None:
            return False
    return True


def _normalize_cast_lines(
    lines: list[str], *, default_bucket: str = "other", allow_headings: bool = True
) -> dict[str, list[str]]:
    active_bucket = default_bucket
    buckets: dict[str, list[str]] = {k: [] for k in CAST_BUCKETS}
    i = 0
    while i < len(lines):
        line = lines[i]
        heading_bucket = _bucket_for_heading(line) if allow_headings else None
        if heading_bucket:
            active_bucket = heading_bucket
            i += 1
            continue
        if line.lower() in {"spoiler", "spoiler:"}:
            i += 1
            continue

        if line.lower().startswith("as "):
            role = line[3:].strip()
            if role and buckets[active_bucket]:
                buckets[active_bucket][-1] = f"{buckets[active_bucket][-1]} as {role}"
            i += 1
            continue

        if (
            _looks_like_name_line(line)
            and i + 1 < len(lines)
            and lines[i + 1].lower().startswith("as ")
        ):
            role = lines[i + 1][3:].strip()
            if role:
                buckets[active_bucket].append(f"{line} as {role}")
                i += 2
                continue
            buckets[active_bucket].append(line)
            i += 2
            continue

        if line.lower() == "as" and buckets[active_bucket]:
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if not _bucket_for_heading(nxt) and nxt.lower() not in {"spoiler", "spoiler:"}:
                    buckets[active_bucket][-1] = f"{buckets[active_bucket][-1]} as {nxt}"
                    i += 2
                    continue
            i += 1
            continue

        if line.startswith("(") and line.endswith(")") and buckets[active_bucket]:
            buckets[active_bucket][-1] = f"{buckets[active_bucket][-1]} {line}"
            i += 1
            continue

        buckets[active_bucket].append(line)
        i += 1

    return buckets


def _parse_cast_buckets(cast_text: str) -> dict[str, list[str]]:
    if not cast_text.strip():
        return {k: [] for k in CAST_BUCKETS}
    return _normalize_cast_lines(_split_cast_lines(cast_text))


def _append_raw_text_lines(target: list[str], raw_text: str) -> None:
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line in {"[", "]"}:
            continue
        target.append(line)


def _parse_cast_buckets_from_html(article_html: str) -> dict[str, list[str]] | None:
    soup = BeautifulSoup(article_html, "lxml")
    root = soup.select_one(".mw-parser-output")
    if root is None:
        return None

    cast_h2: Tag | None = None
    for h2 in root.find_all("h2"):
        span = h2.find("span", class_="mw-headline")
        if span and span.get("id") == "Cast":
            cast_h2 = h2
            break
    if cast_h2 is None:
        return None

    raw_by_bucket: dict[str, list[str]] = {k: [] for k in CAST_BUCKETS}
    global_lines: list[str] = []
    saw_structured_heading = False
    active_bucket = "other"
    for sib in cast_h2.find_next_siblings():
        name = getattr(sib, "name", None)
        if name == "h2":
            break
        if not isinstance(sib, Tag):
            continue

        if name in {"h3", "h4"}:
            heading = sib.get_text(" ", strip=True)
            bucket = _bucket_for_heading(heading)
            if bucket:
                active_bucket = bucket
                saw_structured_heading = True
            continue

        if name in {"ul", "ol"}:
            for li in sib.find_all("li", recursive=False):
                raw = li.get_text("\n", strip=True)
                _append_raw_text_lines(raw_by_bucket[active_bucket], raw)
                _append_raw_text_lines(global_lines, raw)
            continue

        if name in {"p", "div"}:
            raw = sib.get_text("\n", strip=True)
            _append_raw_text_lines(raw_by_bucket[active_bucket], raw)
            _append_raw_text_lines(global_lines, raw)

    if not saw_structured_heading:
        return _normalize_cast_lines(global_lines)

    normalized: dict[str, list[str]] = {k: [] for k in CAST_BUCKETS}
    for bucket in CAST_BUCKETS:
        bucket_lines = raw_by_bucket[bucket]
        if not bucket_lines:
            continue
        normalized_bucket = _normalize_cast_lines(
            bucket_lines, default_bucket=bucket, allow_headings=False
        )
        normalized[bucket].extend(normalized_bucket[bucket])
    return normalized


def _parse_air_date_from_infobox(infobox_text: str | None) -> str | None:
    if not infobox_text:
        return None
    m = re.search(r"Airdate\s*\n?\s*([^\n]+)", infobox_text, re.IGNORECASE)
    if not m:
        return None
    s = m.group(1).strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def build_extracted_data(
    ref: EpisodeRef,
    wiki: WikiArticleResult,
    series_config: SeriesConfig,
    *,
    season_number: int | None = None,
    episode_number: int | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    cold_open = _first_non_empty(wiki.sections, COLD_OPEN_IDS)
    synopsis = _first_non_empty(wiki.sections, SUMMARY_IDS)
    cast_text = wiki.sections.get("Cast", "").strip()
    cast = _parse_cast_buckets_from_html(wiki.article_html) or _parse_cast_buckets(cast_text)
    air_date = _parse_air_date_from_infobox(wiki.infobox_text)
    season = season_number if season_number is not None else ref.season_number
    episode = episode_number if episode_number is not None else ref.episode_number
    final_episode_id = episode_id if episode_id is not None else ref.episode_id

    metadata: dict[str, Any] = {
        "episode_id": final_episode_id,
        "series_title": series_config.series_title,
        "series_slug": ref.series_slug,
        "episode_title": wiki.page_title,
        "season_number": season,
        "episode_number": episode,
        "air_date": air_date,
        "provenance": {
            "source": "fandom_wiki",
            "wiki_page_title": ref.wiki_page_title,
            "wiki_origin": ref.wiki_origin,
            "wiki_api_url": wiki.api_url,
        },
    }

    narrative: dict[str, Any] = {
        "cold_open": cold_open,
        "synopsis": synopsis,
        "cast": cast,
    }

    return {
        "episode_id": final_episode_id,
        "metadata": metadata,
        "narrative": narrative,
    }


def write_extracted_data(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
