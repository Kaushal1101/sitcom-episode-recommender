from __future__ import annotations

import re


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate to max_chars at the last word boundary. No ellipsis."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    parts = truncated.rsplit(None, 1)
    return parts[0] if parts else truncated


def _normalize_section(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    return cleaned


def format_for_zero_shot(episode_data: dict) -> str | None:
    """
    Convert extracted_data.json dict to a text string suitable as a DeBERTa NLI premise.
    Returns None if there is no narrative content to classify.
    """
    metadata = episode_data.get("metadata") or {}
    narrative = episode_data.get("narrative") or {}

    synopsis = _normalize_section(narrative.get("synopsis"))
    cold_open = _normalize_section(narrative.get("cold_open"))

    if not synopsis and not cold_open:
        return None

    episode_title = metadata.get("episode_title") or "Unknown"
    series_title = metadata.get("series_title") or "Unknown"
    season_number = metadata.get("season_number", "?")
    episode_number = metadata.get("episode_number", "?")

    header = f"{episode_title} – {series_title}, Season {season_number} Episode {episode_number}"
    parts: list[str] = [header, ""]

    if synopsis:
        parts.append(_truncate_at_word(synopsis, 1200))

    if cold_open:
        if synopsis:
            parts.append("")
        parts.append(f"Cold open: {_truncate_at_word(cold_open, 300)}")

    return "\n".join(parts).strip()
