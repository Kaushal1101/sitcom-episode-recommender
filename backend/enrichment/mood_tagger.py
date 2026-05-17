from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.enrichment.text_formatter import format_for_zero_shot

MODEL_ID = "cross-encoder/nli-deberta-v3-base"

DIMENSION_LABEL_PAIRS: dict[str, tuple[str, str]] = {
    "humor_level": ("funny and humorous", "serious and humorless"),
    "energy_level": ("high energy and chaotic", "slow-paced and calm"),
    "comfort_level": ("wholesome and comforting", "tense and uncomfortable"),
    "sadness_level": ("sad and emotionally heavy", "light-hearted and upbeat"),
}

TONE_CANDIDATES: list[str] = [
    "awkward",
    "chaotic",
    "lighthearted",
    "romantic",
    "tense",
    "heartwarming",
    "cringe",
    "silly",
    "dramatic",
    "emotional",
    "wholesome",
    "dark",
    "bittersweet",
]

TONE_THRESHOLD = 0.4


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scores_dict(result: dict[str, Any]) -> dict[str, float]:
    return dict(zip(result["labels"], result["scores"], strict=True))


def _classify_dimensions(text: str, classifier: Any) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    values: dict[str, float] = {}
    raw: dict[str, dict[str, float]] = {}
    for dimension, (positive, negative) in DIMENSION_LABEL_PAIRS.items():
        result = classifier(text, [positive, negative], multi_label=False)
        scores = _scores_dict(result)
        values[dimension] = round(float(scores[positive]), 4)
        raw[dimension] = {label: round(float(scores[label]), 4) for label in (positive, negative)}
    return values, raw


def _classify_tone(text: str, classifier: Any) -> tuple[list[str], dict[str, float]]:
    result = classifier(text, TONE_CANDIDATES, multi_label=True)
    scores = _scores_dict(result)
    rounded = {label: round(float(scores[label]), 4) for label in TONE_CANDIDATES}
    active = sorted(label for label, score in rounded.items() if score >= TONE_THRESHOLD)
    return active, rounded


def _classify_mood(text: str, classifier: Any) -> dict[str, Any]:
    scalar_values, scalar_raw = _classify_dimensions(text, classifier)
    tone_tags, tone_raw = _classify_tone(text, classifier)
    return {
        **scalar_values,
        "tone": tone_tags,
        "raw_scores": {**scalar_raw, "tone": tone_raw},
    }


def enrich_episode(episode_id: str, data_root: Path, classifier: Any) -> dict[str, Any]:
    """
    Read extracted_data.json, format text, run zero-shot classification,
    write mood_enriched.json, return the written dict.
    """
    episode_dir = data_root / episode_id
    extracted_path = episode_dir / "extracted_data.json"
    if not extracted_path.exists():
        raise FileNotFoundError(f"Missing extracted_data.json for {episode_id}: {extracted_path}")

    episode_data = json.loads(extracted_path.read_text(encoding="utf-8"))
    enriched_at = _utc_now_iso()
    formatted_text = format_for_zero_shot(episode_data)

    if formatted_text is None:
        payload: dict[str, Any] = {
            "episode_id": episode_id,
            "model_id": MODEL_ID,
            "enriched_at": enriched_at,
            "skipped": True,
            "reason": "no_narrative_content",
        }
    else:
        mood = _classify_mood(formatted_text, classifier)
        payload = {
            "episode_id": episode_id,
            "model_id": MODEL_ID,
            "enriched_at": enriched_at,
            "input_text": formatted_text,
            "mood": mood,
        }

    out_path = episode_dir / "mood_enriched.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_classifier() -> Any:
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("zero-shot-classification", model=MODEL_ID)
