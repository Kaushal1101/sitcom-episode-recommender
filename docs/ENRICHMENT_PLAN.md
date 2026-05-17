# Enrichment Plan — Mood Tagging via DeBERTa v3 Zero-Shot

**Owner:** Orchestrator  
**Last updated:** 2026-05-17  
**Scope:** Text formatting + zero-shot mood classification only. No DB, no rec engine.  
**Depends on:** Scraping phase complete (`data/raw/{episode_id}/extracted_data.json` exists)

---

## What This Phase Does

```
extracted_data.json
      ↓
 text_formatter.py     → human-readable text string (the "premise")
      ↓
  mood_tagger.py       → DeBERTa v3 zero-shot classification
      ↓
 mood_enriched.json    → mood float scores + tone tags
```

The output of this phase (`mood_enriched.json`) is what eventually feeds the recommendation engine's episode vectors. This phase does not modify `extracted_data.json`.

---

## 1. The Text Formatter

**File:** `backend/enrichment/text_formatter.py`  
**Function:** `format_for_zero_shot(episode_data: dict) -> str | None`

### Why this component exists
DeBERTa v3 base uses NLI (Natural Language Inference) internally for zero-shot classification. The input text is the "premise" — it must be clean natural language, not raw JSON. The formatter is a pure function: JSON dict in, formatted string out.

### Token budget
DeBERTa v3 base max sequence length = **512 tokens** total, shared between premise + hypothesis. The hypothesis ("This episode is funny and humorous") consumes ~8–12 tokens. That leaves ~480 tokens for the premise — approximately **1,600 characters** of English prose.

Budget allocation:
| Part | Max chars |
|---|---|
| Header (title + series + season/ep) | ~80 |
| Synopsis | **1,200** |
| Cold open | **300** |
| Total | ~1,580 |

### Output format
```
{episode_title} – {series_title}, Season {season} Episode {episode}

{synopsis, truncated to 1200 chars}

Cold open: {cold_open, truncated to 300 chars}
```

Rules:
- If `synopsis` is present, always include it (truncated). It carries the most mood signal.
- If `cold_open` is present, append it after synopsis. Cold opens often set the episode tone distinctly.
- If **both** are absent or empty: return `None`. Caller skips enrichment for this episode and writes a sentinel `{"skipped": true, "reason": "no_narrative_content"}` to `mood_enriched.json`.
- Truncation: cut at last word boundary before the char limit — do not mid-word cut.
- Strip internal excessive whitespace from the wiki text (wiki scrapes often have `\n\n\n` runs).

### Example output
```
The Fire – The Office, Season 2 Episode 4

Michael gives Ryan a glowing checkpoint review. When Ryan expresses his interest in
starting his own business someday, Michael takes it upon himself to teach Ryan the
first of the ten rules of business. While Michael attempts to school Ryan, a fire
breaks out in the office kitchen...

Cold open: Dwight tests the office's fire safety by setting a small fire in a trash can.
The staff panics. Michael tries to take charge.
```

---

## 2. Mood Label Schema

The zero-shot classifier takes `(text, candidate_labels)` and returns a score per label. We run it in two modes:

### Mode A — Scalar dimensions (binary pair, `multi_label=False`)
Run once per dimension. Take the score of the **positive label** as the float value.

| Dimension | Positive label | Negative label |
|---|---|---|
| `humor_level` | `"funny and humorous"` | `"serious and humorless"` |
| `energy_level` | `"high energy and chaotic"` | `"slow-paced and calm"` |
| `comfort_level` | `"wholesome and comforting"` | `"tense and uncomfortable"` |
| `sadness_level` | `"sad and emotionally heavy"` | `"light-hearted and upbeat"` |

Output range: [0.0, 1.0] (the model's softmax score for the positive label).

### Mode B — Tone tags (multi-label, `multi_label=True`)
Run once with all tone candidates. Apply threshold to select active tags.

Candidate labels:
```python
TONE_CANDIDATES = [
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
```

Tags with score ≥ 0.4 are included in `mood.tone`. If no tags exceed the threshold, `mood.tone` is `[]`.

---

## 3. Output Contract — `mood_enriched.json`

Written to `data/raw/{episode_id}/mood_enriched.json`.

### Normal output
```json
{
  "episode_id": "the_office_s02_e04",
  "model_id": "cross-encoder/nli-deberta-v3-base",
  "enriched_at": "2026-05-17T12:00:00Z",
  "input_text": "The Fire – The Office, Season 2 Episode 4\n\n...",
  "mood": {
    "humor_level": 0.82,
    "energy_level": 0.61,
    "comfort_level": 0.54,
    "sadness_level": 0.09,
    "tone": ["awkward", "chaotic", "silly"],
    "raw_scores": {
      "humor_level": {"funny and humorous": 0.82, "serious and humorless": 0.18},
      "energy_level": {"high energy and chaotic": 0.61, "slow-paced and calm": 0.39},
      "comfort_level": {"wholesome and comforting": 0.54, "tense and uncomfortable": 0.46},
      "sadness_level": {"sad and emotionally heavy": 0.09, "light-hearted and upbeat": 0.91},
      "tone": {
        "awkward": 0.73,
        "chaotic": 0.55,
        "silly": 0.48,
        "lighthearted": 0.38,
        "dramatic": 0.12
      }
    }
  }
}
```

### Skipped output (no narrative content)
```json
{
  "episode_id": "the_office_s07_e13",
  "model_id": "cross-encoder/nli-deberta-v3-base",
  "enriched_at": "2026-05-17T12:00:00Z",
  "skipped": true,
  "reason": "no_narrative_content"
}
```

---

## 4. File Structure

```
backend/enrichment/
  __init__.py
  text_formatter.py    ← the formatting component (pure function, no ML)
  mood_tagger.py       ← loads DeBERTa, runs zero-shot, writes mood_enriched.json
  __main__.py          ← CLI entry point

data/raw/{episode_id}/
  extracted_data.json  ← input (unchanged)
  mood_enriched.json   ← output (new)
```

---

## 5. CLI

```bash
# Enrich one episode
python -m backend.enrichment --episode-id the_office_s02_e04

# Enrich a full season
python -m backend.enrichment --season the_office 2

# Enrich all scraped episodes
python -m backend.enrichment --all the_office

# Dry-run: show formatted text without running the model
python -m backend.enrichment --episode-id the_office_s02_e04 --show-text

# Skip already-enriched episodes (re-run safe)
python -m backend.enrichment --all the_office --skip-existing
```

`--skip-existing`: if `mood_enriched.json` already exists for an episode, skip it. Makes the command re-runnable without re-processing everything.

---

## 6. Dependencies

Add to `pyproject.toml`:
```toml
"torch>=2.0",
"transformers>=4.40",
```

**Model:** `cross-encoder/nli-deberta-v3-base` (HuggingFace Hub). Downloaded automatically on first run to `~/.cache/huggingface/`.

**Performance:** On CPU, expect ~3–8 seconds per episode (5 classifier calls: 4 scalar + 1 multi-label tone). For 201 episodes: ~10–25 minutes on CPU. GPU reduces this to under 2 minutes total.

---

## 7. Open Questions

1. **Tone threshold tuning** — 0.4 is a starting point. After running on S2, spot-check whether the active tags feel right. Too many tags = lower threshold, too few = raise it.
2. **Two no-content episodes** (`s07_e13`, `s09_e05`) — enrichment is skipped. These will need manual tags or stay unenriched when rec engine is built. Track them.
3. **Synopsis truncation quality** — at 1,200 chars, long synopses are cut mid-story. This is acceptable for mood classification (tone is usually established early), but verify on a few long episodes.
