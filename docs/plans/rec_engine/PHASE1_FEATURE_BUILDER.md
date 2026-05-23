# Recommendation Engine — Phase 1: Episode Feature Builder

## Goal

Build `backend/recommender/episode_feature_builder.py` — a pure function module
that converts a SQLite episode row into a stable, L2-normalized 17-dim float vector.

This module is the single source of truth for vector construction.
All downstream phases (indexer, retriever, reranker) consume vectors produced here.

---

## Pre-requisite: SQLite Schema Change

The current `episodes` table stores `tone_labels` (a JSON array of label strings that
passed the 0.4 threshold). It does NOT store the raw per-label float scores needed for
the 17-dim vector.

Before building the feature builder, extend the schema and re-ingest.

### 1. Add column to `backend/db/setup.py`

Add `tone_scores TEXT` to the CREATE TABLE statement:

```sql
tone_scores         TEXT,        -- JSON object: {"awkward": 0.82, "chaotic": 0.31, ...}
```

Place it after `tone_labels`.

### 2. Update `backend/db/ingestor.py`

In the mood section, after writing `tone_labels`, also write `tone_scores`:

```python
raw_tone = mood.get("raw_scores", {}).get("tone", {})
row["tone_scores"] = json.dumps(raw_tone, ensure_ascii=False) if raw_tone else None
```

Also add `"tone_scores": None` to the default row dict.

### 3. Rebuild the database

Delete `data/app.sqlite3` and re-run ingest for all episodes:

```bash
rm data/app.sqlite3
python -m backend.db --all the_office
```

Expected result: 201 rows, 199 with non-null `tone_scores`.

---

## Feature Space

### Fixed dimension order (17 dims total)

| Index | Name            | Source column       | Type   |
|-------|-----------------|---------------------|--------|
| 0     | humor_level     | SQLite REAL         | mood   |
| 1     | energy_level    | SQLite REAL         | mood   |
| 2     | comfort_level   | SQLite REAL         | mood   |
| 3     | sadness_level   | SQLite REAL         | mood   |
| 4     | awkward         | tone_scores JSON    | tone   |
| 5     | chaotic         | tone_scores JSON    | tone   |
| 6     | lighthearted    | tone_scores JSON    | tone   |
| 7     | romantic        | tone_scores JSON    | tone   |
| 8     | tense           | tone_scores JSON    | tone   |
| 9     | heartwarming    | tone_scores JSON    | tone   |
| 10    | cringe          | tone_scores JSON    | tone   |
| 11    | silly           | tone_scores JSON    | tone   |
| 12    | dramatic        | tone_scores JSON    | tone   |
| 13    | emotional       | tone_scores JSON    | tone   |
| 14    | wholesome       | tone_scores JSON    | tone   |
| 15    | dark            | tone_scores JSON    | tone   |
| 16    | bittersweet     | tone_scores JSON    | tone   |

This order MUST match `MOOD_DIMENSIONS` and `TONE_DIMENSIONS` in
`backend/embedding/episode_vectorizer.py` exactly. Do not change either order.

### Weights

- Mood weight: `0.7`
- Tone weight: `0.3`

These match the existing vectorizer. Do not change them without updating ChromaDB.

---

## Vector Construction

The math mirrors `backend/embedding/episode_vectorizer.py` exactly.

```python
import math
import numpy as np

MOOD_WEIGHT = 0.7
TONE_WEIGHT = 0.3

def build_vector(mood_vec: np.ndarray, tone_vec: np.ndarray) -> np.ndarray:
    mood_unit = _l2_normalize(mood_vec)   # shape (4,)
    tone_unit = _l2_normalize(tone_vec)   # shape (13,)
    combined = np.concatenate([
        math.sqrt(MOOD_WEIGHT) * mood_unit,
        math.sqrt(TONE_WEIGHT) * tone_unit,
    ])                                     # shape (17,)
    return _l2_normalize(combined)         # shape (17,), unit vector

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
```

If both `mood_vec` and `tone_vec` are all-zeros, return `None` (episode is not indexable).

---

## Module Structure

### Files to create

```
backend/recommender/
├── __init__.py                    (empty)
└── episode_feature_builder.py
```

### `episode_feature_builder.py` public interface

```python
# Constants — frozen, never reorder after vectors are stored
MOOD_DIMENSIONS: list[str]   # ["humor_level", "energy_level", "comfort_level", "sadness_level"]
TONE_DIMENSIONS: list[str]   # [13 labels in fixed order — see table above]
MOOD_WEIGHT: float           # 0.7
TONE_WEIGHT: float           # 0.3
VECTOR_DIM: int              # 17

@dataclass
class EpisodeFeatures:
    episode_id: str
    vector: np.ndarray          # shape (17,), float32, L2-normalized unit vector
    mood_vec: np.ndarray        # shape (4,), raw un-normalized mood values
    tone_vec: np.ndarray        # shape (13,), raw un-normalized tone scores


def build_features(row: dict) -> EpisodeFeatures | None:
    """
    Convert a SQLite row dict into EpisodeFeatures.
    Returns None if mood and tone data are both absent (skipped episodes).
    row must contain: episode_id, humor_level, energy_level, comfort_level,
    sadness_level, tone_scores (JSON string or None).
    """

def load_episode_row(episode_id: str, db_path: Path) -> dict | None:
    """Fetch a single episode row from SQLite. Returns None if not found."""

def load_all_rows(db_path: Path, series_slug: str | None = None) -> list[dict]:
    """
    Fetch all episode rows from SQLite.
    If series_slug is provided, filter to that show only.
    """
```

### Implementation notes

- `build_features` is a pure function: no I/O, no side effects.
- `load_episode_row` and `load_all_rows` do the SQLite I/O; they return plain dicts.
- Parse `tone_scores` with `json.loads` inside `build_features`.
- Missing tone labels default to `0.0` (use `tone_scores_dict.get(label, 0.0)`).
- If `humor_level` is `None` (skipped episode), treat all mood dims as `0.0`.
- If both resulting vectors are all-zeros after filling defaults, return `None`.
- Use `np.float32` throughout.

---

## What This Module Does NOT Do

- No ChromaDB interaction
- No ranking or retrieval
- No user vector logic
- No conversation or session logic
- No file I/O beyond SQLite reads

---

## Tests

### File: `tests/recommender/test_episode_feature_builder.py`

Create the `tests/recommender/` package with an `__init__.py`.

Use `pytest`. No external DB required — tests use in-memory SQLite or plain dicts.

---

### Test 1 — Dimension count

```python
def test_vector_dim():
    row = make_row(humor=0.9, energy=0.8, comfort=0.5, sadness=0.1, tone=full_tone_dict())
    result = build_features(row)
    assert result is not None
    assert len(result.vector) == 17
    assert result.vector.dtype == np.float32
```

---

### Test 2 — L2 normalization

```python
def test_vector_is_unit_length():
    row = make_row(humor=0.9, energy=0.8, comfort=0.5, sadness=0.1, tone=full_tone_dict())
    result = build_features(row)
    assert abs(np.linalg.norm(result.vector) - 1.0) < 1e-5
```

---

### Test 3 — Dimension order matches constants

```python
def test_mood_dims_at_indices_0_to_3():
    """Humor is dim 0, sadness is dim 3."""
    tone = {label: 0.0 for label in TONE_DIMENSIONS}
    row_high_humor = make_row(humor=1.0, energy=0.0, comfort=0.0, sadness=0.0, tone=tone)
    row_high_sadness = make_row(humor=0.0, energy=0.0, comfort=0.0, sadness=1.0, tone=tone)
    r1 = build_features(row_high_humor)
    r2 = build_features(row_high_sadness)
    assert r1.vector[0] > r1.vector[3]   # humor > sadness in high-humor row
    assert r2.vector[3] > r2.vector[0]   # sadness > humor in high-sadness row
```

---

### Test 4 — Tone dims start at index 4

```python
def test_tone_dims_at_indices_4_to_16():
    mood_zero = make_row(humor=0.0, energy=0.0, comfort=0.0, sadness=0.0,
                         tone={"awkward": 1.0, **{l: 0.0 for l in TONE_DIMENSIONS if l != "awkward"}})
    result = build_features(mood_zero)
    # With all-zero mood, the vector is driven entirely by tone
    # awkward is dim 4 (TONE_DIMENSIONS[0]), so dim 4 should be the dominant non-zero
    assert result.vector[4] > 0
    assert result.vector[0] == 0.0  # mood dim should be zero
```

---

### Test 5 — Skipped episode returns None

```python
def test_skipped_episode_returns_none():
    row = make_row(humor=None, energy=None, comfort=None, sadness=None, tone_scores_str=None)
    result = build_features(row)
    assert result is None
```

---

### Test 6 — Missing tone labels default to 0.0

```python
def test_missing_tone_labels_default_to_zero():
    """tone_scores JSON missing some labels — should not raise, missing dims = 0.0."""
    partial_tone = json.dumps({"awkward": 0.8, "chaotic": 0.6})
    row = make_row(humor=0.5, energy=0.5, comfort=0.5, sadness=0.5, tone_scores_str=partial_tone)
    result = build_features(row)
    assert result is not None
    # dims for missing labels should be 0.0 before weighting
    for i, label in enumerate(TONE_DIMENSIONS):
        if label not in ("awkward", "chaotic"):
            assert result.tone_vec[i] == 0.0
```

---

### Test 7 — Feature order stability (regression guard)

```python
def test_dimension_order_matches_vectorizer():
    """
    Regression guard: MOOD_DIMENSIONS and TONE_DIMENSIONS must stay in sync
    with backend.embedding.episode_vectorizer.
    """
    from backend.embedding.episode_vectorizer import (
        MOOD_DIMENSIONS as OLD_MOOD,
        TONE_DIMENSIONS as OLD_TONE,
    )
    assert MOOD_DIMENSIONS == OLD_MOOD
    assert TONE_DIMENSIONS == OLD_TONE
```

---

### Test 8 — Consistent with existing vectorizer output

```python
def test_vector_matches_existing_vectorizer():
    """
    Build a vector using the new feature builder and the old vectorizer from the same
    input data. Results should be identical.
    """
    from backend.embedding.episode_vectorizer import build_episode_vector

    tone_dict = full_tone_dict()  # all dims, some non-zero
    mood_enriched = make_mood_enriched_dict(
        humor=0.9, energy=0.64, comfort=0.28, sadness=0.21, tone=tone_dict
    )
    row = make_row_from_mood_enriched(mood_enriched)

    old_result = build_episode_vector(mood_enriched)
    new_result = build_features(row)

    assert new_result is not None
    np.testing.assert_allclose(new_result.vector, old_result["episode_vec"], atol=1e-5)
```

---

### Helper fixtures (add to `conftest.py` or top of test file)

```python
def make_row(
    humor: float | None,
    energy: float | None,
    comfort: float | None,
    sadness: float | None,
    tone: dict | None = None,
    tone_scores_str: str | None = _SENTINEL,
    episode_id: str = "test_s01_e01",
) -> dict:
    if tone_scores_str is _SENTINEL:
        tone_scores_str = json.dumps(tone) if tone is not None else None
    return {
        "episode_id": episode_id,
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": tone_scores_str,
    }

def full_tone_dict() -> dict:
    """Returns a tone dict with all 13 labels set to plausible non-zero scores."""
    scores = [0.82, 0.31, 0.74, 0.15, 0.44, 0.20, 0.55, 0.60, 0.35, 0.90, 0.18, 0.10, 0.42]
    return dict(zip(TONE_DIMENSIONS, scores))
```

---

## Verification Commands

After implementation, run:

```bash
# Run all Phase 1 tests
pytest tests/recommender/test_episode_feature_builder.py -v

# Quick sanity check: build vector for a known episode
python -c "
import sqlite3, json
from pathlib import Path
from backend.recommender.episode_feature_builder import load_episode_row, build_features

row = load_episode_row('the_office_s02_e01', Path('data/app.sqlite3'))
result = build_features(row)
print('episode_id:', result.episode_id)
print('vector shape:', result.vector.shape)
print('vector norm:', round(sum(x**2 for x in result.vector)**0.5, 6))
print('top mood dim:', result.mood_vec.tolist())
"
```

Expected: vector shape `(17,)`, norm `1.000000`.

---

## Success Criteria

Phase 1 is complete when:

1. `python -m backend.db --all the_office` rebuilds SQLite with `tone_scores` populated.
2. `build_features(row)` returns a 17-dim unit vector for any non-skipped episode.
3. `build_features(row)` returns `None` for the 2 skipped episodes.
4. All 8 tests pass.
5. Test 8 (consistency with existing vectorizer) passes — confirming no math divergence.
