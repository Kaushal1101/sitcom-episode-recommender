# Recommendation Engine — Phase 4: Reranker

## Goal

Build `backend/recommender/reranker.py` — a deterministic scoring module that takes
the flat list of `Candidate` objects from the retriever, fetches their mood/tone data
from SQLite, computes a composite final score, applies hard exclusions, and returns a
`list[RankedCandidate]` sorted by final score descending.

The reranker is the only place where mood and tone alignment are computed separately
from the combined vector similarity. This gives the system a second, decomposed pass
over the same signal — useful for explanation in Phase 5 and for tuning weights later.

---

## Context

- **Phase 1 (done):** `episode_feature_builder.py` — 17-dim vector math, constants.
- **Phase 2 (done):** `vector_indexer.py` — Chroma populated from SQLite.
- **Phase 3 (done):** `retriever.py` — returns `list[Candidate]` sorted by cosine similarity.
  `Candidate` contains: `episode_id`, `similarity`, `series_slug`, `season_number`,
  `episode_number`, `episode_title`. No mood scalars — those are still in SQLite.
- **Phase 4:** `reranker.py` — fetches mood data from SQLite, scores + reranks candidates.

---

## Scoring Formula

```
final_score = W_SIM * similarity
            + W_MOOD * mood_alignment
            + W_TONE * tone_alignment
```

### Weights (MVP starting point — easy to tune)

```python
W_SIM:  float = 0.5   # cosine similarity from Chroma
W_MOOD: float = 0.3   # mood dimension alignment
W_TONE: float = 0.2   # tone dimension alignment
```

These must sum to 1.0. Do not change them without updating this doc.

### mood_alignment

The user_vector encodes mood in dims [0:4] as `sqrt(W_MOOD_VEC) * mood_unit` where
`W_MOOD_VEC = 0.7` (the vectorizer's internal weight). Because the combined vector
is always unit length before final normalization, `user_vector[0:4]` is exactly
`sqrt(0.7) * user_mood_unit`.

```python
ep_mood_vec = np.array(
    [row["humor_level"] or 0.0,
     row["energy_level"] or 0.0,
     row["comfort_level"] or 0.0,
     row["sadness_level"] or 0.0],
    dtype=np.float32,
)
ep_mood_unit = _l2_normalize(ep_mood_vec)
mood_alignment = float(np.dot(user_vector[0:4], ep_mood_unit))
# Range: [-sqrt(0.7), sqrt(0.7)] ≈ [-0.837, 0.837]
# For real episodes (all-positive mood values): [0, ~0.837]
# If ep_mood_vec is all-zero (no mood data): mood_alignment = 0.0
```

### tone_alignment

Same approach using dims [4:17] and the episode's raw tone scores from SQLite:

```python
tone_scores_dict = json.loads(row["tone_scores"] or "{}")
ep_tone_vec = np.array(
    [tone_scores_dict.get(label, 0.0) for label in TONE_DIMENSIONS],
    dtype=np.float32,
)
ep_tone_unit = _l2_normalize(ep_tone_vec)
tone_alignment = float(np.dot(user_vector[4:17], ep_tone_unit))
# Range: [-sqrt(0.3), sqrt(0.3)] ≈ [-0.548, 0.548]
# For real episodes: [0, ~0.548]
# If ep_tone_vec is all-zero: tone_alignment = 0.0
```

---

## Hard Exclusions

Episodes in `excluded_ids` are removed from the output entirely — they do not appear
in `RankedCandidate` results at any score. This is enforced before scoring (skip fetch
and skip output).

`excluded_ids` defaults to an empty set if not provided.

---

## SQLite Fetch

The reranker fetches mood columns for all non-excluded candidates in a single batch
query using `WHERE episode_id IN (...)`. It does NOT fetch text columns (synopsis,
cast) — those are not needed for scoring.

Columns fetched: `episode_id`, `humor_level`, `energy_level`, `comfort_level`,
`sadness_level`, `tone_scores`.

If a candidate's `episode_id` is not found in SQLite (stale Chroma entry), skip it
silently and do not include it in output.

---

## Output Type

```python
@dataclass
class RankedCandidate:
    episode_id:      str
    final_score:     float   # composite score — sort key
    similarity:      float   # raw cosine similarity from retriever (unchanged)
    mood_alignment:  float   # mood component of score
    tone_alignment:  float   # tone component of score
    series_slug:     str
    season_number:   int
    episode_number:  int
    episode_title:   str
```

---

## Module Structure

### Files to create

```
backend/recommender/
└── reranker.py        (new)

tests/recommender/
└── test_reranker.py   (new)
```

Do NOT modify any existing files.

---

## `reranker.py` Public Interface

```python
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.recommender.episode_feature_builder import TONE_DIMENSIONS
from backend.recommender.retriever import Candidate

W_SIM:  float = 0.5
W_MOOD: float = 0.3
W_TONE: float = 0.2


@dataclass
class RankedCandidate:
    episode_id:     str
    final_score:    float
    similarity:     float
    mood_alignment: float
    tone_alignment: float
    series_slug:    str
    season_number:  int
    episode_number: int
    episode_title:  str


def rerank(
    candidates: list[Candidate],
    user_vector: np.ndarray,
    db_path: Path,
    excluded_ids: set[str] | None = None,
) -> list[RankedCandidate]:
    """
    Score and rerank candidates using mood/tone alignment from SQLite.

    - Excluded episodes are removed entirely from output.
    - Candidates whose episode_id is not found in SQLite are silently dropped.
    - Output is sorted by final_score descending.
    """
```

---

## Implementation Notes

### Helper: `_l2_normalize`

```python
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
```

### Helper: `_fetch_mood_rows`

```python
_MOOD_COLUMNS = "episode_id, humor_level, energy_level, comfort_level, sadness_level, tone_scores"

def _fetch_mood_rows(episode_ids: list[str], db_path: Path) -> dict[str, dict]:
    """
    Fetch mood columns for the given episode_ids in one query.
    Returns {episode_id: row_dict}.
    """
    placeholders = ",".join("?" * len(episode_ids))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {_MOOD_COLUMNS} FROM episodes WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
    return {dict(r)["episode_id"]: dict(r) for r in rows}
```

### Helper: `_score`

```python
def _score(
    candidate: Candidate,
    row: dict,
    user_vector: np.ndarray,
) -> tuple[float, float, float]:
    """Returns (final_score, mood_alignment, tone_alignment)."""
    ep_mood = np.array(
        [row["humor_level"] or 0.0, row["energy_level"] or 0.0,
         row["comfort_level"] or 0.0, row["sadness_level"] or 0.0],
        dtype=np.float32,
    )
    tone_dict = json.loads(row["tone_scores"] or "{}")
    ep_tone = np.array(
        [tone_dict.get(label, 0.0) for label in TONE_DIMENSIONS],
        dtype=np.float32,
    )
    mood_alignment = float(np.dot(user_vector[0:4], _l2_normalize(ep_mood)))
    tone_alignment = float(np.dot(user_vector[4:17], _l2_normalize(ep_tone)))
    final_score = W_SIM * candidate.similarity + W_MOOD * mood_alignment + W_TONE * tone_alignment
    return final_score, mood_alignment, tone_alignment
```

### `rerank` implementation sketch

```python
def rerank(candidates, user_vector, db_path, excluded_ids=None):
    excluded = excluded_ids or set()
    active = [c for c in candidates if c.episode_id not in excluded]
    if not active:
        return []

    rows = _fetch_mood_rows([c.episode_id for c in active], db_path)

    ranked = []
    for c in active:
        row = rows.get(c.episode_id)
        if row is None:
            continue   # stale Chroma entry — skip silently
        final_score, mood_aln, tone_aln = _score(c, row, user_vector)
        ranked.append(RankedCandidate(
            episode_id=c.episode_id,
            final_score=round(final_score, 6),
            similarity=c.similarity,
            mood_alignment=round(mood_aln, 6),
            tone_alignment=round(tone_aln, 6),
            series_slug=c.series_slug,
            season_number=c.season_number,
            episode_number=c.episode_number,
            episode_title=c.episode_title,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
```

### What this module does NOT do

- No Chroma interaction
- No user vector updates
- No question selection or session logic
- No free-text generation

---

## Tests

### File: `tests/recommender/test_reranker.py`

Use `pytest` with `tmp_path`. Tests construct `Candidate` objects directly — no need
to go through the retriever. SQLite is set up with `make_db` (same helper pattern as
previous test files — define it locally, do not import from other test files).

Only columns the reranker fetches from SQLite are needed: `episode_id`, `humor_level`,
`energy_level`, `comfort_level`, `sadness_level`, `tone_scores`. The `make_db` helper
can insert minimal rows with just these columns plus the required primary key.

---

### Fixtures

```python
def make_candidate(
    episode_id: str = "show_s01_e01",
    similarity: float = 0.9,
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
) -> Candidate:
    return Candidate(
        episode_id=episode_id,
        similarity=similarity,
        series_slug=series_slug,
        season_number=season,
        episode_number=number,
        episode_title=title,
    )


def make_mood_row(
    episode_id: str,
    humor: float = 0.5,
    energy: float = 0.5,
    comfort: float = 0.5,
    sadness: float = 0.5,
    tone: dict | None = None,
) -> dict:
    if tone is None:
        tone = {label: 0.5 for label in TONE_DIMENSIONS}
    return {
        "episode_id": episode_id,
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": json.dumps(tone),
    }
```

For `make_db`, insert rows using only the columns the reranker needs:

```python
def make_db(tmp_path: Path, rows: list[dict]) -> Path:
    from backend.db.setup import setup_db
    db_path = tmp_path / "app.sqlite3"
    setup_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for r in rows:
            conn.execute(
                "INSERT INTO episodes (episode_id, humor_level, energy_level, "
                "comfort_level, sadness_level, tone_scores) "
                "VALUES (:episode_id, :humor_level, :energy_level, "
                ":comfort_level, :sadness_level, :tone_scores)",
                r,
            )
        conn.commit()
    return db_path
```

### User vector helper

```python
def user_vec_from_mood(
    humor: float = 0.9,
    energy: float = 0.5,
    comfort: float = 0.5,
    sadness: float = 0.1,
    tone: dict | None = None,
) -> np.ndarray:
    """Build a realistic user_vector using the feature builder math."""
    if tone is None:
        tone = {label: 0.5 for label in TONE_DIMENSIONS}
    row = {
        "episode_id": "user",
        "humor_level": humor,
        "energy_level": energy,
        "comfort_level": comfort,
        "sadness_level": sadness,
        "tone_scores": json.dumps(tone),
    }
    from backend.recommender.episode_feature_builder import build_features
    return build_features(row).vector
```

---

### Test 1 — Output count matches non-excluded input

```python
def test_rerank_preserves_all_non_excluded(tmp_path):
    candidates = [make_candidate(f"ep_{i}", similarity=0.9 - i * 0.1) for i in range(3)]
    rows = [make_mood_row(f"ep_{i}") for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path)
    assert len(results) == 3
```

---

### Test 2 — Output sorted by final_score descending

```python
def test_rerank_sorted_by_final_score(tmp_path):
    candidates = [make_candidate(f"ep_{i}", similarity=0.9 - i * 0.1) for i in range(4)]
    rows = [make_mood_row(f"ep_{i}") for i in range(4)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path)
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)
```

---

### Test 3 — Episode matching user mood ranks first

```python
def test_rerank_mood_match_ranks_first(tmp_path):
    """
    Episode with exact same mood profile as user should outscore
    an episode with a very different mood, even if initial similarities differ.
    """
    user_vec = user_vec_from_mood(humor=0.9, energy=0.1, comfort=0.8, sadness=0.1)
    # high_match: mood matches user (high humor/comfort)
    # low_match: opposite mood (low humor, high sadness)
    candidates = [
        make_candidate("high_match", similarity=0.85),
        make_candidate("low_match",  similarity=0.90),  # higher similarity but wrong mood
    ]
    rows = [
        make_mood_row("high_match", humor=0.9, energy=0.1, comfort=0.8, sadness=0.1),
        make_mood_row("low_match",  humor=0.1, energy=0.9, comfort=0.1, sadness=0.9),
    ]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec, db_path)
    assert results[0].episode_id == "high_match"
```

---

### Test 4 — Excluded IDs removed from output

```python
def test_rerank_excluded_ids_removed(tmp_path):
    candidates = [make_candidate(f"ep_{i}") for i in range(3)]
    rows = [make_mood_row(f"ep_{i}") for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path, excluded_ids={"ep_1"})
    ids = [r.episode_id for r in results]
    assert "ep_1" not in ids
    assert len(results) == 2
```

---

### Test 5 — All excluded returns empty list

```python
def test_rerank_all_excluded_returns_empty(tmp_path):
    candidates = [make_candidate(f"ep_{i}") for i in range(3)]
    rows = [make_mood_row(f"ep_{i}") for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = rerank(candidates, user_vec_from_mood(), db_path,
                     excluded_ids={"ep_0", "ep_1", "ep_2"})
    assert results == []
```

---

### Test 6 — RankedCandidate has all required fields

```python
def test_rerank_output_fields(tmp_path):
    c = make_candidate("ep_0", similarity=0.88, season=2, number=4, title="The Fire")
    db_path = make_db(tmp_path, [make_mood_row("ep_0")])
    results = rerank([c], user_vec_from_mood(), db_path)
    assert len(results) == 1
    r = results[0]
    assert r.episode_id == "ep_0"
    assert r.series_slug == "the_office"
    assert r.season_number == 2
    assert r.episode_number == 4
    assert r.episode_title == "The Fire"
    assert isinstance(r.final_score, float)
    assert isinstance(r.mood_alignment, float)
    assert isinstance(r.tone_alignment, float)
    assert r.similarity == 0.88
```

---

### Test 7 — Empty candidates returns empty list

```python
def test_rerank_empty_candidates(tmp_path):
    db_path = make_db(tmp_path, [])
    results = rerank([], user_vec_from_mood(), db_path)
    assert results == []
```

---

### Test 8 — final_score formula verified

```python
def test_rerank_score_formula(tmp_path):
    """Verify final_score = W_SIM * sim + W_MOOD * mood_aln + W_TONE * tone_aln."""
    from backend.recommender.reranker import W_SIM, W_MOOD, W_TONE
    c = make_candidate("ep_0", similarity=0.80)
    db_path = make_db(tmp_path, [make_mood_row("ep_0")])
    results = rerank([c], user_vec_from_mood(), db_path)
    r = results[0]
    expected = W_SIM * r.similarity + W_MOOD * r.mood_alignment + W_TONE * r.tone_alignment
    assert abs(r.final_score - expected) < 1e-5
```

---

### Test 9 — Stale Chroma entry (episode_id not in SQLite) silently dropped

```python
def test_rerank_missing_sqlite_row_dropped(tmp_path):
    candidates = [
        make_candidate("ep_0"),
        make_candidate("ep_ghost"),  # not in SQLite
    ]
    db_path = make_db(tmp_path, [make_mood_row("ep_0")])
    results = rerank(candidates, user_vec_from_mood(), db_path)
    ids = [r.episode_id for r in results]
    assert "ep_0" in ids
    assert "ep_ghost" not in ids
```

---

## Verification Commands

After implementation, run all tests:

```bash
pytest tests/recommender/ -v
```

Then do a live end-to-end check using the real database:

```bash
python -c "
import numpy as np
from pathlib import Path
from backend.recommender.episode_feature_builder import load_episode_row, build_features
from backend.recommender.retriever import retrieve
from backend.recommender.reranker import rerank

DB = Path('data/app.sqlite3')
CHROMA = Path('data/chroma')

# Use The Dundies (S2E01) as the user vector
row = load_episode_row('the_office_s02_e01', DB)
user_vec = build_features(row).vector

candidates = retrieve(user_vec, CHROMA, top_k=20)
ranked = rerank(candidates, user_vec, DB)

for r in ranked[:5]:
    print(f'{r.final_score:.4f}  sim={r.similarity:.3f}  mood={r.mood_alignment:.3f}  tone={r.tone_alignment:.3f}  S{r.season_number}E{r.episode_number}  {r.episode_title}')
"
```

Expected: `the_office_s02_e01` ranks first (or very close), scores decomposed visibly
across similarity, mood, and tone columns.

---

## Success Criteria

Phase 4 is complete when:

1. All 9 tests pass.
2. Test 3 confirms reranking can reorder candidates (mood-matched episode beats
   a higher-similarity but mood-mismatched episode).
3. Test 8 confirms the score formula is numerically correct.
4. The live end-to-end check produces a ranked list with visible score components.
5. Full test suite (all phases) still passes with no regressions.
