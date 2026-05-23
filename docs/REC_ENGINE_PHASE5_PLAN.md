# Recommendation Engine — Phase 5: Explanation Builder + Engine

## Goal

Build two files:

1. `backend/recommender/explanation_builder.py` — deterministic, structured match
   reasons for each `RankedCandidate`. No free-form text generation.
2. `backend/recommender/engine.py` — thin public entry point that chains retrieve →
   rerank → explain into a single `recommend()` call.

After Phase 5, the recommendation engine is callable end-to-end:
```python
from backend.recommender.engine import recommend
result = recommend(user_vector, db_path, chroma_path, top_k=10)
# result.ranked      → list[RankedCandidate]
# result.explanations → list[MatchExplanation], one per ranked episode
```

---

## Context

- **Phase 1–4 (done):** feature builder → vector indexer → retriever → reranker.
- **Phase 5:** adds explanation layer and the public engine entry point.
- The explanation builder reads mood/tone scalars from SQLite (same columns as the
  reranker). It does NOT call Chroma.
- The engine is a thin orchestrator — no new logic, just composition.

---

## Part A: Explanation Builder

### What explanations contain

```python
@dataclass
class MatchExplanation:
    episode_id:              str
    mood_matches:            list[str]   # mood dims with strong user-episode alignment
    tone_matches:            list[str]   # tone labels with strong user-episode alignment
    dominant_episode_traits: list[str]   # episode's strongest dims, regardless of user
    score_breakdown:         dict[str, float]  # {"similarity": 0.92, "mood": 0.71, "tone": 0.45}
```

#### `mood_matches`
Mood dimensions where BOTH the user wants and the episode has that quality.
Computed per-dim: `contribution = user_vector[i] * ep_mood_unit[i]`.
Include dim name if contribution > `mood_threshold` (default 0.15).

```python
user_mood = user_vector[0:4]            # sqrt(0.7) * user_mood_unit
ep_mood_unit = _l2_normalize(ep_mood_vec)
per_dim = user_mood * ep_mood_unit      # element-wise
mood_matches = [MOOD_DIMENSIONS[i] for i in range(4) if per_dim[i] > mood_threshold]
```

#### `tone_matches`
Same approach for tone dimensions.
Include tone label if contribution > `tone_threshold` (default 0.10).

```python
user_tone = user_vector[4:17]           # sqrt(0.3) * user_tone_unit
ep_tone_unit = _l2_normalize(ep_tone_vec)
per_dim = user_tone * ep_tone_unit
tone_matches = [TONE_DIMENSIONS[i] for i in range(13) if per_dim[i] > tone_threshold]
```

#### `dominant_episode_traits`
The episode's strongest mood and tone dims, regardless of user preference.
Combine raw `ep_mood_vec` and `ep_tone_vec` values, threshold at 0.5, take top N.

```python
mood_scored = [(MOOD_DIMENSIONS[i], ep_mood_vec[i]) for i in range(4)]
tone_scored = [(TONE_DIMENSIONS[i], ep_tone_vec[i]) for i in range(13)]
all_scored = sorted(mood_scored + tone_scored, key=lambda x: x[1], reverse=True)
dominant_episode_traits = [
    name for name, score in all_scored
    if score > trait_threshold   # default 0.5
][:top_n_traits]                 # default 3
```

#### `score_breakdown`
Pulled directly from the `RankedCandidate`:
```python
score_breakdown = {
    "similarity":      ranked.similarity,
    "mood_alignment":  ranked.mood_alignment,
    "tone_alignment":  ranked.tone_alignment,
    "final_score":     ranked.final_score,
}
```

---

### SQLite fetch

Same batch query pattern as the reranker. Columns:
`episode_id, humor_level, energy_level, comfort_level, sadness_level, tone_scores`

Use a private `_fetch_mood_rows` (same logic as reranker's — define it locally, do
not import from reranker).

---

### Public interface

```python
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.recommender.episode_feature_builder import MOOD_DIMENSIONS, TONE_DIMENSIONS
from backend.recommender.reranker import RankedCandidate

MOOD_THRESHOLD:  float = 0.15
TONE_THRESHOLD:  float = 0.10
TRAIT_THRESHOLD: float = 0.50
TOP_N_TRAITS:    int   = 3


@dataclass
class MatchExplanation:
    episode_id:              str
    mood_matches:            list[str]
    tone_matches:            list[str]
    dominant_episode_traits: list[str]
    score_breakdown:         dict[str, float] = field(default_factory=dict)


def explain(
    ranked: RankedCandidate,
    user_vector: np.ndarray,
    db_path: Path,
    mood_threshold: float = MOOD_THRESHOLD,
    tone_threshold: float = TONE_THRESHOLD,
    trait_threshold: float = TRAIT_THRESHOLD,
    top_n_traits: int = TOP_N_TRAITS,
) -> MatchExplanation | None:
    """
    Build a MatchExplanation for a single RankedCandidate.
    Returns None if the episode is not found in SQLite.
    """


def explain_all(
    ranked: list[RankedCandidate],
    user_vector: np.ndarray,
    db_path: Path,
    mood_threshold: float = MOOD_THRESHOLD,
    tone_threshold: float = TONE_THRESHOLD,
    trait_threshold: float = TRAIT_THRESHOLD,
    top_n_traits: int = TOP_N_TRAITS,
) -> list[MatchExplanation]:
    """
    Build explanations for a list of RankedCandidates in a single SQLite query.
    Episodes not found in SQLite are silently omitted.
    Order matches the input ranked list.
    """
```

---

### Implementation notes

- `explain_all` fetches all rows in one batch query, then iterates in input order.
- If a `RankedCandidate`'s episode_id is missing from SQLite, omit it silently.
- `mood_matches` and `tone_matches` may be empty lists (no strong alignment on any dim).
- `dominant_episode_traits` may be shorter than `top_n_traits` if fewer dims exceed the threshold.
- All computation is pure numpy — no randomness, no model calls.

---

## Part B: Engine

### `engine.py` public interface

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.recommender.explanation_builder import MatchExplanation, explain_all
from backend.recommender.reranker import RankedCandidate, rerank
from backend.recommender.retriever import retrieve

RETRIEVAL_MULTIPLIER: int = 5   # retrieve top_k * 5 candidates before reranking


@dataclass
class RecommendationResult:
    ranked:       list[RankedCandidate]
    explanations: list[MatchExplanation]


def recommend(
    user_vector: np.ndarray,
    db_path: Path,
    chroma_path: Path,
    top_k: int = 10,
    excluded_ids: set[str] | None = None,
    series_slug: str | None = None,
) -> RecommendationResult:
    """
    Full recommendation pipeline: retrieve → rerank → explain.

    Retrieves top_k * RETRIEVAL_MULTIPLIER candidates from Chroma,
    reranks them, trims to top_k, then builds explanations.
    """
```

### Implementation

```python
def recommend(user_vector, db_path, chroma_path, top_k=10,
              excluded_ids=None, series_slug=None):
    candidates = retrieve(
        user_vector, chroma_path,
        top_k=top_k * RETRIEVAL_MULTIPLIER,
        series_slug=series_slug,
    )
    ranked = rerank(candidates, user_vector, db_path, excluded_ids=excluded_ids)
    ranked = ranked[:top_k]
    explanations = explain_all(ranked, user_vector, db_path)
    return RecommendationResult(ranked=ranked, explanations=explanations)
```

### What the engine does NOT do

- No user vector updates
- No session state management
- No question selection
- No confidence computation (future phase)

---

## Module Structure

### Files to create

```
backend/recommender/
├── explanation_builder.py   (new)
└── engine.py                (new)

tests/recommender/
├── test_explanation_builder.py   (new)
└── test_engine.py                (new)
```

Do NOT modify any existing files.

---

## Tests

### File A: `tests/recommender/test_explanation_builder.py`

Use `pytest` with `tmp_path`. Construct `RankedCandidate` objects directly and set up
SQLite with `make_db`. Define all helpers locally.

---

#### Fixtures

```python
def make_ranked(
    episode_id: str = "ep_0",
    final_score: float = 0.80,
    similarity: float = 0.85,
    mood_alignment: float = 0.60,
    tone_alignment: float = 0.40,
    series_slug: str = "the_office",
    season: int = 1,
    number: int = 1,
    title: str = "Test Episode",
) -> RankedCandidate:
    return RankedCandidate(
        episode_id=episode_id,
        final_score=final_score,
        similarity=similarity,
        mood_alignment=mood_alignment,
        tone_alignment=tone_alignment,
        series_slug=series_slug,
        season_number=season,
        episode_number=number,
        episode_title=title,
    )

def make_db_row(episode_id, humor, energy, comfort, sadness, tone=None):
    # same pattern as test_reranker.py
    ...

def make_db(tmp_path, rows):
    # same pattern as test_reranker.py
    ...

def user_vec_from_mood(humor=0.9, energy=0.5, comfort=0.5, sadness=0.1, tone=None):
    # same pattern as test_reranker.py — use build_features
    ...
```

---

#### Test 1 — explain_all returns one explanation per ranked candidate

```python
def test_explain_all_count(tmp_path):
    ranked = [make_ranked(f"ep_{i}") for i in range(3)]
    rows = [make_db_row(f"ep_{i}", 0.8, 0.5, 0.6, 0.2) for i in range(3)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    assert len(results) == 3
```

---

#### Test 2 — episode_id matches ranked candidate

```python
def test_explain_episode_id_matches(tmp_path):
    ranked = [make_ranked("ep_0"), make_ranked("ep_1")]
    rows = [make_db_row("ep_0", 0.8, 0.5, 0.5, 0.1),
            make_db_row("ep_1", 0.2, 0.5, 0.5, 0.8)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    assert [r.episode_id for r in results] == ["ep_0", "ep_1"]
```

---

#### Test 3 — mood match detected when user and episode agree on a dimension

```python
def test_explain_mood_match_detected(tmp_path):
    """High-humor user + high-humor episode → 'humor_level' in mood_matches."""
    user_vec = user_vec_from_mood(humor=0.95, energy=0.1, comfort=0.1, sadness=0.1)
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", humor=0.9, energy=0.1, comfort=0.1, sadness=0.1)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec, db_path)
    assert "humor_level" in results[0].mood_matches
```

---

#### Test 4 — tone match detected when user and episode agree on a tone

```python
def test_explain_tone_match_detected(tmp_path):
    """User with strong emotional preference + emotional episode → 'emotional' in tone_matches."""
    tone = {label: 0.1 for label in TONE_DIMENSIONS}
    tone["emotional"] = 0.95
    user_vec = user_vec_from_mood(tone=tone)
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", 0.5, 0.5, 0.5, 0.5, tone=tone)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec, db_path)
    assert "emotional" in results[0].tone_matches
```

---

#### Test 5 — dominant traits reflect episode's strongest dims

```python
def test_explain_dominant_traits(tmp_path):
    """Episode with very high humor and low everything else → 'humor_level' is dominant."""
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", humor=0.95, energy=0.1, comfort=0.1, sadness=0.1)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    assert "humor_level" in results[0].dominant_episode_traits
```

---

#### Test 6 — score_breakdown contains required keys

```python
def test_explain_score_breakdown_keys(tmp_path):
    ranked = [make_ranked("ep_0")]
    rows = [make_db_row("ep_0", 0.8, 0.5, 0.5, 0.2)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    bd = results[0].score_breakdown
    assert "similarity" in bd
    assert "mood_alignment" in bd
    assert "tone_alignment" in bd
    assert "final_score" in bd
```

---

#### Test 7 — empty ranked returns empty list

```python
def test_explain_empty_ranked(tmp_path):
    db_path = make_db(tmp_path, [])
    results = explain_all([], user_vec_from_mood(), db_path)
    assert results == []
```

---

#### Test 8 — missing SQLite row silently omitted

```python
def test_explain_missing_row_omitted(tmp_path):
    ranked = [make_ranked("ep_0"), make_ranked("ep_ghost")]
    rows = [make_db_row("ep_0", 0.8, 0.5, 0.5, 0.2)]
    db_path = make_db(tmp_path, rows)
    results = explain_all(ranked, user_vec_from_mood(), db_path)
    ids = [r.episode_id for r in results]
    assert "ep_0" in ids
    assert "ep_ghost" not in ids
```

---

### File B: `tests/recommender/test_engine.py`

Engine tests use full SQLite + Chroma setup (via `index_all`). Define helpers locally.

---

#### Test 9 — recommend returns RecommendationResult with both fields

```python
def test_engine_returns_result(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 6)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=3)
    assert isinstance(result, RecommendationResult)
    assert len(result.ranked) == 3
    assert len(result.explanations) == 3
```

---

#### Test 10 — top_k respected

```python
def test_engine_top_k_respected(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 8)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=3)
    assert len(result.ranked) <= 3
```

---

#### Test 11 — excluded_ids not in output

```python
def test_engine_excluded_ids(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", episode_id=f"show_s01_e0{i}", number=i)
                for i in range(1, 5)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=10,
                       excluded_ids={"show_s01_e01"})
    ids = [r.episode_id for r in result.ranked]
    assert "show_s01_e01" not in ids
```

---

#### Test 12 — empty collection returns empty result

```python
def test_engine_empty_collection(tmp_path):
    db_path = make_db(tmp_path, [])
    chroma_path = tmp_path / "chroma"
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=5)
    assert result.ranked == []
    assert result.explanations == []
```

---

#### Test 13 — ranked and explanations are aligned (same order, same IDs)

```python
def test_engine_ranked_and_explanations_aligned(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 5)]
    db_path, chroma_path = make_indexed(tmp_path, episodes)
    user_vec = user_vec_from_mood()
    result = recommend(user_vec, db_path, chroma_path, top_k=4)
    ranked_ids = [r.episode_id for r in result.ranked]
    explanation_ids = [e.episode_id for e in result.explanations]
    assert ranked_ids == explanation_ids
```

---

## Verification Commands

After implementation, run all tests:

```bash
pytest tests/recommender/ -v
```

Then run the full end-to-end live check against real data:

```bash
python -c "
import numpy as np
from pathlib import Path
from backend.recommender.engine import recommend
from backend.recommender.episode_feature_builder import load_episode_row, build_features

DB = Path('data/app.sqlite3')
CHROMA = Path('data/chroma')

row = load_episode_row('the_office_s02_e01', DB)
user_vec = build_features(row).vector

result = recommend(user_vec, DB, CHROMA, top_k=5)

for ranked, expl in zip(result.ranked, result.explanations):
    print(f'S{ranked.season_number}E{ranked.episode_number} {ranked.episode_title}')
    print(f'  score={ranked.final_score:.3f}  sim={ranked.similarity:.3f}  mood={ranked.mood_alignment:.3f}  tone={ranked.tone_alignment:.3f}')
    print(f'  mood matches:    {expl.mood_matches}')
    print(f'  tone matches:    {expl.tone_matches}')
    print(f'  dominant traits: {expl.dominant_episode_traits}')
    print()
"
```

Expected output: 5 ranked episodes with structured explanations showing which mood and
tone dimensions drove each match.

---

## Success Criteria

Phase 5 is complete when:

1. All 13 tests pass (7 explanation builder + 6 engine... wait, 8 + 5 = 13).
2. Test 3 confirms mood matches are populated when user and episode agree on a dim.
3. Test 4 confirms tone matches are populated when user and episode agree on a tone.
4. Test 13 confirms `ranked` and `explanations` are always aligned in order and IDs.
5. The live end-to-end check produces structured explanations for real episodes.
6. Full test suite (all phases, 35 existing + new) passes with no regressions.
