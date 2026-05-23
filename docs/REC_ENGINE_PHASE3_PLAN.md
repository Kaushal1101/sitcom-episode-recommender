# Recommendation Engine — Phase 3: Retriever

## Goal

Build `backend/recommender/retriever.py` — a module that accepts a 17-dim user vector,
queries ChromaDB for the nearest episode vectors, and returns a ranked list of candidate
episodes with similarity scores and metadata.

The retriever's only job is fast nearest-neighbour lookup. It does not touch SQLite,
does not rerank, and does not update any vectors.

---

## Context

- **Phase 1 (done):** `episode_feature_builder.py` — builds 17-dim unit vectors from SQLite rows.
- **Phase 2 (done):** `vector_indexer.py` — populates ChromaDB with 199 episode vectors + metadata.
  Chroma stores per-entry metadata: `series_slug`, `season_number`, `episode_number`, `episode_title`.
- **Phase 3:** `retriever.py` — queries Chroma with a user vector, returns `list[Candidate]`.

---

## Architecture Decisions

- The retriever is **Chroma-only**: it does not fetch from SQLite.
  Full episode detail (synopsis, mood scalars) is the reranker's concern (Phase 4).
- Output is a `list[Candidate]` sorted by similarity descending — Chroma already returns
  results in this order for cosine space.
- Filters are passed as explicit kwargs (`series_slug`, `season_number`), not as raw
  Chroma `where` dicts. The module builds the Chroma `where` clause internally.
- If `top_k` exceeds the collection size, cap `n_results` at `collection.count()` to
  avoid a Chroma error. If the collection is empty, return `[]` immediately.

---

## Output Type

```python
@dataclass
class Candidate:
    episode_id:     str
    similarity:     float   # cosine similarity in [0, 1]; higher = more similar
    series_slug:    str
    season_number:  int
    episode_number: int
    episode_title:  str
```

`similarity` is computed as `1 - cosine_distance` (Chroma returns distances, not similarities).

---

## Module Structure

### Files to create

```
backend/recommender/
└── retriever.py       (new)

tests/recommender/
└── test_retriever.py  (new)
```

Do NOT modify any existing files.

---

## `retriever.py` Public Interface

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.recommender.vector_indexer import COLLECTION_NAME, get_collection

DEFAULT_TOP_K: int = 50


@dataclass
class Candidate:
    episode_id:     str
    similarity:     float
    series_slug:    str
    season_number:  int
    episode_number: int
    episode_title:  str


def retrieve(
    user_vector: np.ndarray,
    chroma_path: Path,
    top_k: int = DEFAULT_TOP_K,
    series_slug: str | None = None,
    season_number: int | None = None,
) -> list[Candidate]:
    """
    Query ChromaDB with user_vector and return up to top_k nearest candidates.

    Results are sorted by similarity descending (most similar first).
    Returns an empty list if the collection is empty or no results match the filters.
    """
```

---

## Implementation Notes

### Where clause builder

```python
def _build_where(
    series_slug: str | None,
    season_number: int | None,
) -> dict | None:
    conditions = []
    if series_slug is not None:
        conditions.append({"series_slug": series_slug})
    if season_number is not None:
        conditions.append({"season_number": season_number})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
```

### Retrieve implementation

```python
def retrieve(user_vector, chroma_path, top_k=DEFAULT_TOP_K,
             series_slug=None, season_number=None) -> list[Candidate]:
    collection = get_collection(chroma_path)
    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)
    where = _build_where(series_slug, season_number)

    query_kwargs = dict(
        query_embeddings=[user_vector.tolist()],
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    candidates = []
    for episode_id, distance, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
        strict=True,
    ):
        candidates.append(Candidate(
            episode_id=episode_id,
            similarity=round(1.0 - distance, 6),
            series_slug=meta["series_slug"],
            season_number=meta["season_number"],
            episode_number=meta["episode_number"],
            episode_title=meta["episode_title"],
        ))
    return candidates
```

### What this module does NOT do

- No SQLite reads
- No vector math or normalization
- No reranking or scoring beyond cosine similarity
- No session or user state

---

## Tests

### File: `tests/recommender/test_retriever.py`

Use `pytest` with `tmp_path`. Each test sets up a fresh Chroma index using
`index_all` from Phase 2, then calls `retrieve`.

Define local helpers `make_db`, `ep`, `skipped_ep` in this file (same pattern
as `test_vector_indexer.py` — do not import from it).

---

### Helper: `make_indexed_chroma`

```python
def make_indexed_chroma(tmp_path: Path, episodes: list[dict]) -> tuple[Path, Path]:
    """Create SQLite DB, index into Chroma, return (db_path, chroma_path)."""
    db_path = make_db(tmp_path / "db", episodes)
    chroma_path = tmp_path / "chroma"
    index_all(db_path, chroma_path)
    return db_path, chroma_path
```

---

### Test 1 — Returns correct number of results

```python
def test_retrieve_returns_top_k(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 6)]  # 5 episodes
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=3)
    assert len(results) == 3
```

---

### Test 2 — Results sorted by similarity descending

```python
def test_retrieve_sorted_by_similarity(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 6)]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=5)
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)
```

---

### Test 3 — Similarity values in valid range

```python
def test_retrieve_similarity_in_range(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 4)]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=3)
    for r in results:
        assert 0.0 <= r.similarity <= 1.0
```

---

### Test 4 — Candidate has all required fields

```python
def test_retrieve_candidate_fields(tmp_path):
    episode = ep("show_s02_e04", season=2, number=4, title="The Fire")
    _, chroma_path = make_indexed_chroma(tmp_path, [episode])
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=1)
    assert len(results) == 1
    c = results[0]
    assert c.episode_id == "show_s02_e04"
    assert c.series_slug == "the_office"
    assert c.season_number == 2
    assert c.episode_number == 4
    assert c.episode_title == "The Fire"
    assert isinstance(c.similarity, float)
```

---

### Test 5 — Exact vector match ranks first with similarity ≈ 1.0

```python
def test_retrieve_exact_match_ranks_first(tmp_path):
    """Querying with an episode's own vector should return it as the top result."""
    target = ep("show_s01_e01", humor=0.9, energy=0.8, comfort=0.5, sadness=0.1)
    others = [ep(f"show_s01_e0{i}", number=i, humor=0.1, energy=0.1) for i in range(2, 5)]
    _, chroma_path = make_indexed_chroma(tmp_path, [target] + others)

    from backend.recommender.episode_feature_builder import build_features
    query_vec = build_features(target).vector

    results = retrieve(query_vec, chroma_path, top_k=4)
    assert results[0].episode_id == "show_s01_e01"
    assert results[0].similarity > 0.99
```

---

### Test 6 — series_slug filter excludes other shows

```python
def test_retrieve_series_slug_filter(tmp_path):
    episodes = [
        ep("office_s01_e01", series_slug="the_office"),
        ep("office_s01_e02", series_slug="the_office", number=2),
        ep("friends_s01_e01", series_slug="friends"),
    ]
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=10, series_slug="the_office")
    assert len(results) == 2
    assert all(r.series_slug == "the_office" for r in results)
```

---

### Test 7 — top_k larger than collection returns all available

```python
def test_retrieve_top_k_larger_than_collection(tmp_path):
    episodes = [ep(f"show_s01_e0{i}", number=i) for i in range(1, 4)]  # 3 episodes
    _, chroma_path = make_indexed_chroma(tmp_path, episodes)
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=100)
    assert len(results) == 3
```

---

### Test 8 — Empty collection returns empty list

```python
def test_retrieve_empty_collection(tmp_path):
    chroma_path = tmp_path / "chroma"
    user_vec = np.ones(VECTOR_DIM, dtype=np.float32)
    user_vec /= np.linalg.norm(user_vec)
    results = retrieve(user_vec, chroma_path, top_k=10)
    assert results == []
```

---

## Verification Commands

After implementation, run all tests:

```bash
pytest tests/recommender/ -v
```

Then do a live sanity check against the real database:

```bash
python -c "
import numpy as np
from pathlib import Path
from backend.recommender.retriever import retrieve
from backend.recommender.episode_feature_builder import load_episode_row, build_features

# Use The Dundies (S2E01) vector as the user vector
row = load_episode_row('the_office_s02_e01', Path('data/app.sqlite3'))
user_vec = build_features(row).vector

results = retrieve(user_vec, Path('data/chroma'), top_k=5)
for r in results:
    print(f'{r.similarity:.4f}  S{r.season_number}E{r.episode_number}  {r.episode_title}')
"
```

Expected: top result is `the_office_s02_e01` itself (similarity ≈ 1.0), followed by
episodes with similar high-humor, awkward tone profiles.

---

## Success Criteria

Phase 3 is complete when:

1. All 8 tests pass.
2. Querying with an episode's own vector returns it as rank 1 with similarity > 0.99.
3. The `series_slug` filter correctly excludes other shows.
4. `top_k` larger than collection size does not raise.
5. The live sanity check returns plausible neighbours for `the_office_s02_e01`.
