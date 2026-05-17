# Vector Plan — Mood & Tone Episode Vectors

**Owner:** Orchestrator  
**Last updated:** 2026-05-17  
**Scope:** Building and storing episode vectors from enriched mood data. No rec engine logic, no user vectors.  
**Depends on:** Enrichment phase complete (`data/raw/{episode_id}/mood_enriched.json` exists)

---

## What This Phase Does

```
mood_enriched.json
      ↓
 episode_vectorizer.py   → mood_vec (4-dim) + tone_vec (13-dim)
                         → weighted combination → episode_vec (17-dim)
      ↓
   chroma_store.py       → upsert into ChromaDB
      +
 mood_vector.json        → written to disk for inspectability
```

**No embedding model is needed.** The DeBERTa scores are already continuous numbers. This phase is pure linear algebra — normalize, scale, concatenate.

---

## 1. Episode ID

Episode ID (`the_office_s02_e05`) is **not part of the vector**. It is the document key used to identify and retrieve the vector in ChromaDB. It does not affect the math.

---

## 2. Vector Construction

### Mood sub-vector — 4 dimensions

Fixed order:
```
[humor_level, energy_level, comfort_level, sadness_level]
```

Source: `mood_enriched.json → mood.{field}`.

### Tone sub-vector — 13 dimensions

Fixed order (must never change — order determines what each dimension means):
```
[awkward, chaotic, lighthearted, romantic, tense,
 heartwarming, cringe, silly, dramatic, emotional,
 wholesome, dark, bittersweet]
```

Source: `mood_enriched.json → mood.raw_scores.tone.{label}`.

Use `raw_scores.tone` (continuous scores), **not** the thresholded `tone` list. Continuous scores carry more signal for similarity than binary tags.

### Combining the sub-vectors

Step 1 — L2-normalize each sub-vector independently:
```
mood_unit = mood_vec / ||mood_vec||₂
tone_unit = tone_vec / ||tone_vec||₂
```

Step 2 — Scale by square-root weights and concatenate:
```
episode_vec = [ √0.7 × mood_unit  |  √0.3 × tone_unit ]   # 17-dim
```

Step 3 — L2-normalize the combined vector:
```
episode_vec = episode_vec / ||episode_vec||₂
```

**Why √0.7 / √0.3?** Cosine similarity squares each component. Using √w means the effective weight of each sub-space in cosine similarity is exactly w. The final normalize ensures the stored vector has unit length, which is required for correct cosine similarity.

**Why normalize per sub-vector first?** Without it, a sub-vector with larger magnitude would dominate regardless of the intended weights. Per-sub-vector normalization puts both sub-spaces on equal footing before the weights are applied.

### Handling zero vectors

If `mood_vec` or `tone_vec` is all zeros (shouldn't happen with DeBERTa but guard against it): skip normalization for that sub-vector and leave it as zeros. The combined vector will still be valid — just with zero contribution from that sub-space.

---

## 3. Output

### On disk — `mood_vector.json`

Written to `data/raw/{episode_id}/mood_vector.json`. Keeps the vector inspectable without querying ChromaDB.

```json
{
  "episode_id": "the_office_s02_e05",
  "mood_vec": [0.4694, 0.584, 0.1096, 0.7558],
  "tone_vec": [0.066, 0.1906, 0.0344, 0.0575, 0.3399, 0.0001, 0.0536, 0.0171, 0.0461, 0.9978, 0.0001, 0.089, 0.2294],
  "episode_vec": [0.xxx, ...],
  "mood_dimensions": ["humor_level", "energy_level", "comfort_level", "sadness_level"],
  "tone_dimensions": ["awkward", "chaotic", "lighthearted", "romantic", "tense", "heartwarming", "cringe", "silly", "dramatic", "emotional", "wholesome", "dark", "bittersweet"],
  "weights": {"mood": 0.7, "tone": 0.3}
}
```

`mood_dimensions` and `tone_dimensions` are included so the vector is self-describing — no need to look up dimension order elsewhere.

### In ChromaDB

- **Collection:** `episode_mood_vectors`
- **Document ID:** `episode_id`
- **Vector:** `episode_vec` (17-dim float list)
- **Metadata stored alongside:**
  ```json
  {
    "series_slug": "the_office",
    "season_number": 2,
    "episode_number": 5
  }
  ```

Metadata is stored for filtering at query time (e.g. "only recommend from seasons 1–3"). It is not part of the vector.

ChromaDB is local and file-backed. DB path: `data/chroma/`.

### Skipped episodes

Two episodes have no mood data (`skipped: true` in `mood_enriched.json`). These are not added to ChromaDB. Log a warning and continue.

---

## 4. File Structure

```
backend/embedding/
  __init__.py
  episode_vectorizer.py  ← pure math: mood_enriched.json → mood_vec, tone_vec, episode_vec
  chroma_store.py        ← ChromaDB client: upsert, query by vector
  __main__.py            ← CLI

data/
  raw/{episode_id}/
    mood_vector.json     ← written per episode
  chroma/                ← ChromaDB persistent storage
```

---

## 5. CLI

```bash
# Vectorize one episode
python -m backend.embedding --episode-id the_office_s02_e05

# Vectorize a full season
python -m backend.embedding --season the_office 2

# Vectorize all enriched episodes
python -m backend.embedding --all the_office

# Skip already-vectorized episodes
python -m backend.embedding --all the_office --skip-existing

# Query: find the 5 most similar episodes to a given one
python -m backend.embedding --similar-to the_office_s02_e05 --top-k 5
```

The `--similar-to` flag is a sanity check tool — lets you verify the vectors produce sensible nearest neighbours before the rec engine is built.

---

## 6. Dependencies

Add to `pyproject.toml`:
```toml
"numpy>=1.26",
"chromadb>=0.5",
```

---

## 7. Dimension Summary

| Sub-vector | Dims | Source field | Weight |
|---|---|---|---|
| mood | 4 | `mood.{humor,energy,comfort,sadness}_level` | 0.7 |
| tone | 13 | `mood.raw_scores.tone.{label}` (fixed order) | 0.3 |
| **episode_vec** | **17** | combined | — |

---

## 8. Open Questions

1. **Weight validation** — 0.7/0.3 is a starting assumption. After `--similar-to` spot-checks, if nearest neighbours feel off (e.g. tone is dominating or being ignored), adjust weights here before re-running.
2. **Future semantic vectors** — if a text embedding model (sentence-transformers) is added later, that would produce a separate collection in ChromaDB, not replace this one. The mood/tone vectors capture structured signal; semantic vectors capture narrative signal. They serve different purposes.
3. **User vector shape** — when the rec engine is built, the user preference vector must be in the same 17-dim space with the same dimension ordering. This file is the reference for that contract.
