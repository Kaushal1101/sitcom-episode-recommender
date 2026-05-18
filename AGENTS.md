# Agent Instructions

## Current task: none — data pipeline complete

All data pipeline phases are implemented and verified for The Office (9 seasons, 201 episodes). The next session will begin the recommendation engine.

---

## Before starting the recommendation engine

Run vectorization on any episodes not yet in ChromaDB (safe to re-run with --skip-existing):

```bash
python -m backend.embedding --all the_office --skip-existing
```

---

## Completed tasks

### Scraping layer ✓
Built `backend/scraping/` and `backend/pipeline/`. All 201 Office episodes scraped across 9 seasons. Season discovery is automatic via `SeasonDiscoverer`.

### Enrichment layer ✓
Built `backend/enrichment/`. DeBERTa v3 zero-shot mood tagging. Outputs `mood_enriched.json` per episode (199/201 episodes enriched; s07e13 and s09e05 were skipped by the model).

### Mood vector layer ✓
Built `backend/embedding/`. Pure linear algebra — no ML model. Produces 17-dim episode vectors (4 mood + 13 tone). Writes `mood_vector.json`. Stores in ChromaDB collection `episode_mood_vectors`. CLI supports `--episode-id`, `--season`, `--all`, `--skip-existing`, `--similar-to`.

### Episode DB layer ✓
Built `backend/db/`. Plain sqlite3 — no ORM, no migrations. Single `CREATE TABLE IF NOT EXISTS episodes` in `setup.py`. All 201 episodes ingested into `data/app.sqlite3` with mood scalars and tone labels.

### Codebase cleanup ✓
Extracted shared CLI utilities into `backend/cli_utils.py` (SERIES_REGISTRY, repo_root, data_root, resolve_series, parse_episode_id, list_episode_ids). Eliminated duplication across all four `__main__.py` files. Consolidated `_mw_api_parse_url` into a single definition in `providers/fandom_wiki.py`.

---

## Architecture notes for the recommendation engine

The recommendation engine will live in `backend/recommender/` (to be planned). It must:
- Accept a user intent vector (17-dim, same space as `episode_mood_vectors` in ChromaDB)
- Query ChromaDB to retrieve candidate episodes
- Filter by hard constraints (excluded shows, excluded episodes)
- Rerank candidates using a weighted combination of vector similarity and mood scalar alignment
- Return ranked candidates with scores and confidence
- Respect the architecture constraints in `CLAUDE.md`

Do not implement LangGraph. Do not implement RL, Bayesian inference, or Thompson Sampling. See `CLAUDE.md` for full constraints.
