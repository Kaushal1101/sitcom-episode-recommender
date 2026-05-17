# Agent Instructions

## Current task: none — data pipeline complete

All three data pipeline phases (scraping, enrichment, mood vectors) are implemented and verified for The Office. The next session will begin the recommendation engine.

---

## Completed tasks

### Scraping layer ✓
Built `backend/scraping/` and `backend/pipeline/`. All 201 Office episodes scraped across 9 seasons. See `docs/PROJECT_LOG.md` for details.

### Enrichment layer ✓
Built `backend/enrichment/`. DeBERTa v3 zero-shot mood tagging. Outputs `mood_enriched.json` per episode.

### Mood vector layer ✓
Built `backend/embedding/`. Pure linear algebra — no ML model. Produces 17-dim episode vectors (4 mood + 13 tone). Writes `mood_vector.json`. Stores in ChromaDB collection `episode_mood_vectors`. CLI supports `--episode-id`, `--season`, `--all`, `--skip-existing`, `--similar-to`.

---

## Architecture notes for next session

Before starting the recommendation engine, run enrichment and vectorization on all 9 seasons:

```bash
# Enrich all episodes (~10–25 min CPU, skips already-done)
python -m backend.enrichment --all the_office --skip-existing

# Vectorize all enriched episodes
python -m backend.embedding --all the_office --skip-existing
```

The recommendation engine will live in `backend/recommender/` (to be planned). It must:
- Query `episode_mood_vectors` in ChromaDB
- Accept a user intent vector (17-dim, same space as episode vectors)
- Return ranked candidates with similarity scores
- Respect the architecture constraints in `CLAUDE.md`

Do not implement LangGraph. Do not implement RL, Bayesian inference, or Thompson Sampling. See `CLAUDE.md` for full constraints.
