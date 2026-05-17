# Backend & Recommendation Engine Plan

**Owner:** Orchestrator  
**Last updated:** 2026-05-17  
**Scope:** Scraping, Episode DB, Vector DB, Recommendation Engine  
**Out of scope:** UI Layer, Conversation Strategy Module (CSM)

---

## 1. What We're Working With

### Old project (`sitcom-ml-project`) — what exists
| Module | Files | State |
|---|---|---|
| Scraping | `episode_ref.py`, `extracted_data.py`, `http_client.py`, `runner.py`, `providers/wiki_fandom.py` | Working. All 22 The Office S2 episodes scraped. |
| DB | `models.py`, `repositories.py`, `session.py`, `character_repo.py` | SQLite via SQLAlchemy + Alembic. 4 migrations applied. |
| Ingestion | `ingestion/wiki_sqlite.py` | Reads `extracted_data.json`, upserts into SQLite. |
| Pipeline | `pipeline/graph.py`, `pipeline/__main__.py` | CLI entry point (scrape → ingest). |
| Retrieval | `retrieval/embedding_text.py` | Builds embedding text string. Not yet embedded. |

### New project (`sitcom-episode-recommender`) — what's defined
- `docs/schemas/episode-schema.json.txt` — target episode schema including mood/style vectors and embeddings
- `docs/schemas/parsed-preference-schema.json.txt` — user preference shape
- `docs/schemas/recommendation-result-schema.json.txt` — recommendation output contract
- `docs/schemas/session-state.json.txt` — session tracking shape
- `CLAUDE.md` — architecture constraints and module boundaries

### The gap
The old scraper produces `extracted_data.json` with raw text (synopsis, cold_open, cast). The new episode schema requires **mood/style float vectors** (`energy_level`, `humor_level`, etc.) and **semantic embeddings**. These do not exist yet. Bridging this gap is the central work of the backend.

---

## 2. Target Project Structure

```
sitcom-episode-recommender/
├── backend/
│   ├── scraping/                  # Migrated from old project
│   │   ├── __init__.py
│   │   ├── episode_ref.py         # EpisodeRef + registry
│   │   ├── extracted_data.py      # Raw JSON builder
│   │   ├── http_client.py         # HTTP with jitter/UA rotation
│   │   ├── runner.py              # Scrape orchestrator (one fetch, N writes)
│   │   └── providers/
│   │       ├── __init__.py
│   │       └── wiki_fandom.py     # Dunderpedia MediaWiki parse API
│   │
│   ├── enrichment/                # NEW — converts raw text → vectors
│   │   ├── __init__.py
│   │   ├── mood_tagger.py         # Produces mood float fields from synopsis
│   │   └── style_tagger.py        # Produces episode_style float fields
│   │
│   ├── embedding/                 # NEW — converts episode text → semantic vectors
│   │   ├── __init__.py
│   │   └── episode_embedder.py    # Calls embedding model, writes to vector DB
│   │
│   ├── db/                        # Episode DB (SQLite)
│   │   ├── __init__.py
│   │   ├── models.py              # SQLAlchemy models (extended for new schema)
│   │   ├── repositories.py        # Upsert helpers, make_episode_id
│   │   ├── session.py             # DB session factory
│   │   └── character_repo.py      # Character helpers (carry over)
│   │
│   ├── vector_db/                 # NEW — Vector DB abstraction
│   │   ├── __init__.py
│   │   ├── client.py              # ChromaDB client wrapper
│   │   └── episode_collection.py  # Episode vector CRUD
│   │
│   ├── ingestion/                 # Reads raw JSON → SQLite + Vector DB
│   │   ├── __init__.py
│   │   └── episode_ingestor.py    # Replaces wiki_sqlite.py; handles full schema
│   │
│   ├── rec_engine/                # Recommendation Engine (new)
│   │   ├── __init__.py
│   │   ├── retriever.py           # Vector search → candidate set
│   │   ├── filter.py              # Hard constraint filtering
│   │   ├── reranker.py            # Score-based reranking
│   │   ├── confidence.py          # Confidence score computation
│   │   └── engine.py             # Orchestrates retriever → filter → reranker
│   │
│   └── pipeline/                  # CLI entry points
│       ├── __init__.py
│       ├── __main__.py            # scrape → enrich → embed → ingest
│       └── graph.py               # Pipeline step graph
│
├── alembic/                       # DB migrations (continued from old project)
│   └── versions/
│       ├── 001_initial_schema.py
│       ├── 002_episode_source_document.py
│       ├── 003_wiki_only_series_tmdb_nullable.py
│       ├── 004_episode_contract_fields.py
│       └── 005_mood_style_vectors.py   # NEW — adds mood/style float columns
│
├── data/
│   ├── raw/{episode_id}/          # Scraped JSON artifacts (same as old project)
│   └── app.sqlite3                # Episode DB
│
├── docs/
│   └── BACKEND_PLAN.md            # This file
│
├── .env.example
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md
```

---

## 3. Scraping Migration

### What to port directly (no logic changes)
| File | Notes |
|---|---|
| `episode_ref.py` | Port as-is. The Office S2 registry + combined episode support. |
| `http_client.py` | Port as-is. Jitter + User-Agent rotation is correct. |
| `providers/wiki_fandom.py` | Port as-is. Dunderpedia MediaWiki parse API works. |
| `runner.py` | Port as-is. "Scrape once, write N" semantics are correct. |

### What to adapt (`extracted_data.py`)
The old `extracted_data.py` produces a raw JSON with `narrative` (cold_open, synopsis, cast). In the new project this is still the right **raw output** — do not add mood/style tagging here. Keep scraping dumb and fast. The enrichment step handles mood/style separately.

**Keep in `extracted_data.py`:**
- `narrative.cold_open`
- `narrative.synopsis`
- `narrative.cast` (main/supporting/recurring/other buckets)
- `metadata` (episode_id, series_slug, season, episode number, air_date, provenance)

**Remove from `extracted_data.py`:**
- Legacy `plot` field (deprecated concatenation)
- `cast_text_debug`, `lede_text`, `trivia` — these were transitional; drop them in the new project
- `social` stub — not needed until Reddit/social is in scope

### Episode registry strategy
Start with The Office Season 2 (22 episodes, already mapped). The registry pattern (`EpisodeRef` + `_EPISODE_REGISTRY`) is correct and should be the extension point for new shows.

---

## 4. Enrichment Layer (NEW)

This is the critical new piece. After scraping produces raw text, the enrichment layer produces the mood and style float vectors required by the episode schema.

### Mood tagger (`enrichment/mood_tagger.py`)
**Input:** `narrative.synopsis` + `narrative.cold_open`  
**Output:** `mood` object:
```json
{
  "tone": ["awkward", "lighthearted"],
  "energy_level": 0.4,
  "humor_level": 0.3,
  "comfort_level": 0.7,
  "sadness_level": 0.1,
  "tags": ["workplace", "comedy"]
}
```

**Strategy for MVP:** Rule-based keyword scoring against synopsis text. No LLM required yet. Keep it deterministic and inspectable.  
- Build a keyword-to-dimension map (e.g. "funeral", "fired", "breakup" → sadness score contribution)
- Normalize scores to [0, 1]
- Tone tags derived from top-scoring dimensions + show-level defaults

**Future:** Swap in a lightweight classifier if rule-based quality is insufficient.

### Style tagger (`enrichment/style_tagger.py`)
**Input:** `narrative.synopsis`, `metadata.season_number`, `metadata.episode_number`, `narrative.cast`  
**Output:** `episode_style` object:
```json
{
  "novelty_level": 0.8,
  "specialness": 0.9,
  "setting_variation": 0.7,
  "continuity_heaviness": 0.6,
  "stakes_level": 0.7,
  "experimental_style": 0.4
}
```

**Strategy for MVP:** Rule-based, seeded by:
- `specialness` — holiday/event keyword detection in title or synopsis
- `continuity_heaviness` — character arc / season finale signals
- `setting_variation` — location keywords (road trip, convention, etc.)
- `novelty_level` — inverse of standard workplace format signals
- `stakes_level` — conflict intensity signals
- `experimental_style` — mockumentary-breaking signals (talking heads count, direct address)

---

## 5. Episode DB (SQLite) — Schema Extension

The existing SQLite schema (migrations 001–004) covers:
- `series`, `episode`, `character`, `episode_character`
- `episode_source_document`
- `cold_open`, `synopsis`, `cast_json` (migration 004)

### Migration 005: mood and style vector columns
Add to the `episode` table:

```sql
-- mood dimensions
energy_level        REAL,
humor_level         REAL,
comfort_level       REAL,
sadness_level       REAL,
mood_tags           TEXT,   -- JSON array of strings
mood_tone           TEXT,   -- JSON array of strings

-- episode style dimensions
novelty_level       REAL,
specialness         REAL,
setting_variation   REAL,
continuity_heaviness REAL,
stakes_level        REAL,
experimental_style  REAL,

-- structure
runtime_minutes     INTEGER,
is_standalone       BOOLEAN,
is_special_episode  BOOLEAN,
special_type        TEXT,

-- characters
characters_involved TEXT,   -- JSON array of strings
primary_characters  TEXT,   -- JSON array of strings
character_focus     TEXT,

-- provenance
embedding_source    TEXT,   -- e.g. "summary_v1"
enrichment_version  TEXT,   -- for re-enrichment tracking
confidence_score    REAL
```

**Principle:** SQLite remains the single source of truth for structured metadata and mood/style scalars. The vector DB holds embeddings only and references back to `episode_id`.

---

## 6. Vector DB

**Choice: ChromaDB (local, file-backed)**  
Rationale: fits the MVP constraint (no distributed infra), already in scope from old project's chroma references, inspectable, zero-ops.

### Collection: `episodes`
- **ID:** `episode_id` (e.g. `the_office_s02_e04`)
- **Embedding:** 768-dim or 1536-dim float vector from episode text
- **Metadata stored in Chroma:** `series_id`, `season_number`, `episode_number` — enough to join back to SQLite for full metadata
- **Embedding text source:** `episode_embedding_text()` from `retrieval/embedding_text.py` (port from old project) — concatenation of synopsis + cold_open + key mood tags

### Embedding model
**MVP:** `sentence-transformers/all-MiniLM-L6-v2` (local, fast, 384-dim). No API dependency.  
**Future:** Swap to OpenAI `text-embedding-3-small` if quality demands it.

### User vector
User vectors live **in memory during a session** — they are not persisted to ChromaDB. The recommendation engine holds the current user vector and computes cosine similarity against episode vectors at query time.

---

## 7. Recommendation Engine

### Module: `rec_engine/engine.py` — the only public interface

```python
def recommend(
    user_vector: np.ndarray,
    session_state: SessionState,
    top_k: int = 20,
) -> list[RankedEpisode]:
    candidates = retriever.retrieve(user_vector, top_k=top_k * 3)
    filtered = filter.apply(candidates, session_state.hard_constraints)
    ranked = reranker.rank(filtered, user_vector, session_state)
    return ranked[:top_k]
```

### `rec_engine/retriever.py`
- Queries ChromaDB with user vector
- Returns broad candidate set (episode_ids + raw scores)
- Fetches structured metadata from SQLite for each candidate

### `rec_engine/filter.py`
- Hard constraint application: `excluded_shows`, `excluded_episodes`, `excluded_tags`, `max_runtime_minutes`
- Pure function: `filter(candidates, constraints) → candidates`
- No ML, no vectors — deterministic

### `rec_engine/reranker.py`
- Combines vector similarity score with structured mood/style scalar alignment
- Applies session context (penalize previously recommended, boost accepted patterns)
- Score formula (MVP):
  ```
  final_score = (w1 * vector_similarity) + (w2 * mood_alignment) + (w3 * style_alignment)
  ```
  Where weights start at `[0.5, 0.3, 0.2]` and can be tuned.

### `rec_engine/confidence.py`
- Computes confidence from: vector similarity distribution, number of answered questions, constraint coverage
- Returns float in [0, 1]
- Used by CSM (via interface) to decide whether to recommend or ask more questions

### User vector updates
The recommendation engine is the **only** module that updates user vectors:
1. **From preference answers:** Parsed preferences (energy, humor, etc.) map directly to user vector dimensions
2. **From critique feedback:** Accepted/rejected episodes shift the user vector toward/away from episode vectors
3. **Update rule (MVP):** Exponential moving average — `user_vec = alpha * new_signal + (1 - alpha) * user_vec` with `alpha = 0.3`

---

## 8. Data Pipeline (Full Flow)

```
CLI: python -m backend.pipeline --episode-id <id>

Step 1: Scrape
  wiki_fandom.py → MediaWiki parse API
  runner.py → writes data/raw/{episode_id}/extracted_data.json

Step 2: Enrich
  mood_tagger.py → reads synopsis, writes mood fields
  style_tagger.py → reads synopsis + metadata, writes style fields
  → writes data/raw/{episode_id}/enriched_data.json

Step 3: Embed
  episode_embedder.py → builds embedding text, calls embedding model
  → upserts into ChromaDB episodes collection

Step 4: Ingest
  episode_ingestor.py → reads enriched_data.json
  → upserts into SQLite (all columns including mood/style)
```

### CLI flags (to be implemented)
```
--episode-id <id>      Target episode
--skip-scrape          Use existing raw JSON
--skip-enrich          Use existing enriched JSON
--skip-embed           Skip vector DB upsert
--skip-ingest          Skip SQLite upsert
--all-s2               Run full Season 2 batch
--json                 Print final state as JSON
```

---

## 9. Interfaces Between Modules

The recommendation engine exposes these interfaces to the CSM:

```python
# Input: user vector + session state → output: ranked episodes + confidence
def recommend(user_vector, session_state, top_k) -> list[RankedEpisode]: ...

# Input: session state + feedback → output: updated user vector
def update_user_vector(user_vector, feedback: EpisodeFeedback) -> np.ndarray: ...

# Input: user vector → output: confidence float
def compute_confidence(user_vector, session_state) -> float: ...

# Input: parsed preference update → output: updated user vector
def apply_preference(user_vector, preference: ParsedPreference) -> np.ndarray: ...
```

The CSM calls these — it never touches vectors or DBs directly. This boundary is enforced by module structure.

---

## 10. Phased Delivery

### Phase 1 — Scraping migration (immediate)
- Port scraping modules into new project structure
- Drop legacy fields (`plot`, `trivia`, `cast_text_debug`, `lede_text`)
- Verify The Office S2 scrape still works end-to-end
- **Deliverable:** `data/raw/{episode_id}/extracted_data.json` for all S2 episodes

### Phase 2 — Episode DB extension
- Write migration 005 (mood/style/structure columns)
- Extend `models.py` and `repositories.py`
- Write `episode_ingestor.py` (replaces `wiki_sqlite.py`)
- **Deliverable:** SQLite schema ready for full episode schema

### Phase 3 — Enrichment layer
- Build `mood_tagger.py` with keyword-based scoring
- Build `style_tagger.py` with rule-based signals
- Wire into pipeline as Step 2
- **Deliverable:** All S2 episodes have mood/style floats in SQLite

### Phase 4 — Embedding + Vector DB
- Set up ChromaDB client and episodes collection
- Build `episode_embedder.py`
- Wire into pipeline as Step 3
- **Deliverable:** All S2 episodes queryable by vector similarity

### Phase 5 — Recommendation Engine
- Build `retriever.py`, `filter.py`, `reranker.py`, `confidence.py`, `engine.py`
- Expose clean interface for CSM
- **Deliverable:** `engine.recommend()` returns ranked episodes given a user vector

---

## 11. Constraints (from CLAUDE.md)

- No LangChain, LangGraph, RL, Bayesian inference, Thompson Sampling, or GNNs
- No microservices or distributed infrastructure
- ChromaDB: local file-backed only
- SQLite remains episode DB (no Postgres migration in MVP)
- All recommendation logic must be deterministic and inspectable
- Recommendation engine is the **only** module that updates intent/user vectors

---

## 12. Open Questions (to resolve before Phase 3)

1. **Mood tagger quality bar** — what's the minimum acceptable accuracy for mood tags before we consider a classifier? Define a spot-check protocol.
2. **Embedding text composition** — synopsis only, or synopsis + cold_open + cast? This affects retrieval quality significantly. Start with synopsis-only and measure.
3. **Multi-show expansion** — when do we add a second show beyond The Office? This drives whether the registry pattern scales or needs rethinking.
4. **Guest star enrichment** — `guest_features.guest_stars` and `guest_star_prominence` are in the episode schema but not in the scraper output. Is this needed for MVP?
5. **Confidence thresholds** — the CSM flow doc defines thresholds (0.4, 0.6, 0.7, 0.85, 0.9). These are the recommendation engine's output contract. Confirm these are fixed before building `confidence.py`.
