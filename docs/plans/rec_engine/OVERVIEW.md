# Recommendation Engine Plan

## Project Goal

Build the recommendation engine for a conversational sitcom episode recommender.

The recommendation engine is responsible for:
- converting episode records into model-ready vectors
- storing and querying episode vectors
- retrieving candidate episodes for a given user intent vector
- reranking retrieved candidates
- returning recommendations with explanation-ready signals

The recommendation engine is NOT responsible for:
- UI rendering
- question generation
- conversation strategy
- free-text parsing
- raw enrichment / scraping
- storing session history outside the current request flow

---

## Core Architecture Decisions

### 1. Feature space
The MVP uses a **17-dimension mood/tone space**.

This is the primary episode representation.

The engine should NOT depend on semantic plot embeddings for MVP.

Use the following idea:
- **mood** = primary matching signal
- **tone** = secondary matching signal
- **metadata** = filtering and tie-breaking, not the core vector space

### 2. Episode vectors
Each episode should be represented by:
- a compact 17-dim vector derived from mood + tone
- separate metadata fields from SQLite
- optional raw enrichment data for debugging only

### 3. Retrieval store
Use **ChromaDB** for vector storage and nearest-neighbor retrieval.

Chroma stores vectors; it does not generate them.

### 4. Source of truth
SQLite is the source of truth for episode records.

Chroma is a derived index.

If SQLite changes, Chroma should be rebuildable from SQLite + enrichment artifacts.

---

## Inputs and Outputs

### Recommendation engine input
The engine receives:
- user vector
- metadata preferences
- hard exclusions
- optional session context
- optional candidate constraints from the strategy module

### Recommendation engine output
The engine returns:
- ranked candidate episodes
- per-candidate scores
- candidate metadata
- confidence signals
- explanation-ready trait matches

---

## Data Model Assumptions

The engine will consume episode rows from SQLite.

Expected episode fields include:
- episode_id
- series_slug
- series_title
- season_number
- episode_number
- episode_title
- air_date
- synopsis
- cold_open
- cast_main
- cast_supporting
- cast_recurring
- humor_level
- energy_level
- comfort_level
- sadness_level
- tone labels / tone scores
- enrichment metadata

The engine may ignore raw enrichment metadata during vector creation, but must preserve it for debugging and traceability.

---

## Recommendation Engine Responsibilities

### A. Episode vector construction
Convert SQLite rows into a compact, fixed-length 17-dim vector.

The vector should include:
- mood dimensions
- tone dimensions

It should NOT include:
- episode_id
- timestamps
- model_id
- raw text blobs
- raw JSON fields that are only for provenance

Those may remain in SQLite, but should not be part of the similarity vector.

### B. Vector index population
Build a script to:
- read all episodes from SQLite
- generate vectors
- upsert vectors into Chroma
- attach metadata for filtering and inspection

### C. Candidate retrieval
Given a user vector:
- retrieve top-K candidates from Chroma
- K should be configurable
- default K should be around 50 for MVP

### D. Filtering
Apply deterministic filters before final ranking when needed:
- show exclusions
- runtime constraints
- special episode constraints
- season / metadata preferences
- any explicit user hard exclusions

### E. Reranking
Rerank the retrieved candidates using a deterministic scoring function.

The reranker should consider:
- core vector similarity
- tone alignment
- metadata preferences
- hard penalties / exclusions
- diversity / variety if needed

### F. Candidate summaries
Return compact explanation-ready summaries such as:
- matched on comfort
- matched on humor
- low chaos
- early-season preference
- highly rated episode

---

## Feature Space Rules

### Mood = primary
Mood dimensions should dominate the core similarity score.

Examples:
- humor
- energy
- comfort
- sadness
- chaos

### Tone = secondary
Tone dimensions should refine the score, but not dominate it.

Examples:
- emotional
- awkward
- bittersweet
- tense
- wholesome
- cringe
- lighthearted

### Metadata = separate
Metadata should be used for:
- filters
- bonuses
- tie-breaks
- user-specific preferences

Examples:
- runtime
- season
- rating
- release date
- show title
- special episode flag

---

## Planned Modules

### 1. `episode_feature_builder`
Purpose:
- convert SQLite episode rows into normalized vectors

Responsibilities:
- map schema fields to fixed feature order
- normalize mood/tone values
- produce the final 17-dim vector
- keep feature order stable

Non-goals:
- no retrieval
- no ranking
- no UI logic

---

### 2. `vector_indexer`
Purpose:
- populate Chroma from SQLite

Responsibilities:
- read all episodes
- build vectors
- insert or update vectors in Chroma
- store metadata alongside vectors

Non-goals:
- no conversation logic
- no question selection
- no free-text parsing

---

### 3. `retriever`
Purpose:
- query Chroma using a user vector

Responsibilities:
- return top-K candidate episodes
- support configurable K
- support optional metadata filters

Non-goals:
- no question selection
- no user vector updates
- no deterministic reranking logic beyond nearest neighbor retrieval

---

### 4. `reranker`
Purpose:
- rerank retrieved candidates with deterministic scoring

Responsibilities:
- combine vector similarity and metadata preferences
- apply penalties for hard exclusions
- optionally diversify top results
- produce final ranked list

Non-goals:
- no retrieval indexing
- no UI formatting
- no raw enrichment

---

### 5. `explanation_builder`
Purpose:
- generate structured match reasons for each candidate

Responsibilities:
- summarize why an episode matched
- expose trait-level alignment
- remain deterministic where possible

Non-goals:
- no natural-language chatbot generation
- no free-form model output dependence

---

## Implementation Order

Build in this order:

### Phase 1 — Feature builder
- define the exact 17-dim feature order
- implement episode row -> vector conversion
- write unit tests for dimension order and normalization

### Phase 2 — Chroma indexing
- read from SQLite
- build vectors
- store vectors in Chroma
- attach metadata

### Phase 3 — Retrieval
- implement top-K retrieval for a mock user vector
- verify that returned candidates are plausible

### Phase 4 — Reranking
- implement deterministic reranking
- combine mood and tone with weights
- support basic metadata bonuses and penalties

### Phase 5 — Explanation output
- return match reasons and candidate summaries
- make output compatible with the strategy module

---

## Weighting Guidance

For MVP, use weighted scoring such as:
- mood similarity = primary weight
- tone similarity = secondary weight
- metadata bonuses = small adjustments

Suggested starting point:
- mood: 70–80%
- tone: 20–30%

Do NOT overfit the initial weights.
Keep them easy to inspect and tune.

---

## Hard Constraints

Hard exclusions must be enforced before final ranking.

Examples:
- excluded shows
- excluded episodes
- max/min runtime
- explicit no-special-episode preference
- any other deterministic user restriction

Hard constraints should never be overridden by similarity score.

---

## Debugging Requirements

The engine should be inspectable.

For every retrieval or rerank run, it should be possible to log:
- input user vector
- retrieved candidate IDs
- scores before rerank
- scores after rerank
- applied filters
- top match reasons

This is critical for development.

---

## Current MVP Constraints

Do NOT add:
- semantic summary embeddings
- graph-based episode traversal
- learned reranker models
- Bayesian updating
- RL / Thompson Sampling
- free-text intent parsing
- microservices

These can be future work.

The MVP should remain:
- deterministic
- inspectable
- easy to test
- easy to refactor

---

## Success Criteria

The recommendation engine is considered working when:

1. It can read enriched episode rows from SQLite.
2. It can convert each episode into a stable 17-dim vector.
3. It can populate Chroma with those vectors.
4. It can retrieve top-K candidates for a mock user vector.
5. It can rerank candidates deterministically.
6. It can return match reasons and scores.
7. It can respect hard exclusions reliably.

---

## Future Extensions

Later versions may add:
- semantic summary embeddings
- graph relationships between episodes
- learned reranking
- user feedback-based vector updating
- better confidence estimation
- adaptive question selection feedback loops

Do not implement these until the MVP retrieval + reranking pipeline works end-to-end.