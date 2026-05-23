# Project Log

Append-only session reports. Newest entries at the top. Each entry records what changed, who (role), and how it was verified.

Do not put secrets or `.env` contents here.

---

## 2026-05-23 — Confidence calculator

**Summary:** Designed and implemented the confidence calculator — a deterministic sub-component
of the recommendation engine that produces a `[0.0, 1.0]` confidence score consumed by the CSM.
Established `ConfidenceResult` as the stable contract between the two modules.

**Changes:**
- `backend/recommender/confidence.py` — `compute_confidence()`, `ConfidenceResult`, `TopCandidate`; three-signal formula: top_score (0.5) + normalized_gap (0.3) + normalized_coverage (0.2)
- `tests/recommender/test_confidence.py` — 8 unit tests including worked-example regression
- `docs/plans/CONFIDENCE_CALCULATOR_PLAN.md` — new plan; architecture position, formula, edge cases, output contract, CSM threshold alignment
- `docs/schemas/confidence-result-schema.json.txt` — formal contract schema between confidence calculator and CSM
- `docs/CAVEATS.md` — new file; small observations and debugging notes (venv requirement, skipped episodes, near-duplicate two-parter, calibration notes)

**Commands / verification:**
```bash
source .venv/bin/activate && python -m pytest tests/recommender/test_confidence.py -v
# 8/8 passing
```

**Edge cases / notes:**
- `MAX_EXPECTED_GAP = 0.15` calibrated for single-show (The Office) corpus; retune after adding more shows
- Coverage signal is raw question count for MVP; future: weight by question type or derive from vector entropy
- All calibration constants are kwargs — overridable in tests without monkey-patching

---

## 2026-05-23 — Recommendation engine (Phases 1–5)

**Summary:** Built the full recommendation engine across five phases. The engine is
callable end-to-end via `engine.recommend()` and tested with 48 unit tests.

**Changes:**
- `backend/db/setup.py` — added `tone_scores TEXT` column to `episodes` table
- `backend/db/ingestor.py` — writes `tone_scores` from `mood["raw_scores"]["tone"]`
- `backend/recommender/episode_feature_builder.py` — converts SQLite rows to 17-dim L2-normalized vectors; exports `load_mood_rows` (shared DB loader)
- `backend/recommender/vector_indexer.py` — populates ChromaDB from SQLite; CLI: `python -m backend.recommender.vector_indexer --all the_office --wipe`
- `backend/recommender/retriever.py` — top-K Chroma query; supports `series_slug` / `season_number` filters
- `backend/recommender/reranker.py` — deterministic rescoring: `0.5*sim + 0.3*mood + 0.2*tone`; hard exclusions
- `backend/recommender/explanation_builder.py` — structured match reasons (mood/tone matches, dominant traits, score breakdown)
- `backend/recommender/engine.py` — public entry point: retrieve → rerank → explain
- `tests/recommender/` — 48 unit tests across all five phases
- `.gitignore` — added `data/app.sqlite3`
- `docs/RECOMMENDATION_ENGINE_PLAN.md` — renamed from ENGINER typo

**Commands / verification:**
```bash
# Rebuild SQLite with tone_scores
rm data/app.sqlite3 && python -m backend.db --all the_office  # 201 rows, 199 with tone_scores

# Rebuild ChromaDB index
python -m backend.recommender.vector_indexer --all the_office --wipe  # 199 indexed, 2 skipped

# Run full test suite
pytest tests/recommender/ -v  # 48/48 passing

# Live end-to-end check
python -c "
from pathlib import Path
from backend.recommender.engine import recommend
from backend.recommender.episode_feature_builder import load_episode_row, build_features
row = load_episode_row('the_office_s02_e01', Path('data/app.sqlite3'))
result = recommend(build_features(row).vector, Path('data/app.sqlite3'), Path('data/chroma'), top_k=5)
for r in result.ranked: print(r.episode_title, r.final_score)
"
```

**Edge cases / notes:**
- 2 skipped episodes (`s07_e13`, `s09_e05`) have NULL mood data — excluded from Chroma index, return None from `build_features`
- S5E14/E15 (Stress Relief two-parter) share identical mood scores — will appear as near-duplicates; handle via `excluded_ids` in the session layer
- Heartwarming/wholesome mood profile returns weaker results — thin signal in enrichment data, not an engine bug
- `load_mood_rows` extracted from reranker/explanation_builder to eliminate duplication

---

## 2026-05-18 — SQLite episode DB

**Summary:** Completed full-show mood enrichment (177 new + 23 existing + 1 skipped) and built `backend/db/` to ingest `extracted_data.json` + `mood_enriched.json` into `data/app.sqlite3`.

**Changes:**
- `backend/db/setup.py` — `setup_db()` creates `episodes` table (no Alembic)
- `backend/db/ingestor.py` — `ingest_episode()` merges JSON files via `INSERT OR REPLACE`
- `backend/db/__main__.py` — CLI: `--episode-id`, `--season`, `--all`

**Commands / verification:**
```bash
python -m backend.enrichment --all the_office --skip-existing   # ~12 min CPU, 177 enriched, 1 skipped
python -m backend.db --episode-id the_office_s02_e01
python -m backend.db --all the_office
sqlite3 data/app.sqlite3 "SELECT COUNT(*), COUNT(humor_level) FROM episodes;"  # 201 rows, 199 with mood
```

**Edge cases:**
- `the_office_s07_e13` and `the_office_s09_e05` have `mood_enriched.json` with `skipped: true` — mood columns NULL in SQLite (expected).

---

## 2026-05-17 — Session close (data pipeline)

**Summary:** No new code this session. Reviewed the completed 17-dim mood vector layer. Discussed adding a BAAI/bge-small-en-v1.5 semantic embedding layer (384-dim, separate ChromaDB collection) — decided against it; the structured mood/tone vectors are sufficient for the current phase. Cleaned up repo and pushed initial commit to git.

**State at close:**
- Scraping: all 201 episodes scraped ✓
- Enrichment: S2 enriched; remaining seasons still need `--all the_office --skip-existing`
- Mood vectors: S2 vectorized in ChromaDB; remaining seasons still need `--all the_office --skip-existing`

---

## 2026-05-17 — Embedding (mood & tone vectors)

**Summary:** Implemented the embedding layer: pure numpy vectorization from `mood_enriched.json` (4 mood + 13 tone dims → 17-dim L2-normalized `episode_vec`), ChromaDB persistence with cosine space, and a plain-Python CLI including `--similar-to` sanity checks.

**Changes:**
- Added `backend/embedding/episode_vectorizer.py` (math only — no chromadb import)
- Added `backend/embedding/chroma_store.py` (ChromaDB upsert/query only — no vector math)
- Added `backend/embedding/__main__.py` CLI (`--episode-id`, `--season`, `--all`, `--skip-existing`, `--similar-to`, `--top-k`)
- Updated `pyproject.toml` with `numpy>=1.26` and `chromadb>=0.5`
- Outputs `data/raw/{episode_id}/mood_vector.json` + `data/chroma/` collection `episode_mood_vectors`

**Sample `--similar-to` results (Season 2, 22 episodes indexed):**

*Halloween (S2E05)* — high sadness (0.76), low comfort (0.11), dominant `emotional` tone:
1. `the_office_s02_e19` Michael's Birthday (0.96) — emotional, moderate sadness
2. `the_office_s02_e12` The Injury (0.95) — emotional
3. `the_office_s02_e08` Performance Review (0.90)
4. `the_office_s02_e11` Booze Cruise (0.90)
5. `the_office_s02_e09` Email Surveillance (0.88)

*The Fire (S2E04)* — high humor (0.91), max energy (1.00), chaotic/bittersweet tones:
1. `the_office_s02_e16` Valentine's Day (0.96)
2. `the_office_s02_e13` The Secret (0.93)
3. `the_office_s02_e07` The Client (0.93)
4. `the_office_s02_e14` The Carpet (0.91)
5. `the_office_s02_e22` Casino Night (0.90)

Neighbours cluster by mood profile (emotional/sad vs chaotic/high-energy) rather than random season order — sanity check passed.

**Commands / verification:**
```bash
pip install -e .
python -m backend.embedding --episode-id the_office_s02_e05
python -c "import json, numpy as np; d=json.load(open('data/raw/the_office_s02_e05/mood_vector.json')); assert len(d['episode_vec'])==17; assert abs(np.linalg.norm(d['episode_vec'])-1)<1e-5"
python -m backend.enrichment --season the_office 2 --skip-existing   # prerequisite for full S2 index
python -m backend.embedding --season the_office 2
python -m backend.embedding --similar-to the_office_s02_e05 --top-k 5
```

**Edge cases / follow-ups:**
- Skipped enrichment episodes (`s07_e13`, `s09_e05`) are logged and skipped during vectorization.
- `MOOD_DIMENSIONS` / `TONE_DIMENSIONS` order is fixed — do not reorder after vectors are stored.
- Mood/tone weights (0.7/0.3) are a starting point; tune if `--similar-to` clusters feel off on other seasons.

---

## 2026-05-17 — Enrichment (mood tagging)

**Summary:** Implemented the enrichment layer: text formatting for DeBERTa NLI premises, zero-shot mood classification via `cross-encoder/nli-deberta-v3-base`, and a plain-Python CLI. Writes `mood_enriched.json` alongside existing `extracted_data.json` without modifying scrape output.

**Changes:**
- Added `backend/enrichment/text_formatter.py` (pure string processing, no ML imports)
- Added `backend/enrichment/mood_tagger.py` (4 scalar dimensions + 13 tone tags, skip records for no-content episodes)
- Added `backend/enrichment/__main__.py` CLI (`--episode-id`, `--season`, `--all`, `--show-text`, `--skip-existing`)
- Updated `pyproject.toml` with `torch>=2.0` and `transformers>=4.40`

**Sample mood scores (CPU, model loaded once per run):**
| Episode | humor | energy | comfort | sadness | tone tags |
|---|---|---|---|---|---|
| `the_office_s02_e01` (Dundies) | 0.95 | 0.64 | 0.28 | 0.21 | awkward, chaotic, cringe, lighthearted, silly |
| `the_office_s02_e04` (The Fire) | 0.91 | 1.00 | 0.02 | 0.42 | bittersweet, chaotic, emotional, lighthearted, romantic, tense |
| `the_office_s02_e22` (Casino Night) | 0.57 | 0.57 | 0.31 | 0.35 | emotional |

**Commands / verification:**
```bash
pip install -e .
python -m backend.enrichment --episode-id the_office_s02_e04 --show-text   # 1460 chars
python -m backend.enrichment --episode-id the_office_s02_e04
python -c "import json; d=json.load(open('data/raw/the_office_s02_e04/mood_enriched.json')); ..."
python -m backend.enrichment --episode-id the_office_s07_e13
python -c "assert json.load(open('data/raw/the_office_s07_e13/mood_enriched.json')).get('skipped')"
```

**Performance:** ~2.5–4.2 seconds per episode on CPU after model load (5 classifier calls: 4 binary dimensions + 1 multi-label tone pass). First run downloads ~400MB model to HuggingFace cache.

**Edge cases / follow-ups:**
- `the_office_s07_e13` and `the_office_s09_e05` have null synopsis/cold_open → `{"skipped": true, "reason": "no_narrative_content"}`.
- Tone threshold 0.4 may over-tag on some episodes (e.g. S2E04 got 6 tone tags); tune after a full-season run.
- `--all the_office` not run in this session (~10–25 min CPU for 201 episodes).

---

## 2026-05-17 — Scraper

**Summary:** Implemented the full scraping layer: series-agnostic Fandom wiki provider, dynamic season discovery, plain-Python pipeline CLI, and cleaned `extracted_data.json` output (no legacy fields, no DB/LangGraph).

**Changes:**
- Added `backend/scraping/` (`series_config`, `episode_ref`, `season_discoverer`, `http_client`, `extracted_data`, `runner`, `providers/fandom_wiki`)
- Added `backend/pipeline/__main__.py` CLI (`--episode-id`, `--season`, `--all-seasons`, `--skip-scrape`, `--json`)
- `SeasonDiscoverer` parses `table.episodelist` on fandom season pages (episode link in column 2, number in column 3; supports `1/2` combined notation)
- `FandomWikiProvider` replaces `DunderpediaWikiProvider`; `make_episode_id` lives in `episode_ref` (no DB import)
- `THE_OFFICE_COMBINED_OVERRIDES` for S4E01/E02 (`Fun_Run`); slash notation handles other combined rows without extra overrides

**Commands / verification:**
```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
python -m backend.pipeline --season the_office 2
ls data/raw/the_office_s02_e01/extracted_data.json
ls data/raw/the_office_s02_e22/extracted_data.json
python -c "import json; d=json.load(open('data/raw/the_office_s02_e04/extracted_data.json')); assert list(d.keys())==['episode_id','metadata','narrative']; assert list(d['narrative'].keys())==['cold_open','synopsis','cast']; assert 'social' not in d; print('output shape: OK')"
python -m backend.pipeline --season the_office 4
ls data/raw/the_office_s04_e01/extracted_data.json
ls data/raw/the_office_s04_e02/extracted_data.json
```

**Edge cases / follow-ups:**
- Season page uses `table.episodelist` with 5-column episode rows plus 1-column description rows (description rows skipped).
- Combined episodes on season pages use `1/2` notation (optional `‡` suffix); parser expands to `linked_episode_numbers`.
- Apostrophe titles (e.g. `Valentine's_Day`, `Dwight's_Speech`) work via URL-encoded hrefs from discovery.
- S4 has additional combined pairs (3/4, 5/6, 7/8, 18/19) handled via slash parsing without manual overrides; only S4E01/E02 has explicit `COMBINED_OVERRIDES` per spec.
- Some episodes may lack `cold_open` or `synopsis` sections (null in output per data spec).
- `--all-seasons` scrapes seasons 1–9 (~190+ HTTP requests); not run in this session.

---

## Entry template (copy for new sessions)

```markdown
## YYYY-MM-DD — <Role: Scraper / Storage / Rec Engine / Orchestrator>

**Summary:**

**Changes:**

**Commands / verification:**

**Edge cases / follow-ups:**

---
```
