# Scraping Migration Plan

**Owner:** Orchestrator  
**Last updated:** 2026-05-17  
**Scope:** Scraping only — no DB, no Alembic, no enrichment, no rec engine  
**Source:** `sitcom-ml-project/backend/scraping/`  
**Target:** `sitcom-episode-recommender/backend/scraping/`

**No database in this phase.** The scraper's only output is JSON files written to `data/raw/{episode_id}/`. No SQLite, no Alembic, no ORM. Those belong to a future DB phase.  
**Runtime dependencies:** `httpx`, `beautifulsoup4`, `lxml` only.

---

## 1. What We're Starting With

The old scraper works but has three limitations to fix:

| Problem | Where | Impact |
|---|---|---|
| Registry is hardcoded for S2 only | `episode_ref.py` | Can't scrape S1, S3–S9 without manually registering every episode |
| Provider is named/wired for The Office only | `wiki_fandom.py` — `DunderpediaWikiProvider` | Can't add a second show without structural changes |
| Pipeline uses LangGraph | `pipeline/graph.py` | Banned by CLAUDE.md |
| `make_episode_id` imported from DB layer | `episode_ref.py` | Scraping shouldn't depend on DB |
| Legacy fields in output JSON | `extracted_data.py` | `plot`, `trivia`, `cast_text_debug`, `lede_text` were transitional; they're noise now |

---

## 2. Target File Structure

```
backend/
└── scraping/
    ├── __init__.py
    ├── series_config.py        # NEW — SeriesConfig dataclass
    ├── episode_ref.py          # ADAPTED — decouple from DB, remove hardcoded registry
    ├── season_discoverer.py    # NEW — auto-discover episodes from fandom wiki season page
    ├── http_client.py          # PORT UNCHANGED
    ├── extracted_data.py       # ADAPTED — drop legacy fields, make series-agnostic
    ├── runner.py               # ADAPTED — use generic provider, decouple from DB
    └── providers/
        ├── __init__.py
        └── fandom_wiki.py      # ADAPTED — generalized from wiki_fandom.py
```

```
backend/
└── pipeline/
    ├── __init__.py
    └── __main__.py             # REWRITTEN — no LangGraph, simple Python
```

```
data/
└── raw/
    └── {episode_id}/
        ├── wiki_parse_api.json
        ├── wiki_article_from_api.html
        ├── wiki_extracted.json
        └── extracted_data.json     # The output we care about
```

---

## 3. Design Decisions

### 3.1 `make_episode_id` moves out of the DB layer

**Old:** `episode_ref.py` imports `make_episode_id` from `backend.db.repositories` — scraping depends on DB.  
**New:** `make_episode_id` lives in `episode_ref.py` itself. It's a pure string function (`f"{slug}_s{s:02d}_e{e:02d}"`), not DB logic.

---

### 3.2 `SeriesConfig` — the extension point for new shows

A new `series_config.py` defines:

```
SeriesConfig:
    series_slug: str          # e.g. "the_office"
    series_title: str         # e.g. "The Office"
    wiki_origin: str          # e.g. "https://theoffice.fandom.com"
    season_page_template: str # e.g. "Season_{n}" (used for discovery)
```

Adding a new sitcom = adding a `SeriesConfig`. Nothing else changes.

Pre-defined configs ship for The Office. Others added as needed.

---

### 3.3 Season discovery — the key new capability

**Problem:** Manually registering 200 episodes across 9 seasons is not viable.  
**Solution:** `SeasonDiscoverer` fetches the fandom wiki season page and parses the episode list.

How it works:
1. Build season page URL from `series_config.wiki_origin` + `series_config.season_page_template` (e.g. `https://theoffice.fandom.com/wiki/Season_2`)
2. Fetch the page HTML via the MediaWiki parse API (same approach as episode scraping — avoids Cloudflare)
3. Parse the episode table: extract episode number and the wiki page title (link href) for each row
4. Return a `list[EpisodeRef]` — one per discovered episode

**Combined episode overrides:** Some episodes share one wiki page (e.g. S4E01+E02 on "Fun_Run"). These can't be auto-detected from the season page (which shows them as separate rows). A static override dict per series handles this:

```
THE_OFFICE_COMBINED_EPISODES = {
    ("the_office", 4, 1): EpisodeRef(..., linked_episode_numbers=(2,)),
    ("the_office", 4, 2): EpisodeRef(..., linked_episode_numbers=(1,)),
    ...
}
```

Discovery runs first; overrides are applied after. Only combined episodes need manual registration — not every episode.

---

### 3.4 `FandomWikiProvider` — replaces `DunderpediaWikiProvider`

**Old:** `DunderpediaWikiProvider` is hardcoded for The Office.  
**New:** `FandomWikiProvider(series_config: SeriesConfig)` — works for any fandom wiki. The MediaWiki parse API is identical across all fandom wikis; only `wiki_origin` differs.

The underlying fetch/parse logic (`_mw_api_parse_url`, `_parse_sections_from_html`, etc.) is untouched.

---

### 3.5 `extracted_data.py` — drop legacy fields

Remove from the output JSON:
- `narrative.plot` (was cold_open + synopsis concatenated — redundant)
- `narrative.plot_sections_used`
- `narrative.trivia`
- `narrative.cast_text_debug`
- `narrative.lede_text`
- `social` (not relevant until Reddit is in scope)

The `build_extracted_data` function takes `SeriesConfig` instead of hardcoded `series_title`.

---

### 3.6 Pipeline — no LangGraph

**Old:** `pipeline/graph.py` uses LangGraph `StateGraph` — banned by CLAUDE.md.  
**New:** `pipeline/__main__.py` is a plain Python CLI. The scrape step is just a function call. No graph, no state machine.

```
python -m backend.pipeline --episode-id the_office_s03_e01
python -m backend.pipeline --season the_office 3
python -m backend.pipeline --all-seasons the_office
```

---

## 4. Output Contract

`extracted_data.json` shape in the new project (cleaned — no legacy fields):

```json
{
  "episode_id": "the_office_s03_e01",
  "metadata": {
    "episode_id": "the_office_s03_e01",
    "series_title": "The Office",
    "series_slug": "the_office",
    "episode_title": "Gay Witch Hunt",
    "season_number": 3,
    "episode_number": 1,
    "air_date": "2006-09-21",
    "provenance": {
      "source": "fandom_wiki",
      "wiki_page_title": "Gay_Witch_Hunt",
      "wiki_origin": "https://theoffice.fandom.com",
      "wiki_api_url": "https://theoffice.fandom.com/api.php?..."
    }
  },
  "narrative": {
    "cold_open": "...",
    "synopsis": "...",
    "cast": {
      "main": ["Steve Carell as Michael Scott", "..."],
      "supporting": ["..."],
      "recurring": ["..."],
      "other": []
    }
  }
}
```

---

## 5. What Cursor Implements (Ordered)

### Step 1 — Project scaffolding
- Create `backend/scraping/` directory with `__init__.py`
- Create `backend/pipeline/` with `__init__.py`
- Copy `requirements.txt`/`pyproject.toml` from old project, strip LangGraph dependency

### Step 2 — Port `http_client.py` unchanged
- Exact copy. No changes.

### Step 3 — Write `series_config.py`
- `SeriesConfig` dataclass with the four fields above
- `THE_OFFICE` constant (slug, title, wiki_origin, season_page_template)

### Step 4 — Adapt `episode_ref.py`
- Move `make_episode_id` into this file as a standalone function (remove DB import)
- Keep `EpisodeRef` dataclass exactly as-is
- Remove the hardcoded S2 registry (`_THE_OFFICE_S02_PAGE_TITLES`, `_THE_OFFICE_S02_REFS`, `_EPISODE_REGISTRY`)
- Add `THE_OFFICE_COMBINED_EPISODES` dict (static overrides for known combined wiki pages — start with S4E01/E02)
- Remove `ref_for_episode_id` (replaced by discovery)

### Step 5 — Write `season_discoverer.py`
- `SeasonDiscoverer(series_config: SeriesConfig, data_root: Path)`
- `discover(season_number: int) -> list[EpisodeRef]`
  - Fetches season page via MediaWiki parse API
  - Parses episode table: extracts episode number + wiki page title per row
  - Applies combined episode overrides from `THE_OFFICE_COMBINED_EPISODES`
  - Returns list of `EpisodeRef`

### Step 6 — Adapt `providers/fandom_wiki.py` (from `wiki_fandom.py`)
- Rename `DunderpediaWikiProvider` → `FandomWikiProvider`
- Constructor: `FandomWikiProvider(series_config: SeriesConfig, data_root: Path)`
- Rename `fetch_dunderpedia_article` → `fetch_fandom_article`
- Remove all Dunderpedia-specific naming; `wiki_origin` comes from `series_config`
- All parsing logic (`_mw_api_parse_url`, `_parse_sections_from_html`, etc.) untouched

### Step 7 — Adapt `extracted_data.py`
- Remove all legacy fields from `build_extracted_data` output
- Replace `series_title: str = "The Office"` parameter with `series_config: SeriesConfig`
- Set `provenance.source = "fandom_wiki"` (generic, not dunderpedia-specific)
- Keep all cast parsing logic unchanged (`_normalize_cast_lines`, `_parse_cast_buckets_from_html`, etc.)

### Step 8 — Adapt `runner.py`
- Replace `DunderpediaWikiProvider` with `FandomWikiProvider`
- Remove `from backend.db.repositories import make_episode_id` — use the one in `episode_ref.py`
- Signature: `run_scrape_for_ref(ref: EpisodeRef, series_config: SeriesConfig, data_root: Path) -> dict`
- "Scrape once, write N" logic is unchanged

### Step 9 — Write `pipeline/__main__.py`
- No LangGraph — plain Python
- CLI args:
  - `--episode-id <id>` — scrape one episode (requires discovery for that episode's ref)
  - `--season <series_slug> <season_number>` — discover + scrape a full season
  - `--all-seasons <series_slug>` — discover + scrape all seasons (1–9 for The Office)
  - `--skip-scrape` — dry-run, print what would be scraped
  - `--json` — print result summary as JSON
- Calls `SeasonDiscoverer.discover()` then `run_scrape_for_ref()` for each ref

---

## 6. Open Questions

1. **Season page format** — The season page title template `Season_{n}` is an assumption. Needs verification on theoffice.fandom.com before Step 5 is implemented. Cursor should confirm the actual URL before writing the parser.

2. **Episode table structure** — Fandom season pages use different table formats per show. Step 5 should handle the case where the episode table format isn't recognized (log a warning + return empty list rather than crash). Cursor must check the actual HTML structure of `Season_2` page before writing the parser.

3. **All combined episodes** — Only S4E01/E02 (`Fun_Run`) is registered in the old project. Are there other combined-episode wiki pages across S1–S9? These need to be identified and added to the override dict. A scrape run across all seasons will surface them (discovery will produce refs with mismatched episode counts vs expected).

4. **Rate limiting** — The existing `jitter_sleep(0.35, 1.2)` in `http_client.py` should be sufficient for fandom, but a full-season scrape (20+ requests) should be validated with a real run before bulk-scraping all seasons.
