# Caveats & Debugging Notes

Small observations, gotchas, and non-obvious behaviour worth remembering.
Add entries as they surface — these are things not obvious from reading the code.

---

## Environment

### Always run pytest with the project venv
```bash
source .venv/bin/activate && python -m pytest ...
```
The import chain `confidence → reranker → retriever → vector_indexer → chromadb` means
running pytest with the system Python will fail with `ModuleNotFoundError: No module named 'chromadb'`
even though the confidence calculator itself has no direct chromadb dependency.

---

## Data

### Two episodes permanently excluded from ChromaDB
`the_office_s07_e13` and `the_office_s09_e05` have `skipped: true` in their
`mood_enriched.json` (no synopsis/cold_open content). Mood columns are NULL in SQLite.
They are silently skipped during vector indexing and will never appear as candidates.
This is expected — not a bug.

### Stress Relief two-parter appears as near-duplicate candidates
`the_office_s05_e14` and `the_office_s05_e15` share identical mood scores (they were
enriched as a combined episode). They will always appear together near the top of any
ranking that favours their mood profile. Mitigation: pass the second one via `excluded_ids`
once the first has been recommended.

### Heartwarming/wholesome queries return weaker results
The DeBERTa enrichment tagger scores few Office episodes highly on warmth/comfort labels.
This is a thin-signal problem in the enrichment data, not an engine bug. Will improve when
more shows or a retuned tone threshold are added.

---

## Calibration

### `MAX_EXPECTED_GAP = 0.15` is calibrated for The Office only
With a single-show corpus, reranker score gaps are typically 0.03–0.10. When shows with
distinct mood profiles are added (e.g. high-energy comedy vs. slow dramatic comedy),
real gaps can reach 0.30+. Retune `MAX_EXPECTED_GAP` in `confidence.py` empirically
after expanding the corpus.

### `MOOD_DIMENSIONS` / `TONE_DIMENSIONS` order is fixed
Do not reorder these lists in `episode_feature_builder.py` after vectors have been stored
in ChromaDB. The 17-dim vectors are positional — reordering silently corrupts all similarity
results. If a reorder is ever needed, wipe and rebuild the Chroma index.
