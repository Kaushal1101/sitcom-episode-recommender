# Confidence Calculator Plan

## Architecture Position

The confidence calculator is a sub-component of the **Recommendation Engine** (not the CSM).

```
Recommendation Engine
├── retrieve        (retriever.py)
├── rerank          (reranker.py)
├── explain         (explanation_builder.py)
└── confidence      (confidence.py)  ← lives here

Conversation Strategy Module (CSM)
└── calls compute_confidence()
    reads ConfidenceResult
    decides: ask question or recommend
```

The CSM consumes `ConfidenceResult` but never reaches into the calculator's internals.
`ConfidenceResult` is the stable contract between the two sides — see
`docs/schemas/confidence-result-schema.json.txt`.

Changing the confidence formula (weights, signals, normalization) only requires editing
`confidence.py`. The CSM only needs to change if the `ConfidenceResult` schema changes.

---

## Goal

Implement a deterministic confidence calculator inside the recommendation engine.

The confidence score represents:

> "How confident is the system that it can make a good recommendation without asking another question?"

The confidence score is normalized to:
```text
[0.0, 1.0]
```

The confidence calculator should NOT:
- perform question selection
- decide whether to ask or recommend
- update vectors
- manage exploration/exploitation

Those responsibilities belong to the Conversation Strategy Module (CSM).

The confidence calculator ONLY computes confidence signals.

---

# Confidence Inputs

The calculator depends on three signals:

## 1. Top Score
How strong is the best recommendation candidate?

Reflects the reranker's `final_score` for the top-ranked episode.

```python
top_score = 0.91
```

Higher = more confidence.

---

## 2. Score Gap
How clearly does the best episode stand out from the rest?

```python
score_gap = top_score - mean(next_n_scores)
n = 3
```

### High confidence example
```text
Episode A = 0.92
Episode B = 0.61
Episode C = 0.58
Episode D = 0.55
```
```python
score_gap = 0.92 - mean([0.61, 0.58, 0.55]) = 0.34
```

### Low confidence example
```text
Episode A = 0.81
Episode B = 0.79
Episode C = 0.77
Episode D = 0.75
```
```python
score_gap = 0.81 - mean([0.79, 0.77, 0.75]) = 0.04
```

Small gaps indicate ambiguity; large gaps indicate a clear dominant candidate.

> **Calibration note:** With only The Office in the corpus, real reranker gaps are typically
> 0.03–0.10. `MAX_EXPECTED_GAP = 0.15` is calibrated for this. When more shows with different
> mood profiles are added (e.g. Brooklyn Nine-Nine, Schitt's Creek), gaps can reach 0.30+.
> Retune `MAX_EXPECTED_GAP` empirically after expanding the corpus.

---

## 3. Vector Coverage
How much has the user expressed their preferences so far?

Tracked as the number of questions the user has answered in the session.

```python
normalized_coverage = min(questions_answered / COVERAGE_SATURATION_POINT, 1.0)
```

```python
COVERAGE_SATURATION_POINT = 5
```

This means:
- 0 questions answered → coverage = 0.0
- 3 questions answered → coverage = 0.6
- 5+ questions answered → coverage = 1.0

This is intentionally simple. It acknowledges that early in a conversation the system
cannot be confident regardless of how the scores look — it hasn't heard enough from the user.

> **Future extension:** Replace raw question count with a richer signal: weighted by question
> type (attribute questions carry more signal than item Y/N questions), or derived from
> user vector entropy. Not needed for MVP.

---

# Confidence Formula

```python
confidence = (
    TOP_SCORE_WEIGHT * top_score
    + GAP_WEIGHT * normalized_gap
    + COVERAGE_WEIGHT * normalized_coverage
)
```

## Default weights

```python
TOP_SCORE_WEIGHT  = 0.5
GAP_WEIGHT        = 0.3
COVERAGE_WEIGHT   = 0.2
```

## Gap normalization

```python
normalized_gap = min(max(score_gap / MAX_EXPECTED_GAP, 0.0), 1.0)
MAX_EXPECTED_GAP = 0.15
```

Both `normalized_gap` and `normalized_coverage` are clipped to `[0.0, 1.0]`.
`top_score` is already in `[0.0, 1.0]` from the reranker.

The final `confidence` is therefore guaranteed to be in `[0.0, 1.0]`.

---

# Worked Example

## Inputs
```python
scores            = [0.86, 0.81, 0.79, 0.78]
questions_answered = 3
```

## Gap
```python
score_gap = 0.86 - mean([0.81, 0.79, 0.78])
          = 0.86 - 0.793
          = 0.067
```

## Normalize
```python
normalized_gap      = min(0.067 / 0.15, 1.0) = 0.447
normalized_coverage = min(3 / 5, 1.0)        = 0.60
```

## Final confidence
```python
confidence = (0.5 * 0.86) + (0.3 * 0.447) + (0.2 * 0.60)
           = 0.430 + 0.134 + 0.120
           = 0.684
```

Under the CSM thresholds this lands in the "item comparison question" band (0.6–0.7),
which is correct — the engine has a reasonable leader but a tight cluster behind it.

---

# Edge Cases

| Situation | Behaviour |
|---|---|
| No candidates | Return confidence = 0.0, all signals = 0.0 |
| Single candidate | score_gap = 0.0 (no peers to compare); formula still runs |
| Fewer than n+1 candidates | Compute gap over however many candidates exist |
| questions_answered = 0 | coverage = 0.0; confidence is driven entirely by score signals |

---

# Output Contract

```python
@dataclass
class TopCandidate:
    episode_id: str
    episode_title: str
    series_slug: str
    season_number: int
    episode_number: int
    score: float


@dataclass
class ConfidenceResult:
    confidence: float            # final score [0.0, 1.0]
    top_score: float             # best candidate's reranker score
    score_gap: float             # raw gap (before normalization)
    normalized_gap: float        # gap after normalization [0.0, 1.0]
    normalized_coverage: float   # coverage after normalization [0.0, 1.0]
    questions_answered: int      # raw input (for debugging / logging)
    top_candidates: list[TopCandidate]  # top N candidates passed in
```

The CSM uses:
- `confidence` to select the next interaction branch
- `top_candidates` for item-comparison questions and final recommendation
- The breakdown fields (`top_score`, `score_gap`, etc.) for logging and debugging

---

# CSM Threshold Alignment

These thresholds are defined in `docs/architecture/conversation-strategy.txt` and must
stay in sync with the CSM implementation. They are NOT enforced by this module.

```text
confidence < 0.4   → broad attribute question (Multiple Choice)
confidence < 0.6   → Y/N attribute refinement
confidence < 0.7   → item comparison question (Multiple Choice)
confidence < 0.85  → Y/N item comparison
confidence ≥ 0.9   → recommend episode
```

---

# File Location

```
backend/recommender/confidence.py
```

## Responsibilities
- compute `ConfidenceResult` from a ranked candidate list + questions_answered count
- normalize gap and coverage signals
- expose a single public function

## Should NOT
- perform retrieval or reranking
- manage session state
- select questions

---

# Public API

```python
def compute_confidence(
    ranked: list[RankedCandidate],
    questions_answered: int,
    n_gap_peers: int = 3,
    max_expected_gap: float = MAX_EXPECTED_GAP,
    coverage_saturation: int = COVERAGE_SATURATION_POINT,
    top_score_weight: float = TOP_SCORE_WEIGHT,
    gap_weight: float = GAP_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> ConfidenceResult:
    ...
```

All weights and calibration constants are keyword arguments with defaults so they
can be overridden in tests without monkey-patching module globals.

---

# MVP Constraints

DO NOT:
- implement Bayesian confidence
- implement probabilistic uncertainty estimation
- implement RL-based confidence
- implement learned confidence models

Use:
- deterministic heuristics only

---

# Future Extensions

- Replace `questions_answered` count with a weighted coverage signal (question type matters)
- Entropy-based candidate spread as an alternative to score gap
- Confidence calibration based on empirical recommendation acceptance rates
- Retune `MAX_EXPECTED_GAP` after adding multi-show corpus
- Bayesian or RL confidence (post-MVP only)
