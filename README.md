# Multi-Modal Evidence Review System

An 8-stage insurance damage claim verification pipeline that reads submitted photos, a short claim conversation, user history, and minimum evidence requirements — and produces structured predictions for each claim.

Built for the **HackerRank Orchestrate** hackathon (June 2026).

---

## Quick Start

### Prerequisites
- Python 3.10 or newer
- A free **Groq** API key — [console.groq.com](https://console.groq.com) (no credit card required)

### Installation

```bash
cd /path/to/hackerrank-orchestrate-june26

# Install dependencies
pip install -r code/requirements.txt

# Configure API key
cp code/.env.example code/.env
# Edit code/.env and set:  GROQ_API_KEY=your_key_here
```

### Run on test set

```bash
python3 code/main.py
# Reads:  dataset/claims.csv
# Writes: output.csv  (repo root)
```

### Run with Strategy C (observe-first)

```bash
python3 code/main.py c
# Uses VisionAnalyzerC — no claim context sent to vision model
```

### Run evaluation (Strategy A vs B vs C comparison)

```bash
python3 code/evaluation/main.py
# Reads:  dataset/sample_claims.csv  (labeled ground truth)
# Writes: code/evaluation/evaluation_report.md
#         code/evaluation/strategy_a_predictions.csv
#         code/evaluation/strategy_b_predictions.csv
#         code/evaluation/strategy_c_predictions.csv
```

### Quick options

```bash
python3 code/main.py --limit 5          # first 5 rows only (fast test)
python3 code/main.py --sample           # run on sample_claims.csv
python3 code/main.py c --limit 5        # Strategy C, first 5 rows
python3 code/evaluation/main.py --limit 3
python3 code/smoke_test.py              # no API key needed — runs 23 unit checks
```

---

## Architecture Overview

```
claims.csv
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1  │  Claim Extractor          │  AI  (text call)    │
│           │  claim_extractor.py        │  → StructuredClaim  │
├─────────────────────────────────────────────────────────────┤
│  Stage 2  │  Image Quality Checker    │  Deterministic      │
│           │  image_quality_checker.py  │  → ImageQualityResult│
├─────────────────────────────────────────────────────────────┤
│  Stage 3  │  Evidence Checker         │  Deterministic      │
│           │  evidence_checker.py       │  → EvidenceStandard │
├─────────────────────────────────────────────────────────────┤
│  Stage 4  │  Risk Scorer              │  Deterministic      │
│           │  risk_scorer.py            │  → UserRisk         │
├─────────────────────────────────────────────────────────────┤
│  Stage 5  │  Vision Analyzer          │  AI  (vision call)  │
│           │  vision_analyzer.py        │  → PerImageAnalysis │
│           │  vision_analyzer_c.py      │  (observe-first)    │
├─────────────────────────────────────────────────────────────┤
│  Stage 6  │  Image Aggregator         │  Deterministic      │
│           │  image_aggregator.py       │  → AggregatedVision │
├─────────────────────────────────────────────────────────────┤
│  Stage 7  │  Decision Engine          │  Deterministic      │
│           │  decision_engine.py        │  → ClaimDecision    │
├─────────────────────────────────────────────────────────────┤
│  Stage 8  │  Output Formatter         │  Deterministic      │
│           │  output_formatter.py       │  → OutputRow        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
output.csv
```

**The AI is called in exactly two places:**
1. **Stage 1** — one text-only call per claim (extracts what is being claimed)
2. **Stage 5** — one vision call per valid image (checks what is visible)

All other stages are pure Python with no external calls.

---

## Strategies

Three strategies are implemented and compared. Select at runtime via a positional argument to `main.py`.

| Strategy | Positional arg | Description |
|----------|---------------|-------------|
| **A** — Baseline | *(evaluation only)* | Single monolithic prompt with all context. One API call per claim. No quality pre-filtering, no evidence checks, no risk context. |
| **B** — Pipeline *(default)* | `b` or omit | Full 8-stage pipeline. Vision model receives the full claim context (anchored). Conservative `contradicted` rule requires all 5 conditions. |
| **C** — Observe-First | `c` | Full 8-stage pipeline. Vision model receives **no** claim context — it freely describes what it observes. Damage type matching and claim assessment are performed by deterministic semantic matching (`utils/semantic_matcher.py`) after the vision call, eliminating anchoring bias. |

### Evaluation Results (sample_claims.csv, 20 labeled rows)

| Metric | Strategy A | Strategy B | Strategy C |
|--------|-----------|-----------|-----------|
| claim_status_accuracy | 0.5500 | 0.4500 | 0.3500 |
| issue_type_accuracy | 0.4000 | 0.4000 | 0.5000 |
| object_part_accuracy | 0.5000 | 0.7000 | 0.5500 |
| severity_accuracy | 0.4000 | 0.3500 | 0.3000 |
| evidence_standard_met_accuracy | 0.7000 | 0.8500 | 0.8500 |
| valid_image_accuracy | 0.7500 | 0.9000 | 0.9000 |
| risk_flags_jaccard | 0.4454 | 0.2983 | 0.3700 |
| **Overall Score** | **0.5098** | **0.5174** | **0.4810** |
| Runtime | 94.7s | 217.6s | 289.4s |

**Strategy B selected for final `output.csv`** (highest overall score).

---

## Design Decisions

- **Two-call AI architecture.** Separating claim extraction (text) from image analysis (vision) keeps each prompt focused and allows independent retry and fallback strategies per call type.

- **Deterministic decision layer.** All business logic — evidence standard checks, risk scoring, majority voting, priority rule tree — is pure Python. The LLM never makes the final verdict. This makes the system auditable and reproducible.

- **Conservative `contradicted` rule (Strategy B).** The decision engine only marks a claim `contradicted` when all five conditions are met: high confidence, correct object detected, claimed part is visible, no damage found, and consensus verdict is `contradicted`. This avoids false negatives from uncertain vision responses.

- **Observe-First vision (Strategy C).** `VisionAnalyzerC` sends no claim context to the vision model. The free-form observation is then matched against a `DAMAGE_FAMILIES` vocabulary using `utils/semantic_matcher.py`, removing the anchoring bias where the model searches for the *absence* of a specific named damage type rather than describing what is actually present.

- **Evidence standard as a hard gate.** Images are quality-checked before any API call. Claims with no valid images skip the vision stage entirely, saving quota.

- **Graceful degradation.** Every stage catches exceptions and returns a typed fallback. If the Groq API is unavailable, keyword-based claim extraction + conservative defaults still produce a valid output row for every input.

- **Single data access point.** `DataStore` loads all four CSVs once at startup. No other module reads CSVs directly. This avoids repeated I/O and makes the data layer easy to test.

- **Typed contracts between stages.** Every stage exchanges typed dataclasses (defined in `models/schemas.py`), not raw dicts. Field names and types are enforced at the boundary before they reach the output formatter.

---

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | User who submitted the claim |
| `image_paths` | string | Semicolon-separated paths (passthrough from input) |
| `user_claim` | string | Claim transcript (passthrough from input) |
| `claim_object` | string | `car`, `laptop`, or `package` |
| `evidence_standard_met` | `true`/`false` | Whether enough valid images were submitted |
| `evidence_standard_met_reason` | string | Short reason for the evidence decision |
| `risk_flags` | string | Semicolon-separated risk flags, or `none` |
| `issue_type` | string | Visible damage type (e.g. `dent`, `crack`) |
| `object_part` | string | Relevant part (e.g. `front_bumper`, `screen`) |
| `claim_status` | string | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | string | Image-grounded explanation |
| `supporting_image_ids` | string | Semicolon-separated IDs of supporting images, or `none` |
| `valid_image` | `true`/`false` | Whether the image set is usable for automated review |
| `severity` | string | `none`, `low`, `medium`, `high`, or `unknown` |

### Allowed enum values

| Field | Values |
|-------|--------|
| `claim_status` | `supported`, `contradicted`, `not_enough_information` |
| `issue_type` | `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`, `missing_part`, `torn_packaging`, `crushed_packaging`, `water_damage`, `stain`, `none`, `unknown` |
| `severity` | `none`, `low`, `medium`, `high`, `unknown` |
| `risk_flags` | `blurry_image`, `cropped_or_obstructed`, `low_light_or_glare`, `wrong_angle`, `wrong_object`, `wrong_object_part`, `damage_not_visible`, `claim_mismatch`, `possible_manipulation`, `non_original_image`, `text_instruction_present`, `user_history_risk`, `manual_review_required`, `none` |

---

## API Backend & Rate Limits

The system uses **Groq** as the AI backend (free tier, no credit card required).

| Parameter | Value |
|-----------|-------|
| Text model | `llama-3.1-8b-instant` |
| Vision model | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Tier | Free (Groq) |
| Rate limit enforced | 25 calls/minute |
| Min call interval | 2.5 s (enforced in `GeminiClient`) |
| Retry strategy | Exponential backoff on 429: 2 s → 4 s → up to 30 s |
| Fallback | Keyword extraction if text call fails; `not_enough_information` if vision call fails |
| Estimated cost | **$0.00** (within free tier limits) |

At ~2 API calls per claim (1 text + ~1 vision), the 44-row test set requires ~88 calls and takes roughly 7–10 minutes on Strategy B.

---

## File Layout

```
.
├── AGENTS.md
├── problem_statement.md
├── README.md
├── output.csv                         ← generated by:  python3 code/main.py
├── dataset/
│   ├── claims.csv                     ← 44 test rows (inputs only)
│   ├── sample_claims.csv              ← 20 labeled rows (inputs + ground truth)
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/
│       └── test/
└── code/
    ├── main.py                        ← pipeline entry point (strategy b/c)
    ├── smoke_test.py                  ← 23 unit checks, no API key needed
    ├── config.py                      ← all constants (single source of truth)
    ├── requirements.txt
    ├── .env.example                   ← copy to .env and add GROQ_API_KEY
    ├── models/
    │   ├── schemas.py                 ← all typed dataclass contracts
    │   ├── gemini_client.py           ← Groq API boundary (rate limiting, retry)
    │   └── prompts.py                 ← all LLM prompt templates (incl. FREE_OBSERVATION)
    ├── pipeline/
    │   ├── claim_extractor.py         ← Stage 1 (AI — text call)
    │   ├── image_quality_checker.py   ← Stage 2 (deterministic)
    │   ├── evidence_checker.py        ← Stage 3 (deterministic)
    │   ├── risk_scorer.py             ← Stage 4 (deterministic)
    │   ├── vision_analyzer.py         ← Stage 5 (AI — anchored, Strategy B)
    │   ├── vision_analyzer_c.py       ← Stage 5 (AI — observe-first, Strategy C)
    │   ├── image_aggregator.py        ← Stage 6 (deterministic)
    │   ├── decision_engine.py         ← Stage 7 (deterministic, conservative contradicted rule)
    │   └── output_formatter.py        ← Stage 8 (deterministic)
    ├── utils/
    │   ├── csv_loader.py              ← DataStore (single CSV access point)
    │   ├── image_utils.py             ← image loading helpers
    │   ├── semantic_matcher.py        ← DAMAGE_FAMILIES vocab + match_damage_type (Strategy C)
    │   └── validators.py              ← enum constants + validation functions
    └── evaluation/
        ├── main.py                    ← evaluation entry point (3-strategy runner)
        ├── metrics.py                 ← accuracy + Jaccard metrics
        ├── compare_strategies.py      ← Strategy A / B / C runner
        ├── strategy_a_predictions.csv ← generated predictions
        ├── strategy_b_predictions.csv ← generated predictions
        ├── strategy_c_predictions.csv ← generated predictions
        └── evaluation_report.md       ← generated operational analysis
```
