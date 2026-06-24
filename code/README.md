# Code Directory

> **See the main [`README.md`](../README.md) at the repo root for full documentation** —
> quick start, architecture, strategies, output schema, evaluation, and submission instructions.

## Entry Points

| Command | What it does |
|---------|-------------|
| `python3 code/main.py` | Run pipeline on `dataset/claims.csv` → `output.csv` |
| `python3 code/main.py c` | Same, using Strategy C (observe-first) |
| `python3 code/main.py --limit 5` | First 5 rows only (fast test) |
| `python3 code/main.py --sample` | Run on `dataset/sample_claims.csv` |
| `python3 code/evaluation/main.py` | 3-strategy evaluation on sample set |
| `python3 code/smoke_test.py` | 23 unit checks, no API key needed |

## Setup

```bash
pip install -r code/requirements.txt
cp code/.env.example code/.env
# Set GROQ_API_KEY in code/.env
```
