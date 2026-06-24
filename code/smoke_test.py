"""Smoke test — no API key, no internet, no image reads required.

Verifies that every module can be imported and core components
initialized correctly.  Run this before any live pipeline run to
catch environment or dependency issues early.

Usage:
    python3 code/smoke_test.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root and code/ are on the path
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def check(label: str, fn) -> bool:
    """Run *fn*, record PASS/FAIL, return bool."""
    try:
        detail = fn()
        results.append((label, True, str(detail) if detail else ""))
        return True
    except Exception as exc:
        results.append((label, False, str(exc)))
        return False


# ── Check 1-8: all module imports ─────────────────────────────────────────

check("import config", lambda: __import__("config"))
check("import models.schemas", lambda: __import__("models.schemas"))
check("import models.gemini_client", lambda: __import__("models.gemini_client"))
check("import models.prompts", lambda: __import__("models.prompts"))
check("import utils.csv_loader", lambda: __import__("utils.csv_loader"))
check("import utils.image_utils", lambda: __import__("utils.image_utils"))
check("import utils.validators", lambda: __import__("utils.validators"))
check("import pipeline.claim_extractor", lambda: __import__("pipeline.claim_extractor"))
check("import pipeline.image_quality_checker", lambda: __import__("pipeline.image_quality_checker"))
check("import pipeline.evidence_checker", lambda: __import__("pipeline.evidence_checker"))
check("import pipeline.risk_scorer", lambda: __import__("pipeline.risk_scorer"))
check("import pipeline.vision_analyzer", lambda: __import__("pipeline.vision_analyzer"))
check("import pipeline.image_aggregator", lambda: __import__("pipeline.image_aggregator"))
check("import pipeline.decision_engine", lambda: __import__("pipeline.decision_engine"))
check("import pipeline.output_formatter", lambda: __import__("pipeline.output_formatter"))
check("import evaluation.metrics", lambda: __import__("evaluation.metrics"))
check("import evaluation.compare_strategies", lambda: __import__("evaluation.compare_strategies"))

# ── Check: DataStore loads all CSVs ───────────────────────────────────────

def _datastore_check():
    from utils.csv_loader import DataStore
    ds = DataStore()
    assert len(ds.claims) > 0, "claims.csv is empty"
    assert len(ds.sample_claims) > 0, "sample_claims.csv is empty"
    assert len(ds._user_history_df) > 0, "user_history.csv is empty"
    assert len(ds._evidence_requirements_df) > 0, "evidence_requirements.csv is empty"
    return (
        f"claims={len(ds.claims)}, sample={len(ds.sample_claims)}, "
        f"history={len(ds._user_history_df)}, requirements={len(ds._evidence_requirements_df)}"
    )

check("DataStore loads all CSVs", _datastore_check)

# ── Check: ImageQualityChecker instantiates ────────────────────────────────

check(
    "ImageQualityChecker instantiates",
    lambda: __import__("pipeline.image_quality_checker", fromlist=["ImageQualityChecker"]).ImageQualityChecker(),
)

# ── Check: OutputRow.get_columns() — 14 columns, correct order ────────────

EXPECTED_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]

def _output_row_check():
    from models.schemas import OutputRow
    cols = OutputRow.get_columns()
    assert len(cols) == 14, f"Expected 14 columns, got {len(cols)}"
    assert cols == EXPECTED_COLUMNS, f"Column order mismatch: {cols}"
    return f"{len(cols)} columns in correct order"

check("OutputRow.get_columns() — 14 cols, correct order", _output_row_check)

# ── Check: validate_enum works ────────────────────────────────────────────

def _validate_enum_check():
    from utils.validators import VALID_ISSUE_TYPES, validate_enum
    result = validate_enum("unknown", VALID_ISSUE_TYPES)
    assert result == "unknown", f"Expected 'unknown', got '{result}'"
    invalid = validate_enum("INVALID_VALUE", VALID_ISSUE_TYPES)
    assert invalid == "unknown", f"Expected fallback 'unknown', got '{invalid}'"
    return "validate_enum('unknown') == 'unknown', invalid → 'unknown'"

check("validate_enum works correctly", _validate_enum_check)

# ── Check: output.csv exists and has correct headers (optional) ───────────

def _output_csv_check():
    output_path = REPO_ROOT / "output.csv"
    if not output_path.exists():
        return "SKIP — output.csv not yet generated (run python3 code/main.py)"
    import csv
    with open(output_path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
    assert headers == EXPECTED_COLUMNS, f"Header mismatch: {headers}"
    rows = list(reader)
    assert len(rows) > 0, "output.csv has no data rows"
    for row in rows:
        status = row[EXPECTED_COLUMNS.index("claim_status")].strip().lower()
        assert status in {"supported", "contradicted", "not_enough_information"}, (
            f"Invalid claim_status: {status}"
        )
    return f"{len(rows)} rows, all claim_status values valid"

check("output.csv exists and has valid headers/claim_status", _output_csv_check)

# ── Check: evaluation_report.md exists and is non-empty (optional) ────────

def _report_check():
    report_path = CODE_DIR / "evaluation" / "evaluation_report.md"
    if not report_path.exists():
        return "SKIP — evaluation_report.md not yet generated (run python3 code/evaluation/main.py)"
    content = report_path.read_text(encoding="utf-8").strip()
    assert len(content) > 100, "evaluation_report.md is suspiciously short"
    return f"{len(content)} characters"

check("evaluation_report.md exists and is non-empty", _report_check)

# ── Print results ──────────────────────────────────────────────────────────

print()
print("=== SMOKE TEST RESULTS ===")
print()
passed = 0
for label, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if ok:
        passed += 1

total = len(results)
print()
print(f"{passed}/{total} checks passed")
print()

if passed == total:
    print("✅  All checks passed.")
    sys.exit(0)
else:
    failed = total - passed
    print(f"❌  {failed} check(s) failed. See above for details.")
    sys.exit(1)
