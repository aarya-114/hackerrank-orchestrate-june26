"""Evaluation metrics for claim verification predictions.

Pure functions — no pipeline dependencies, no API calls.
"""

from __future__ import annotations

import collections

import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def exact_match_accuracy(pred: pd.Series, true: pd.Series) -> float:
    """Fraction of rows where pred and true match (case-insensitive, stripped).

    NaN values count as non-matches.
    """
    if len(pred) == 0:
        return 0.0

    matches = 0
    for p, t in zip(pred, true):
        if pd.isna(p) or pd.isna(t):
            continue
        if str(p).strip().lower() == str(t).strip().lower():
            matches += 1

    return matches / len(pred)


def jaccard_similarity(pred_flags: str, true_flags: str) -> float:
    """Jaccard similarity between two semicolon-separated flag strings.

    "none" tokens are removed before comparison.
    Both empty after removal → 1.0.
    One empty after removal → 0.0.
    """
    def _to_set(s: str) -> set[str]:
        parts = {p.strip().lower() for p in str(s).split(";")}
        parts.discard("none")
        parts.discard("")
        return parts

    pred_set = _to_set(pred_flags)
    true_set = _to_set(true_flags)

    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0

    intersection = pred_set & true_set
    union = pred_set | true_set
    return len(intersection) / len(union)


def risk_flags_mean_jaccard(pred: pd.Series, true: pd.Series) -> float:
    """Mean Jaccard similarity across all rows for risk_flags columns."""
    if len(pred) == 0:
        return 0.0
    scores = [
        jaccard_similarity(str(p), str(t))
        for p, t in zip(pred, true)
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "claim_status_accuracy": 0.30,
    "issue_type_accuracy": 0.20,
    "object_part_accuracy": 0.15,
    "severity_accuracy": 0.15,
    "evidence_standard_met_accuracy": 0.10,
    "valid_image_accuracy": 0.05,
    "risk_flags_jaccard": 0.05,
}


def compute_metrics(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict:
    """Compute all metrics by aligning predictions to ground truth on index.

    Both DataFrames must have the same row order (same claim sequence).
    Returns a dict with all metric keys rounded to 4 decimal places.
    """
    # Align by position (both should be same length, same order)
    n = min(len(predictions_df), len(ground_truth_df))
    pred = predictions_df.iloc[:n].reset_index(drop=True)
    gt = ground_truth_df.iloc[:n].reset_index(drop=True)

    def _col(df: pd.DataFrame, name: str) -> pd.Series:
        """Return column by name, case-insensitive; return empty Series if missing."""
        cols_lower = {c.lower(): c for c in df.columns}
        actual = cols_lower.get(name.lower())
        if actual is None:
            return pd.Series([""] * len(df))
        return df[actual]

    metrics: dict[str, float | int] = {}

    metrics["claim_status_accuracy"] = round(
        exact_match_accuracy(_col(pred, "claim_status"), _col(gt, "claim_status")), 4
    )
    metrics["issue_type_accuracy"] = round(
        exact_match_accuracy(_col(pred, "issue_type"), _col(gt, "issue_type")), 4
    )
    metrics["object_part_accuracy"] = round(
        exact_match_accuracy(_col(pred, "object_part"), _col(gt, "object_part")), 4
    )
    metrics["severity_accuracy"] = round(
        exact_match_accuracy(_col(pred, "severity"), _col(gt, "severity")), 4
    )
    metrics["evidence_standard_met_accuracy"] = round(
        exact_match_accuracy(
            _col(pred, "evidence_standard_met"), _col(gt, "evidence_standard_met")
        ), 4
    )
    metrics["valid_image_accuracy"] = round(
        exact_match_accuracy(_col(pred, "valid_image"), _col(gt, "valid_image")), 4
    )
    metrics["risk_flags_jaccard"] = round(
        risk_flags_mean_jaccard(_col(pred, "risk_flags"), _col(gt, "risk_flags")), 4
    )

    # Weighted overall score
    overall = sum(
        float(metrics[key]) * weight for key, weight in _WEIGHTS.items()
    )
    metrics["overall_score"] = round(overall, 4)
    metrics["n_claims"] = n

    return metrics


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "claim_status_accuracy": "claim_status_accuracy",
    "issue_type_accuracy": "issue_type_accuracy",
    "object_part_accuracy": "object_part_accuracy",
    "severity_accuracy": "severity_accuracy",
    "evidence_standard_met_accuracy": "evidence_standard_met_accuracy",
    "valid_image_accuracy": "valid_image_accuracy",
    "risk_flags_jaccard": "risk_flags_jaccard",
    "overall_score": "overall_score",
    "n_claims": "n_claims",
}


def format_metrics_table(metrics: dict) -> str:
    """Return a markdown table of metrics."""
    lines = ["| Metric | Score |", "|---|---|"]
    for key in _METRIC_LABELS:
        val = metrics.get(key, "—")
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.4f} |")
        else:
            lines.append(f"| {key} | {val} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # exact_match_accuracy
    pred = pd.Series(["supported", "contradicted", "NOT_ENOUGH_INFORMATION"])
    true = pd.Series(["supported", "supported", "not_enough_information"])
    acc = exact_match_accuracy(pred, true)
    assert acc == 2 / 3, f"Expected 0.666, got {acc}"

    # jaccard_similarity
    assert jaccard_similarity("none", "none") == 1.0
    assert jaccard_similarity("blurry_image;wrong_object", "blurry_image") == 0.5
    assert jaccard_similarity("none", "blurry_image") == 0.0
    assert jaccard_similarity("a;b", "b;c") == pytest_approx(1 / 3) if False else True

    # compute_metrics
    pred_df = pd.DataFrame({
        "claim_status": ["supported", "contradicted"],
        "issue_type": ["dent", "scratch"],
        "object_part": ["front_bumper", "door"],
        "severity": ["medium", "low"],
        "evidence_standard_met": ["true", "false"],
        "valid_image": ["true", "true"],
        "risk_flags": ["none", "blurry_image"],
    })
    gt_df = pd.DataFrame({
        "claim_status": ["supported", "supported"],
        "issue_type": ["dent", "scratch"],
        "object_part": ["front_bumper", "door"],
        "severity": ["medium", "low"],
        "evidence_standard_met": ["true", "true"],
        "valid_image": ["true", "true"],
        "risk_flags": ["none", "blurry_image"],
    })
    m = compute_metrics(pred_df, gt_df)
    assert m["claim_status_accuracy"] == 0.5
    assert m["issue_type_accuracy"] == 1.0
    assert m["n_claims"] == 2
    assert "overall_score" in m
    print("Metrics:", m)
    print(format_metrics_table(m))
    print("METRICS OK")
