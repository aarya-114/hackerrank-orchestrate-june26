"""Stage 6 — Image Aggregator.

Combines per-image PerImageAnalysis results using majority-voting logic
to produce a single AggregatedVision consensus.  Pure Python — zero API
calls.
"""

from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import AggregatedVision, PerImageAnalysis  # noqa: E402
from utils.validators import VALID_RISK_FLAGS  # noqa: E402

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"high": 4, "medium": 3, "low": 2, "none": 1, "unknown": 0}


def _majority(values: list) -> object:
    """Return the value that appears in more than half of *values*, or None."""
    if not values:
        return None
    counter = collections.Counter(values)
    top_value, top_count = counter.most_common(1)[0]
    if top_count > len(values) / 2:
        return top_value
    return None


def _most_common_non_unknown(values: list[str]) -> str:
    """Return the most common value that is not 'unknown', or 'unknown'."""
    filtered = [v for v in values if v != "unknown"]
    if not filtered:
        return "unknown"
    counter = collections.Counter(filtered)
    return counter.most_common(1)[0][0]


def _most_common_severity(values: list[str]) -> str:
    """Most common non-'unknown' severity; tie-break prefers higher severity."""
    filtered = [v for v in values if v != "unknown"]
    if not filtered:
        return "unknown"
    counter = collections.Counter(filtered)
    # Find the maximum count
    max_count = counter.most_common(1)[0][1]
    # Among all severities with that count, pick the highest-ranked one
    candidates = [sev for sev, cnt in counter.items() if cnt == max_count]
    return max(candidates, key=lambda s: _SEVERITY_RANK.get(s, 0))


class ImageAggregator:
    """Stage 6 — aggregates per-image analyses into a single consensus."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def aggregate(
        self,
        analyses: list[PerImageAnalysis],
        claim_object: str,
    ) -> AggregatedVision:
        """Return an AggregatedVision; never raises."""
        if not analyses:
            return AggregatedVision(
                per_image_analyses=[],
                aggregation_notes="No images to aggregate",
            )

        successful = [a for a in analyses if not a.analysis_error]

        if not successful:
            return AggregatedVision(
                per_image_analyses=analyses,
                aggregation_notes="All image analyses failed or were skipped",
                vision_risk_flags=["damage_not_visible"],
            )

        n = len(successful)

        # ── Consensus: object correct ──────────────────────────────────────
        object_correct_votes = [a.correct_object_in_image for a in successful]
        consensus_object_correct = object_correct_votes.count(True) > n / 2

        # ── Consensus: part visible ────────────────────────────────────────
        consensus_part_visible = any(a.claimed_part_visible for a in successful)

        # ── Consensus: damage visible ──────────────────────────────────────
        damage_visible_votes = [a.damage_visible for a in successful]
        consensus_damage_visible = damage_visible_votes.count(True) > n / 2

        # ── Consensus: damage type ─────────────────────────────────────────
        consensus_damage_type = _most_common_non_unknown(
            [a.actual_damage_type for a in successful]
        )

        # ── Consensus: object part ─────────────────────────────────────────
        consensus_object_part = _most_common_non_unknown(
            [a.actual_part_identified for a in successful]
        )

        # ── Consensus: severity ────────────────────────────────────────────
        consensus_severity = _most_common_severity(
            [a.severity for a in successful]
        )

        # ── Consensus: assessment ──────────────────────────────────────────
        assessments = [a.assessment for a in successful]
        assessment_counter = collections.Counter(assessments)
        supported_count = assessment_counter.get("supported", 0)
        contradicted_count = assessment_counter.get("contradicted", 0)
        insufficient_count = assessment_counter.get("insufficient", 0)

        if supported_count > n / 2:
            consensus_assessment = "supported"
        elif contradicted_count > n / 2:
            consensus_assessment = "contradicted"
        elif insufficient_count > n / 2:
            consensus_assessment = "insufficient"
        elif supported_count > 0 and contradicted_count > 0:
            # Split — err on the side of caution
            consensus_assessment = "contradicted"
        else:
            consensus_assessment = "insufficient"

        # ── Overall confidence ─────────────────────────────────────────────
        confidences = [a.confidence for a in successful]
        confidence_majority = _majority(confidences)
        if confidence_majority == "high":
            overall_confidence = "high"
        elif confidence_majority == "medium":
            overall_confidence = "medium"
        else:
            overall_confidence = "low"

        # ── Supporting image IDs ───────────────────────────────────────────
        supporting_image_ids = [
            a.image_id
            for a in successful
            if a.assessment == "supported" and a.confidence != "low"
        ]

        # ── Vision risk flags ──────────────────────────────────────────────
        risk_flag_set: list[str] = []
        seen_flags: set[str] = set()

        for a in analyses:
            for flag in a.image_quality_issues:
                if flag not in seen_flags:
                    risk_flag_set.append(flag)
                    seen_flags.add(flag)

        if not consensus_object_correct:
            if "wrong_object" not in seen_flags:
                risk_flag_set.append("wrong_object")
                seen_flags.add("wrong_object")

        if not consensus_damage_visible and consensus_object_correct:
            if "damage_not_visible" not in seen_flags:
                risk_flag_set.append("damage_not_visible")
                seen_flags.add("damage_not_visible")

        # Keep only flags in VALID_RISK_FLAGS
        vision_risk_flags = [f for f in risk_flag_set if f in VALID_RISK_FLAGS]

        notes = (
            f"Aggregated {n} successful analysis(es) out of {len(analyses)} total. "
            f"Assessment: {consensus_assessment}, confidence: {overall_confidence}."
        )

        return AggregatedVision(
            per_image_analyses=analyses,
            consensus_object_correct=consensus_object_correct,
            consensus_part_visible=consensus_part_visible,
            consensus_damage_visible=consensus_damage_visible,
            consensus_damage_type=consensus_damage_type,
            consensus_object_part=consensus_object_part,
            consensus_severity=consensus_severity,
            consensus_assessment=consensus_assessment,
            overall_confidence=overall_confidence,
            supporting_image_ids=supporting_image_ids,
            vision_risk_flags=vision_risk_flags,
            aggregation_notes=notes,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.schemas import PerImageAnalysis

    logging.basicConfig(level=logging.WARNING)
    agg = ImageAggregator()

    # Empty list
    result = agg.aggregate([], "car")
    assert result.aggregation_notes == "No images to aggregate"

    # All errored
    errored = [
        PerImageAnalysis(image_id="img_1", analysis_error="load failed"),
        PerImageAnalysis(image_id="img_2", analysis_error="api error"),
    ]
    result = agg.aggregate(errored, "car")
    assert result.consensus_assessment == "insufficient"
    assert "damage_not_visible" in result.vision_risk_flags

    # Clear support scenario
    supported = [
        PerImageAnalysis(
            image_id="img_1",
            correct_object_in_image=True,
            claimed_part_visible=True,
            damage_visible=True,
            actual_damage_type="dent",
            actual_part_identified="front_bumper",
            severity="medium",
            assessment="supported",
            confidence="high",
        ),
        PerImageAnalysis(
            image_id="img_2",
            correct_object_in_image=True,
            claimed_part_visible=False,
            damage_visible=True,
            actual_damage_type="dent",
            actual_part_identified="front_bumper",
            severity="medium",
            assessment="supported",
            confidence="medium",
        ),
    ]
    result = agg.aggregate(supported, "car")
    print(result)
    assert result.consensus_assessment == "supported"
    assert result.consensus_damage_type == "dent"
    assert result.consensus_object_correct is True
    assert "img_1" in result.supporting_image_ids

    # Contradiction scenario
    contradicted = [
        PerImageAnalysis(
            image_id="img_1",
            correct_object_in_image=True,
            damage_visible=False,
            actual_damage_type="none",
            severity="none",
            assessment="contradicted",
            confidence="high",
        ),
    ]
    result = agg.aggregate(contradicted, "car")
    assert result.consensus_assessment == "contradicted"
    assert "damage_not_visible" in result.vision_risk_flags

    print("IMAGE AGGREGATOR OK")
