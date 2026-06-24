"""Stage 7 — Decision Engine.

Applies a priority-ordered rule tree to produce a final ClaimDecision.
Pure Python — zero API calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import (  # noqa: E402
    AggregatedVision,
    ClaimDecision,
    EvidenceStandard,
    ImageQualityResult,
    StructuredClaim,
    UserRisk,
)
from utils.validators import (  # noqa: E402
    VALID_ISSUE_TYPES,
    VALID_RISK_FLAGS,
    validate_enum,
    validate_object_part,
)


class DecisionEngine:
    """Stage 7 — deterministic rule tree that produces a ClaimDecision."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def decide(
        self,
        structured_claim: StructuredClaim,
        evidence_standard: EvidenceStandard,
        user_risk: UserRisk,
        aggregated_vision: AggregatedVision,
        image_quality_results: list[ImageQualityResult],
        claim_object: str,
    ) -> ClaimDecision:
        """Return a ClaimDecision; never raises."""

        # ── Step 1: Determine valid_image ──────────────────────────────────
        valid_image = any(q.is_valid for q in image_quality_results)

        # ── Step 2: Collect all risk flags ─────────────────────────────────
        all_flags: set[str] = set()

        for q in image_quality_results:
            for flag in q.quality_flags:
                if flag in VALID_RISK_FLAGS:
                    all_flags.add(flag)

        for flag in aggregated_vision.vision_risk_flags:
            if flag in VALID_RISK_FLAGS:
                all_flags.add(flag)

        for flag in user_risk.risk_flags:
            if flag in VALID_RISK_FLAGS:
                all_flags.add(flag)

        # ── Step 3: Determine claim_status (PRIORITY ORDER) ────────────────
        claim_status: str
        justification: str

        # RULE 1 — No valid images
        if not valid_image:
            claim_status = "not_enough_information"
            justification = "No valid images could be processed."
            evidence_standard = EvidenceStandard(
                claim_object=evidence_standard.claim_object,
                issue_family=evidence_standard.issue_family,
                requirements=evidence_standard.requirements,
                images_provided=evidence_standard.images_provided,
                valid_images_provided=evidence_standard.valid_images_provided,
                standard_met=False,
                reason=evidence_standard.reason,
            )
            return self._build_decision(
                claim_status, justification, evidence_standard,
                aggregated_vision, all_flags, valid_image,
                structured_claim, claim_object,
                user_risk,
            )

        # RULE 2 — Evidence standard not met
        if not evidence_standard.standard_met:
            claim_status = "not_enough_information"
            justification = f"Evidence standard not met: {evidence_standard.reason}"
            return self._build_decision(
                claim_status, justification, evidence_standard,
                aggregated_vision, all_flags, valid_image,
                structured_claim, claim_object,
                user_risk,
            )

        # RULE 3 — Wrong object in images
        if not aggregated_vision.consensus_object_correct and valid_image:
            claim_status = "not_enough_information"
            justification = "Submitted images do not show the claimed object type."
            all_flags.add("wrong_object")
            return self._build_decision(
                claim_status, justification, evidence_standard,
                aggregated_vision, all_flags, valid_image,
                structured_claim, claim_object,
                user_risk,
            )

        # RULE 4 — Claim mismatch detection (informational flag only)
        if (
            aggregated_vision.consensus_damage_visible
            and aggregated_vision.consensus_assessment != "supported"
        ):
            visible_type = aggregated_vision.consensus_damage_type
            claimed_type = structured_claim.claimed_issue_type
            if visible_type not in ("unknown", "none") and visible_type != claimed_type:
                all_flags.add("claim_mismatch")

        # RULE 5 — Main decision from vision consensus
        consensus = aggregated_vision.consensus_assessment
        img_ids_str = self._format_image_ids(aggregated_vision.supporting_image_ids)
        damage_type = aggregated_vision.consensus_damage_type
        object_part = aggregated_vision.consensus_object_part
        confidence = aggregated_vision.overall_confidence

        if consensus == "supported":
            claim_status = "supported"
            part_str = object_part if object_part != "unknown" else structured_claim.claimed_object_part
            dmg_str = damage_type if damage_type not in ("unknown", "none") else structured_claim.claimed_issue_type
            if img_ids_str:
                justification = (
                    f"{img_ids_str} shows visible {dmg_str} on {part_str}. "
                    f"Confidence: {confidence}."
                )
            else:
                justification = (
                    f"Visual evidence supports {dmg_str} on {part_str}. "
                    f"Confidence: {confidence}."
                )

        # Conservative contradicted rule: only contradict when we are
        # certain the right part is visible and clearly undamaged.
        # A single ambiguous image cannot produce contradicted verdict.
        elif consensus == "contradicted" and (
            aggregated_vision.overall_confidence == "high"
            and aggregated_vision.consensus_object_correct
            and aggregated_vision.consensus_part_visible
            and not aggregated_vision.consensus_damage_visible
        ):
            claim_status = "contradicted"
            part_str = object_part if object_part != "unknown" else structured_claim.claimed_object_part
            justification = (
                f"Images clearly show {part_str} with no visible damage "
                f"matching the claimed {structured_claim.claimed_issue_type}."
            )

        else:
            claim_status = "not_enough_information"
            reason = aggregated_vision.aggregation_notes or "insufficient visual evidence"
            justification = f"Could not verify claim. {reason}"

        # ── Step 4: Add high-risk flag if needed ───────────────────────────
        if user_risk.risk_level == "high":
            all_flags.add("manual_review_required")

        return self._build_decision(
            claim_status, justification, evidence_standard,
            aggregated_vision, all_flags, valid_image,
            structured_claim, claim_object,
            user_risk,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_decision(
        self,
        claim_status: str,
        justification: str,
        evidence_standard: EvidenceStandard,
        aggregated_vision: AggregatedVision,
        all_flags: set[str],
        valid_image: bool,
        structured_claim: StructuredClaim,
        claim_object: str,
        user_risk: UserRisk,
    ) -> ClaimDecision:
        # High-risk user always gets manual_review_required
        if user_risk.risk_level == "high":
            all_flags.add("manual_review_required")

        # ── Step 5: issue_type and object_part ─────────────────────────────
        if claim_status == "supported":
            raw_issue = aggregated_vision.consensus_damage_type
            if raw_issue == "unknown":
                raw_issue = structured_claim.claimed_issue_type
        elif claim_status == "contradicted":
            raw_issue = aggregated_vision.consensus_damage_type
            if raw_issue == "unknown":
                raw_issue = "none"
        else:
            raw_issue = structured_claim.claimed_issue_type

        issue_type = validate_enum(raw_issue, VALID_ISSUE_TYPES, fallback="unknown")

        raw_part = aggregated_vision.consensus_object_part
        if raw_part == "unknown":
            raw_part = structured_claim.claimed_object_part
        object_part = validate_object_part(raw_part, claim_object)

        # ── Step 6: severity ───────────────────────────────────────────────
        if claim_status == "supported":
            severity = aggregated_vision.consensus_severity
            if severity == "unknown":
                severity = structured_claim.claimed_severity_hint
        elif claim_status == "contradicted":
            severity = "none"
        else:
            severity = "unknown"

        from utils.validators import VALID_SEVERITIES
        severity = validate_enum(severity, VALID_SEVERITIES, fallback="unknown")

        return ClaimDecision(
            claim_status=claim_status,
            claim_status_justification=justification,
            evidence_standard_met=evidence_standard.standard_met,
            evidence_standard_met_reason=evidence_standard.reason,
            issue_type=issue_type,
            object_part=object_part,
            severity=severity,
            risk_flags=sorted(all_flags),
            supporting_image_ids=aggregated_vision.supporting_image_ids,
            valid_image=valid_image,
            decision_confidence=aggregated_vision.overall_confidence,
        )

    @staticmethod
    def _format_image_ids(image_ids: list[str]) -> str:
        if not image_ids:
            return ""
        if len(image_ids) == 1:
            return image_ids[0]
        return ", ".join(image_ids[:-1]) + " and " + image_ids[-1]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.schemas import (
        AggregatedVision,
        EvidenceStandard,
        ImageQualityResult,
        PerImageAnalysis,
        StructuredClaim,
        UserRisk,
    )

    logging.basicConfig(level=logging.WARNING)
    engine = DecisionEngine()

    def _make_evidence(standard_met=True):
        return EvidenceStandard(
            claim_object="car",
            issue_family="dent or scratch",
            requirements=["Clear image of bumper required"],
            images_provided=1,
            valid_images_provided=1 if standard_met else 0,
            standard_met=standard_met,
            reason="Valid image(s) submitted" if standard_met else "No valid images",
        )

    def _make_claim():
        return StructuredClaim(
            claimed_issue_type="dent",
            claimed_object_part="front_bumper",
            claimed_severity_hint="medium",
            claim_summary="Dent on front bumper after accident.",
        )

    def _make_quality(is_valid=True):
        return [
            ImageQualityResult(
                image_id="img_1",
                image_path="images/test/case_001/img_1.jpg",
                is_valid=is_valid,
                is_readable=is_valid,
                quality_flags=[],
            )
        ]

    # ── Scenario 1: Clear support ──────────────────────────────────────────
    agg_supported = AggregatedVision(
        per_image_analyses=[
            PerImageAnalysis(
                image_id="img_1",
                correct_object_in_image=True,
                damage_visible=True,
                actual_damage_type="dent",
                actual_part_identified="front_bumper",
                severity="medium",
                assessment="supported",
                confidence="high",
            )
        ],
        consensus_object_correct=True,
        consensus_part_visible=True,
        consensus_damage_visible=True,
        consensus_damage_type="dent",
        consensus_object_part="front_bumper",
        consensus_severity="medium",
        consensus_assessment="supported",
        overall_confidence="high",
        supporting_image_ids=["img_1"],
        vision_risk_flags=[],
    )
    d1 = engine.decide(
        _make_claim(), _make_evidence(True),
        UserRisk(user_id="user_001", risk_level="low"),
        agg_supported, _make_quality(True), "car",
    )
    print("Scenario 1:", d1.claim_status, d1.claim_status_justification)
    assert d1.claim_status == "supported", f"Got: {d1.claim_status}"
    assert d1.severity == "medium"
    assert d1.valid_image is True

    # ── Scenario 2: No valid images ────────────────────────────────────────
    agg_empty = AggregatedVision(
        per_image_analyses=[],
        consensus_assessment="insufficient",
        overall_confidence="low",
        supporting_image_ids=[],
        vision_risk_flags=[],
    )
    d2 = engine.decide(
        _make_claim(), _make_evidence(False),
        UserRisk(user_id="user_002", risk_level="low"),
        agg_empty, _make_quality(False), "car",
    )
    print("Scenario 2:", d2.claim_status, d2.claim_status_justification)
    assert d2.claim_status == "not_enough_information", f"Got: {d2.claim_status}"
    assert d2.valid_image is False

    # ── Scenario 3: High-risk user, contradiction ──────────────────────────
    agg_contradicted = AggregatedVision(
        per_image_analyses=[
            PerImageAnalysis(
                image_id="img_1",
                correct_object_in_image=True,
                damage_visible=False,
                actual_damage_type="none",
                actual_part_identified="front_bumper",
                severity="none",
                assessment="contradicted",
                confidence="high",
            )
        ],
        consensus_object_correct=True,
        consensus_damage_visible=False,
        consensus_damage_type="none",
        consensus_object_part="front_bumper",
        consensus_severity="none",
        consensus_assessment="contradicted",
        overall_confidence="high",
        supporting_image_ids=[],
        vision_risk_flags=["damage_not_visible"],
    )
    d3 = engine.decide(
        _make_claim(), _make_evidence(True),
        UserRisk(user_id="user_003", risk_level="high",
                 risk_flags=["manual_review_required"]),
        agg_contradicted, _make_quality(True), "car",
    )
    print("Scenario 3:", d3.claim_status, d3.risk_flags)
    assert d3.claim_status == "contradicted", f"Got: {d3.claim_status}"
    assert "manual_review_required" in d3.risk_flags, f"Flags: {d3.risk_flags}"

    print("DECISION ENGINE OK")
