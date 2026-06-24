from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.validators import (  # noqa: E402
    OBJECT_PARTS,
    VALID_CLAIM_STATUS,
    VALID_ISSUE_TYPES,
    VALID_RISK_FLAGS,
    VALID_SEVERITIES,
)


@dataclass
class ImageQualityResult:
    image_id: str
    image_path: str
    is_valid: bool
    is_readable: bool
    width: int = 0
    height: int = 0
    blur_score: float = 0.0
    brightness: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    quality_notes: str = ""


@dataclass
class StructuredClaim:
    claimed_issue_type: str = "unknown"
    claimed_object_part: str = "unknown"
    claimed_severity_hint: str = "unknown"
    claim_summary: str = ""
    incident_context: str = ""
    confidence: str = "unknown"
    extraction_method: str = "llm"


@dataclass
class EvidenceStandard:
    claim_object: str
    issue_family: str
    requirements: list[str]
    images_provided: int
    valid_images_provided: int
    standard_met: bool
    reason: str


@dataclass
class UserRisk:
    user_id: str
    past_claim_count: int = 0
    rejected_claim_count: int = 0
    last_90_days_count: int = 0
    history_flags: str = ""
    risk_level: str = "low"
    risk_flags: list[str] = field(default_factory=list)
    risk_summary: str = ""
    user_found: bool = True


@dataclass
class PerImageAnalysis:
    image_id: str
    object_visible: bool = False
    correct_object_in_image: bool = False
    claimed_part_visible: bool = False
    actual_part_identified: str = "unknown"
    damage_visible: bool = False
    actual_damage_type: str = "unknown"
    damage_matches_claim: bool = False
    severity: str = "unknown"
    image_quality_issues: list[str] = field(default_factory=list)
    assessment: str = "insufficient"
    visual_evidence_summary: str = ""
    confidence: str = "low"
    raw_response: str = ""
    analysis_error: str = ""


@dataclass
class AggregatedVision:
    per_image_analyses: list[PerImageAnalysis]
    consensus_object_correct: bool = False
    consensus_part_visible: bool = False
    consensus_damage_visible: bool = False
    consensus_damage_type: str = "unknown"
    consensus_object_part: str = "unknown"
    consensus_severity: str = "unknown"
    consensus_assessment: str = "insufficient"
    overall_confidence: str = "low"
    supporting_image_ids: list[str] = field(default_factory=list)
    vision_risk_flags: list[str] = field(default_factory=list)
    aggregation_notes: str = ""


@dataclass
class ClaimDecision:
    claim_status: str
    claim_status_justification: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    issue_type: str
    object_part: str
    severity: str
    risk_flags: list[str]
    supporting_image_ids: list[str]
    valid_image: bool
    decision_confidence: str = "low"


@dataclass
class OutputRow:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str

    @classmethod
    def get_columns(cls) -> list[str]:
        return [field_info.name for field_info in fields(cls)]


def create_error_output_row(
    user_id: str,
    image_paths: str,
    user_claim: str,
    claim_object: str,
    error_reason: str,
) -> OutputRow:
    return OutputRow(
        user_id=user_id,
        image_paths=image_paths,
        user_claim=user_claim,
        claim_object=claim_object,
        evidence_standard_met="false",
        evidence_standard_met_reason=f"Processing error: {error_reason}",
        risk_flags="manual_review_required",
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification=f"Unable to process claim: {error_reason}",
        supporting_image_ids="none",
        valid_image="false",
        severity="unknown",
    )


if __name__ == "__main__":
    expected_columns = [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]

    samples = [
        ImageQualityResult(
            image_id="img_1",
            image_path="dataset/images/test/case_001/img_1.jpg",
            is_valid=True,
            is_readable=True,
            width=800,
            height=600,
            blur_score=250.0,
            brightness=128.0,
            quality_flags=[],
            quality_notes="Readable image.",
        ),
        StructuredClaim(
            claimed_issue_type="dent",
            claimed_object_part="front_bumper",
            claimed_severity_hint="medium",
            claim_summary="Customer reports a dent on the front bumper.",
            incident_context="Parked near office.",
            confidence="high",
        ),
        EvidenceStandard(
            claim_object="car",
            issue_family="dent",
            requirements=[
                "The claimed car panel or bumper should be visible clearly."
            ],
            images_provided=2,
            valid_images_provided=2,
            standard_met=True,
            reason="Relevant part is visible in at least one image.",
        ),
        UserRisk(
            user_id="user_001",
            past_claim_count=2,
            rejected_claim_count=0,
            last_90_days_count=1,
            history_flags="none",
            risk_level="low",
            risk_summary="Low-risk user.",
        ),
        PerImageAnalysis(
            image_id="img_1",
            object_visible=True,
            correct_object_in_image=True,
            claimed_part_visible=True,
            actual_part_identified="front_bumper",
            damage_visible=True,
            actual_damage_type="dent",
            damage_matches_claim=True,
            severity="medium",
            assessment="supported",
            visual_evidence_summary="Dent is visible on the front bumper.",
            confidence="high",
        ),
    ]
    samples.append(
        AggregatedVision(
            per_image_analyses=[samples[-1]],
            consensus_object_correct=True,
            consensus_part_visible=True,
            consensus_damage_visible=True,
            consensus_damage_type="dent",
            consensus_object_part="front_bumper",
            consensus_severity="medium",
            consensus_assessment="supported",
            overall_confidence="high",
            supporting_image_ids=["img_1"],
        )
    )
    samples.extend(
        [
            ClaimDecision(
                claim_status="supported",
                claim_status_justification="Visual evidence supports the claim.",
                evidence_standard_met=True,
                evidence_standard_met_reason="Evidence standard met.",
                issue_type="dent",
                object_part="front_bumper",
                severity="medium",
                risk_flags=[],
                supporting_image_ids=["img_1"],
                valid_image=True,
                decision_confidence="high",
            ),
            OutputRow(
                user_id="user_001",
                image_paths="images/test/case_001/img_1.jpg",
                user_claim="Front bumper dent.",
                claim_object="car",
                evidence_standard_met="true",
                evidence_standard_met_reason="Evidence standard met.",
                risk_flags="none",
                issue_type="dent",
                object_part="front_bumper",
                claim_status="supported",
                claim_status_justification="Visual evidence supports the claim.",
                supporting_image_ids="img_1",
                valid_image="true",
                severity="medium",
            ),
        ]
    )

    error_row = create_error_output_row(
        user_id="user_001",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="Front bumper dent.",
        claim_object="car",
        error_reason="sample failure",
    )

    for sample in samples:
        print(sample)
    print(error_row)

    assert OutputRow.get_columns() == expected_columns
    assert len(OutputRow.get_columns()) == 14
    assert "supported" in VALID_CLAIM_STATUS
    assert "dent" in VALID_ISSUE_TYPES
    assert "medium" in VALID_SEVERITIES
    assert "manual_review_required" in VALID_RISK_FLAGS
    assert "front_bumper" in OBJECT_PARTS["car"]
    print("ALL SCHEMAS OK")
