"""Stage 8 — Output Formatter.

Maps a ClaimDecision + the original pandas row into a validated OutputRow.
Enforces all allowed enum values defined in validators.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import ClaimDecision, OutputRow  # noqa: E402
from utils.validators import (  # noqa: E402
    VALID_CLAIM_STATUS,
    VALID_ISSUE_TYPES,
    VALID_RISK_FLAGS,
    VALID_SEVERITIES,
    validate_enum,
    validate_object_part,
    validate_risk_flags,
)


class OutputFormatter:
    """Stage 8 — converts a ClaimDecision + raw row into a validated OutputRow."""

    def format(
        self,
        row: pd.Series,
        decision: ClaimDecision,
    ) -> OutputRow:
        """Return a fully validated OutputRow; never raises."""

        # Validate claim_status
        claim_status = validate_enum(
            decision.claim_status,
            VALID_CLAIM_STATUS,
            "not_enough_information",
        )

        # Validate issue_type
        issue_type = validate_enum(
            decision.issue_type,
            VALID_ISSUE_TYPES,
            "unknown",
        )

        # Validate object_part
        claim_object = str(row.get("claim_object", "")).lower().strip()
        object_part = validate_object_part(decision.object_part, claim_object)

        # Validate severity
        severity = validate_enum(
            decision.severity,
            VALID_SEVERITIES,
            "unknown",
        )

        # Format risk_flags — filter out invalid values and "none" before joining
        valid_flags = [
            f for f in decision.risk_flags
            if f in VALID_RISK_FLAGS and f != "none"
        ]
        risk_flags_str = validate_risk_flags(valid_flags)

        # Format supporting_image_ids
        if decision.supporting_image_ids:
            supporting_ids = ";".join(decision.supporting_image_ids)
        else:
            supporting_ids = "none"

        # Format boolean fields as lowercase strings
        evidence_met_str = "true" if decision.evidence_standard_met else "false"
        valid_image_str = "true" if decision.valid_image else "false"

        # Truncate long text fields
        justification = str(decision.claim_status_justification)[:1000]
        reason = str(decision.evidence_standard_met_reason)[:500]

        return OutputRow(
            user_id=str(row.get("user_id", "")),
            image_paths=str(row.get("image_paths", "")),
            user_claim=str(row.get("user_claim", "")),
            claim_object=claim_object,
            evidence_standard_met=evidence_met_str,
            evidence_standard_met_reason=reason,
            risk_flags=risk_flags_str,
            issue_type=issue_type,
            object_part=object_part,
            claim_status=claim_status,
            claim_status_justification=justification,
            supporting_image_ids=supporting_ids,
            valid_image=valid_image_str,
            severity=severity,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.schemas import ClaimDecision

    formatter = OutputFormatter()

    row = pd.Series({
        "user_id": "user_001",
        "image_paths": "images/test/case_001/img_1.jpg",
        "user_claim": "Front bumper dent.",
        "claim_object": "car",
    })

    decision = ClaimDecision(
        claim_status="supported",
        claim_status_justification="img_1 shows a dent on front_bumper.",
        evidence_standard_met=True,
        evidence_standard_met_reason="Valid image submitted.",
        issue_type="dent",
        object_part="front_bumper",
        severity="medium",
        risk_flags=[],
        supporting_image_ids=["img_1"],
        valid_image=True,
        decision_confidence="high",
    )

    output = formatter.format(row, decision)
    assert output.claim_status == "supported"
    assert output.evidence_standard_met == "true"
    assert output.valid_image == "true"
    assert output.risk_flags == "none"
    assert output.supporting_image_ids == "img_1"
    assert output.claim_object == "car"
    print(output)

    # Validate bad enum values are normalised
    bad_decision = ClaimDecision(
        claim_status="INVALID",
        claim_status_justification="x",
        evidence_standard_met=False,
        evidence_standard_met_reason="x",
        issue_type="INVALID",
        object_part="INVALID",
        severity="INVALID",
        risk_flags=["INVALID", "blurry_image"],
        supporting_image_ids=[],
        valid_image=False,
    )
    bad_row = pd.Series({"user_id": "u2", "image_paths": "", "user_claim": "", "claim_object": "laptop"})
    bad_out = formatter.format(bad_row, bad_decision)
    assert bad_out.claim_status == "not_enough_information"
    assert bad_out.issue_type == "unknown"
    assert bad_out.object_part == "unknown"
    assert bad_out.severity == "unknown"
    assert "blurry_image" in bad_out.risk_flags

    print("OUTPUT FORMATTER OK")
