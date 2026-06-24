"""Stage 5 — Vision Analyzer.

Sends each valid image + structured claim context to the Gemini Vision
API (Stage 5 of the 8-stage pipeline).  Returns a PerImageAnalysis for
every image path passed in.  Images that fail quality check or the API
call get a safe "insufficient" result rather than crashing the pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gemini_client import GeminiClient  # noqa: E402
from models.schemas import ImageQualityResult, PerImageAnalysis, StructuredClaim  # noqa: E402
from utils.image_utils import get_image_id, load_image_as_base64  # noqa: E402
from utils.validators import (  # noqa: E402
    OBJECT_PARTS,
    VALID_ISSUE_TYPES,
    VALID_SEVERITIES,
    validate_enum,
    validate_object_part,
)


class VisionAnalyzer:
    """Stage 5 — analyses each valid image against the structured claim."""

    def __init__(self, client: GeminiClient):
        self.client = client
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_allowed_parts_string(self, claim_object: str) -> str:
        parts = OBJECT_PARTS.get(claim_object)
        if parts is None:
            return "unknown"
        return ", ".join(sorted(parts))

    def _map_quality_issue_to_flag(self, quality_issue: str) -> str | None:
        mapping = {
            "blurry": "blurry_image",
            "dark": "low_light_or_glare",
            "obstructed": "cropped_or_obstructed",
            "wrong_angle": "wrong_angle",
            "cropped": "cropped_or_obstructed",
        }
        return mapping.get(str(quality_issue).strip().lower())

    def _parse_vision_response(
        self,
        raw_dict: dict,
        image_id: str,
        claim_object: str,
        raw_text: str = "",
    ) -> PerImageAnalysis:
        actual_part = validate_object_part(
            raw_dict.get("actual_part_identified", "unknown"),
            claim_object,
        )
        actual_damage = validate_enum(
            raw_dict.get("actual_damage_type", "unknown"),
            VALID_ISSUE_TYPES,
        )
        severity = validate_enum(
            raw_dict.get("severity", "unknown"),
            VALID_SEVERITIES,
        )

        assessment_raw = raw_dict.get("assessment", "insufficient")
        assessment = (
            assessment_raw
            if assessment_raw in ("supported", "contradicted", "insufficient")
            else "insufficient"
        )

        quality_issues_raw = raw_dict.get("image_quality_issues", [])
        if not isinstance(quality_issues_raw, list):
            quality_issues_raw = []
        quality_flags = [
            self._map_quality_issue_to_flag(q) for q in quality_issues_raw
        ]
        quality_flags = [f for f in quality_flags if f is not None]

        confidence = raw_dict.get("confidence", "low")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        return PerImageAnalysis(
            image_id=image_id,
            object_visible=bool(raw_dict.get("object_visible", False)),
            correct_object_in_image=bool(
                raw_dict.get("correct_object_in_image", False)
            ),
            claimed_part_visible=bool(raw_dict.get("claimed_part_visible", False)),
            actual_part_identified=actual_part,
            damage_visible=bool(raw_dict.get("damage_visible", False)),
            actual_damage_type=actual_damage,
            damage_matches_claim=bool(raw_dict.get("damage_matches_claim", False)),
            severity=severity,
            image_quality_issues=quality_flags,
            assessment=assessment,
            visual_evidence_summary=str(
                raw_dict.get("visual_evidence_summary", "")
            )[:500],
            confidence=confidence,
            raw_response=raw_text,
        )

    def _error_analysis(self, image_id: str, error_reason: str) -> PerImageAnalysis:
        return PerImageAnalysis(
            image_id=image_id,
            analysis_error=error_reason,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_single_image(
        self,
        image_path: str,
        structured_claim: StructuredClaim,
        claim_object: str,
    ) -> PerImageAnalysis:
        """Analyse one image; never raises."""
        image_id = get_image_id(image_path)

        try:
            b64_result = load_image_as_base64(image_path)
            if b64_result is None:
                self.logger.warning("Cannot load image: %s", image_path)
                return self._error_analysis(image_id, "Image could not be loaded")

            image_base64, mime_type = b64_result

            result = self.client.analyze_image(
                image_base64=image_base64,
                mime_type=mime_type,
                claim_object=claim_object,
                claimed_issue_type=structured_claim.claimed_issue_type,
                claimed_object_part=structured_claim.claimed_object_part,
                claim_summary=structured_claim.claim_summary,
                allowed_parts=self._get_allowed_parts_string(claim_object),
            )

            if result is None:
                return self._error_analysis(
                    image_id, "Vision API returned no result"
                )

            analysis = self._parse_vision_response(result, image_id, claim_object, raw_text="")
            self.logger.info(
                "Vision: %s → %s (conf=%s)", image_id, analysis.assessment, analysis.confidence
            )
            return analysis

        except Exception as exc:
            self.logger.exception("Vision analysis failed for %s: %s", image_id, exc)
            return self._error_analysis(image_id, str(exc))

    def analyze_all_images(
        self,
        image_paths: list[str],
        quality_results: list[ImageQualityResult],
        structured_claim: StructuredClaim,
        claim_object: str,
    ) -> list[PerImageAnalysis]:
        """
        Analyse all images, skipping those that failed quality check.
        Returns one PerImageAnalysis per entry in image_paths (same order).
        """
        analyses: list[PerImageAnalysis] = []
        for i, path in enumerate(image_paths):
            image_id = get_image_id(path)
            if i < len(quality_results) and not quality_results[i].is_valid:
                self.logger.info(
                    "Skipping invalid image %s (quality check failed)", image_id
                )
                analyses.append(
                    self._error_analysis(
                        image_id, "Skipped: failed image quality check"
                    )
                )
            else:
                analyses.append(
                    self.analyze_single_image(path, structured_claim, claim_object)
                )
        return analyses


# ---------------------------------------------------------------------------
# Self-test (no real API calls)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.WARNING)

    # Use a no-op client so we don't need a real key for unit tests
    class _NoOpClient:
        def analyze_image(self, **kw):
            return None
        def get_call_count(self):
            return 0

    analyzer = VisionAnalyzer(_NoOpClient())  # type: ignore[arg-type]

    # Test _get_allowed_parts_string
    parts_str = analyzer._get_allowed_parts_string("car")
    print("Car parts:", parts_str)
    assert "front_bumper" in parts_str

    # Test _map_quality_issue_to_flag
    assert analyzer._map_quality_issue_to_flag("blurry") == "blurry_image"
    assert analyzer._map_quality_issue_to_flag("dark") == "low_light_or_glare"
    assert analyzer._map_quality_issue_to_flag("obstructed") == "cropped_or_obstructed"
    assert analyzer._map_quality_issue_to_flag("wrong_angle") == "wrong_angle"
    assert analyzer._map_quality_issue_to_flag("cropped") == "cropped_or_obstructed"
    assert analyzer._map_quality_issue_to_flag("unknown_flag") is None

    # Test _error_analysis
    err = analyzer._error_analysis("img_1", "test error")
    print("Error analysis:", err)
    assert err.image_id == "img_1"
    assert err.analysis_error == "test error"
    assert err.assessment == "insufficient"

    # Test analyze_single_image with a non-existent path → error result
    result = analyzer.analyze_single_image(
        "/nonexistent/path/img_1.jpg",
        StructuredClaim(),
        "car",
    )
    assert result.image_id == "img_1"
    assert result.analysis_error  # should be set

    # Test analyze_all_images skips invalid images
    from models.schemas import ImageQualityResult

    quality = [
        ImageQualityResult(
            image_id="img_1", image_path="img_1.jpg", is_valid=False, is_readable=False
        ),
        ImageQualityResult(
            image_id="img_2", image_path="img_2.jpg", is_valid=True, is_readable=True
        ),
    ]
    results = analyzer.analyze_all_images(
        ["img_1.jpg", "img_2.jpg"],
        quality,
        StructuredClaim(),
        "car",
    )
    assert len(results) == 2
    assert "Skipped" in results[0].analysis_error
    assert results[1].image_id == "img_2"

    print("VISION ANALYZER OK")
