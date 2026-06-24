"""Stage 5 (Strategy C) — Observe-First Vision Analyzer.

The vision model receives NO claim context and describes what it observes
freely. Damage-type matching and claim assessment are then performed by
deterministic semantic similarity code (utils/semantic_matcher.py).

This eliminates anchoring bias where the model searches for absence of a
specific named damage type rather than describing what is actually present.
"""

from __future__ import annotations

import logging

from config import Config
from models.gemini_client import GeminiClient
from models.prompts import FREE_OBSERVATION_SYSTEM, FREE_OBSERVATION_USER
from models.schemas import ImageQualityResult, PerImageAnalysis
from pipeline.image_quality_checker import ImageQualityChecker  # noqa: F401 (used by callers)
from utils.image_utils import get_image_id, load_image_as_base64
from utils.semantic_matcher import (
    DAMAGE_FAMILIES,
    PART_KEYWORDS,
    match_damage_type,
    match_object_part,
    map_severity,
)
from utils.validators import (
    VALID_ISSUE_TYPES,
    VALID_SEVERITIES,
    validate_enum,
    validate_object_part,
)


class VisionAnalyzerC:
    """Observe-first vision analyzer for Strategy C.

    Calls the vision model with NO claim context — only the object type is
    shared so the model can frame its vocabulary. All claim-matching logic
    is handled deterministically via semantic_matcher.
    """

    def __init__(self, client: GeminiClient) -> None:
        self.client = client
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_single_image(
        self,
        image_path: str,
        claim_object: str,
        claimed_issue_type: str,
        claimed_object_part: str,
        claim_summary: str,  # kept in signature for interface parity; not sent to model
    ) -> PerImageAnalysis:
        """Analyze one image via free observation and semantic matching."""
        image_id = get_image_id(image_path)

        # Step 1 — Load image
        result = load_image_as_base64(image_path)
        if result is None:
            return PerImageAnalysis(
                image_id=image_id,
                analysis_error="Image could not be loaded",
            )
        image_base64, mime_type = result

        # Step 2 — Build free observation prompt (no claim context)
        text_prompt = (
            FREE_OBSERVATION_SYSTEM.strip()
            + "\n\n"
            + FREE_OBSERVATION_USER.format(claim_object=claim_object)
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            }
        ]

        # Step 3 — Call API
        raw = self.client._call_with_retry(messages, model=Config.VISION_MODEL)
        obs_dict = self.client._extract_json(raw)

        if obs_dict is None:
            self.logger.warning("VisionAnalyzerC: no parseable response for %s", image_id)
            return PerImageAnalysis(
                image_id=image_id,
                analysis_error="Vision API returned no parseable response",
            )

        # Step 4 — Extract raw observations
        object_visible = bool(obs_dict.get("object_visible", False))
        damage_present = bool(obs_dict.get("damage_present", False))
        damage_description = str(obs_dict.get("damage_description", ""))
        damage_location = str(obs_dict.get("damage_location", ""))
        observed_severity = str(obs_dict.get("damage_severity", "unknown"))
        image_quality = str(obs_dict.get("image_quality", "good"))
        confidence = str(obs_dict.get("confidence", "low"))
        observation_summary = str(obs_dict.get("observation_summary", ""))
        primary_part = str(obs_dict.get("primary_part_visible", ""))

        full_observation = f"{damage_description} {damage_location} {observation_summary}"

        # Step 5 — Semantic matching (deterministic, no AI)
        damage_matches, match_score = match_damage_type(
            full_observation, claimed_issue_type
        )

        # Identify the actual damage type from observation vocabulary
        actual_damage_type = "none"
        if damage_present:
            best_type = "unknown"
            best_count = 0
            for dtype, keywords in DAMAGE_FAMILIES.items():
                count = sum(1 for kw in keywords if kw in full_observation.lower())
                if count > best_count:
                    best_count = count
                    best_type = dtype
            actual_damage_type = best_type if best_count > 0 else "unknown"

        actual_damage_type = validate_enum(actual_damage_type, VALID_ISSUE_TYPES, "unknown")

        # Match object part from primary_part + damage_location
        part_from_observation = match_object_part(
            f"{primary_part} {damage_location}", claim_object
        )
        actual_part = validate_object_part(part_from_observation, claim_object)

        # Map severity
        severity = map_severity(observed_severity)
        severity = validate_enum(severity, VALID_SEVERITIES, "unknown")

        # Step 6 — Determine assessment (DETERMINISTIC, no AI verdict)
        quality_flags: list[str] = []
        if image_quality == "blurry":
            quality_flags.append("blurry_image")
        elif image_quality == "dark":
            quality_flags.append("low_light_or_glare")
        elif image_quality == "obstructed":
            quality_flags.append("cropped_or_obstructed")
        elif image_quality == "wrong_angle":
            quality_flags.append("wrong_angle")

        if not object_visible:
            assessment = "insufficient"
        elif confidence == "low":
            assessment = "insufficient"
        elif not damage_present:
            # No damage seen — only contradict if image quality was good and we're confident
            if image_quality == "good" and confidence == "high":
                assessment = "contradicted"
            else:
                assessment = "insufficient"
        elif damage_present and match_score >= 0.7:
            assessment = "supported"
        elif damage_present and match_score >= 0.3:
            # Some damage, but type doesn't match cleanly
            assessment = "insufficient"
        else:
            assessment = "insufficient"

        # Determine if claimed part was visible in the observation
        part_map = PART_KEYWORDS.get(claim_object, {})
        claimed_kws = part_map.get(claimed_object_part, [claimed_object_part])
        claimed_part_visible = any(
            kw in full_observation.lower() for kw in claimed_kws
        )

        # Step 7 — Return PerImageAnalysis
        return PerImageAnalysis(
            image_id=image_id,
            object_visible=object_visible,
            correct_object_in_image=object_visible,
            claimed_part_visible=claimed_part_visible,
            actual_part_identified=actual_part,
            damage_visible=damage_present,
            actual_damage_type=actual_damage_type,
            damage_matches_claim=damage_matches,
            severity=severity,
            image_quality_issues=quality_flags,
            assessment=assessment,
            visual_evidence_summary=observation_summary[:500],
            confidence=confidence,
            raw_response=raw,
        )

    def analyze_all_images(
        self,
        image_paths: list[str],
        quality_results: list[ImageQualityResult],
        claim_object: str,
        claimed_issue_type: str,
        claimed_object_part: str,
        claim_summary: str,
    ) -> list[PerImageAnalysis]:
        """Analyze all images, skipping those that failed quality check."""
        analyses: list[PerImageAnalysis] = []

        for i, image_path in enumerate(image_paths):
            quality = quality_results[i] if i < len(quality_results) else None

            if quality is not None and not quality.is_valid:
                image_id = get_image_id(image_path)
                analyses.append(
                    PerImageAnalysis(
                        image_id=image_id,
                        analysis_error="Skipped: failed image quality check",
                    )
                )
                continue

            analysis = self.analyze_single_image(
                image_path=image_path,
                claim_object=claim_object,
                claimed_issue_type=claimed_issue_type,
                claimed_object_part=claimed_object_part,
                claim_summary=claim_summary,
            )
            analyses.append(analysis)

        return analyses
