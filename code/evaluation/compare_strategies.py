"""Strategy A vs Strategy B comparison runner.

Strategy A — Baseline: single monolithic Gemini call per claim with all
context in one prompt.  No decomposition, no deterministic layers.

Strategy B — Full pipeline: calls main.main() which runs the 8-stage
decomposed pipeline.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from models.gemini_client import GeminiClient  # noqa: E402
from models.schemas import OutputRow, create_error_output_row  # noqa: E402
from utils.csv_loader import DataStore  # noqa: E402
from utils.image_utils import get_image_id, load_image_as_base64, load_image_paths  # noqa: E402
from utils.validators import (  # noqa: E402
    VALID_CLAIM_STATUS,
    VALID_ISSUE_TYPES,
    VALID_SEVERITIES,
    validate_enum,
    validate_object_part,
    validate_risk_flags,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy A — Monolithic single-prompt baseline
# ---------------------------------------------------------------------------

_STRATEGY_A_SYSTEM = (
    "You are an insurance claim verifier. "
    "Return ONLY valid JSON. No markdown. No code blocks."
)

_STRATEGY_A_USER = """Analyze this damage claim and return a JSON verdict.

Object: {claim_object}
Claim: {user_claim}

Return JSON with EXACTLY these fields (no others):
{{
  "claim_status": "<supported|contradicted|not_enough_information>",
  "issue_type": "<dent|scratch|crack|glass_shatter|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown>",
  "object_part": "<specific part, e.g. front_bumper, screen, box — or unknown>",
  "severity": "<none|low|medium|high|unknown>",
  "evidence_standard_met": "<true|false>",
  "valid_image": "<true|false>",
  "risk_flags": "<semicolon-separated flags from: blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, user_history_risk, manual_review_required — or none>",
  "claim_status_justification": "<one sentence>",
  "evidence_standard_met_reason": "<one sentence>",
  "supporting_image_ids": "<semicolon-separated image IDs or none>"
}}"""


def _parse_strategy_a_response(
    raw: dict,
    row: pd.Series,
    first_image_id: str,
) -> OutputRow:
    """Map the monolithic LLM response to a validated OutputRow."""
    claim_object = str(row.get("claim_object", "")).lower().strip()

    claim_status = validate_enum(
        raw.get("claim_status", "not_enough_information"),
        VALID_CLAIM_STATUS,
        "not_enough_information",
    )
    issue_type = validate_enum(
        raw.get("issue_type", "unknown"), VALID_ISSUE_TYPES, "unknown"
    )
    object_part = validate_object_part(raw.get("object_part", "unknown"), claim_object)
    severity = validate_enum(raw.get("severity", "unknown"), VALID_SEVERITIES, "unknown")

    evidence_met_raw = str(raw.get("evidence_standard_met", "false")).strip().lower()
    evidence_met = "true" if evidence_met_raw == "true" else "false"
    valid_image_raw = str(raw.get("valid_image", "false")).strip().lower()
    valid_image = "true" if valid_image_raw == "true" else "false"

    raw_flags = str(raw.get("risk_flags", "none"))
    flags_list = [f.strip().lower() for f in raw_flags.split(";")]
    risk_flags_str = validate_risk_flags(flags_list)

    supporting_raw = str(raw.get("supporting_image_ids", "none")).strip()
    supporting = supporting_raw if supporting_raw else "none"

    return OutputRow(
        user_id=str(row.get("user_id", "")),
        image_paths=str(row.get("image_paths", "")),
        user_claim=str(row.get("user_claim", "")),
        claim_object=claim_object,
        evidence_standard_met=evidence_met,
        evidence_standard_met_reason=str(raw.get("evidence_standard_met_reason", ""))[:500],
        risk_flags=risk_flags_str,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=str(raw.get("claim_status_justification", ""))[:1000],
        supporting_image_ids=supporting,
        valid_image=valid_image,
        severity=severity,
    )


def run_strategy_a(
    sample_df: pd.DataFrame,
    client: GeminiClient,
) -> pd.DataFrame:
    """Strategy A: one monolithic Gemini call per claim (no image in text-only mode).

    Uses the first valid image if available via vision; otherwise text-only.
    No decomposition, no deterministic layers.
    """
    results: list[OutputRow] = []
    total = len(sample_df)

    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        logger.info("Strategy A: claim %d/%d", i, total)
        user_id = str(row.get("user_id", "")).strip()
        image_paths_raw = str(row.get("image_paths", ""))
        user_claim = str(row.get("user_claim", ""))
        claim_object = str(row.get("claim_object", "")).lower().strip()

        try:
            image_paths = load_image_paths(image_paths_raw)
            first_image_id = get_image_id(image_paths[0]) if image_paths else "none"

            # Build prompt
            prompt_text = _STRATEGY_A_USER.format(
                claim_object=claim_object,
                user_claim=user_claim,
            )

            # Try to include the first image if loadable
            b64_result = None
            if image_paths:
                b64_result = load_image_as_base64(image_paths[0])

            if b64_result:
                image_base64, mime_type = b64_result
                import base64
                contents = [
                    _STRATEGY_A_SYSTEM,
                    {"mime_type": mime_type, "data": base64.b64decode(image_base64)},
                    prompt_text,
                ]
            else:
                contents = [_STRATEGY_A_SYSTEM + "\n\n" + prompt_text]

            raw_text = client._call_with_retry(contents)
            raw_dict = client._extract_json(raw_text) if raw_text else None

            if raw_dict:
                output_row = _parse_strategy_a_response(raw_dict, row, first_image_id)
            else:
                logger.warning("Strategy A: no JSON for user %s", user_id)
                output_row = create_error_output_row(
                    user_id, image_paths_raw, user_claim, claim_object,
                    "Strategy A: no JSON returned"
                )

        except Exception as exc:
            logger.exception("Strategy A failed for user %s: %s", user_id, exc)
            output_row = create_error_output_row(
                user_id, image_paths_raw, user_claim, claim_object, str(exc)
            )

        results.append(output_row)

    columns = OutputRow.get_columns()
    return pd.DataFrame([vars(r) for r in results], columns=columns)


# ---------------------------------------------------------------------------
# Strategy B — Full 8-stage decomposed pipeline
# ---------------------------------------------------------------------------

def run_strategy_b(
    sample_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Strategy B: calls the full 8-stage pipeline via main.main()."""
    import main as pipeline_main  # noqa: E402 (import from code/)
    from config import SAMPLE_CLAIMS_CSV  # noqa: E402

    logger.info("Strategy B: running full pipeline on sample_claims.csv")

    # Run the pipeline (passes limit if set)
    output_df = pipeline_main.main(
        input_csv_path=str(SAMPLE_CLAIMS_CSV),
        output_csv_path=output_path,
    )
    return output_df


# ---------------------------------------------------------------------------
# Strategy C — Observe-First Pipeline
# ---------------------------------------------------------------------------

def run_strategy_c(
    sample_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Strategy C: full pipeline with observe-first VisionAnalyzerC.

    The vision model receives NO claim context and freely describes what it
    observes. Damage-type matching is performed by deterministic semantic
    similarity code, eliminating anchoring bias.
    """
    import sys
    from pathlib import Path
    CODE_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(CODE_DIR))

    from models.gemini_client import GeminiClient
    from models.schemas import OutputRow, create_error_output_row
    from pipeline.claim_extractor import ClaimExtractor
    from pipeline.decision_engine import DecisionEngine
    from pipeline.evidence_checker import EvidenceChecker
    from pipeline.image_aggregator import ImageAggregator
    from pipeline.image_quality_checker import ImageQualityChecker
    from pipeline.output_formatter import OutputFormatter
    from pipeline.risk_scorer import RiskScorer
    from pipeline.vision_analyzer_c import VisionAnalyzerC
    from utils.csv_loader import DataStore
    from utils.image_utils import load_image_paths

    client = GeminiClient()
    data_store = DataStore()
    quality_checker = ImageQualityChecker()
    claim_extractor = ClaimExtractor(client)
    evidence_checker = EvidenceChecker(data_store)
    risk_scorer = RiskScorer(data_store)
    vision_analyzer_c = VisionAnalyzerC(client)
    image_aggregator = ImageAggregator()
    decision_engine = DecisionEngine()
    formatter = OutputFormatter()

    results: list[OutputRow] = []
    total = len(sample_df)

    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        logger.info("Strategy C: claim %d/%d", i, total)
        user_id = str(row.get("user_id", "")).strip()
        image_paths_raw = str(row.get("image_paths", ""))
        user_claim = str(row.get("user_claim", ""))
        claim_object = str(row.get("claim_object", "")).lower().strip()

        try:
            image_paths = load_image_paths(image_paths_raw)
            quality_results = quality_checker.check_all_images(image_paths)
            structured_claim = claim_extractor.extract(user_claim, claim_object)
            evidence_standard = evidence_checker.check(
                claim_object, structured_claim.claimed_issue_type, quality_results
            )
            user_risk = risk_scorer.score(user_id)

            # Strategy C: observe-first vision analysis
            vision_analyses = vision_analyzer_c.analyze_all_images(
                image_paths=image_paths,
                quality_results=quality_results,
                claim_object=claim_object,
                claimed_issue_type=structured_claim.claimed_issue_type,
                claimed_object_part=structured_claim.claimed_object_part,
                claim_summary=structured_claim.claim_summary,
            )

            aggregated = image_aggregator.aggregate(vision_analyses, claim_object)
            decision = decision_engine.decide(
                structured_claim, evidence_standard, user_risk,
                aggregated, quality_results, claim_object,
            )
            output_row = formatter.format(row, decision)

        except Exception as exc:
            logger.exception("Strategy C failed for user %s: %s", user_id, exc)
            output_row = create_error_output_row(
                user_id, image_paths_raw, user_claim, claim_object, str(exc)
            )

        results.append(output_row)

    columns = OutputRow.get_columns()
    result_df = pd.DataFrame([vars(r) for r in results], columns=columns)

    from pathlib import Path as _Path
    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    return result_df

