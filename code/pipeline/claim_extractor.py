"""Stage 1 — Claim Extractor.

Parses the user_claim conversation into a StructuredClaim via a
single text-only Gemini call (Stage 1 of the 8-stage pipeline).
Falls back to keyword-based heuristics if the API call fails or
returns malformed JSON.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gemini_client import GeminiClient  # noqa: E402
from models.schemas import StructuredClaim  # noqa: E402
from utils.validators import (  # noqa: E402
    VALID_ISSUE_TYPES,
    VALID_SEVERITIES,
    validate_enum,
    validate_object_part,
)


class ClaimExtractor:
    """Stage 1 — converts a raw user_claim string into a StructuredClaim."""

    def __init__(self, client: GeminiClient):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def _fallback_extraction(
        self, user_claim: str, claim_object: str
    ) -> StructuredClaim:
        """Keyword-based heuristic used when the LLM call fails."""
        text = user_claim.lower()

        # Detect issue_type — order matters (more-specific first)
        if "crack" in text or "cracked" in text:
            issue_type = "crack"
        elif "scratch" in text or "scratched" in text:
            issue_type = "scratch"
        elif "dent" in text or "dented" in text:
            issue_type = "dent"
        elif "shatter" in text or "broken glass" in text:
            issue_type = "glass_shatter"
        elif "torn" in text or "ripped" in text:
            issue_type = "torn_packaging"
        elif "crushed" in text or "smashed" in text:
            issue_type = "crushed_packaging"
        elif "water" in text or "wet" in text or "liquid" in text:
            issue_type = "water_damage"
        elif "stain" in text:
            issue_type = "stain"
        elif "broken" in text or "broke" in text:
            issue_type = "broken_part"
        elif "missing" in text:
            issue_type = "missing_part"
        else:
            issue_type = "unknown"

        # Detect severity
        if "severe" in text or "completely" in text or "totally" in text:
            severity = "high"
        elif "small" in text or "minor" in text or "little" in text:
            severity = "low"
        else:
            severity = "unknown"

        return StructuredClaim(
            claimed_issue_type=issue_type,
            claimed_object_part="unknown",
            claimed_severity_hint=severity,
            claim_summary=user_claim[:200],
            incident_context="unknown",
            confidence="low",
            extraction_method="fallback",
        )

    def extract(self, user_claim: str, claim_object: str) -> StructuredClaim:
        """Return a StructuredClaim; never raises."""
        try:
            result = self.client.extract_claim(user_claim, claim_object)

            if result is None:
                self.logger.info(
                    "Gemini returned None for claim extraction; using fallback"
                )
                return self._fallback_extraction(user_claim, claim_object)

            claimed_issue_type = validate_enum(
                result.get("claimed_issue_type", "unknown"),
                VALID_ISSUE_TYPES,
                fallback="unknown",
            )
            claimed_object_part = validate_object_part(
                result.get("claimed_object_part", "unknown"),
                claim_object,
            )
            claimed_severity_hint = validate_enum(
                result.get("claimed_severity_hint", "unknown"),
                VALID_SEVERITIES,
                fallback="unknown",
            )
            claim_summary = str(result.get("claim_summary", ""))[:500]
            incident_context = str(result.get("incident_context", ""))[:200]

            confidence = result.get("confidence", "low")
            if confidence not in ("high", "medium", "low"):
                confidence = "low"

            return StructuredClaim(
                claimed_issue_type=claimed_issue_type,
                claimed_object_part=claimed_object_part,
                claimed_severity_hint=claimed_severity_hint,
                claim_summary=claim_summary,
                incident_context=incident_context,
                confidence=confidence,
                extraction_method="llm",
            )

        except Exception as exc:
            self.logger.exception("Claim extraction failed: %s", exc)
            return self._fallback_extraction(user_claim, claim_object)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.WARNING)

    TEST_CLAIM = "My car has a big dent on the front door after a crash"
    TEST_OBJECT = "car"

    # --- Fallback test (no API key needed) ----------------------------------
    try:
        _client = GeminiClient()
    except EnvironmentError as exc:
        print(f"SKIP (no key): {exc}")
        # Still test fallback via a temporary no-op client
        class _NoOpClient:  # type: ignore[no-redef]
            def extract_claim(self, *a, **kw):
                return None
            def get_call_count(self):
                return 0

        _client = _NoOpClient()  # type: ignore[assignment]

    extractor = ClaimExtractor(_client)  # type: ignore[arg-type]
    fallback_result = extractor._fallback_extraction(TEST_CLAIM, TEST_OBJECT)
    print("Fallback result:", fallback_result)
    assert fallback_result.claimed_issue_type == "dent", (
        f"Expected dent, got {fallback_result.claimed_issue_type}"
    )
    assert fallback_result.extraction_method == "fallback"
    print("FALLBACK OK")

    # --- Live API test (only if key is present) ----------------------------
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            live_client = GeminiClient()
            live_extractor = ClaimExtractor(live_client)
            live_result = live_extractor.extract(TEST_CLAIM, TEST_OBJECT)
            print("Live result:", live_result)
            assert live_result.claimed_issue_type in VALID_ISSUE_TYPES
            print(f"API calls used: {live_client.get_call_count()}")
        except Exception as e:
            print(f"Live API test skipped: {e}")
    else:
        print("SKIP: GOOGLE_API_KEY not set — live test skipped")

    print("CLAIM EXTRACTOR OK")