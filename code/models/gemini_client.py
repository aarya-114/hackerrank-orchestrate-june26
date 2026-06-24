"""
Groq-backed API client.
Keeps GeminiClient class name and all method signatures
so no other module requires changes.
"""

import os
import time
import json
import logging
from dotenv import load_dotenv
from groq import Groq
from models.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    VISION_ANALYSIS_SYSTEM,
    VISION_ANALYSIS_USER,
    JSON_REPAIR_PROMPT,
)
from config import Config

logger = logging.getLogger("gemini_client")


class GeminiClient:

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file. "
                "Get a free key at console.groq.com"
            )
        self._client = Groq(api_key=api_key)
        self._last_call_time: float = 0.0
        self._call_count: int = 0

    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < Config.MIN_CALL_INTERVAL:
            time.sleep(Config.MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()
        self._call_count += 1

    @staticmethod
    def _normalize_messages(messages: list) -> tuple[list, str]:
        """Convert legacy Gemini parts list into Groq chat messages.

        compare_strategies.py passes either:
          - [str]                              (text-only)
          - [str, {"mime_type":..,"data":bytes}, str]  (vision)

        Both need to become Groq-compatible message dicts.
        Returns (groq_messages, chosen_model).
        """
        # Already looks like proper chat messages (list of dicts with "role")
        if messages and isinstance(messages[0], dict) and "role" in messages[0]:
            # Detect whether any message contains an image_url → use vision model
            has_image = any(
                isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for m in messages
                if isinstance(m, dict)
            )
            model = Config.VISION_MODEL if has_image else Config.GEMINI_MODEL
            return messages, model

        # Legacy Gemini parts list — rebuild as Groq messages
        import base64
        text_parts: list[str] = []
        image_part: dict | None = None

        for part in messages:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                # Gemini inline image: {"mime_type": "image/jpeg", "data": bytes}
                raw_data = part.get("data", b"")
                mime = part.get("mime_type", "image/jpeg")
                b64 = base64.b64encode(raw_data).decode() if isinstance(raw_data, bytes) else raw_data
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }

        combined_text = "\n\n".join(text_parts)

        if image_part:
            content = [{"type": "text", "text": combined_text}, image_part]
            model = Config.VISION_MODEL
        else:
            content = combined_text
            model = Config.GEMINI_MODEL

        return [{"role": "user", "content": content}], model

    def _call_with_retry(
        self,
        messages: list,
        model: str = Config.GEMINI_MODEL,
        max_retries: int = Config.MAX_RETRIES,
    ) -> str:
        # Normalise legacy Gemini parts format if needed; may override model
        normalized, detected_model = self._normalize_messages(messages)
        # Only override model if the caller used the default (i.e., didn't pass one explicitly)
        if model == Config.GEMINI_MODEL:
            model = detected_model

        for attempt in range(1, max_retries + 1):
            self._rate_limit_wait()
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=normalized,
                    max_tokens=1000,
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                is_rate_limit = (
                    "429" in err
                    or "rate" in err.lower()
                    or "quota" in err.lower()
                )
                if is_rate_limit and attempt < max_retries:
                    delay = min(
                        (2 ** attempt) * Config.RETRY_BASE_DELAY,
                        Config.MAX_RETRY_DELAY,
                    )
                    logger.warning(
                        f"Groq rate limit on attempt {attempt}/{max_retries}; "
                        f"sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Groq API call failed on attempt "
                        f"{attempt}/{max_retries}: {e}"
                    )
                    if attempt == max_retries:
                        return ""
        return ""

    def _extract_json(self, raw_text: str) -> dict | None:
        if not raw_text:
            return None
        try:
            return json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass
        try:
            start = raw_text.find("```")
            if start != -1:
                end = raw_text.rfind("```")
                if end > start:
                    inner = raw_text[start + 3:end].strip()
                    if inner.startswith("json"):
                        inner = inner[4:].strip()
                    return json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end > start:
                return json.loads(raw_text[start:end + 1])
        except json.JSONDecodeError:
            pass
        return None

    def extract_claim(
        self,
        user_claim: str,
        claim_object: str,
    ) -> dict | None:
        prompt = (
            CLAIM_EXTRACTION_SYSTEM
            + "\n\n"
            + CLAIM_EXTRACTION_USER.format(
                user_claim=user_claim,
                claim_object=claim_object,
            )
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self._call_with_retry(messages, model=Config.GEMINI_MODEL)
        return self._extract_json(raw)

    def analyze_image(
        self,
        image_base64: str,
        mime_type: str,
        claim_object: str,
        claimed_issue_type: str,
        claimed_object_part: str,
        claim_summary: str,
        allowed_parts: str,
    ) -> dict | None:
        text_prompt = (
            VISION_ANALYSIS_SYSTEM
            + "\n\n"
            + VISION_ANALYSIS_USER.format(
                claim_object=claim_object,
                claimed_issue_type=claimed_issue_type,
                claimed_object_part=claimed_object_part,
                claim_summary=claim_summary,
                allowed_parts=allowed_parts,
            )
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            }
        ]
        raw = self._call_with_retry(
            messages, model=Config.VISION_MODEL
        )
        return self._extract_json(raw)

    def repair_json_field(
        self,
        original_json: str,
        field_name: str,
        bad_value: str,
        allowed_values: list[str],
    ) -> dict | None:
        prompt = JSON_REPAIR_PROMPT.format(
            field_name=field_name,
            bad_value=bad_value,
            allowed_values=", ".join(allowed_values),
            original_json=original_json,
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self._call_with_retry(messages, model=Config.GEMINI_MODEL)
        return self._extract_json(raw)

    def get_call_count(self) -> int:
        return self._call_count


if __name__ == "__main__":
    import sys
    try:
        client = GeminiClient()
    except EnvironmentError as e:
        print(f"SKIP: {e}")
        sys.exit(0)

    result = client.extract_claim(
        "My laptop screen cracked after I dropped it.",
        "laptop",
    )
    print("Extracted claim:", result)
    assert result is not None, "Expected a dict result"
    assert "claimed_issue_type" in result
    print(f"API call count: {client.get_call_count()}")
    print("GROQ CLIENT OK")
