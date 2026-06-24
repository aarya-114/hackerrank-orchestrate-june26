VALID_CLAIM_STATUS = frozenset(
    {"supported", "contradicted", "not_enough_information"}
)

VALID_ISSUE_TYPES = frozenset(
    {
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
        "stain",
        "none",
        "unknown",
    }
)

VALID_SEVERITIES = frozenset({"none", "low", "medium", "high", "unknown"})

VALID_RISK_FLAGS = frozenset(
    {
        "none",
        "blurry_image",
        "cropped_or_obstructed",
        "low_light_or_glare",
        "wrong_angle",
        "wrong_object",
        "wrong_object_part",
        "damage_not_visible",
        "claim_mismatch",
        "possible_manipulation",
        "non_original_image",
        "text_instruction_present",
        "user_history_risk",
        "manual_review_required",
    }
)

OBJECT_PARTS = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}


def validate_enum(value: str, valid_set: set, fallback: str = "unknown") -> str:
    normalized_value = str(value).strip().lower()
    if normalized_value in valid_set:
        return normalized_value
    return fallback


def validate_object_part(part: str, claim_object: str) -> str:
    normalized_object = str(claim_object).strip().lower()
    valid_parts = OBJECT_PARTS.get(normalized_object)
    if valid_parts is None:
        return "unknown"

    return validate_enum(part, valid_parts)


def validate_risk_flags(flags: list[str]) -> str:
    valid_flags = []
    seen = set()

    for flag in flags:
        normalized_flag = str(flag).strip().lower()
        if normalized_flag in VALID_RISK_FLAGS and normalized_flag not in seen:
            valid_flags.append(normalized_flag)
            seen.add(normalized_flag)

    if not valid_flags:
        return "none"

    if len(valid_flags) > 1 and "none" in seen:
        valid_flags = [flag for flag in valid_flags if flag != "none"]

    return ";".join(valid_flags) if valid_flags else "none"
