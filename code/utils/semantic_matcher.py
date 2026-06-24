"""Semantic matching utilities for Strategy C observe-first pipeline.

Pure stdlib — no external AI library dependencies.
Provides vocabulary-based damage type matching and object part detection.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Damage family vocabulary
# ---------------------------------------------------------------------------

DAMAGE_FAMILIES: dict[str, list[str]] = {
    "dent": [
        "dent", "dented", "deformation", "deformed", "indent",
        "indentation", "dimple", "depression", "buckle", "buckled",
        "crumple", "crumpled", "impact", "collision damage",
    ],
    "scratch": [
        "scratch", "scratched", "scuff", "scuffed", "mark", "marks",
        "abrasion", "scrape", "scraped", "surface damage", "paint damage",
        "paint loss", "coating damage",
    ],
    "crack": [
        "crack", "cracked", "fracture", "fractured", "split", "fissure",
        "line", "hairline", "break", "broken", "shatter", "shattered",
        "spider", "spiderweb",
    ],
    "glass_shatter": [
        "shatter", "shattered", "broken glass", "glass break",
        "windshield damage", "screen shatter", "fragments", "glass pieces",
    ],
    "broken_part": [
        "broken", "broke", "snapped", "snapped off", "detached",
        "separated", "missing piece", "piece missing", "component broken",
    ],
    "missing_part": [
        "missing", "absent", "not present", "gone", "removed",
        "detached", "fallen off", "no longer attached",
    ],
    "torn_packaging": [
        "torn", "tear", "rip", "ripped", "puncture", "punctured",
        "hole", "opening", "damaged packaging", "box damage",
        "cardboard damage",
    ],
    "crushed_packaging": [
        "crushed", "crush", "compressed", "flattened", "collapsed",
        "squashed", "deformed box", "crushed box",
    ],
    "water_damage": [
        "water", "wet", "moisture", "liquid", "soaked", "damp",
        "flood", "leak", "rust", "corrosion", "oxidation",
    ],
    "stain": [
        "stain", "stained", "discolor", "discoloration", "mark",
        "blemish", "spot", "spill", "contamination",
    ],
}

SEVERITY_MAP: dict[str, str] = {
    "none": "none",
    "minor": "low",
    "moderate": "medium",
    "severe": "high",
    "unknown": "unknown",
}

PART_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "car": {
        "front_bumper": ["front bumper", "bumper front", "front end"],
        "rear_bumper": ["rear bumper", "back bumper", "rear end"],
        "door": ["door", "car door", "side door", "passenger door", "driver door"],
        "hood": ["hood", "bonnet", "engine cover"],
        "windshield": ["windshield", "windscreen", "front glass", "front window"],
        "side_mirror": ["mirror", "side mirror", "wing mirror"],
        "headlight": ["headlight", "headlamp", "front light"],
        "taillight": ["taillight", "tail light", "rear light", "brake light"],
        "fender": ["fender", "wing panel", "wheel arch"],
        "quarter_panel": ["quarter panel", "rear panel", "side panel"],
        "body": ["body", "panel", "exterior", "side"],
    },
    "laptop": {
        "screen": ["screen", "display", "monitor", "lcd", "panel"],
        "keyboard": ["keyboard", "keys", "keypad"],
        "trackpad": ["trackpad", "touchpad"],
        "hinge": ["hinge", "joint", "connector"],
        "lid": ["lid", "top cover", "screen cover", "back cover"],
        "corner": ["corner", "edge"],
        "port": ["port", "usb", "hdmi", "jack", "connector"],
        "base": ["base", "bottom", "underside"],
        "body": ["body", "chassis", "casing", "shell"],
    },
    "package": {
        "box": ["box", "carton", "container"],
        "package_corner": ["corner", "edge"],
        "package_side": ["side", "face", "panel"],
        "seal": ["seal", "tape", "closure"],
        "label": ["label", "sticker", "address"],
        "contents": ["contents", "inside", "item inside"],
        "item": ["item", "product", "goods"],
    },
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def match_damage_type(
    observation_text: str,
    claimed_issue_type: str,
) -> tuple[bool, float]:
    """Return (matches, confidence_score) for the claimed damage type.

    Score:
      1.0  — two or more keyword matches (strong)
      0.7  — exactly one keyword match (moderate)
      0.3  — no claimed-type match but some other damage vocabulary found
      0.0  — no damage vocabulary at all
    """
    observation_lower = observation_text.lower()

    claimed_keywords = DAMAGE_FAMILIES.get(claimed_issue_type, [claimed_issue_type])
    matches = sum(1 for kw in claimed_keywords if kw in observation_lower)

    if matches >= 2:
        return True, 1.0
    if matches == 1:
        return True, 0.7

    # Check if ANY damage type matches (some damage, wrong type)
    for keywords in DAMAGE_FAMILIES.values():
        for kw in keywords:
            if kw in observation_lower:
                return False, 0.3

    return False, 0.0


def match_object_part(
    observation_text: str,
    claim_object: str,
) -> str:
    """Return the best-matching object part name for the given observation."""
    observation_lower = observation_text.lower()
    parts = PART_KEYWORDS.get(claim_object, {})

    best_part = "unknown"
    best_count = 0

    for part_name, keywords in parts.items():
        count = sum(1 for kw in keywords if kw in observation_lower)
        if count > best_count:
            best_count = count
            best_part = part_name

    return best_part


def map_severity(observed_severity: str) -> str:
    """Map free-text observed severity to a validated severity enum value."""
    return SEVERITY_MAP.get(observed_severity.lower(), "unknown")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test match_damage_type
    r1 = match_damage_type("visible dent on the door panel", "dent")
    assert r1[0] is True and r1[1] >= 0.7, f"Expected (True, >=0.7) got {r1}"
    print(f"  dent match: {r1}")

    r2 = match_damage_type("large crack running across screen", "crack")
    assert r2[0] is True and r2[1] >= 0.7, f"Expected (True, >=0.7) got {r2}"
    print(f"  crack match: {r2}")

    r3 = match_damage_type("no damage visible", "dent")
    assert r3[0] is False and r3[1] == 0.0, f"Expected (False, 0.0) got {r3}"
    print(f"  no-damage match: {r3}")

    # Test match_object_part
    r4 = match_object_part("damage on the front bumper area", "car")
    assert r4 == "front_bumper", f"Expected 'front_bumper' got '{r4}'"
    print(f"  part match: {r4}")

    # Test map_severity
    assert map_severity("severe") == "high"
    assert map_severity("minor") == "low"
    assert map_severity("moderate") == "medium"
    assert map_severity("none") == "none"
    print("  severity mappings: OK")

    print("SEMANTIC MATCHER OK")
