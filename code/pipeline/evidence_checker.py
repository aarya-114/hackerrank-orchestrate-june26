from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.schemas import EvidenceStandard, ImageQualityResult  # noqa: E402
from utils.csv_loader import DataStore  # noqa: E402


class EvidenceChecker:
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.requirements_df = data_store.get_all_requirements()

    def _get_issue_family(self, issue_type: str) -> str:
        issue_family_map = {
            "dent": "dent or scratch",
            "scratch": "dent or scratch",
            "crack": "crack or break",
            "glass_shatter": "crack or break",
            "broken_part": "crack or break",
            "missing_part": "crack or break",
            "torn_packaging": "packaging damage",
            "crushed_packaging": "packaging damage",
            "water_damage": "water damage",
            "stain": "water damage",
            "unknown": "general",
            "none": "general",
        }
        return issue_family_map.get(str(issue_type).strip().lower(), "general")

    def check(
        self,
        claim_object: str,
        claimed_issue_type: str,
        image_quality_results: list[ImageQualityResult],
    ) -> EvidenceStandard:
        issue_family = self._get_issue_family(claimed_issue_type)
        requirements = self.data_store.get_evidence_requirements(
            claim_object, issue_family
        )
        images_provided = len(image_quality_results)
        valid_images_provided = sum(
            1 for result in image_quality_results if result.is_valid
        )

        if not requirements:
            standard_met = valid_images_provided >= 1
            reason = "No specific requirements; 1+ valid image sufficient"
        elif images_provided == 0:
            standard_met = False
            reason = "No images submitted"
        elif valid_images_provided == 0:
            standard_met = False
            reason = "All submitted images failed quality check"
        else:
            standard_met = True
            reason = f"Valid image(s) submitted for {issue_family} claim"

        return EvidenceStandard(
            claim_object=claim_object,
            issue_family=issue_family,
            requirements=requirements,
            images_provided=images_provided,
            valid_images_provided=valid_images_provided,
            standard_met=standard_met,
            reason=reason,
        )


if __name__ == "__main__":
    data_store = DataStore()
    checker = EvidenceChecker(data_store)

    one_valid_image = [
        ImageQualityResult(
            image_id="img_1",
            image_path="images/test/case_001/img_1.jpg",
            is_valid=True,
            is_readable=True,
        )
    ]
    zero_valid_images = [
        ImageQualityResult(
            image_id="img_1",
            image_path="images/test/case_001/img_1.jpg",
            is_valid=False,
            is_readable=True,
            quality_flags=["blurry_image"],
        )
    ]

    valid_result = checker.check("car", "dent", one_valid_image)
    invalid_result = checker.check("car", "dent", zero_valid_images)

    print(valid_result)
    print(invalid_result)

    assert valid_result.standard_met is True
    assert invalid_result.standard_met is False
    print("EVIDENCE CHECKER OK")
