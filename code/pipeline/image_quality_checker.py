from pathlib import Path
import sys

import numpy as np
from PIL import ImageFilter


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CLAIMS_CSV, Config  # noqa: E402
from models.schemas import ImageQualityResult  # noqa: E402
from utils.image_utils import get_image_id, load_image_as_pil, load_image_paths  # noqa: E402
from utils.validators import VALID_RISK_FLAGS  # noqa: E402


class ImageQualityChecker:
    def check_single_image(self, image_path: str) -> ImageQualityResult:
        image_id = get_image_id(image_path)
        pil_image = load_image_as_pil(image_path)

        if pil_image is None:
            return ImageQualityResult(
                image_id=image_id,
                image_path=image_path,
                is_valid=False,
                is_readable=False,
                quality_flags=["damage_not_visible"],
                quality_notes="Image file unreadable or missing",
            )

        width, height = pil_image.size
        quality_flags = []

        if (
            width < Config.IMAGE_MIN_DIMENSION
            or height < Config.IMAGE_MIN_DIMENSION
        ):
            quality_flags.append("cropped_or_obstructed")

        gray = pil_image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges, dtype=float)
        blur_score = float(edge_arr.var())

        if blur_score < Config.BLUR_THRESHOLD:
            quality_flags.append("blurry_image")

        gray_arr = np.array(gray, dtype=float)
        brightness = float(gray_arr.mean())

        if (
            brightness < Config.BRIGHTNESS_MIN
            or brightness > Config.BRIGHTNESS_MAX
        ):
            quality_flags.append("low_light_or_glare")

        quality_flags = [
            flag for flag in dict.fromkeys(quality_flags) if flag in VALID_RISK_FLAGS
        ]
        is_valid = (
            "cropped_or_obstructed" not in quality_flags
            and not ("blurry_image" in quality_flags and blur_score < 50.0)
        )

        notes = "Image readable"
        if quality_flags:
            notes = f"Image readable with quality flags: {';'.join(quality_flags)}"

        return ImageQualityResult(
            image_id=image_id,
            image_path=image_path,
            is_valid=is_valid,
            is_readable=True,
            width=width,
            height=height,
            blur_score=blur_score,
            brightness=brightness,
            quality_flags=quality_flags,
            quality_notes=notes,
        )

    def check_all_images(self, image_paths: list[str]) -> list[ImageQualityResult]:
        return [self.check_single_image(image_path) for image_path in image_paths]

    def any_valid(self, results: list[ImageQualityResult]) -> bool:
        return any(result.is_valid for result in results)

    def get_valid_paths(
        self, image_paths: list[str], results: list[ImageQualityResult]
    ) -> list[str]:
        return [
            image_path
            for image_path, result in zip(image_paths, results)
            if result.is_valid
        ]


def _get_first_claim_image_paths() -> list[str]:
    first_data_line = CLAIMS_CSV.read_text(encoding="utf-8").splitlines()[1]
    raw_image_paths = first_data_line.split('","')[1]
    return load_image_paths(raw_image_paths)


if __name__ == "__main__":
    checker = ImageQualityChecker()
    image_paths = _get_first_claim_image_paths()
    results = checker.check_all_images(image_paths)

    for result in results:
        print(result)

    print("QUALITY CHECK OK")
