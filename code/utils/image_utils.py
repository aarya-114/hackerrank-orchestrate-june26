from pathlib import Path
import base64
import io
import sys

from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CLAIMS_CSV, DATASET_DIR  # noqa: E402


def load_image_paths(raw_paths: str) -> list[str]:
    return [path.strip() for path in str(raw_paths).split(";") if path.strip()]


def get_image_id(image_path: str) -> str:
    return Path(image_path).stem


def get_image_ids(image_paths: list[str]) -> list[str]:
    return [get_image_id(image_path) for image_path in image_paths]


def image_path_to_absolute(image_path: str) -> Path:
    repo_root = Path(__file__).parent.parent.parent
    repo_relative_path = (repo_root / image_path).resolve()
    if repo_relative_path.exists():
        return repo_relative_path

    dataset_relative_path = (DATASET_DIR / image_path).resolve()
    if dataset_relative_path.exists():
        return dataset_relative_path

    return repo_relative_path


def load_image_as_pil(image_path: str) -> Image.Image | None:
    absolute_path = image_path_to_absolute(image_path)
    if not absolute_path.exists():
        return None

    try:
        with Image.open(absolute_path) as image:
            loaded_image = image.copy()
    except (OSError, ValueError):
        return None

    if loaded_image.mode in {"RGBA", "P"}:
        loaded_image = loaded_image.convert("RGB")

    return loaded_image


def load_image_as_base64(image_path: str) -> tuple[str, str] | None:
    absolute_path = image_path_to_absolute(image_path)
    if not absolute_path.exists():
        return None

    try:
        image_bytes = absolute_path.read_bytes()
    except OSError:
        return None

    suffix = absolute_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")

    return base64.b64encode(image_bytes).decode("utf-8"), mime_type


def _get_first_claim_image_paths() -> list[str]:
    first_data_line = CLAIMS_CSV.read_text(encoding="utf-8").splitlines()[1]
    raw_image_paths = first_data_line.split('","')[1]
    return load_image_paths(raw_image_paths)


if __name__ == "__main__":
    first_image_path = _get_first_claim_image_paths()[0]
    pil_image = load_image_as_pil(first_image_path)
    base64_result = load_image_as_base64(first_image_path)

    if pil_image is None or base64_result is None:
        raise RuntimeError(f"Unable to load image: {first_image_path}")

    encoded_image, mime_type = base64_result
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")

    print(f"image_id: {get_image_id(first_image_path)}")
    print(f"dimensions: {pil_image.width}x{pil_image.height}")
    print(f"base64: {encoded_image[:30]} {mime_type}")
    print("IMAGE UTILS OK")
