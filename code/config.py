from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_CSV = REPO_ROOT / "output.csv"


@dataclass(frozen=True)
class Config:
    GEMINI_MODEL: str = "llama-3.1-8b-instant"
    VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 2.0
    MAX_RETRY_DELAY: float = 30.0
    API_CALL_TIMEOUT: float = 20.0
    CALLS_PER_MINUTE_LIMIT: int = 25
    MIN_CALL_INTERVAL: float = 2.5
    IMAGE_MIN_DIMENSION: int = 50
    BLUR_THRESHOLD: float = 100.0
    BRIGHTNESS_MIN: float = 20.0
    BRIGHTNESS_MAX: float = 235.0
