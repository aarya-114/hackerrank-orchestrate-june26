"""Main entry point — 8-stage claim verification pipeline.

Reads dataset/claims.csv, processes each row through all pipeline stages,
and writes predictions to output.csv.

Usage:
    python code/main.py                        # claims.csv → output.csv
    python code/main.py --sample               # sample_claims.csv → output_sample.csv
    python code/main.py --limit N              # first N rows only (quick test)
    python code/main.py --input PATH --output PATH
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from config import CLAIMS_CSV, OUTPUT_CSV, SAMPLE_CLAIMS_CSV  # noqa: E402
from models.gemini_client import GeminiClient  # noqa: E402
from models.schemas import OutputRow, create_error_output_row  # noqa: E402
from pipeline.claim_extractor import ClaimExtractor  # noqa: E402
from pipeline.decision_engine import DecisionEngine  # noqa: E402
from pipeline.evidence_checker import EvidenceChecker  # noqa: E402
from pipeline.image_aggregator import ImageAggregator  # noqa: E402
from pipeline.image_quality_checker import ImageQualityChecker  # noqa: E402
from pipeline.output_formatter import OutputFormatter  # noqa: E402
from pipeline.risk_scorer import RiskScorer  # noqa: E402
from pipeline.vision_analyzer import VisionAnalyzer  # noqa: E402
from pipeline.vision_analyzer_c import VisionAnalyzerC  # noqa: E402
from utils.csv_loader import DataStore  # noqa: E402
from utils.image_utils import load_image_paths  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


def process_claim(
    row: pd.Series,
    client: GeminiClient,
    data_store: DataStore,
    quality_checker: ImageQualityChecker,
    claim_extractor: ClaimExtractor,
    evidence_checker: EvidenceChecker,
    risk_scorer: RiskScorer,
    vision_analyzer,  # VisionAnalyzer or VisionAnalyzerC
    image_aggregator: ImageAggregator,
    decision_engine: DecisionEngine,
    formatter: OutputFormatter,
) -> OutputRow:
    """Run all 8 pipeline stages for one claim row; never raises."""

    user_id = str(row.get("user_id", "")).strip()
    image_paths_raw = str(row.get("image_paths", ""))
    user_claim = str(row.get("user_claim", ""))
    claim_object = str(row.get("claim_object", "")).lower().strip()

    try:
        # Stage 1 — parse image path list
        image_paths = load_image_paths(image_paths_raw)

        # Stage 2 — image quality check (deterministic)
        quality_results = quality_checker.check_all_images(image_paths)

        # Stage 1 — claim extraction (Gemini text call)
        structured_claim = claim_extractor.extract(user_claim, claim_object)
        logger.info(
            "Claim extracted: %s on %s",
            structured_claim.claimed_issue_type,
            structured_claim.claimed_object_part,
        )

        # Stage 3 — evidence requirements (deterministic)
        evidence_standard = evidence_checker.check(
            claim_object,
            structured_claim.claimed_issue_type,
            quality_results,
        )

        # Stage 4 — risk scoring (deterministic)
        user_risk = risk_scorer.score(user_id)

        # Stage 5 — vision analysis
        # VisionAnalyzer: pass structured_claim for anchored analysis
        # VisionAnalyzerC: pass individual fields for observe-first analysis
        if isinstance(vision_analyzer, VisionAnalyzerC):
            vision_analyses = vision_analyzer.analyze_all_images(
                image_paths=image_paths,
                quality_results=quality_results,
                claim_object=claim_object,
                claimed_issue_type=structured_claim.claimed_issue_type,
                claimed_object_part=structured_claim.claimed_object_part,
                claim_summary=structured_claim.claim_summary,
            )
        else:
            vision_analyses = vision_analyzer.analyze_all_images(
                image_paths, quality_results, structured_claim, claim_object
            )
        logger.info("Analyzed %d images", len(vision_analyses))

        # Stage 6 — aggregate vision results (deterministic)
        aggregated = image_aggregator.aggregate(vision_analyses, claim_object)

        # Stage 7 — decision engine (deterministic)
        decision = decision_engine.decide(
            structured_claim,
            evidence_standard,
            user_risk,
            aggregated,
            quality_results,
            claim_object,
        )
        logger.info("Decision: %s", decision.claim_status)

        # Stage 8 — format output row
        return formatter.format(row, decision)

    except Exception as exc:
        logger.exception(
            "Unhandled error processing claim for user %s: %s", user_id, exc
        )
        return create_error_output_row(
            user_id, image_paths_raw, user_claim, claim_object, str(exc)
        )


def main(
    input_csv_path: str | None = None,
    output_csv_path: str | None = None,
) -> pd.DataFrame:
    """Process all rows in *input_csv_path* and write *output_csv_path*.

    Returns the output DataFrame so the evaluation module can call this
    directly with sample_claims.csv.
    """
    if input_csv_path is None:
        input_csv_path = str(CLAIMS_CSV)
    if output_csv_path is None:
        output_csv_path = str(OUTPUT_CSV)

    logger.info("Starting Multi-Modal Evidence Review Pipeline")
    logger.info("Input:  %s", input_csv_path)
    logger.info("Output: %s", output_csv_path)

    # Strategy selection (set via module attribute or "c" positional arg)
    strategy = getattr(main, "_strategy", "b")

    # 1. Initialise all components
    data_store = DataStore()
    client = GeminiClient()
    quality_checker = ImageQualityChecker()
    claim_extractor = ClaimExtractor(client)
    evidence_checker = EvidenceChecker(data_store)
    risk_scorer = RiskScorer(data_store)
    if strategy == "c":
        vision_analyzer: VisionAnalyzer | VisionAnalyzerC = VisionAnalyzerC(client)
        logger.info("Using Strategy C (observe-first vision analyzer)")
    else:
        vision_analyzer = VisionAnalyzer(client)
        logger.info("Using Strategy B (anchored vision analyzer)")
    image_aggregator = ImageAggregator()
    decision_engine = DecisionEngine()
    formatter = OutputFormatter()

    # 2. Load input CSV
    df = pd.read_csv(input_csv_path)

    # Apply row limit if set (via module-level attribute, set by CLI wrapper)
    limit = getattr(main, "_limit", None)
    if limit is not None:
        df = df.head(limit)

    total = len(df)
    logger.info("Processing %d claims", total)

    # 3. Process each row
    results: list[OutputRow] = []
    start_time = time.time()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        logger.info(
            "Processing claim %d/%d (user: %s)",
            i,
            total,
            row.get("user_id", "unknown"),
        )
        output_row = process_claim(
            row,
            client,
            data_store,
            quality_checker,
            claim_extractor,
            evidence_checker,
            risk_scorer,
            vision_analyzer,
            image_aggregator,
            decision_engine,
            formatter,
        )
        results.append(output_row)

    elapsed = time.time() - start_time

    # 4. Write output CSV
    columns = OutputRow.get_columns()
    output_df = pd.DataFrame(
        [vars(r) for r in results],
        columns=columns,
    )
    output_path = pathlib.Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info("Output written to %s", output_csv_path)
    logger.info(
        "Finished %d claim(s) in %.1fs (avg %.1fs/claim). Total API calls: %d",
        total,
        elapsed,
        elapsed / total if total else 0,
        client.get_call_count(),
    )

    return output_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Claim verification pipeline")
    parser.add_argument(
        "strategy",
        nargs="?",
        choices=["b", "c"],
        default="b",
        help="Strategy to use: 'b' (default, anchored vision) or 'c' (observe-first)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_claims.csv instead of claims.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="PATH",
        help="Override input CSV path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Override output CSV path",
    )
    args = parser.parse_args()

    # Pass strategy selection via function attribute
    main._strategy = args.strategy  # type: ignore[attr-defined]

    if args.input:
        _input = args.input
        _output = args.output or str(REPO_ROOT / "output_custom.csv")
    elif args.sample:
        _input = str(SAMPLE_CLAIMS_CSV)
        _output = args.output or str(REPO_ROOT / "output_sample.csv")
    else:
        _input = str(CLAIMS_CSV)
        _output = args.output or str(OUTPUT_CSV)

    # Pass limit via function attribute so main() signature stays clean
    if args.limit is not None:
        main._limit = args.limit  # type: ignore[attr-defined]

    main(input_csv_path=_input, output_csv_path=_output)
