"""
Run the full Phase 2 data pipeline:
  1. Ingest raw CSVs from data/raw/
  2. Validate with Pydantic schemas
  3. ETL: clean + feature engineer + save to data/processed/

Usage:
    python scripts/run_pipeline.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.etl import run_etl  # noqa: E402
from src.data.ingest import load_all_raw  # noqa: E402
from src.data.validate import validate_all  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("pipeline")


def main():
    logger.info("=== K-Drama Compass: Data Pipeline Starting ===")

    # Step 1: Ingest
    logger.info("Step 1/3: Ingesting raw data...")
    raw = load_all_raw()

    # Step 2: Validate
    logger.info("Step 2/3: Validating raw data...")
    results = validate_all(raw)
    for dataset, passed in results.items():
        status = "PASSED" if passed else "WARNINGS (check logs)"
        logger.info(f"  {dataset}: {status}")

    # Step 3: ETL
    logger.info("Step 3/3: Running ETL...")
    cleaned = run_etl(raw)

    logger.info("=== Pipeline complete ===")
    for name, df in cleaned.items():
        logger.info(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols")


if __name__ == "__main__":
    main()
