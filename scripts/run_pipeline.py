"""
Run the full Phase 2 data pipeline:
  1. Ingest raw CSVs from data/raw/
  2. Validate with Pydantic schemas
  3. ETL: clean + feature engineer + save to data/processed/

Usage (from project root, after: pip install -e .):
    python scripts/run_pipeline.py
"""

from src.data.etl import run_etl
from src.data.ingest import load_all_raw
from src.data.validate import validate_all
from src.utils.logger import get_logger

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
