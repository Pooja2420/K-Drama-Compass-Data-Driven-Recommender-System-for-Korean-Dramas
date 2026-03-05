"""
Run full EDA on all three cleaned datasets.
Loads from data/processed/ and saves plots to reports/figures/.

Usage:
    python scripts/run_eda.py
"""

import pandas as pd

from src.features.eda import run_eda
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_eda")


def main():
    logger.info("=== K-Drama Compass: EDA Starting ===")
    cfg = load_config()
    processed = cfg["data"]["processed_dir"]

    df_dramas = pd.read_csv(
        f"{processed}/cleaned_dramas.csv", parse_dates=["start_dt", "end_dt"]
    )
    df_reviews = pd.read_csv(f"{processed}/cleaned_reviews.csv")
    df_actors = pd.read_csv(f"{processed}/cleaned_actors.csv")
    df_merged = pd.read_csv(f"{processed}/merged_actors_reviews.csv")

    logger.info(
        f"Loaded: dramas={len(df_dramas)}, "
        f"reviews={len(df_reviews)}, "
        f"actors={len(df_actors)}, "
        f"merged={len(df_merged)}"
    )

    run_eda(df_dramas, df_reviews, df_actors, df_merged)
    logger.info("=== EDA complete ===")


if __name__ == "__main__":
    main()
