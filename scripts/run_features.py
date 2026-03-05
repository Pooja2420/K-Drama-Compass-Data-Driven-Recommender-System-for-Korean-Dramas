"""
Run the full feature engineering pipeline.
Loads cleaned data from data/processed/ and saves the feature store.

Usage:
    python scripts/run_features.py
"""

import pandas as pd

from src.features.feature_engineering import build_feature_store, save_feature_store
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_features")


def main():
    logger.info("=== K-Drama Compass: Feature Engineering Starting ===")
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
        f"actors={len(df_actors)}"
    )

    feature_store = build_feature_store(df_dramas, df_reviews, df_actors, df_merged)
    save_feature_store(feature_store)

    logger.info(f"=== Feature engineering complete: {feature_store.shape} ===")


if __name__ == "__main__":
    main()
