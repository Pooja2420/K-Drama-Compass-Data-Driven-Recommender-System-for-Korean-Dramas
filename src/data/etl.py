"""
ETL pipeline module.
Cleans and transforms raw DataFrames → saves to data/processed/.
Based on logic from notebooks/KDrama.ipynb, refactored for production.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("etl")

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Dramas ETL
# ---------------------------------------------------------------------------


def clean_dramas(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning dramas dataset...")
    df = df.copy()

    # Fill missing categorical fields
    df["director"].fillna("Unknown", inplace=True)
    df["screenwriter"].fillna("Unknown", inplace=True)
    df["synopsis"].fillna("No synopsis available", inplace=True)

    # Fill aired_on and org_net with mode or Unknown
    for col in ["aired_on", "org_net"]:
        mode = df[col].mode()
        df[col].fillna(mode[0] if not mode.empty else "Unknown", inplace=True)

    # Fill duration with median
    df["duration"].fillna(df["duration"].median(), inplace=True)

    # Parse dates
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_dt"], errors="coerce")

    # Impute missing dates with median
    df["start_dt"].fillna(df["start_dt"].median(), inplace=True)
    df["end_dt"].fillna(df["end_dt"].median(), inplace=True)

    logger.info("Dramas cleaned. Running feature engineering...")
    df = _feature_engineer_dramas(df)
    return df


def _feature_engineer_dramas(df: pd.DataFrame) -> pd.DataFrame:
    # Temporal features
    df["start_month"] = df["start_dt"].dt.month
    df["start_day_of_week"] = df["start_dt"].dt.dayofweek

    # Duration in days (min 1 day)
    df["duration_days"] = (df["end_dt"] - df["start_dt"]).dt.days
    df["duration_days"] = df["duration_days"].replace(0, 1).fillna(0).astype(int)

    # Duration category (in seconds: short <30min, medium 30-60min, long >60min)
    bins = [0, 1800, 3600, float("inf")]
    labels = ["short", "medium", "long"]
    df["duration_category"] = pd.cut(df["duration"], bins=bins, labels=labels)

    logger.info("Feature engineering complete for dramas.")
    return df


# ---------------------------------------------------------------------------
# Reviews ETL
# ---------------------------------------------------------------------------


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning reviews dataset...")
    df = df.copy()

    df["review_text"].fillna("No review provided", inplace=True)

    # Parse ep_watched into numeric columns
    df["episodes_watched"] = df["ep_watched"].apply(_parse_ep_watched)
    df["total_episodes"] = df["ep_watched"].apply(_parse_total_eps)

    # Normalize title for merging
    df["title"] = df["title"].str.strip().str.lower()

    # Sentiment label from overall_score
    df["sentiment_label"] = df["overall_score"].apply(_label_sentiment)

    logger.info(f"Reviews cleaned: {len(df)} records.")
    return df


def _parse_ep_watched(ep_str: str) -> int:
    try:
        return int(str(ep_str).split(" ")[0])
    except (ValueError, IndexError):
        return 0


def _parse_total_eps(ep_str: str) -> float:
    try:
        parts = str(ep_str).split(" ")
        return int(parts[2]) if len(parts) > 2 else np.nan
    except (ValueError, IndexError):
        return np.nan


def _label_sentiment(score: float) -> int:
    if score >= 7:
        return 2  # Positive
    elif score >= 4:
        return 1  # Neutral
    return 0  # Negative


# ---------------------------------------------------------------------------
# Actors ETL
# ---------------------------------------------------------------------------


def clean_actors(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning actors dataset...")
    df = df.copy()

    # Standardize role casing
    df["role"] = df["role"].str.title()

    # Normalize drama_name for merging
    df["drama_name"] = df["drama_name"].str.strip().str.lower()

    logger.info(f"Actors cleaned: {len(df)} records.")
    return df


# ---------------------------------------------------------------------------
# Save & orchestrate
# ---------------------------------------------------------------------------


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(df)} rows)")
    return out_path


def run_etl(raw_datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Run full ETL on all three datasets.
    Returns dict of cleaned DataFrames and saves them to data/processed/.
    """
    cleaned = {}

    cleaned["dramas"] = clean_dramas(raw_datasets["dramas"])
    save_processed(cleaned["dramas"], "cleaned_dramas.csv")

    cleaned["reviews"] = clean_reviews(raw_datasets["reviews"])
    save_processed(cleaned["reviews"], "cleaned_reviews.csv")

    cleaned["actors"] = clean_actors(raw_datasets["actors"])
    save_processed(cleaned["actors"], "cleaned_actors.csv")

    # Merged dataset (actors + reviews joined on drama name)
    merged = pd.merge(
        cleaned["actors"],
        cleaned["reviews"],
        left_on="drama_name",
        right_on="title",
        how="inner",
    )
    save_processed(merged, "merged_actors_reviews.csv")
    logger.info(f"Merged actors+reviews: {len(merged)} rows")

    return cleaned
