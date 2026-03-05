"""
TextBlob sentiment baseline.
Fast, rule-based sentiment scoring — used as a lightweight alternative
to BERT or as a sanity-check baseline.

Outputs per review:
  - polarity    : float in [-1, 1]  (negative → positive)
  - subjectivity: float in [ 0, 1]  (objective → subjective)
  - tb_label    : 0=Negative, 1=Neutral, 2=Positive
"""

import pandas as pd
from textblob import TextBlob

from src.utils.logger import get_logger

logger = get_logger("textblob_model")


def _polarity(text: str) -> float:
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def _subjectivity(text: str) -> float:
    try:
        return TextBlob(str(text)).sentiment.subjectivity
    except Exception:
        return 0.0


def _label(polarity: float) -> int:
    if polarity > 0.1:
        return 2  # Positive
    elif polarity < -0.1:
        return 0  # Negative
    return 1  # Neutral


def run_textblob(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Run TextBlob sentiment on df_reviews['review_text'].
    Adds columns: tb_polarity, tb_subjectivity, tb_label.
    Returns a copy of df_reviews with the new columns.
    """
    logger.info(f"Running TextBlob sentiment on {len(df_reviews)} reviews...")
    df = df_reviews.copy()

    df["tb_polarity"] = df["review_text"].apply(_polarity)
    df["tb_subjectivity"] = df["review_text"].apply(_subjectivity)
    df["tb_label"] = df["tb_polarity"].apply(_label)

    dist = df["tb_label"].value_counts().to_dict()
    logger.info(f"TextBlob label distribution: {dist}")
    return df


def aggregate_textblob_per_drama(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate TextBlob scores per drama title.
    Returns a DataFrame indexed by normalised title.
    """
    agg = (
        df_reviews.groupby("title")
        .agg(
            tb_avg_polarity=("tb_polarity", "mean"),
            tb_avg_subjectivity=("tb_subjectivity", "mean"),
            tb_pct_positive=("tb_label", lambda x: (x == 2).mean()),
            tb_pct_neutral=("tb_label", lambda x: (x == 1).mean()),
            tb_pct_negative=("tb_label", lambda x: (x == 0).mean()),
        )
        .reset_index()
    )
    logger.info(f"TextBlob aggregated for {len(agg)} dramas.")
    return agg
