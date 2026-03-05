"""
Feature engineering module.
Transforms cleaned datasets into a unified feature matrix for the recommendation engine.

Features produced:
  - Drama content features  : TF-IDF on synopsis, encoded network/rating/aired_on
  - Drama numeric features  : normalized rank, pop, tot_eps, duration, temporal cols
  - Review aggregation      : per-drama avg scores, review count, sentiment distribution
  - Actor reputation        : avg score & appearance count per actor, mapped to dramas
  - Final feature store     : all features merged on kdrama_id / drama_name
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger("features")

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# 1. Drama content features (TF-IDF on synopsis)
# ---------------------------------------------------------------------------


def build_synopsis_tfidf(
    df_dramas: pd.DataFrame, max_features: int = 500
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on drama synopses.
    Returns (tfidf_df with kdrama_id index, fitted vectorizer).
    """
    logger.info(f"Building synopsis TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(df_dramas["synopsis"].fillna(""))
    cols = [f"tfidf_{t}" for t in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        matrix.toarray(), index=df_dramas["kdrama_id"], columns=cols
    )
    logger.info(f"TF-IDF matrix shape: {tfidf_df.shape}")
    return tfidf_df, vectorizer


# ---------------------------------------------------------------------------
# 2. Drama categorical encoding (network, content rating, aired_on)
# ---------------------------------------------------------------------------


def encode_drama_categoricals(df_dramas: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode org_net, content_rt, aired_on; keep kdrama_id as index."""
    logger.info("Encoding drama categorical features...")
    cat_cols = ["org_net", "content_rt", "aired_on"]
    encoded = pd.get_dummies(
        df_dramas[["kdrama_id"] + cat_cols].set_index("kdrama_id"),
        columns=cat_cols,
        prefix=cat_cols,
        drop_first=False,
    )
    logger.info(f"Categorical encoded shape: {encoded.shape}")
    return encoded


# ---------------------------------------------------------------------------
# 3. Drama numeric features (normalized)
# ---------------------------------------------------------------------------


def build_drama_numeric_features(df_dramas: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric drama columns with MinMaxScaler.
    Returns DataFrame indexed by kdrama_id.
    """
    logger.info("Building drama numeric features...")
    num_cols = [
        "year",
        "tot_eps",
        "duration",
        "rank",
        "pop",
        "start_month",
        "start_day_of_week",
        "duration_days",
    ]
    existing = [c for c in num_cols if c in df_dramas.columns]
    df_num = df_dramas[["kdrama_id"] + existing].set_index("kdrama_id").copy()

    scaler = MinMaxScaler()
    df_num[existing] = scaler.fit_transform(df_num[existing].fillna(0))
    logger.info(f"Numeric features shape: {df_num.shape}")
    return df_num


# ---------------------------------------------------------------------------
# 4. Review aggregation per drama
# ---------------------------------------------------------------------------


def aggregate_reviews(
    df_reviews: pd.DataFrame, df_dramas: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate review scores and sentiment per drama.
    Joins on normalized drama_name → kdrama_id.
    Returns DataFrame indexed by kdrama_id.
    """
    logger.info("Aggregating review features per drama...")

    # Normalize drama names for joining
    df_reviews = df_reviews.copy()
    df_reviews["title_norm"] = df_reviews["title"].str.strip().str.lower()

    df_dramas = df_dramas.copy()
    df_dramas["name_norm"] = df_dramas["drama_name"].str.strip().str.lower()

    agg = (
        df_reviews.groupby("title_norm")
        .agg(
            avg_overall=("overall_score", "mean"),
            avg_story=("story_score", "mean"),
            avg_acting=("acting_cast_score", "mean"),
            avg_music=("music_score", "mean"),
            avg_rewatch=("rewatch_value_score", "mean"),
            review_count=("user_id", "count"),
            pct_positive=("sentiment_label", lambda x: (x == 2).mean()),
            pct_neutral=("sentiment_label", lambda x: (x == 1).mean()),
            pct_negative=("sentiment_label", lambda x: (x == 0).mean()),
        )
        .reset_index()
    )

    # Map back to kdrama_id
    name_to_id = df_dramas.set_index("name_norm")["kdrama_id"].to_dict()
    agg["kdrama_id"] = agg["title_norm"].map(name_to_id)
    agg = (
        agg.dropna(subset=["kdrama_id"])
        .set_index("kdrama_id")
        .drop(columns=["title_norm"])
    )

    logger.info(
        f"Review aggregation shape: {agg.shape} ({len(agg)} dramas with reviews)"
    )
    return agg


# ---------------------------------------------------------------------------
# 5. Actor reputation features per drama
# ---------------------------------------------------------------------------


def build_actor_features(
    df_actors: pd.DataFrame, df_merged: pd.DataFrame, df_dramas: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-actor reputation (avg score, total appearances).
    Then aggregate actor reputation per drama.
    Returns DataFrame indexed by kdrama_id.
    """
    logger.info("Building actor reputation features...")

    # Actor-level reputation
    actor_rep = (
        df_merged.groupby("actor_name")
        .agg(
            actor_avg_score=("overall_score", "mean"),
            actor_appearances=("drama_name", "nunique"),
        )
        .reset_index()
    )

    # Join reputation back to actors
    actors_with_rep = df_actors.merge(actor_rep, on="actor_name", how="left")

    # Aggregate per drama: avg reputation of cast, count of main/support roles
    drama_actor = (
        actors_with_rep.groupby("drama_name")
        .agg(
            cast_avg_score=("actor_avg_score", "mean"),
            cast_size=("actor_name", "nunique"),
            main_role_count=("role", lambda x: (x == "Main Role").sum()),
            support_role_count=("role", lambda x: (x == "Support Role").sum()),
        )
        .reset_index()
    )

    # Map drama_name (normalised) → kdrama_id
    df_dramas = df_dramas.copy()
    df_dramas["name_norm"] = df_dramas["drama_name"].str.strip().str.lower()
    name_to_id = df_dramas.set_index("name_norm")["kdrama_id"].to_dict()

    drama_actor["kdrama_id"] = drama_actor["drama_name"].map(name_to_id)
    drama_actor = (
        drama_actor.dropna(subset=["kdrama_id"])
        .set_index("kdrama_id")
        .drop(columns=["drama_name"])
    )

    logger.info(f"Actor features shape: {drama_actor.shape}")
    return drama_actor


# ---------------------------------------------------------------------------
# 6. Build unified feature store
# ---------------------------------------------------------------------------


def build_feature_store(
    df_dramas: pd.DataFrame,
    df_reviews: pd.DataFrame,
    df_actors: pd.DataFrame,
    df_merged: pd.DataFrame,
    tfidf_max_features: int = 300,
) -> pd.DataFrame:
    """
    Combine all feature groups into a single feature matrix.
    Index = kdrama_id.
    """
    logger.info("Building unified feature store...")

    # Individual feature blocks
    tfidf_df, _ = build_synopsis_tfidf(df_dramas, max_features=tfidf_max_features)
    cat_df = encode_drama_categoricals(df_dramas)
    num_df = build_drama_numeric_features(df_dramas)
    review_df = aggregate_reviews(df_reviews, df_dramas)
    actor_df = build_actor_features(df_actors, df_merged, df_dramas)

    # Merge all on kdrama_id
    feature_store = (
        num_df.join(cat_df, how="left")
        .join(tfidf_df, how="left")
        .join(review_df, how="left")
        .join(actor_df, how="left")
    )

    # Fill any remaining NaNs
    feature_store = feature_store.fillna(0)

    logger.info(f"Feature store shape: {feature_store.shape}")
    return feature_store


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_feature_store(feature_store: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "feature_store.csv"
    feature_store.to_csv(out_path)
    logger.info(f"Feature store saved: {out_path}")
    return out_path
