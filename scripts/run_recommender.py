"""
Run the full Phase 5 recommendation engine pipeline.

Steps:
  1. Load feature store + cleaned data from data/processed/
  2. Train Content-Based model  → save to models/artifacts/content_based/
  3. Train Collaborative model  → save to models/artifacts/collaborative/
  4. Build Hybrid model         → save to models/artifacts/hybrid/
  5. Run a demo query and print results

Usage:
    python scripts/run_recommender.py
    python scripts/run_recommender.py --query "Crash Landing on You" --top-n 10
    python scripts/run_recommender.py --alpha 0.7   # 70% content, 30% CF
"""

import argparse
from pathlib import Path

import pandas as pd

from src.models.recommender.collaborative import CollaborativeRecommender
from src.models.recommender.content_based import (
    ContentBasedRecommender,
    catalog_coverage,
)
from src.models.recommender.hybrid import HybridRecommender
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_recommender")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Drama Recommender Engine")
    parser.add_argument(
        "--query",
        type=str,
        default="Celebrity",
        help="Drama name to get recommendations for (default: 'Celebrity')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="CB vs CF blend weight: 1.0=pure CB, 0.0=pure CF (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    processed = cfg["data"]["processed_dir"]

    logger.info("=== K-Drama Compass: Recommendation Engine Starting ===")

    # Step 1: Load data
    logger.info("Step 1/4: Loading data...")
    feature_store = pd.read_csv(f"{processed}/feature_store.csv", index_col="kdrama_id")
    df_dramas = pd.read_csv(
        f"{processed}/cleaned_dramas.csv",
        parse_dates=["start_dt", "end_dt"],
    )
    df_reviews = pd.read_csv(f"{processed}/cleaned_reviews.csv")

    logger.info(
        f"  feature_store={feature_store.shape}, "
        f"dramas={len(df_dramas)}, reviews={len(df_reviews)}"
    )

    # Step 2: Content-Based model
    logger.info("Step 2/4: Training content-based model...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(feature_store, df_dramas)
    cb_model.save()

    # Step 3: Collaborative Filtering model
    logger.info("Step 3/4: Training collaborative filtering model...")
    cf_model = CollaborativeRecommender(n_components=50)
    cf_model.fit(df_reviews)
    cf_model.save()

    # Step 4: Hybrid model
    logger.info("Step 4/4: Building hybrid model...")
    hybrid = HybridRecommender(alpha=args.alpha)
    hybrid.fit(cb_model, cf_model)
    hybrid.save()

    # Demo query
    logger.info(f"\n{'='*60}")
    logger.info(f"RECOMMENDATIONS FOR: '{args.query}' (top {args.top_n})")
    logger.info(f"{'='*60}")

    logger.info("\n--- Content-Based ---")
    cb_recs = cb_model.recommend(args.query, top_n=args.top_n)
    print(cb_recs.to_string(index=False))

    logger.info("\n--- Collaborative Filtering ---")
    cf_recs = cf_model.recommend_for_drama(args.query, top_n=args.top_n)
    print(cf_recs.to_string(index=False))

    logger.info(f"\n--- Hybrid (alpha={args.alpha}) ---")
    hybrid_recs = hybrid.recommend(args.query, top_n=args.top_n)
    print(hybrid_recs.to_string(index=False))

    # Catalog coverage
    all_recs = [
        cb_model.recommend(row["drama_name"], top_n=10)["drama_name"].tolist()
        for _, row in df_dramas.head(100).iterrows()
    ]
    coverage = catalog_coverage(all_recs, catalog_size=len(df_dramas))
    logger.info(f"\nCatalog coverage (sample 100 queries): {coverage:.2%}")

    logger.info("=== Recommendation Engine Complete ===")


if __name__ == "__main__":
    main()
