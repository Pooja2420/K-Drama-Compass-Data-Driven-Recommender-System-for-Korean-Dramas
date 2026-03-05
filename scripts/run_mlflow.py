"""
MLOps pipeline runner — trains all models and tracks with MLflow.

For each model (content-based, collaborative, hybrid):
  1. Train the model
  2. Evaluate with Precision@K, Recall@K, NDCG@K, Coverage
  3. Log params + metrics to MLflow
  4. Save artifact to MLflow + local registry

Usage:
    # Start MLflow UI first (in another terminal):
    #   mlflow ui --port 5000
    #   Then open http://localhost:5000

    python scripts/run_mlflow.py
    python scripts/run_mlflow.py --top-k 10 --cf-components 100
"""

import argparse
from pathlib import Path

import pandas as pd

from src.models.evaluate import evaluate_recommender
from src.models.mlflow_tracker import MLflowTracker, get_best_run
from src.models.recommender.collaborative import CollaborativeRecommender
from src.models.recommender.content_based import ContentBasedRecommender
from src.models.recommender.hybrid import HybridRecommender
from src.models.registry import compare_versions, save_to_registry
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_mlflow")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Drama MLOps Pipeline")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cf-components", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eval-sample", type=int, default=50)
    return parser.parse_args()


def load_data(processed: str) -> tuple:
    feature_store = pd.read_csv(f"{processed}/feature_store.csv", index_col="kdrama_id")
    df_dramas = pd.read_csv(
        f"{processed}/cleaned_dramas.csv",
        parse_dates=["start_dt", "end_dt"],
    )
    df_reviews = pd.read_csv(f"{processed}/cleaned_reviews.csv")
    return feature_store, df_dramas, df_reviews


def run_content_based(feature_store, df_dramas, df_reviews, args):
    params = {"n_features": feature_store.shape[1], "top_k": args.top_k}

    with MLflowTracker("content_based") as tracker:
        tracker.log_params(params)

        cb = ContentBasedRecommender()
        cb.fit(feature_store, df_dramas)

        metrics = evaluate_recommender(
            cb,
            df_dramas,
            df_reviews,
            k=args.top_k,
            sample_n=args.eval_sample,
        )
        tracker.log_metrics(metrics)

        artifact_path = cb.save()
        tracker.log_artifact(artifact_path)

        save_to_registry(cb, "content_based", metadata={**params, **metrics})
        logger.info(f"Content-based run id: {tracker.run_id}")

    return cb, metrics


def run_collaborative(df_dramas, df_reviews, args):
    params = {
        "n_components": args.cf_components,
        "top_k": args.top_k,
    }

    with MLflowTracker("collaborative") as tracker:
        tracker.log_params(params)

        cf = CollaborativeRecommender(n_components=args.cf_components)
        cf.fit(df_reviews)

        metrics = evaluate_recommender(
            cf,
            df_dramas,
            df_reviews,
            k=args.top_k,
            sample_n=args.eval_sample,
        )
        tracker.log_metrics(metrics)

        artifact_path = cf.save()
        tracker.log_artifact(artifact_path)

        save_to_registry(cf, "collaborative", metadata={**params, **metrics})
        logger.info(f"Collaborative run id: {tracker.run_id}")

    return cf, metrics


def run_hybrid(cb, cf, df_dramas, df_reviews, args):
    params = {"alpha": args.alpha, "top_k": args.top_k}

    with MLflowTracker("hybrid") as tracker:
        tracker.log_params(params)

        hybrid = HybridRecommender(alpha=args.alpha)
        hybrid.fit(cb, cf)

        metrics = evaluate_recommender(
            hybrid,
            df_dramas,
            df_reviews,
            k=args.top_k,
            sample_n=args.eval_sample,
        )
        tracker.log_metrics(metrics)

        artifact_path = hybrid.save()
        tracker.log_artifact(artifact_path)

        save_to_registry(hybrid, "hybrid", metadata={**params, **metrics})
        logger.info(f"Hybrid run id: {tracker.run_id}")

    return hybrid, metrics


def main():
    args = parse_args()
    cfg = load_config()
    processed = cfg["data"]["processed_dir"]

    logger.info("=== K-Drama Compass: MLOps Pipeline Starting ===")
    feature_store, df_dramas, df_reviews = load_data(processed)

    # Train & track all three models
    cb, cb_metrics = run_content_based(feature_store, df_dramas, df_reviews, args)
    cf, cf_metrics = run_collaborative(df_dramas, df_reviews, args)
    _, hybrid_metrics = run_hybrid(cb, cf, df_dramas, df_reviews, args)

    # Summary
    logger.info("\n=== Results Summary ===")
    for name, m in [
        ("Content-Based", cb_metrics),
        ("Collaborative", cf_metrics),
        ("Hybrid", hybrid_metrics),
    ]:
        p = m.get(f"precision@{args.top_k}", 0)
        n = m.get(f"ndcg@{args.top_k}", 0)
        c = m.get("catalog_coverage", 0)
        logger.info(
            f"{name}: precision@{args.top_k}={p:.4f} | "
            f"ndcg@{args.top_k}={n:.4f} | coverage={c:.4f}"
        )

    # Best run across all experiments
    best = get_best_run(metric=f"precision@{args.top_k}")
    if best:
        logger.info(f"\nBest run overall: {best['run_name']}")

    # Registry comparison
    compare_versions("content_based", metric=f"precision@{args.top_k}")

    logger.info("=== MLOps Pipeline Complete ===")
    logger.info("View results: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
