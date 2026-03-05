"""
MLflow experiment tracking wrapper.

Provides a thin, reusable layer over MLflow so every training run:
  - Is logged under a named experiment
  - Records all hyperparameters
  - Records all evaluation metrics
  - Stores model artifacts
  - Tags the run with model type and timestamp

Usage:
    with MLflowTracker("content_based") as tracker:
        model = ContentBasedRecommender()
        model.fit(feature_store, df_dramas)
        tracker.log_params({"n_features": feature_store.shape[1]})
        tracker.log_metrics({"precision@10": 0.42, "coverage": 0.65})
        tracker.log_artifact(model.save())
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

from src.utils.logger import get_logger

logger = get_logger("mlflow_tracker")

EXPERIMENT_NAME = "kdrama-compass"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


class MLflowTracker:
    """Context-manager wrapper around an MLflow run."""

    def __init__(self, run_name: str, tags: dict[str, str] | None = None):
        self.run_name = run_name
        self.tags = tags or {}
        self._run = None

    def __enter__(self) -> "MLflowTracker":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        full_run_name = f"{self.run_name}_{timestamp}"

        default_tags = {
            "model_type": self.run_name,
            "project": "kdrama-compass",
        }
        default_tags.update(self.tags)

        self._run = mlflow.start_run(run_name=full_run_name, tags=default_tags)
        logger.info(
            f"MLflow run started: {full_run_name} " f"(id={self._run.info.run_id})"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            logger.error(f"Run failed: {exc_val}")
        else:
            mlflow.set_tag("status", "completed")
        mlflow.end_run()
        logger.info("MLflow run ended.")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dict of hyperparameters."""
        mlflow.log_params(params)
        for k, v in params.items():
            logger.info(f"  param  {k}={v}")

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log a dict of evaluation metrics."""
        mlflow.log_metrics(metrics, step=step)
        for k, v in metrics.items():
            logger.info(
                f"  metric {k}={v:.4f}" if isinstance(v, float) else f"  metric {k}={v}"
            )

    def log_artifact(self, path: Path) -> None:
        """Log a file or directory as an MLflow artifact."""
        mlflow.log_artifact(str(path))
        logger.info(f"  artifact logged: {path}")

    def log_artifacts_dir(self, dir_path: Path) -> None:
        """Log an entire directory as MLflow artifacts."""
        mlflow.log_artifacts(str(dir_path))
        logger.info(f"  artifact dir logged: {dir_path}")

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None


# ---------------------------------------------------------------------------
# Convenience: list recent runs
# ---------------------------------------------------------------------------


def get_best_run(metric: str = "precision@10") -> dict:
    """
    Return the run with the highest value of `metric`
    from the kdrama-compass experiment.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.warning(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.`{metric}` > 0",
        order_by=[f"metrics.`{metric}` DESC"],
        max_results=1,
    )

    if not runs:
        logger.warning(f"No runs found with metric '{metric}'.")
        return {}

    best = runs[0]
    logger.info(
        f"Best run: {best.info.run_name} | "
        f"{metric}={best.data.metrics.get(metric, 'N/A')}"
    )
    return {
        "run_id": best.info.run_id,
        "run_name": best.info.run_name,
        "metrics": best.data.metrics,
        "params": best.data.params,
    }
