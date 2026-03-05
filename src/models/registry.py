"""
Model registry module.

Manages versioned model artifacts locally under models/registry/.
Each saved version is stamped with a timestamp and metadata JSON.

Structure:
    models/registry/
    ├── content_based/
    │   ├── v1_20260101_120000/
    │   │   ├── model.joblib
    │   │   └── metadata.json
    │   └── latest -> v1_20260101_120000   (symlink)
    ├── collaborative/
    └── hybrid/
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.utils.logger import get_logger

logger = get_logger("registry")

REGISTRY_DIR = Path("models/registry")


def _version_dir(model_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = f"v1_{timestamp}"
    return REGISTRY_DIR / model_name / version


def save_to_registry(
    model,
    model_name: str,
    metadata: dict | None = None,
) -> Path:
    """
    Save a model to the registry with versioned directory and metadata.

    Args:
        model      : any picklable model object
        model_name : logical name (e.g. 'content_based', 'collaborative')
        metadata   : optional dict of params/metrics to store alongside

    Returns:
        Path to the versioned model directory
    """
    version_dir = _version_dir(model_name)
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    meta = {
        "model_name": model_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version_dir": str(version_dir),
    }
    if metadata:
        meta.update(metadata)

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Update 'latest' symlink
    latest_link = REGISTRY_DIR / model_name / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    os.symlink(version_dir.resolve(), latest_link)

    logger.info(f"Saved to registry: {version_dir}")
    return version_dir


def load_from_registry(model_name: str, version: str = "latest"):
    """
    Load a model from the registry.

    Args:
        model_name : e.g. 'content_based'
        version    : version folder name or 'latest' (default)

    Returns:
        Loaded model object
    """
    if version == "latest":
        model_path = REGISTRY_DIR / model_name / "latest" / "model.joblib"
    else:
        model_path = REGISTRY_DIR / model_name / version / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n" f"Run the training pipeline first."
        )

    logger.info(f"Loading from registry: {model_path}")
    return joblib.load(model_path)


def list_versions(model_name: str) -> list[dict]:
    """
    List all saved versions for a model, newest first.

    Returns:
        List of dicts with version info and metrics.
    """
    model_dir = REGISTRY_DIR / model_name
    if not model_dir.exists():
        return []

    versions = []
    for v_dir in sorted(model_dir.iterdir(), reverse=True):
        if v_dir.is_symlink() or not v_dir.is_dir():
            continue
        meta_path = v_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                versions.append(json.load(f))

    return versions


def compare_versions(model_name: str, metric: str = "precision@10") -> dict:
    """
    Compare all saved versions of a model on a given metric.

    Returns:
        dict with best version info.
    """
    versions = list_versions(model_name)
    if not versions:
        logger.warning(f"No versions found for '{model_name}'.")
        return {}

    best = max(
        (v for v in versions if metric in v),
        key=lambda v: v[metric],
        default=None,
    )

    if best:
        logger.info(
            f"Best '{model_name}' by {metric}: "
            f"{best['version_dir']} | {metric}={best[metric]:.4f}"
        )
    else:
        logger.info(f"Metric '{metric}' not recorded for '{model_name}'.")

    return best or {}
