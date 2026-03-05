"""
Hybrid Recommender.

Combines Content-Based and Collaborative Filtering scores using
a configurable weighted average:
    final_score = alpha * cb_score + (1 - alpha) * cf_score

Both scores are min-max normalised to [0, 1] before blending.

Artifacts saved to: models/artifacts/hybrid/
"""

from pathlib import Path

import joblib
import pandas as pd

from src.models.recommender.collaborative import CollaborativeRecommender
from src.models.recommender.content_based import ContentBasedRecommender
from src.utils.logger import get_logger

logger = get_logger("hybrid")

ARTIFACT_DIR = Path("models/artifacts/hybrid")


def _minmax_norm(series: pd.Series) -> pd.Series:
    """Normalise a Series to [0, 1]. Returns zeros if range is 0."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


class HybridRecommender:
    """
    Weighted hybrid of content-based and collaborative filtering.

    Args:
        alpha : weight for content-based score (0–1).
                alpha=1.0 → pure content-based
                alpha=0.0 → pure collaborative
                alpha=0.5 → equal blend (default)
    """

    def __init__(self, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.cb_model: ContentBasedRecommender | None = None
        self.cf_model: CollaborativeRecommender | None = None

    def fit(
        self,
        cb_model: ContentBasedRecommender,
        cf_model: CollaborativeRecommender,
    ) -> "HybridRecommender":
        """Attach pre-fitted CB and CF models."""
        self.cb_model = cb_model
        self.cf_model = cf_model
        logger.info(
            f"Hybrid model ready (alpha={self.alpha}: "
            f"{self.alpha:.0%} CB + {1 - self.alpha:.0%} CF)"
        )
        return self

    def recommend(
        self,
        drama_name: str,
        top_n: int = 10,
        candidate_pool: int = 50,
    ) -> pd.DataFrame:
        """
        Return top-N hybrid recommendations for a given drama.

        Strategy:
          1. Fetch top-`candidate_pool` results from both CB and CF.
          2. Normalise each model's scores to [0, 1].
          3. Merge on drama_name and compute weighted blend.
          4. Return top-N by blended score.

        Args:
            drama_name     : query drama title (case-insensitive)
            top_n          : final number of recommendations
            candidate_pool : how many candidates each sub-model provides

        Returns:
            DataFrame [drama_name, cb_score, cf_score, hybrid_score]
        """
        if self.cb_model is None or self.cf_model is None:
            raise RuntimeError("Models not attached. Call .fit() first.")

        # Fetch candidates
        cb_recs = self.cb_model.recommend(drama_name, top_n=candidate_pool)
        cf_recs = self.cf_model.recommend_for_drama(drama_name, top_n=candidate_pool)

        if cb_recs.empty and cf_recs.empty:
            logger.warning(f"No recommendations found for '{drama_name}'.")
            return pd.DataFrame(
                columns=["drama_name", "cb_score", "cf_score", "hybrid_score"]
            )

        # Normalise scores
        if not cb_recs.empty:
            cb_recs = cb_recs.rename(columns={"similarity_score": "cb_score"})[
                ["drama_name", "cb_score"]
            ]
            cb_recs["cb_score"] = _minmax_norm(cb_recs["cb_score"])

        if not cf_recs.empty:
            cf_recs = cf_recs.rename(columns={"cf_score": "cf_score"})[
                ["drama_name", "cf_score"]
            ]
            cf_recs["cf_score"] = _minmax_norm(cf_recs["cf_score"])

        # Merge (outer join so we keep dramas from either model)
        merged = pd.merge(cb_recs, cf_recs, on="drama_name", how="outer")
        merged["cb_score"] = merged["cb_score"].fillna(0.0)
        merged["cf_score"] = merged["cf_score"].fillna(0.0)

        # Blend
        merged["hybrid_score"] = (
            self.alpha * merged["cb_score"] + (1 - self.alpha) * merged["cf_score"]
        )

        result = (
            merged.sort_values("hybrid_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        logger.info(
            f"Hybrid: {top_n} recommendations for '{drama_name}' "
            f"(alpha={self.alpha})"
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / "model.joblib"
        joblib.dump(self, path)
        logger.info(f"Hybrid model saved: {path}")
        return path

    @classmethod
    def load(cls) -> "HybridRecommender":
        path = ARTIFACT_DIR / "model.joblib"
        logger.info(f"Loading hybrid model from {path}")
        return joblib.load(path)
