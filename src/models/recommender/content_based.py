"""
Content-Based Filtering recommender.

Uses the unified feature store (TF-IDF synopsis + numeric + categorical features)
to compute cosine similarity between dramas and return top-N recommendations
for a given drama title.

Artifacts saved to: models/artifacts/content_based/
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger("content_based")

ARTIFACT_DIR = Path("models/artifacts/content_based")


class ContentBasedRecommender:
    """
    Recommends dramas similar to a given drama based on feature similarity.

    Attributes:
        similarity_matrix : (n_dramas x n_dramas) cosine similarity DataFrame
        drama_index       : mapping from kdrama_id → drama_name
        name_to_id        : mapping from normalised name → kdrama_id
    """

    def __init__(self):
        self.similarity_matrix: pd.DataFrame | None = None
        self.drama_index: dict = {}
        self.name_to_id: dict = {}

    def fit(
        self,
        feature_store: pd.DataFrame,
        df_dramas: pd.DataFrame,
    ) -> "ContentBasedRecommender":
        """
        Build cosine similarity matrix from feature_store.

        Args:
            feature_store : DataFrame indexed by kdrama_id (from feature_engineering)
            df_dramas     : cleaned dramas DataFrame (for name lookups)
        """
        logger.info(
            f"Fitting content-based model on "
            f"{len(feature_store)} dramas x {feature_store.shape[1]} features..."
        )

        sim = cosine_similarity(feature_store.values)
        self.similarity_matrix = pd.DataFrame(
            sim,
            index=feature_store.index,
            columns=feature_store.index,
        )

        # Build lookup maps
        df_dramas = df_dramas.copy()
        df_dramas["name_norm"] = df_dramas["drama_name"].str.strip().str.lower()
        self.drama_index = df_dramas.set_index("kdrama_id")["drama_name"].to_dict()
        self.name_to_id = df_dramas.set_index("name_norm")["kdrama_id"].to_dict()

        logger.info("Content-based model fitted.")
        return self

    def recommend(
        self,
        drama_name: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return top-N most similar dramas for a given drama name.

        Args:
            drama_name : drama title (case-insensitive)
            top_n      : number of recommendations to return

        Returns:
            DataFrame with columns [drama_name, kdrama_id, similarity_score]
        """
        if self.similarity_matrix is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        key = drama_name.strip().lower()
        kdrama_id = self.name_to_id.get(key)

        if kdrama_id is None:
            logger.warning(f"Drama '{drama_name}' not found in index.")
            return pd.DataFrame(columns=["drama_name", "kdrama_id", "similarity_score"])

        if kdrama_id not in self.similarity_matrix.index:
            logger.warning(f"'{drama_name}' not in feature store.")
            return pd.DataFrame(columns=["drama_name", "kdrama_id", "similarity_score"])

        scores = (
            self.similarity_matrix.loc[kdrama_id]
            .drop(labels=[kdrama_id])
            .sort_values(ascending=False)
            .head(top_n)
        )

        results = pd.DataFrame(
            {
                "kdrama_id": scores.index,
                "similarity_score": scores.values,
            }
        )
        results["drama_name"] = results["kdrama_id"].map(self.drama_index)
        logger.info(f"Content-based: {top_n} recommendations for '{drama_name}'")
        return results[["drama_name", "kdrama_id", "similarity_score"]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / "model.joblib"
        joblib.dump(self, path)
        logger.info(f"Content-based model saved: {path}")
        return path

    @classmethod
    def load(cls) -> "ContentBasedRecommender":
        path = ARTIFACT_DIR / "model.joblib"
        logger.info(f"Loading content-based model from {path}")
        return joblib.load(path)


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------


def precision_at_k(
    recommended: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Fraction of top-k recommendations that are relevant."""
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def catalog_coverage(
    all_recommendations: list[list[str]],
    catalog_size: int,
) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    unique = set(item for recs in all_recommendations for item in recs)
    return len(unique) / catalog_size if catalog_size > 0 else 0.0
