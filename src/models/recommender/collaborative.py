"""
Collaborative Filtering recommender using SVD (Matrix Factorisation).

Builds a user-item matrix from review scores, decomposes it with
TruncatedSVD, and recommends dramas a user is likely to enjoy
based on latent factor similarity.

Artifacts saved to: models/artifacts/collaborative/
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

logger = get_logger("collaborative")

ARTIFACT_DIR = Path("models/artifacts/collaborative")


class CollaborativeRecommender:
    """
    SVD-based collaborative filtering recommender.

    Attributes:
        user_enc     : LabelEncoder for user_id
        drama_enc    : LabelEncoder for drama title
        svd          : fitted TruncatedSVD model
        user_factors : (n_users x n_components) latent user matrix
        item_factors : (n_dramas x n_components) latent item matrix
        drama_titles : list of drama titles (in encoder order)
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.user_enc = LabelEncoder()
        self.drama_enc = LabelEncoder()
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.drama_titles: list[str] = []
        self._user_item_matrix: np.ndarray | None = None

    def fit(
        self,
        df_reviews: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> "CollaborativeRecommender":
        """
        Build user-item matrix and fit SVD.

        Args:
            df_reviews   : cleaned reviews with [user_id, title, overall_score]
            test_size    : fraction held out for RMSE evaluation
            random_state : reproducibility seed
        """
        df = df_reviews[["user_id", "title", "overall_score"]].dropna().copy()
        df["title"] = df["title"].str.strip().str.lower()

        # Encode users and dramas
        df["user_idx"] = self.user_enc.fit_transform(df["user_id"])
        df["drama_idx"] = self.drama_enc.fit_transform(df["title"])
        self.drama_titles = list(self.drama_enc.classes_)

        n_users = df["user_idx"].nunique()
        n_dramas = df["drama_idx"].nunique()
        logger.info(f"Building user-item matrix: {n_users} users x {n_dramas} dramas")

        # Train / test split for evaluation
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        # Build sparse matrix from training data
        train_sparse = csr_matrix(
            (
                train_df["overall_score"].values,
                (train_df["user_idx"].values, train_df["drama_idx"].values),
            ),
            shape=(n_users, n_dramas),
        )

        logger.info(f"Fitting SVD (n_components={self.n_components})...")
        self.user_factors = self.svd.fit_transform(train_sparse)
        self.item_factors = self.svd.components_.T
        self._user_item_matrix = train_sparse.toarray()

        # Evaluate on test set
        rmse = self._evaluate(test_df)
        logger.info(f"Collaborative RMSE on test set: {rmse:.4f}")
        logger.info(
            f"Explained variance: " f"{self.svd.explained_variance_ratio_.sum():.4f}"
        )

        return self

    def _evaluate(self, test_df: pd.DataFrame) -> float:
        """Compute RMSE on held-out ratings."""
        valid = test_df[
            (test_df["user_idx"] < len(self.user_factors))
            & (test_df["drama_idx"] < len(self.item_factors))
        ]
        if valid.empty:
            return float("nan")

        preds = np.array(
            [
                self.user_factors[r["user_idx"]] @ self.item_factors[r["drama_idx"]]
                for _, r in valid.iterrows()
            ]
        )
        preds = np.clip(preds, 1, 10)
        return float(np.sqrt(mean_squared_error(valid["overall_score"], preds)))

    def recommend_for_drama(
        self,
        drama_name: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Return top-N dramas most similar to a given drama
        based on item latent factors (item-item collaborative filtering).

        Args:
            drama_name : drama title (case-insensitive)
            top_n      : number of recommendations

        Returns:
            DataFrame with [drama_name, cf_score]
        """
        if self.item_factors is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        key = drama_name.strip().lower()
        if key not in self.drama_enc.classes_:
            logger.warning(f"Drama '{drama_name}' not found in CF index.")
            return pd.DataFrame(columns=["drama_name", "cf_score"])

        idx = self.drama_enc.transform([key])[0]
        drama_vec = self.item_factors[idx]

        scores = self.item_factors @ drama_vec
        scores[idx] = -np.inf  # exclude self

        top_indices = np.argsort(scores)[::-1][:top_n]
        top_titles = self.drama_enc.inverse_transform(top_indices)
        top_scores = scores[top_indices]

        result = pd.DataFrame({"drama_name": top_titles, "cf_score": top_scores})
        logger.info(f"Collaborative: {top_n} recommendations for '{drama_name}'")
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / "model.joblib"
        joblib.dump(self, path)
        logger.info(f"Collaborative model saved: {path}")
        return path

    @classmethod
    def load(cls) -> "CollaborativeRecommender":
        path = ARTIFACT_DIR / "model.joblib"
        logger.info(f"Loading collaborative model from {path}")
        return joblib.load(path)
