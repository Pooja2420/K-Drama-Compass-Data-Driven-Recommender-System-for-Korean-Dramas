"""Unit tests for the ContentBasedRecommender and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.recommender.content_based import (
    ContentBasedRecommender,
    catalog_coverage,
    precision_at_k,
)

# ── ContentBasedRecommender ───────────────────────────────────────────────────


class TestContentBasedRecommender:
    def test_fit_returns_self(self, dramas_df: pd.DataFrame):
        rng = np.random.default_rng(0)
        n = len(dramas_df)
        feature_store = pd.DataFrame(
            rng.random((n, 5)),
            index=dramas_df["kdrama_id"].values,
        )
        model = ContentBasedRecommender()
        result = model.fit(feature_store, dramas_df)
        assert result is model

    def test_recommend_returns_dataframe(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs = fitted_content_model.recommend(drama_name, top_n=5)
        assert isinstance(recs, pd.DataFrame)

    def test_recommend_correct_columns(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs = fitted_content_model.recommend(drama_name, top_n=5)
        assert "drama_name" in recs.columns
        assert "similarity_score" in recs.columns

    def test_recommend_top_n_respected(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs = fitted_content_model.recommend(drama_name, top_n=3)
        assert len(recs) <= 3

    def test_recommend_excludes_self(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs = fitted_content_model.recommend(drama_name, top_n=5)
        assert drama_name not in recs["drama_name"].values

    def test_recommend_unknown_drama_returns_empty(
        self, fitted_content_model: ContentBasedRecommender
    ):
        recs = fitted_content_model.recommend("This Drama Does Not Exist XYZ")
        assert len(recs) == 0

    def test_recommend_similarity_scores_in_range(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs = fitted_content_model.recommend(drama_name, top_n=5)
        if len(recs) > 0:
            assert (recs["similarity_score"] >= -1.0).all()
            assert (recs["similarity_score"] <= 1.0).all()

    def test_recommend_scores_descending(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs = fitted_content_model.recommend(drama_name, top_n=10)
        if len(recs) > 1:
            scores = recs["similarity_score"].values
            assert (scores[:-1] >= scores[1:]).all(), "Scores should be in descending order"

    def test_unfitted_model_raises(self):
        model = ContentBasedRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend("Some Drama")

    def test_fit_builds_name_to_id_map(self, dramas_df: pd.DataFrame):
        rng = np.random.default_rng(0)
        n = len(dramas_df)
        feature_store = pd.DataFrame(
            rng.random((n, 5)),
            index=dramas_df["kdrama_id"].values,
        )
        model = ContentBasedRecommender()
        model.fit(feature_store, dramas_df)
        assert len(model.name_to_id) == n

    def test_case_insensitive_lookup(
        self, fitted_content_model: ContentBasedRecommender, dramas_df: pd.DataFrame
    ):
        drama_name = dramas_df["drama_name"].iloc[0]
        recs_lower = fitted_content_model.recommend(drama_name.lower(), top_n=5)
        recs_upper = fitted_content_model.recommend(drama_name.upper(), top_n=5)
        assert len(recs_lower) == len(recs_upper)

    def test_save_and_load(
        self,
        fitted_content_model: ContentBasedRecommender,
        tmp_path,
        dramas_df: pd.DataFrame,
        monkeypatch,
    ):
        """Save and reload should preserve recommendations."""
        from src.models.recommender import content_based as cb_module

        monkeypatch.setattr(cb_module, "ARTIFACT_DIR", tmp_path / "content_based")
        saved_path = fitted_content_model.save()
        assert saved_path.exists()

        loaded = ContentBasedRecommender.load()
        drama_name = dramas_df["drama_name"].iloc[0]
        recs_orig = fitted_content_model.recommend(drama_name, top_n=5)
        recs_loaded = loaded.recommend(drama_name, top_n=5)
        pd.testing.assert_frame_equal(
            recs_orig.reset_index(drop=True), recs_loaded.reset_index(drop=True)
        )


# ── Evaluation helpers ────────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_perfect_precision(self):
        recs = ["A", "B", "C"]
        relevant = ["A", "B", "C", "D"]
        assert precision_at_k(recs, relevant, k=3) == pytest.approx(1.0)

    def test_zero_precision(self):
        recs = ["X", "Y", "Z"]
        relevant = ["A", "B", "C"]
        assert precision_at_k(recs, relevant, k=3) == pytest.approx(0.0)

    def test_partial_precision(self):
        recs = ["A", "X", "B"]
        relevant = ["A", "B"]
        assert precision_at_k(recs, relevant, k=3) == pytest.approx(2 / 3)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["A", "B"], ["A"], k=0) == pytest.approx(0.0)

    def test_k_truncates_recommendations(self):
        recs = ["A", "B", "C", "D"]
        relevant = ["D"]  # only the 4th item is relevant
        assert precision_at_k(recs, relevant, k=2) == pytest.approx(0.0)


class TestCatalogCoverage:
    def test_full_coverage(self):
        recs = [["A", "B"], ["C", "D"]]
        assert catalog_coverage(recs, catalog_size=4) == pytest.approx(1.0)

    def test_partial_coverage(self):
        recs = [["A", "B"], ["A", "B"]]
        assert catalog_coverage(recs, catalog_size=4) == pytest.approx(0.5)

    def test_zero_catalog_size_returns_zero(self):
        assert catalog_coverage([["A"]], catalog_size=0) == pytest.approx(0.0)

    def test_empty_recommendations(self):
        assert catalog_coverage([], catalog_size=10) == pytest.approx(0.0)
