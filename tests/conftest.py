"""Shared pytest fixtures for the K-Drama Compass test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── Synthetic data ─────────────────────────────────────────────────────────────

N_DRAMAS = 20


def _make_dramas_df(n: int = N_DRAMAS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"Drama {i}" for i in range(n)]
    return pd.DataFrame(
        {
            "kdrama_id": range(n),
            "drama_name": names,
            "genre": rng.choice(["Romance", "Action", "Thriller", "Comedy"], n),
            "rating": rng.uniform(6.0, 10.0, n).round(1),
            "episodes": rng.integers(8, 24, n),
            "synopsis": [f"A story about drama {i} with romance and action." for i in range(n)],
        }
    )


def _make_reviews_df(dramas_df: pd.DataFrame, n_reviews: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drama_names = dramas_df["drama_name"].tolist()
    return pd.DataFrame(
        {
            "drama_name": rng.choice(drama_names, n_reviews),
            "review": [
                f"This drama is really good and entertaining! Episode {i}" for i in range(n_reviews)
            ],
            "rating": rng.uniform(1.0, 10.0, n_reviews).round(1),
        }
    )


@pytest.fixture(scope="session")
def dramas_df() -> pd.DataFrame:
    return _make_dramas_df()


@pytest.fixture(scope="session")
def reviews_df(dramas_df: pd.DataFrame) -> pd.DataFrame:
    return _make_reviews_df(dramas_df)


# ── Content-based recommender fixture ─────────────────────────────────────────


@pytest.fixture(scope="session")
def fitted_content_model(dramas_df: pd.DataFrame):
    """Fit a ContentBasedRecommender on synthetic drama data."""
    from src.models.recommender.content_based import ContentBasedRecommender

    rng = np.random.default_rng(42)
    n = len(dramas_df)
    feature_store = pd.DataFrame(
        rng.random((n, 10)),
        index=dramas_df["kdrama_id"].values,
        columns=[f"feat_{i}" for i in range(10)],
    )
    model = ContentBasedRecommender()
    model.fit(feature_store, dramas_df)
    return model


# ── API client fixture ─────────────────────────────────────────────────────────


@pytest.fixture()
def api_client() -> TestClient:
    """Return a TestClient with no models loaded (models return None)."""
    from src.api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture()
def auth_token(api_client: TestClient) -> str:
    """Obtain a valid JWT token from the demo user."""
    response = api_client.post(
        "/auth/token",
        data={"username": "demo", "password": "kdrama123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture()
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}
