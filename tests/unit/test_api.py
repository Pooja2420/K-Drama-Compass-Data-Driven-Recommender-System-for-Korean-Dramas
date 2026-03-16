"""Unit / integration tests for the K-Drama Compass FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── Health ─────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, api_client: TestClient):
        assert api_client.get("/health").status_code == 200

    def test_health_status_ok(self, api_client: TestClient):
        body = api_client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_has_models_loaded_dict(self, api_client: TestClient):
        body = api_client.get("/health").json()
        assert "models_loaded" in body
        assert isinstance(body["models_loaded"], dict)

    def test_health_models_loaded_is_false_without_artifacts(self, api_client: TestClient):
        body = api_client.get("/health").json()
        # Without trained models on disk all should be False
        for loaded in body["models_loaded"].values():
            assert loaded is False

    def test_health_has_version(self, api_client: TestClient):
        body = api_client.get("/health").json()
        assert "version" in body
        assert body["version"] != ""

    def test_health_process_time_header_present(self, api_client: TestClient):
        response = api_client.get("/health")
        assert "x-process-time-ms" in response.headers


# ── Auth ───────────────────────────────────────────────────────────────────────


class TestAuthEndpoint:
    def test_valid_login_returns_token(self, api_client: TestClient):
        response = api_client.post(
            "/auth/token",
            data={"username": "demo", "password": "kdrama123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_invalid_credentials_returns_401(self, api_client: TestClient):
        response = api_client.post(
            "/auth/token",
            data={"username": "demo", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_missing_credentials_returns_422(self, api_client: TestClient):
        response = api_client.post("/auth/token", data={})
        assert response.status_code == 422

    def test_admin_login_works(self, api_client: TestClient):
        response = api_client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200

    def test_token_type_is_bearer(self, api_client: TestClient):
        body = api_client.post(
            "/auth/token",
            data={"username": "demo", "password": "kdrama123"},
        ).json()
        assert body["token_type"] == "bearer"


# ── Recommend (JWT required) ──────────────────────────────────────────────────


class TestRecommendEndpoint:
    def test_recommend_without_token_returns_401(self, api_client: TestClient):
        response = api_client.get("/recommend?drama_name=Crash+Landing+on+You")
        assert response.status_code == 401

    def test_recommend_invalid_token_returns_401(self, api_client: TestClient):
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = api_client.get(
            "/recommend?drama_name=Crash+Landing+on+You",
            headers=headers,
        )
        assert response.status_code == 401

    def test_recommend_invalid_model_returns_400(self, api_client: TestClient, auth_headers: dict):
        response = api_client.get(
            "/recommend?drama_name=Crash+Landing+on+You&model=nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_recommend_no_model_returns_503(self, api_client: TestClient, auth_headers: dict):
        """When no model is trained, /recommend returns 503."""
        response = api_client.get(
            "/recommend?drama_name=Crash+Landing+on+You",
            headers=auth_headers,
        )
        assert response.status_code == 503

    def test_recommend_top_n_validation(self, api_client: TestClient, auth_headers: dict):
        """top_n must be between 1 and 50."""
        response = api_client.get(
            "/recommend?drama_name=Test&top_n=999",
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_recommend_with_mock_model(
        self,
        api_client: TestClient,
        auth_headers: dict,
        fitted_content_model,
    ):
        """Inject a fitted model via dependency override and check response shape."""
        from src.api.dependencies import get_content_based_model
        from src.api.main import create_app

        app = create_app()
        app.dependency_overrides[get_content_based_model] = lambda: fitted_content_model

        with TestClient(app, raise_server_exceptions=True) as client:
            # Get token from the new client
            token = client.post(
                "/auth/token",
                data={"username": "demo", "password": "kdrama123"},
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get(
                "/recommend?drama_name=Drama+0&model=content_based",
                headers=headers,
            )

        # Either 200 with recs or 404 (drama not in the tiny synthetic set)
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            body = response.json()
            assert "recommendations" in body
            assert "query" in body


# ── Search ────────────────────────────────────────────────────────────────────


class TestSearchEndpoint:
    def test_search_no_data_returns_503(self, api_client: TestClient):
        """Without processed CSVs on disk, search returns 503."""
        response = api_client.get("/search?q=crash")
        assert response.status_code == 503

    def test_search_missing_query_returns_422(self, api_client: TestClient):
        response = api_client.get("/search")
        assert response.status_code == 422

    def test_search_with_mock_data(self, dramas_df: pd.DataFrame):
        """Inject synthetic dramas and verify search logic."""
        from src.api.dependencies import get_dramas_df
        from src.api.main import create_app

        app = create_app()
        app.dependency_overrides[get_dramas_df] = lambda: dramas_df

        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.get("/search?q=Drama")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] > 0
        assert "query" in body
        assert "results" in body

    def test_search_no_match_returns_empty(self, dramas_df: pd.DataFrame):
        from src.api.dependencies import get_dramas_df
        from src.api.main import create_app

        app = create_app()
        app.dependency_overrides[get_dramas_df] = lambda: dramas_df

        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.get("/search?q=xyznonexistentdrama999")

        assert response.status_code == 200
        assert response.json()["total"] == 0


# ── Sentiment ─────────────────────────────────────────────────────────────────


class TestSentimentEndpoint:
    def test_sentiment_no_data_returns_503(self, api_client: TestClient):
        """Without sentiment CSV on disk, endpoint returns 503."""
        response = api_client.get("/sentiment/Crash Landing on You")
        assert response.status_code == 503

    def test_sentiment_with_mock_data(self, dramas_df: pd.DataFrame):
        """Inject a mock sentiment CSV and verify response shape."""
        import pandas as pd

        from src.api.main import create_app
        from src.api.routers import sentiment as s_module

        sentiment_data = pd.DataFrame(
            {
                "drama_name": dramas_df["drama_name"].tolist(),
                "tb_polarity": [0.2] * len(dramas_df),
                "tb_subjectivity": [0.5] * len(dramas_df),
                "tb_label": ["positive"] * len(dramas_df),
                "review_count": [10] * len(dramas_df),
            }
        )

        drama_name = dramas_df["drama_name"].iloc[0]

        with patch.object(s_module, "_load_sentiment", return_value=sentiment_data):
            app = create_app()
            with TestClient(app, raise_server_exceptions=True) as client:
                response = client.get(f"/sentiment/{drama_name}")

        assert response.status_code == 200
        body = response.json()
        assert "polarity" in body
        assert "subjectivity" in body
        assert "label" in body
        assert "review_count" in body
        assert -1.0 <= body["polarity"] <= 1.0
        assert 0.0 <= body["subjectivity"] <= 1.0


# ── Schemas ───────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_health_response_defaults(self):
        from src.api.schemas import HealthResponse

        h = HealthResponse(status="ok")
        assert h.version == "1.0.0"
        assert h.models_loaded == {}

    def test_token_schema(self):
        from src.api.schemas import Token

        t = Token(access_token="abc123")
        assert t.token_type == "bearer"

    def test_drama_out_schema(self):
        from src.api.schemas import DramaOut

        d = DramaOut(drama_name="Goblin", score=0.92)
        assert d.drama_name == "Goblin"
        assert d.score == pytest.approx(0.92)

    def test_recommend_response_schema(self):
        from src.api.schemas import DramaOut, RecommendResponse

        r = RecommendResponse(
            query="Goblin",
            model="hybrid",
            top_n=5,
            recommendations=[DramaOut(drama_name="Crash Landing on You", score=0.91)],
        )
        assert len(r.recommendations) == 1
