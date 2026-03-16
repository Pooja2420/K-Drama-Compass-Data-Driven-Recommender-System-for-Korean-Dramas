"""Unit tests for data utilities: config settings, schema validation, ETL helpers."""

from __future__ import annotations

import pandas as pd
import pytest

# ── Settings / Config ─────────────────────────────────────────────────────────


class TestSettings:
    def test_settings_loads(self):
        from src.utils.config import settings

        assert settings is not None

    def test_default_app_name(self):
        from src.utils.config import settings

        assert "K-Drama" in settings.app_name

    def test_default_port(self):
        from src.utils.config import settings

        assert settings.port == 8000

    def test_default_rate_limit(self):
        from src.utils.config import settings

        assert settings.rate_limit_calls > 0
        assert settings.rate_limit_period > 0

    def test_model_weights_sum_to_one(self):
        from src.utils.config import settings

        total = settings.model.content_weight + settings.model.collab_weight
        assert total == pytest.approx(1.0, abs=0.01)

    def test_paths_are_path_objects(self):
        from pathlib import Path

        from src.utils.config import settings

        assert isinstance(settings.paths.data_raw, Path)
        assert isinstance(settings.paths.models_artifacts, Path)

    def test_env_override(self, monkeypatch):
        """Environment variable should override the default."""
        monkeypatch.setenv("PORT", "9999")
        from importlib import reload

        import src.utils.config as cfg_module

        reload(cfg_module)
        from src.utils.config import Settings

        fresh = Settings()
        assert fresh.port == 9999


# ── Auth utilities ────────────────────────────────────────────────────────────


class TestAuthUtils:
    def test_create_and_decode_token(self):
        from src.api.auth import create_access_token, decode_token

        token = create_access_token(data={"sub": "testuser"})
        token_data = decode_token(token)
        assert token_data.username == "testuser"

    def test_invalid_token_returns_none_username(self):
        from src.api.auth import decode_token

        token_data = decode_token("invalid.jwt.token")
        assert token_data.username is None

    def test_verify_correct_password(self):
        from src.api.auth import pwd_context, verify_password

        hashed = pwd_context.hash("mysecret")
        assert verify_password("mysecret", hashed) is True

    def test_verify_wrong_password(self):
        from src.api.auth import pwd_context, verify_password

        hashed = pwd_context.hash("mysecret")
        assert verify_password("wrongsecret", hashed) is False

    def test_authenticate_demo_user(self):
        from src.api.auth import authenticate_user

        assert authenticate_user("demo", "kdrama123") is True

    def test_authenticate_unknown_user(self):
        from src.api.auth import authenticate_user

        assert authenticate_user("nobody", "nopassword") is False

    def test_token_has_expiry(self):
        """Decoded payload should include an 'exp' claim."""
        from jose import jwt

        from src.api.auth import ALGORITHM, SECRET_KEY, create_access_token

        token = create_access_token(data={"sub": "testuser"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload


# ── Data fixtures sanity ──────────────────────────────────────────────────────


class TestFixtureData:
    def test_dramas_df_has_required_columns(self, dramas_df: pd.DataFrame):
        for col in ("kdrama_id", "drama_name", "genre", "rating"):
            assert col in dramas_df.columns

    def test_dramas_df_no_nulls_in_id(self, dramas_df: pd.DataFrame):
        assert dramas_df["kdrama_id"].isnull().sum() == 0

    def test_dramas_df_unique_ids(self, dramas_df: pd.DataFrame):
        assert dramas_df["kdrama_id"].nunique() == len(dramas_df)

    def test_dramas_df_rating_in_range(self, dramas_df: pd.DataFrame):
        assert (dramas_df["rating"] >= 0).all()
        assert (dramas_df["rating"] <= 10).all()

    def test_reviews_df_has_required_columns(self, reviews_df: pd.DataFrame):
        for col in ("drama_name", "review"):
            assert col in reviews_df.columns

    def test_reviews_df_non_empty(self, reviews_df: pd.DataFrame):
        assert len(reviews_df) > 0
