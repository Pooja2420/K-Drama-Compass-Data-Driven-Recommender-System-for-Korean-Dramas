"""Application settings via pydantic-settings.

Values are read from environment variables (or a .env file).
All fields have sensible defaults so the app runs without a .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathSettings(BaseSettings):
    data_raw: Path = Field(default=Path("data/raw"))
    data_processed: Path = Field(default=Path("data/processed"))
    data_external: Path = Field(default=Path("data/external"))
    models_artifacts: Path = Field(default=Path("models/artifacts"))
    models_registry: Path = Field(default=Path("models/registry"))
    logs: Path = Field(default=Path("logs"))


class DataSettings(BaseSettings):
    dramas_file: str = Field(default="korean_drama.csv")
    reviews_file: str = Field(default="reviews.csv")
    actors_file: str = Field(default="wiki_actors.csv")


class ModelSettings(BaseSettings):
    content_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    collab_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = Field(default="K-Drama Compass API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    # Security
    secret_key: str = Field(
        default="kdrama-compass-dev-secret-change-in-prod",
        description="JWT signing secret — override in production",
    )
    access_token_expire_minutes: int = Field(default=60, ge=1)
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Comma-separated list via ALLOWED_ORIGINS env var",
    )

    # Database
    database_url: str = Field(default="postgresql://user:password@localhost:5432/kdrama")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # Logging
    log_level: str = Field(default="INFO")

    # Rate limiting
    rate_limit_calls: int = Field(default=60, ge=1)
    rate_limit_period: int = Field(default=60, ge=1)

    # Sub-settings (nested via __  delimiter)
    paths: PathSettings = Field(default_factory=PathSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)


settings = Settings()
