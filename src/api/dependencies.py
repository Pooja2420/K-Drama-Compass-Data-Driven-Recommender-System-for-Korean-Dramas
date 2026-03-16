"""Shared FastAPI dependencies: cached model loaders and auth guard."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.api.auth import decode_token
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("api.dependencies")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

_processed = settings.paths.data_processed


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_dramas_df() -> pd.DataFrame:
    """Load and cache the cleaned dramas DataFrame."""
    path = _processed / settings.data.dramas_file.replace(".csv", "") / "cleaned_dramas.csv"
    # Fallback: look directly in processed dir
    direct = _processed / "cleaned_dramas.csv"
    target = direct if direct.exists() else path
    if not target.exists():
        logger.warning("Dramas CSV not found: %s", target)
        return pd.DataFrame()
    return pd.read_csv(target)


@lru_cache(maxsize=1)
def get_reviews_df() -> pd.DataFrame:
    """Load and cache the cleaned reviews DataFrame."""
    path = _processed / "cleaned_reviews.csv"
    if not path.exists():
        logger.warning("Reviews CSV not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _safe_load(loader_fn):
    """Call loader_fn(); return None on any exception."""
    try:
        return loader_fn()
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_content_based_model():
    """Load and cache the content-based recommender (None if not trained)."""
    from src.models.recommender.content_based import ContentBasedRecommender

    return _safe_load(ContentBasedRecommender.load)


@lru_cache(maxsize=1)
def get_collaborative_model():
    """Load and cache the collaborative recommender (None if not trained)."""
    from src.models.recommender.collaborative import CollaborativeRecommender

    return _safe_load(CollaborativeRecommender.load)


@lru_cache(maxsize=1)
def get_hybrid_model():
    """Load and cache the hybrid recommender (None if not trained)."""
    from src.models.recommender.hybrid import HybridRecommender

    return _safe_load(HybridRecommender.load)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> str:
    """Validate the Bearer JWT token and return the username."""
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = decode_token(token)
    if not token_data.username:
        raise exc
    return token_data.username
