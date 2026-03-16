"""Health-check endpoint — GET /health."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import (
    get_collaborative_model,
    get_content_based_model,
    get_hybrid_model,
)
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return API health status and which ML models are currently loaded."""
    models_loaded = {
        "content_based": get_content_based_model() is not None,
        "collaborative": get_collaborative_model() is not None,
        "hybrid": get_hybrid_model() is not None,
    }
    return HealthResponse(status="ok", models_loaded=models_loaded)
