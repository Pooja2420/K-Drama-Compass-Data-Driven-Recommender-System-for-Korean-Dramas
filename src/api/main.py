"""K-Drama Compass — FastAPI application factory.

Endpoints
---------
POST /auth/token            — obtain a JWT access token
GET  /health                — service health + model availability
GET  /recommend             — drama recommendations  (JWT required)
GET  /search                — search dramas by title (public)
GET  /sentiment/{name}      — drama sentiment scores (public)

Swagger UI  : http://localhost:8000/docs
ReDoc       : http://localhost:8000/redoc

Run locally
-----------
    python scripts/run_api.py
    # or
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routers.auth import router as auth_router
from src.api.routers.health import router as health_router
from src.api.routers.recommend import router as recommend_router
from src.api.routers.search import router as search_router
from src.api.routers.sentiment import router as sentiment_router
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → serve → shutdown."""
    logger.info("K-Drama Compass API starting up…")
    yield
    logger.info("K-Drama Compass API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns a fully wired FastAPI instance.  Calling this function (rather
    than importing the module-level ``app`` object) makes it easy to create
    isolated instances for testing.
    """
    application = FastAPI(
        title=settings.app_name,
        description=(
            "Production-grade REST API for the K-Drama Compass recommendation system. "
            "Provides personalised drama recommendations, full-text search, "
            "and sentiment analysis powered by TextBlob and BERT."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware  (outermost wrapper executes first on requests) ──────────

    # CORS — restrict to configured origins in production
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    application.add_middleware(
        RateLimitMiddleware,
        calls=settings.rate_limit_calls,
        period=settings.rate_limit_period,
    )

    # ── Process-time header ─────────────────────────────────────────────────

    @application.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ── Global exception handler ────────────────────────────────────────────

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    # ── Routers ─────────────────────────────────────────────────────────────

    application.include_router(auth_router)
    application.include_router(health_router)
    application.include_router(recommend_router)
    application.include_router(search_router)
    application.include_router(sentiment_router)

    return application


# Module-level app used by uvicorn / gunicorn entry-points
app = create_app()
