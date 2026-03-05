"""Pydantic request / response schemas for the K-Drama Compass API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: str | None = None


class DramaOut(BaseModel):
    drama_name: str
    score: float


class RecommendResponse(BaseModel):
    query: str
    model: str
    top_n: int
    recommendations: list[DramaOut]


class SearchResult(BaseModel):
    drama_name: str
    genre: str | None = None
    rating: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class SentimentResponse(BaseModel):
    drama_name: str
    polarity: float
    subjectivity: float
    label: str
    review_count: int
