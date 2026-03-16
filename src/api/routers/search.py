"""Drama search endpoint — GET /search."""

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import get_dramas_df
from src.api.schemas import SearchResponse, SearchResult
from src.utils.logger import get_logger

logger = get_logger("api.search")

router = APIRouter(prefix="/search", tags=["search"])

# Column name candidates (ordered by preference)
_NAME_COLS = ("drama_name", "name", "title")
_GENRE_COLS = ("genre", "genres", "Genre")
_RATING_COLS = ("score", "rating", "overall_score", "Rating")


def _first_col(row: pd.Series, candidates: tuple) -> str | None:
    """Return the value of the first candidate column that exists and is not NaN."""
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return None


def _first_float(row: pd.Series, candidates: tuple) -> float | None:
    """Return the float value of the first candidate column that is numeric."""
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (ValueError, TypeError):
                pass
    return None


@router.get("", response_model=SearchResponse)
async def search_dramas(
    q: str = Query(..., min_length=1, description="Search query (title substring)"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
) -> SearchResponse:
    """
    Search dramas by title using a case-insensitive substring match.

    This endpoint is public (no auth required).
    """
    df = get_dramas_df()
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("Drama data is not available. " "Run scripts/run_pipeline.py first."),
        )

    # Resolve name column
    name_col = next((c for c in _NAME_COLS if c in df.columns), df.columns[0])

    mask = df[name_col].str.contains(q, case=False, na=False)
    matched = df[mask].head(limit)

    results = [
        SearchResult(
            drama_name=str(row[name_col]),
            genre=_first_col(row, _GENRE_COLS),
            rating=_first_float(row, _RATING_COLS),
        )
        for _, row in matched.iterrows()
    ]

    logger.info(f"Search '{q}' returned {len(results)} results")
    return SearchResponse(query=q, results=results, total=len(results))
