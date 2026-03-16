"""Recommendation endpoint — GET /recommend (JWT required)."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import (
    get_collaborative_model,
    get_content_based_model,
    get_current_user,
    get_hybrid_model,
)
from src.api.schemas import DramaOut, RecommendResponse
from src.utils.logger import get_logger

logger = get_logger("api.recommend")

router = APIRouter(prefix="/recommend", tags=["recommendations"])

_MODEL_LOADERS = {
    "content_based": get_content_based_model,
    "collaborative": get_collaborative_model,
    "hybrid": get_hybrid_model,
}


def _call_recommend(model, drama_name: str, top_n: int) -> pd.DataFrame:
    """
    Dispatch to the correct recommend method.

    - ContentBasedRecommender  → .recommend()
    - HybridRecommender        → .recommend()
    - CollaborativeRecommender → .recommend_for_drama()
    """
    if hasattr(model, "recommend"):
        return model.recommend(drama_name, top_n=top_n)
    if hasattr(model, "recommend_for_drama"):
        return model.recommend_for_drama(drama_name, top_n=top_n)
    return pd.DataFrame(columns=["drama_name"])


@router.get("", response_model=RecommendResponse)
async def recommend(
    drama_name: str = Query(..., description="Source drama title"),
    top_n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    model: str = Query(
        "hybrid",
        description="Model to use: content_based | collaborative | hybrid",
    ),
    _: str = Depends(get_current_user),
) -> RecommendResponse:
    """
    Return top-N drama recommendations for the given source drama.

    Requires a valid Bearer token (obtain one from **POST /auth/token**).
    """
    if model not in _MODEL_LOADERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Unknown model '{model}'. " f"Valid choices: {sorted(_MODEL_LOADERS)}"),
        )

    recommender = _MODEL_LOADERS[model]()
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model}' is not available. "
                "Run scripts/run_mlflow.py to train models first."
            ),
        )

    recs_df = _call_recommend(recommender, drama_name, top_n)
    if recs_df is None or recs_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recommendations found for '{drama_name}'.",
        )

    # Detect the score column (similarity_score / cf_score / hybrid_score)
    score_col = next(
        (c for c in recs_df.columns if "score" in c.lower()),
        recs_df.columns[-1],
    )

    recommendations = [
        DramaOut(drama_name=str(row["drama_name"]), score=float(row[score_col]))
        for _, row in recs_df.iterrows()
        if pd.notna(row["drama_name"])
    ]

    logger.info(f"Served {len(recommendations)} {model} recommendations for '{drama_name}'")
    return RecommendResponse(
        query=drama_name,
        model=model,
        top_n=top_n,
        recommendations=recommendations,
    )
