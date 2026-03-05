"""
Data validation module.
Validates raw DataFrames against Pydantic schemas before ETL.
"""
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from src.utils.logger import get_logger

logger = get_logger("validate")


# ---------------------------------------------------------------------------
# Pydantic schemas (row-level)
# ---------------------------------------------------------------------------


class DramaRow(BaseModel):
    kdrama_id: str
    drama_name: str
    year: int
    country: str
    type: str
    tot_eps: float
    duration: float
    rank: float
    pop: float

    @field_validator("year")
    @classmethod
    def year_in_range(cls, v):
        if not (2000 <= v <= 2030):
            raise ValueError(f"Year {v} is out of expected range 2000-2030")
        return v

    @field_validator("tot_eps", "duration", "rank", "pop")
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return v


class ReviewRow(BaseModel):
    user_id: str
    title: str
    story_score: float
    acting_cast_score: float
    music_score: float
    rewatch_value_score: float
    overall_score: float
    n_helpful: int

    @field_validator(
        "story_score",
        "acting_cast_score",
        "music_score",
        "rewatch_value_score",
        "overall_score",
    )
    @classmethod
    def score_range(cls, v):
        if not (1.0 <= v <= 10.0):
            raise ValueError(f"Score {v} is out of expected range 1-10")
        return v


class ActorRow(BaseModel):
    actor_id: str
    actor_name: str
    drama_name: str
    character_name: str
    role: str


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------


def _validate_df(df: pd.DataFrame, schema: type[BaseModel], name: str) -> bool:
    errors = []
    for idx, row in df.iterrows():
        try:
            schema(**row.to_dict())
        except ValidationError as e:
            errors.append((idx, e))

    if errors:
        logger.warning(f"{name}: {len(errors)} rows failed validation")
        for idx, e in errors[:5]:  # log first 5 only
            logger.debug(f"  Row {idx}: {e}")
        return False

    logger.info(f"{name}: all {len(df)} rows passed validation")
    return True


def validate_dramas(df: pd.DataFrame) -> bool:
    cols = [
        "kdrama_id",
        "drama_name",
        "year",
        "country",
        "type",
        "tot_eps",
        "duration",
        "rank",
        "pop",
    ]
    return _validate_df(df[cols].dropna(subset=["kdrama_id"]), DramaRow, "dramas")


def validate_reviews(df: pd.DataFrame) -> bool:
    cols = [
        "user_id",
        "title",
        "story_score",
        "acting_cast_score",
        "music_score",
        "rewatch_value_score",
        "overall_score",
        "n_helpful",
    ]
    return _validate_df(df[cols], ReviewRow, "reviews")


def validate_actors(df: pd.DataFrame) -> bool:
    cols = ["actor_id", "actor_name", "drama_name", "character_name", "role"]
    return _validate_df(df[cols], ActorRow, "actors")


def validate_all(datasets: dict[str, pd.DataFrame]) -> dict[str, bool]:
    return {
        "dramas": validate_dramas(datasets["dramas"]),
        "reviews": validate_reviews(datasets["reviews"]),
        "actors": validate_actors(datasets["actors"]),
    }
