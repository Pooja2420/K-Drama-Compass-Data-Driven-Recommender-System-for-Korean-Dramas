"""
Data ingestion module.
Loads raw CSV files from data/raw/ and returns DataFrames.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("ingest")

RAW_DIR = Path("data/raw")

EXPECTED_FILES = {
    "dramas": "korean_drama.csv",
    "reviews": "reviews.csv",
    "actors": "wiki_actors.csv",
}


def load_raw(name: str) -> pd.DataFrame:
    """Load a raw CSV by dataset name: 'dramas', 'reviews', or 'actors'."""
    if name not in EXPECTED_FILES:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(EXPECTED_FILES)}"
        )

    path = RAW_DIR / EXPECTED_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Place your CSV files in {RAW_DIR.resolve()}"
        )

    logger.info(f"Loading {name} from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all three raw datasets and return as a dict."""
    return {name: load_raw(name) for name in EXPECTED_FILES}
