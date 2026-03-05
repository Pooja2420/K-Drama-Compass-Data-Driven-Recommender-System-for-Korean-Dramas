"""
Text preprocessing module.
Cleans and normalises review text before feeding into sentiment models.
Steps: lowercase → remove punctuation → tokenise → remove stopwords → stem.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src.utils.logger import get_logger

logger = get_logger("preprocessor")

# Download required NLTK assets (no-op if already present)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single review string.
    Returns a cleaned, stemmed, space-joined string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Tokenise
    tokens = word_tokenize(text)

    # 5. Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]

    # 6. Stem
    tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(texts) -> list[str]:
    """Apply clean_text to an iterable of strings. Returns a list."""
    cleaned = [clean_text(t) for t in texts]
    logger.info(f"Preprocessed {len(cleaned)} texts.")
    return cleaned
