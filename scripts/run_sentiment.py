"""
Run the full Phase 4 sentiment analysis pipeline.

Steps:
  1. Load cleaned reviews from data/processed/
  2. Preprocess review text (tokenise, stem, remove stopwords)
  3. Run TextBlob baseline (fast, always runs)
  4. Optionally fine-tune & run BERT (--bert flag)
  5. Save enriched reviews to data/processed/reviews_with_sentiment.csv
  6. Save per-drama sentiment aggregation to data/processed/drama_sentiment.csv

Usage:
    python scripts/run_sentiment.py            # TextBlob only
    python scripts/run_sentiment.py --bert     # TextBlob + BERT
"""

import argparse
from pathlib import Path

import pandas as pd

from src.models.sentiment.preprocessor import preprocess_series
from src.models.sentiment.textblob_model import (
    aggregate_textblob_per_drama,
    run_textblob,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_sentiment")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Drama Sentiment Pipeline")
    parser.add_argument(
        "--bert",
        action="store_true",
        help="Also fine-tune and run BERT sentiment model",
    )
    parser.add_argument(
        "--bert-sample-frac",
        type=float,
        default=0.1,
        help="Fraction of reviews to use for BERT training (default: 0.1)",
    )
    parser.add_argument(
        "--bert-epochs",
        type=int,
        default=1,
        help="BERT training epochs (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    processed = cfg["data"]["processed_dir"]

    logger.info("=== K-Drama Compass: Sentiment Analysis Starting ===")

    # Step 1: Load
    df_reviews = pd.read_csv(f"{processed}/cleaned_reviews.csv")
    logger.info(f"Loaded {len(df_reviews)} reviews.")

    # Step 2: Preprocess text
    logger.info("Step 1/3: Preprocessing review text...")
    df_reviews["clean_text"] = preprocess_series(df_reviews["review_text"].tolist())

    # Step 3: TextBlob baseline
    logger.info("Step 2/3: Running TextBlob sentiment...")
    df_reviews = run_textblob(df_reviews)

    # Step 4: BERT (optional)
    if args.bert:
        logger.info("Step 3/3: Running BERT sentiment (this may take a while)...")
        from src.models.sentiment.bert_model import predict, save_model, train

        trainer, tokenizer, metrics = train(
            texts=df_reviews["review_text"].tolist(),
            labels=df_reviews["sentiment_label"].tolist(),
            sample_frac=args.bert_sample_frac,
            epochs=args.bert_epochs,
        )
        save_model(trainer, tokenizer)
        logger.info(f"BERT accuracy: {metrics['accuracy']:.4f}")

        # Predict on full dataset
        logger.info("Running BERT inference on all reviews...")
        df_reviews["bert_label"] = predict(
            df_reviews["review_text"].tolist(), trainer.model, tokenizer
        )
    else:
        logger.info("Step 3/3: Skipping BERT (run with --bert to enable).")

    # Step 5: Save enriched reviews
    out_reviews = PROCESSED_DIR / "reviews_with_sentiment.csv"
    df_reviews.to_csv(out_reviews, index=False)
    logger.info(f"Saved enriched reviews: {out_reviews}")

    # Step 6: Aggregate per drama
    drama_sentiment = aggregate_textblob_per_drama(df_reviews)
    out_drama = PROCESSED_DIR / "drama_sentiment.csv"
    drama_sentiment.to_csv(out_drama, index=False)
    logger.info(f"Saved drama-level sentiment: {out_drama}")

    logger.info("=== Sentiment Analysis Complete ===")


if __name__ == "__main__":
    main()
