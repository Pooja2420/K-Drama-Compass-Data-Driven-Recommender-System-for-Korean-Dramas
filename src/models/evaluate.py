"""
Evaluation metrics for the K-Drama recommendation engine.

Metrics:
  - RMSE               : rating prediction error (collaborative filtering)
  - Precision@K        : fraction of top-K recs that are relevant
  - Recall@K           : fraction of relevant items retrieved in top-K
  - NDCG@K             : normalised discounted cumulative gain
  - Catalog coverage   : fraction of catalog appearing in any recommendation
  - Intra-list diversity: avg pairwise dissimilarity within a rec list
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Root Mean Squared Error between actual and predicted ratings."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: list[float], predicted: list[float]) -> float:
    """Mean Absolute Error between actual and predicted ratings."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items that appear in top-K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Penalises relevant items appearing lower in the ranked list.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Average Precision at K (AP@K)."""
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k) if relevant else 0.0


# ---------------------------------------------------------------------------
# System-level metrics
# ---------------------------------------------------------------------------


def catalog_coverage(
    all_recommendations: list[list],
    catalog_size: int,
) -> float:
    """Fraction of catalog that appears in at least one recommendation list."""
    unique = {item for recs in all_recommendations for item in recs}
    return len(unique) / catalog_size if catalog_size > 0 else 0.0


def intra_list_diversity(
    rec_list: list[str],
    feature_store: pd.DataFrame,
) -> float:
    """
    Average pairwise cosine distance between recommended dramas.
    Higher = more diverse recommendations.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    ids = [i for i in rec_list if i in feature_store.index]
    if len(ids) < 2:
        return 0.0

    vecs = feature_store.loc[ids].values
    sim_matrix = cosine_similarity(vecs)
    n = len(ids)
    # Average of upper triangle (pairwise distances = 1 - similarity)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - sim_matrix[i, j]
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------


def evaluate_recommender(
    recommender,
    df_dramas: pd.DataFrame,
    df_reviews: pd.DataFrame,
    k: int = 10,
    sample_n: int = 50,
) -> dict:
    """
    Run full evaluation over a sample of query dramas.

    Args:
        recommender : any model with a .recommend(drama_name, top_n) method
        df_dramas   : cleaned dramas DataFrame
        df_reviews  : cleaned reviews DataFrame
        k           : cut-off for ranking metrics
        sample_n    : number of query dramas to evaluate over

    Returns:
        dict of metric_name → float
    """
    logger.info(f"Evaluating recommender on {sample_n} query dramas (k={k})...")

    # Build ground-truth: for each drama, relevant = dramas reviewed
    # by the same users (proxy for relevance)
    user_dramas = df_reviews.groupby("user_id")["title"].apply(set).to_dict()
    drama_users = df_reviews.groupby("title")["user_id"].apply(set).to_dict()

    sample_dramas = (
        df_dramas["drama_name"]
        .sample(min(sample_n, len(df_dramas)), random_state=42)
        .tolist()
    )

    p_scores, r_scores, ndcg_scores, ap_scores = [], [], [], []
    all_recs = []

    for drama in sample_dramas:
        key = drama.strip().lower()
        relevant_users = drama_users.get(key, set())

        # Relevant = dramas co-watched by the same users
        relevant = set()
        for user in relevant_users:
            relevant |= user_dramas.get(user, set())
        relevant.discard(key)

        recs_df = recommender.recommend(drama, top_n=k)
        if recs_df.empty:
            continue

        rec_names = recs_df["drama_name"].str.strip().str.lower().tolist()
        all_recs.append(rec_names)

        if relevant:
            p_scores.append(precision_at_k(rec_names, relevant, k))
            r_scores.append(recall_at_k(rec_names, relevant, k))
            ndcg_scores.append(ndcg_at_k(rec_names, relevant, k))
            ap_scores.append(average_precision_at_k(rec_names, relevant, k))

    coverage = catalog_coverage(all_recs, catalog_size=len(df_dramas))

    metrics = {
        f"precision@{k}": float(np.mean(p_scores)) if p_scores else 0.0,
        f"recall@{k}": float(np.mean(r_scores)) if r_scores else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"map@{k}": float(np.mean(ap_scores)) if ap_scores else 0.0,
        "catalog_coverage": coverage,
        "queries_evaluated": len(p_scores),
    }

    for name, val in metrics.items():
        logger.info(
            f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}"
        )

    return metrics
