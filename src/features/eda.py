"""
EDA module.
Reusable analysis and visualization functions for all three datasets.
Saves all plots to reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("eda")

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Reviews EDA
# ---------------------------------------------------------------------------


def plot_score_distributions(df_reviews: pd.DataFrame) -> None:
    score_cols = [
        "story_score",
        "acting_cast_score",
        "music_score",
        "rewatch_value_score",
        "overall_score",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(score_cols):
        sns.histplot(df_reviews[col], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution: {col.replace('_', ' ').title()}")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Frequency")

    axes[-1].set_visible(False)
    fig.suptitle("Review Score Distributions", fontsize=16, y=1.01)
    plt.tight_layout()
    _save(fig, "score_distributions.png")


def plot_review_correlation(df_reviews: pd.DataFrame) -> None:
    num_cols = [
        "story_score",
        "acting_cast_score",
        "music_score",
        "rewatch_value_score",
        "overall_score",
        "n_helpful",
    ]
    corr = df_reviews[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Review Scores Correlation Matrix")
    _save(fig, "review_correlation.png")


def plot_sentiment_distribution(df_reviews: pd.DataFrame) -> None:
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    counts = df_reviews["sentiment_label"].map(label_map).value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)
    ax.set_title("Sentiment Label Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    _save(fig, "sentiment_distribution.png")


def plot_review_skewness(df_reviews: pd.DataFrame) -> None:
    num_cols = [
        "story_score",
        "acting_cast_score",
        "music_score",
        "rewatch_value_score",
        "overall_score",
        "n_helpful",
    ]
    skewness = df_reviews[num_cols].skew()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=skewness.index, y=skewness.values, palette="mako", ax=ax)
    ax.set_title("Skewness in Review Dataset")
    ax.set_xlabel("Column")
    ax.set_ylabel("Skewness")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "review_skewness.png")


# ---------------------------------------------------------------------------
# Dramas EDA
# ---------------------------------------------------------------------------


def plot_top_dramas_by_duration(df_dramas: pd.DataFrame, top_n: int = 10) -> None:
    top = df_dramas.nlargest(top_n, "duration")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, x="duration", y="drama_name", palette="magma", ax=ax)
    ax.set_title(f"Top {top_n} K-Dramas by Episode Duration (seconds)")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Drama Name")
    _save(fig, "top_dramas_duration.png")


def plot_yearly_drama_count(df_dramas: pd.DataFrame) -> None:
    counts = df_dramas.groupby("year").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=counts, x="year", y="count", marker="o", ax=ax)
    ax.set_title("Number of K-Dramas Released Per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    _save(fig, "yearly_drama_count.png")


def plot_duration_category_distribution(df_dramas: pd.DataFrame) -> None:
    counts = df_dramas["duration_category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="coolwarm", ax=ax)
    ax.set_title("Drama Duration Category Distribution")
    ax.set_xlabel("Duration Category")
    ax.set_ylabel("Count")
    _save(fig, "duration_category.png")


def plot_top_networks(df_dramas: pd.DataFrame, top_n: int = 10) -> None:
    top = df_dramas["org_net"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top.values, y=top.index, palette="Set2", ax=ax)
    ax.set_title(f"Top {top_n} Broadcasting Networks")
    ax.set_xlabel("Drama Count")
    ax.set_ylabel("Network")
    _save(fig, "top_networks.png")


def plot_drama_correlation(df_dramas: pd.DataFrame) -> None:
    num_cols = [
        "year",
        "tot_eps",
        "duration",
        "rank",
        "pop",
        "start_month",
        "start_day_of_week",
        "duration_days",
    ]
    existing = [c for c in num_cols if c in df_dramas.columns]
    corr = df_dramas[existing].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Drama Features Correlation Matrix")
    _save(fig, "drama_correlation.png")


# ---------------------------------------------------------------------------
# Actors EDA
# ---------------------------------------------------------------------------


def plot_top_actors(df_actors: pd.DataFrame, top_n: int = 10) -> None:
    top = df_actors["actor_name"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
    ax.set_title(f"Top {top_n} Actors by Number of Drama Appearances")
    ax.set_xlabel("Number of Appearances")
    ax.set_ylabel("Actor Name")
    _save(fig, "top_actors.png")


def plot_role_distribution(df_actors: pd.DataFrame) -> None:
    counts = df_actors["role"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, palette="flare", ax=ax)
    ax.set_title("Actor Role Type Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Role")
    _save(fig, "role_distribution.png")


def plot_actor_network(df_actors: pd.DataFrame, top_n: int = 25) -> None:
    """Build and visualise a co-appearance network for the top N actors."""
    collab: dict[str, set] = {}
    for _, row in df_actors.iterrows():
        actor = row["actor_name"]
        drama = row["drama_name"]
        collab.setdefault(actor, set()).add(drama)

    top_actors = sorted(collab, key=lambda a: len(collab[a]), reverse=True)[:top_n]
    top_set = set(top_actors)

    G = nx.Graph()
    for i, row_i in df_actors[df_actors["actor_name"].isin(top_set)].iterrows():
        a1, d1 = row_i["actor_name"], row_i["drama_name"]
        for _, row_j in df_actors[
            (df_actors["actor_name"].isin(top_set))
            & (df_actors["drama_name"] == d1)
            & (df_actors["actor_name"] != a1)
        ].iterrows():
            G.add_edge(a1, row_j["actor_name"])

    logger.info(
        f"Actor network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        edge_color="gray",
        font_size=9,
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Actors Co-Appearance Network")
    _save(fig, "actor_network.png")


# ---------------------------------------------------------------------------
# Merged EDA
# ---------------------------------------------------------------------------


def plot_top_actors_by_score(df_merged: pd.DataFrame, top_n: int = 10) -> None:
    actor_scores = (
        df_merged.groupby("actor_name")
        .agg(avg_score=("overall_score", "mean"), review_count=("user_id", "count"))
        .reset_index()
        .sort_values(["avg_score", "review_count"], ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=actor_scores, x="avg_score", y="actor_name", palette="Blues_d", ax=ax
    )
    ax.set_title(f"Top {top_n} Actors by Average Drama Rating")
    ax.set_xlabel("Average Overall Score")
    ax.set_ylabel("Actor Name")
    _save(fig, "top_actors_by_score.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_eda(
    df_dramas: pd.DataFrame,
    df_reviews: pd.DataFrame,
    df_actors: pd.DataFrame,
    df_merged: pd.DataFrame,
) -> None:
    logger.info("Running full EDA...")

    # Reviews
    plot_score_distributions(df_reviews)
    plot_review_correlation(df_reviews)
    plot_sentiment_distribution(df_reviews)
    plot_review_skewness(df_reviews)

    # Dramas
    plot_top_dramas_by_duration(df_dramas)
    plot_yearly_drama_count(df_dramas)
    plot_duration_category_distribution(df_dramas)
    plot_top_networks(df_dramas)
    plot_drama_correlation(df_dramas)

    # Actors
    plot_top_actors(df_actors)
    plot_role_distribution(df_actors)
    plot_actor_network(df_actors)

    # Merged
    plot_top_actors_by_score(df_merged)

    n_plots = len(list(FIGURES_DIR.glob("*.png")))
    logger.info(f"EDA complete. {n_plots} plots saved to {FIGURES_DIR}")
