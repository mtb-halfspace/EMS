import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def build_feature_matrix(
    df: pd.DataFrame,
    analysis_level: str,
    txn_key: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a per-transaction feature matrix with category counts and scalar features."""
    category_matrix = (
        df.groupby(txn_key)[analysis_level]
        .value_counts()
        .unstack(fill_value=0)
    )
    category_cols = category_matrix.columns.tolist()

    scalars = df.groupby(txn_key).agg(
        n_items=("Item No.", "count"),
        total_spend=("Net Price", "sum"),
        hour_of_day=("hour_of_day", "first"),
        day_of_week=("day_of_week", "first"),
    )
    scalar_cols = scalars.columns.tolist()

    txn_features = category_matrix.join(scalars)
    return txn_features, category_cols, scalar_cols


def fit_archetypes(
    sample_df: pd.DataFrame,
    category_cols: list[str],
    height: float,
) -> np.ndarray:
    """Stage 1: cluster transactions by composition using cosine distance.

    Adds an 'archetype' column to sample_df in-place and returns the linkage matrix Z1.
    """
    dist_cosine = pdist(sample_df[category_cols].values, metric="cosine")
    Z1 = linkage(dist_cosine, method="average")
    sample_df["archetype"] = fcluster(Z1, t=height, criterion="distance")
    return Z1


def fit_gower_clusters(
    sample_df: pd.DataFrame,
    scalar_cols: list[str],
    height: float,
) -> np.ndarray:
    """Stage 2: refine archetypes with Gower distance over archetype label + scalars.

    Adds a 'cluster' column to sample_df in-place and returns the linkage matrix Z2.
    """
    n_features = 1 + len(scalar_cols)

    gower_dist = pdist(sample_df[["archetype"]].values, metric="hamming") / n_features
    for col in scalar_cols:
        col_range = sample_df[col].max() - sample_df[col].min()
        if col_range > 0:
            gower_dist += pdist(sample_df[[col]].values, metric="cityblock") / (col_range * n_features)

    Z2 = linkage(gower_dist, method="average")
    sample_df["cluster"] = fcluster(Z2, t=height, criterion="distance")
    return Z2


def profile_clusters(
    sample_df: pd.DataFrame,
    feature_cols: list[str],
    category_cols: list[str],
    scalar_cols: list[str],
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Profile clusters and return (cluster_profiles, cluster_summary, cluster_size)."""
    cluster_profiles = sample_df.groupby("cluster")[feature_cols].mean()

    cluster_size = sample_df.groupby("cluster").size().rename("n_transactions")
    cluster_summary = cluster_profiles[scalar_cols].join(cluster_size)

    top_categories = cluster_profiles[category_cols].apply(
        lambda row: ", ".join(row.nlargest(top_n).index), axis=1
    ).rename("top_categories")
    cluster_summary = cluster_summary.join(top_categories)

    archetype_dist = (
        sample_df.groupby("cluster")["archetype"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .add_prefix("archetype_")
    )
    cluster_summary = cluster_summary.join(archetype_dist)

    return cluster_profiles, cluster_summary, cluster_size
