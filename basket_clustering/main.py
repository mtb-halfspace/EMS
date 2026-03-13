import numpy as np
from load_dataset import load_dataset
from helpers import load_cache, save_cache
from basket_clustering.pipeline import (
    build_feature_matrix,
    fit_archetypes,
    fit_gower_clusters,
    profile_clusters,
)
from basket_clustering.visualization import (
    plot_dendrogram,
    plot_category_heatmap,
    plot_scalar_medians,
    plot_cluster_hour_distribution,
    plot_cluster_day_of_week_distribution,
    plot_cluster_product_counts,
)

# ── Configuration ────────────────────────────────────────────────────────────
analysis_level = "Retail Product Name"
txn_key = ["Store No.", "Transaction No."]
sample_size = 20_000
archetype_height = 0.9
cluster_height = 0.3

# ── Load data & build features ───────────────────────────────────────────────
sample_df = load_cache("basket_clustering_sample")
if sample_df is not None:
    scalar_cols = ["n_items", "total_spend", "hour_of_day", "day_of_week"]
    category_cols = [c for c in sample_df.columns if c not in scalar_cols]
else:
    df = load_dataset()
    txn_features, category_cols, scalar_cols = build_feature_matrix(df, analysis_level, txn_key)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(txn_features), size=min(sample_size, len(txn_features)), replace=False)
    sample_df = txn_features.iloc[sample_idx].copy()
    save_cache(sample_df, "basket_clustering_sample")

feature_cols = category_cols + scalar_cols

Z1 = fit_archetypes(sample_df, category_cols, archetype_height)
Z2 = fit_gower_clusters(sample_df, scalar_cols, cluster_height)

# ── Profile ──────────────────────────────────────────────────────────────────
cluster_profiles, cluster_summary, cluster_size = profile_clusters(
    sample_df, feature_cols, category_cols, scalar_cols,
)

print("\n── Cluster profiles ───────────────────────────────────────────────")
print(cluster_summary.to_string())
print()

# ── Visualise ────────────────────────────────────────────────────────────────
plot_dendrogram(Z1, ylabel="Cosine distance", title="Stage 1 — Composition archetypes (cosine, average linkage)")
plot_dendrogram(Z2, ylabel="Gower distance", title="Stage 2 — Final clusters (Gower, average linkage)")
plot_category_heatmap(cluster_profiles, category_cols, cluster_size)
plot_scalar_medians(sample_df, scalar_cols)

# ── Cluster deep-dives ────────────────────────────────────────────────────────────────
top_clusters = cluster_size.nlargest(5).index

for c in top_clusters:
    plot_cluster_product_counts(sample_df, cluster_profiles, cluster=c, category_cols=category_cols, support_threshold=0.01)
    plot_cluster_hour_distribution(sample_df, cluster=c)
    plot_cluster_day_of_week_distribution(sample_df, cluster=c)
