from load_dataset import load_dataset
from helpers import build_store_features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = load_dataset()
features, a_categories = build_store_features(df)

feature_names = features.columns.tolist()
store_labels = features.index.astype(str).tolist()

# =============================================================================
# Clustering
# =============================================================================

scaler = StandardScaler()
X = scaler.fit_transform(features)

# --- Hierarchical clustering (Ward linkage) ---
Z = linkage(X, method="ward")
N_CLUSTERS = 3
cluster_labels_hier = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
features["cluster"] = cluster_labels_hier

# =============================================================================
# Visualizations
# =============================================================================

CLUSTER_PALETTE = {
    i: c for i, c in enumerate(
        ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"], start=1
    )
}

# --- Figure 1: Dendrogram ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
dendrogram(
    Z,
    labels=np.array(store_labels),
    leaf_rotation=0,
    leaf_font_size=11,
    ax=ax1,
)
ax1.set_title("Store Dendrogram (Ward Linkage)")
ax1.set_xlabel("Store No.")
ax1.set_ylabel("Distance")
plt.tight_layout()
plt.show()

# --- Figure 2: PCA 2-D scatter ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

fig2, ax2 = plt.subplots(figsize=(8, 6))
for cl in sorted(features["cluster"].unique()):
    mask = features["cluster"].values == cl
    ax2.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cl}",
        color=CLUSTER_PALETTE[cl],
        s=120,
        edgecolors="white",
        linewidths=0.8,
    )
for i, label in enumerate(store_labels):
    ax2.annotate(
        label,
        (X_pca[i, 0], X_pca[i, 1]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
    )
ax2.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
ax2.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
ax2.set_title("Store Clusters — PCA Projection")
ax2.legend()
plt.tight_layout()
plt.show()

loadings = pd.DataFrame(
    pca.components_.T, index=feature_names, columns=["PC1", "PC2"],
)

print("\n=== Top features per principal component ===")
for i, pc in enumerate(["PC1", "PC2"]):
    top = loadings[pc].abs().nlargest(5)
    print(f"\n{pc} ({pca.explained_variance_ratio_[i]:.0%} variance):")
    for feat, val in top.items():
        sign = "+" if loadings.loc[feat, pc] >= 0 else "-"
        print(f"  {sign} {feat:40s} {loadings.loc[feat, pc]:+.3f}")

# --- Figure 3: Radar chart of cluster profiles ---
radar_features = {
    "Morning share": "tod_share_morning",
    "Late-morning share": "tod_share_late_morning",
    "Afternoon share": "tod_share_early_afternoon",
    "Weekend share": "weekend_share",
    "Basket size": "basket_items",
    "Basket value": "basket_value",
    "Promo share": "promo_share",
    "Discount %": "mean_discount_pct",
}

# Add top A-class category shares by overall importance
top_cats = a_categories[:5]
for cat in top_cats:
    col = f"cat_share_{cat}"
    if col in features.columns:
        radar_features[cat] = col

radar_cols = [c for c in radar_features.values() if c in features.columns]
radar_labels = [k for k, v in radar_features.items() if v in features.columns]

# Min-max scale radar values to [0, 1] for visual comparability
radar_data = features.groupby("cluster")[radar_cols].mean()
radar_min = features[radar_cols].min()
radar_max = features[radar_cols].max()
radar_range = radar_max - radar_min
radar_range[radar_range == 0] = 1
radar_scaled = (radar_data - radar_min) / radar_range

n_vars = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]

fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for cl in sorted(radar_scaled.index):
    values = radar_scaled.loc[cl].tolist()
    values += values[:1]
    ax3.plot(angles, values, linewidth=2, label=f"Cluster {cl}", color=CLUSTER_PALETTE[cl])
    ax3.fill(angles, values, alpha=0.15, color=CLUSTER_PALETTE[cl])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_labels, size=8)
ax3.set_title("Cluster Profiles — Radar Chart", y=1.08)
ax3.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.show()

# =============================================================================
# Cluster summary table
# =============================================================================

summary_cols = ["n_transactions", "total_revenue", "basket_items", "basket_value",
                "weekend_share", "promo_share", "mean_discount_pct"] + [
    f"tod_share_{t}" for t in ["morning", "late_morning", "early_afternoon", "late_afternoon"]
    if f"tod_share_{t}" in features.columns
]
summary_cols = [c for c in summary_cols if c in features.columns]

cluster_summary = features.groupby("cluster")[summary_cols].mean().round(3)
print("\n=== Cluster Summary (mean values) ===")
print(cluster_summary.to_string())
