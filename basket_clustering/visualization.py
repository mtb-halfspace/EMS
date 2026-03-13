import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

sns.set_theme(style="whitegrid", font_scale=1.05)


def plot_dendrogram(Z: np.ndarray, ylabel: str, title: str) -> None:
    """Plot a truncated dendrogram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9, ax=ax)
    ax.set_xlabel("Sample cluster size", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_category_heatmap(
    cluster_profiles: pd.DataFrame,
    category_cols: list[str],
    cluster_size: pd.Series,
) -> None:
    """Plot a heatmap of mean item count per category per cluster."""
    n_clusters = len(cluster_size)
    fig, ax = plt.subplots(figsize=(16, max(4, n_clusters * 0.9)))

    heatmap_data = cluster_profiles[category_cols].T
    heatmap_data.columns = [
        f"Cluster {c}  (n={cluster_size.loc[c]:,})" for c in heatmap_data.columns
    ]

    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="mako_r", linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_title("Mean item count per category per cluster", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def plot_scalar_medians(sample_df: pd.DataFrame, scalar_cols: list[str]) -> None:
    """Plot median basket size and total spend per cluster."""
    cluster_medians = sample_df.groupby("cluster")[scalar_cols].median()
    color = sns.color_palette("mako", 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, label in zip(
        axes,
        ["n_items", "total_spend"],
        ["Median basket size (items)", "Median total spend"],
    ):
        ax.bar(
            cluster_medians.index, cluster_medians[col].values,
            color=color, edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13, pad=12)
        ax.set_xticks(cluster_medians.index)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(False)
        sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()


def plot_cluster_hour_distribution(sample_df: pd.DataFrame, cluster: int) -> None:
    """Plot hour-of-day histogram for a single cluster."""
    color = sns.color_palette("mako", 1)[0]
    cluster_data = sample_df.loc[sample_df["cluster"] == cluster, "hour_of_day"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cluster_data, bins=range(6, 19), color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Cluster {cluster} \u2014 hour-of-day distribution (n={len(cluster_data)})", fontsize=13, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_cluster_day_of_week_distribution(sample_df: pd.DataFrame, cluster: int) -> None:
    """Plot day-of-week bar chart for a single cluster."""
    weekday_color, weekend_color = sns.color_palette("mako", 7)[1], sns.color_palette("mako", 7)[5]
    cluster_data = sample_df.loc[sample_df["cluster"] == cluster, "day_of_week"].dropna()

    counts = cluster_data.value_counts().reindex(range(7), fill_value=0)
    bar_colors = [weekend_color if d >= 5 else weekday_color for d in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [_DAY_LABELS[i] for i in counts.index],
        counts.values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Day of week", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Cluster {cluster} \u2014 day-of-week distribution (n={len(cluster_data)})", fontsize=13, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_cluster_product_counts(
    sample_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster: int,
    category_cols: list[str],
    mean_threshold: float = 0.1,
    support_threshold: float = 0.05,
) -> None:
    """Plot frequency of all product-count combinations in a cluster as a horizontal bar chart."""
    means = cluster_profiles.loc[cluster, category_cols]
    selected = means[means >= mean_threshold].sort_values(ascending=False)

    if selected.empty:
        print(f"No products with mean >= {mean_threshold} in cluster {cluster}")
        return

    products = list(selected.index)
    cluster_data = sample_df.loc[sample_df["cluster"] == cluster, products]
    combo_counts = cluster_data.apply(tuple, axis=1).value_counts()
    n_cluster = len(cluster_data)
    combos = combo_counts[combo_counts / n_cluster >= support_threshold]

    def _combo_label(counts: tuple) -> str:
        parts = []
        for name, n in zip(products, counts):
            if n == 1:
                parts.append(name)
            elif n > 1:
                parts.append(f"{n}x {name}")
        return ", ".join(parts) if parts else "(none)"

    labels = [_combo_label(v) for v in combos.index]
    pct = combos.values / n_cluster * 100

    plot_df = pd.DataFrame({"Combination": labels, "Transactions": combos.values, "Pct": pct})
    plot_df = plot_df.sort_values("Transactions")

    palette = sns.color_palette("mako", n_colors=len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.55 * len(plot_df))))
    bars = ax.barh(plot_df["Combination"], plot_df["Transactions"], color=palette, edgecolor="white", linewidth=0.5)

    for bar, p in zip(bars, plot_df["Pct"]):
        ax.text(bar.get_width() + max(combos.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=9, color="0.3")

    ax.set_xlabel(f"Transactions (n={n_cluster})", fontsize=11)
    ax.set_title(
        f"Cluster {cluster} \u2014 product combinations (individual support \u2265 {mean_threshold:.0%}, unique combination support \u2265 {support_threshold:.0%})",
        fontsize=13, pad=12,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()
