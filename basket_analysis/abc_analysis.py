from load_dataset import load_dataset
from helpers import abc_classify

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

THRESHOLD_A = 0.80
THRESHOLD_B = 0.95
ABC_PALETTE = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}

analysis_level = "Retail Product Name"

df = load_dataset()
revenue = abc_classify(df, group_col=analysis_level, value_col="Paid Net Amount", threshold_a=THRESHOLD_A, threshold_b=THRESHOLD_B)

# --- Figure 1: Pareto chart ---

fig, ax = plt.subplots(figsize=(10, max(6, len(revenue) * 0.35)))

ax.barh(
    y=revenue[analysis_level],
    width=revenue["Revenue"],
    color=[ABC_PALETTE[c] for c in revenue["ABC"]],
    edgecolor="white",
    linewidth=0.3,
)
ax.invert_yaxis()
ax.set_xlabel("Revenue (DKK)")
ax.tick_params(axis="y", labelsize=8)

ax2 = ax.twiny()
ax2.plot(
    revenue["CumulativeShare"].values * 100,
    range(len(revenue)),
    color="black",
    linewidth=1.2,
    marker="o",
    markersize=3,
)
ax2.axvline(THRESHOLD_A * 100, color="grey", linestyle="--", linewidth=0.8)
ax2.axvline(THRESHOLD_B * 100, color="grey", linestyle="--", linewidth=0.8)
ax2.set_xlabel("Cumulative revenue %")
ax2.set_xlim(0, 105)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=ABC_PALETTE[c], label=f"Class {c}")
    for c in ["A", "B", "C"]
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True)
ax.set_title("ABC Analysis — Product Revenue Pareto Chart", fontsize=13)

plt.tight_layout()
plt.show()

# --- Figure 2: % of product portfolio per ABC class ---

total_products = len(revenue)
summary = (
    revenue.groupby("ABC")
    .size()
    .reset_index(name="ProductCount")
)
summary["PortfolioShare"] = summary["ProductCount"] / total_products * 100

fig2, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(
    summary["ABC"],
    summary["PortfolioShare"],
    color=[ABC_PALETTE[c] for c in summary["ABC"]],
    edgecolor="white",
    width=0.5,
)
ax.bar_label(bars, fmt="%.1f%%", fontsize=10, padding=3)

ax.set_xlabel("ABC Class")
ax.set_ylabel("% of product portfolio")
ax.set_title("Share of Product Portfolio per ABC Class")
ax.set_ylim(0, summary["PortfolioShare"].max() * 1.15)

plt.tight_layout()
plt.show()
