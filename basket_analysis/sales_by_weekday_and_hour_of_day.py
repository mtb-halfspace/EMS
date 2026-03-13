from load_dataset import load_dataset
from helpers import abc_classify

import math
import seaborn as sns
import matplotlib.pyplot as plt

df = load_dataset()

abc = abc_classify(df, "Retail Product Name", "Paid Net Amount")
a_categories = abc.loc[abc["ABC"] == "A", "Retail Product Name"].tolist()


df_counts = (
    df[df["Retail Product Name"].isin(a_categories)]
    .groupby(
        [
            "Retail Product Name",
            df["Datetime"].dt.floor("h"),
            "hour_of_day",
            "day_of_week",
        ]
    )
    .size()
    .reset_index(name="Count")
)

# layout
n = len(a_categories)
ncols = 2 if n > 1 else 1
nrows = math.floor(n / ncols)

# create subplots
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(6.5 * ncols, 3.8 * nrows),
    sharex=True,
    sharey=False,
)

axes = axes.ravel() if hasattr(axes, "ravel") else list(axes)

# plot each product
for i, product in enumerate(a_categories):
    ax = axes[i]
    data_i = df_counts[df_counts["Retail Product Name"] == product]

    sns.lineplot(
        data=data_i,
        x="hour_of_day",
        y="Count",
        hue="day_of_week",
        ax=ax,
        legend=(i == 0),  # show legend only on first subplot
    )
    ax.set_title(product)
    ax.set_xlim(right=17)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()