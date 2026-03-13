from load_dataset import load_dataset

import seaborn as sns
import matplotlib.pyplot as plt

df = load_dataset()

df_daily_sales = (
    df.groupby(["Date", "Retail Product Name"]).size().reset_index(name="Count")
)

retail_product = "Bolle"

df_daily_sales = df_daily_sales.loc[
    df_daily_sales["Retail Product Name"] == retail_product
][["Date", "Count"]].copy()


moving_avg_window = 7
df_daily_sales["Moving_Avg"] = (
    df_daily_sales["Count"].rolling(window=moving_avg_window).mean()
)
df_daily_sales["Detrended"] = df_daily_sales["Count"] - df_daily_sales["Moving_Avg"]
df_daily_sales["Day of week"] = df_daily_sales["Date"].dt.weekday

df_daily_sales["Seasonal"] = df_daily_sales.groupby("Day of week")[
    "Detrended"
].transform("mean")
df_daily_sales["Remainder"] = df_daily_sales["Detrended"] - df_daily_sales["Seasonal"]

sns.set_style("whitegrid")
palette = sns.color_palette("crest", 4)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(18, 10), sharex=True)

sns.lineplot(data=df_daily_sales, x="Date", y="Count", ax=ax[0])
sns.lineplot(data=df_daily_sales, x="Date", y="Moving_Avg", ax=ax[1])
sns.lineplot(data=df_daily_sales, x="Date", y="Seasonal", ax=ax[2])
sns.lineplot(data=df_daily_sales, x="Date", y="Remainder", ax=ax[3])

fig.suptitle("Daily Sales " + retail_product + " Classical Decomposition")
ax[-1].set_xlabel("Date")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(True, linestyle="--", alpha=0.4)

plt.show()
