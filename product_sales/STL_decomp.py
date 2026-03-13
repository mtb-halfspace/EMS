from load_dataset import load_dataset

import seaborn as sns
from statsmodels.tsa.seasonal import STL
import pandas as pd

df = load_dataset()

retail_products = ["Specialkaffe", "Bolle", "Rugbrød", "Hvedebrød", "Wienerbrød"]

df_daily_sales = (
    df.groupby(["Date", "Retail Product Name"]).size().reset_index(name="Count")
)

df_daily_sales = df_daily_sales.loc[
    df_daily_sales["Retail Product Name"].isin(retail_products)
].copy()


def stl_per_product(g, period=7, seasonal=7, robust=True):
    s = g.set_index("Date")["Count"].asfreq("D").fillna(0).sort_index()
    res = STL(s, period=period, seasonal=seasonal, robust=robust).fit()
    out = pd.DataFrame(
        {
            "Observed": res.observed,
            "Trend": res.trend,
            "Seasonal": res.seasonal,
            "Remainder": res.resid,
        }
    )
    out["Retail Product Name"] = g["Retail Product Name"].iloc[0]
    out.index.name = "Date"
    return out.reset_index(), res


stl_fit_results = {}
stl_dfs = []
for product, group in df_daily_sales.groupby("Retail Product Name"):
    df_out, res = stl_per_product(group)
    stl_dfs.append(df_out)
    stl_fit_results[product] = res

stl_results = pd.concat(stl_dfs, ignore_index=True).sort_values(
    ["Date", "Retail Product Name"]
)

# ------------------ Plot trend per product type ------------------
sns.lineplot(
    data=stl_results, x="Date", y="Trend", hue="Retail Product Name",
    palette="muted",
)


# ------------------ Statsmodels stock STL decomp plot per product type ------------------
for product, res in stl_fit_results.items():
    fig = res.plot()
    fig.suptitle(f"STL Decomposition – {product}", y=1.02)
    fig.tight_layout()


# ------------------ Plot trend per store for specific product ------------------
plot_product = "Bolle"
df_product_store = (
    df.groupby(["Date", "Store No."])
    .size()
    .reset_index(name="Count")
)

store_trend_dfs = []
for store, group in df_product_store.groupby("Store No."):
    s = group.set_index("Date")["Count"].asfreq("D").fillna(0).sort_index()
    res = STL(s, period=7, seasonal=7, robust=True).fit()
    trend_df = pd.DataFrame({"Date": res.trend.index, "Trend": res.trend.values})
    trend_df["Store No."] = store
    store_trend_dfs.append(trend_df)

product_store_trends = pd.concat(store_trend_dfs, ignore_index=True)
product_store_trends["Store No."] = product_store_trends["Store No."].astype(str)

sns.lineplot(data=product_store_trends, x="Date", y="Trend", hue="Store No.")
