import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from sklearn.metrics import root_mean_squared_error

from load_dataset import load_dataset
from helpers import make_daily_sales_dataframe, compute_benchmark_std

sns.set_style("darkgrid")

df = load_dataset()

retail_product = "Bolle"
seasonal_period = 7

y = make_daily_sales_dataframe(df, retail_product)[
    "daily_sales"
]  # .iloc[170:]  # skip first 170 days to skip step

split = int(len(y) * 0.8)
y_train, y_test = y.iloc[:split], y.iloc[split:]
x = np.arange(len(y))
x_train, x_test = x[:split], x[split:]

stl = STL(y_train, period=seasonal_period, seasonal=seasonal_period)
res = stl.fit()

y_mean = pd.Series(y_train.mean(), index=y.index)

y_naive = pd.Series(y_train.iloc[-1], index=y.index)

y_seasonal = y.index.to_series().dt.dayofweek.map(
    res.seasonal.groupby(res.seasonal.index.dayofweek).mean()
)

y_seasonal_mean = pd.Series(y_mean + y_seasonal, index=y.index)

last_by_dow = y_train.groupby(y_train.index.dayofweek).last()
y_seasonal_naive = y.index.to_series().dt.dayofweek.map(last_by_dow)
y_seasonal_naive = pd.Series(
    y_seasonal_naive.values,
    index=y.index,
)

sigma_seasonal_naive = compute_benchmark_std(
    y_train=y_train,
    y_test=y_test,
    y_pred=y_seasonal_naive,
    method="Seasonal Naive",
    seasonal_period=seasonal_period,
)


# Eval
rmse_dict = {
    "Mean": round(root_mean_squared_error(y_test, y_mean.iloc[-len(y_test) :]), 2),
    "Seasonal Mean": round(
        root_mean_squared_error(y_test, y_seasonal_mean.iloc[-len(y_test) :]), 2
    ),
    "Naive": round(root_mean_squared_error(y_test, y_naive.iloc[-len(y_test) :]), 2),
    "Seasonal Naive": round(
        root_mean_squared_error(y_test, y_seasonal_naive.iloc[-len(y_test) :]), 2
    ),
}

df_rmse = (
    pd.DataFrame.from_dict(rmse_dict, orient="index", columns=["rmse"])
    .reset_index()
    .rename(columns={"index": "Estimator"})
)

# plot rmse
plt.figure(figsize=(12, 4))
sns.barplot(data=df_rmse, x="Estimator", y="rmse")
plt.title(retail_product)

# plot time series
plt.figure(figsize=(10, 5))

# observed
sns.lineplot(y_train, color="black", linewidth=1.5, alpha=0.8)
sns.lineplot(y_test, color="gray", linewidth=1.5, alpha=0.4)

# forecast
y_pred_train = y_seasonal_naive.iloc[: -len(y_test)]
y_pred_test = y_seasonal_naive.iloc[-len(y_test) :]
sns.lineplot(
    y_pred_train,
    color="royalblue",
    linewidth=1.5,
    linestyle="--",
)
sns.lineplot(y_pred_test, color="royalblue", linewidth=1.5)

# error bands
plt.fill_between(
    y_test.index,
    y_pred_test - sigma_seasonal_naive,
    y_pred_test + sigma_seasonal_naive,
    color="royalblue",
    alpha=0.25,
    label="±1σ",
)

plt.fill_between(
    y_test.index,
    y_pred_test - 2 * sigma_seasonal_naive,
    y_pred_test + 2 * sigma_seasonal_naive,
    color="royalblue",
    alpha=0.1,
    label="±2σ",
)

plt.title(retail_product)
plt.legend()

plt.show()
