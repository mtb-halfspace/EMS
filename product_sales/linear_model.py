import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from load_dataset import load_dataset
from helpers import make_daily_sales_dataframe
from constants import step_t

sns.set_style("darkgrid")

retail_product = "Bolle"

df = load_dataset()

df_daily_sales = make_daily_sales_dataframe(df, retail_product)

df_daily_sales["t"] = (df_daily_sales.index - df_daily_sales.index.min()).days
df_daily_sales["day_of_week"] = df_daily_sales.index.dayofweek
df_daily_sales["intervention"] = (df_daily_sales["t"] >= step_t).astype(int)

df_daily_sales["t_centered"] = df_daily_sales["t"] - step_t
df_daily_sales["post_slope"] = (
    df_daily_sales["t_centered"] * df_daily_sales["intervention"]
)

dow_ohe = pd.get_dummies(df_daily_sales["day_of_week"], prefix="dow", drop_first=True)

x = pd.concat(
    [
        df_daily_sales[["t_centered", "intervention", "post_slope"]],
        dow_ohe,
    ],
    axis=1,
)

y = df_daily_sales["daily_sales"]

split = int(len(df_daily_sales) * 0.8)
y_train, y_test = y.iloc[:split], y.iloc[split:]
x_train, x_test = x.iloc[:split], x.iloc[split:]

lin = LinearRegression()
lin.fit(x_train, y_train)

y_pred_train = lin.predict(x_train)
y_pred_test = lin.predict(x_test)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_test))

df_daily_sales["y_pred"] = np.concatenate([y_pred_train, y_pred_test])

plt.figure(figsize=(14, 5))
sns.lineplot(
    data=df_daily_sales, x=df_daily_sales.index, y="y_pred", label="Linear model"
)
sns.lineplot(
    data=df_daily_sales, x=df_daily_sales.index, y="daily_sales", label="Actual sales"
)
plt.axvline(x=df_daily_sales.index[split], color="red", label="Train/Test split")
plt.tight_layout()
plt.show()
