import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from load_dataset import load_dataset
from helpers import make_daily_sales_dataframe


sns.set_style("darkgrid")

retail_product = "Bolle"
seasonal_period = 7

df = load_dataset()

df_daily_sales = make_daily_sales_dataframe(df, retail_product)

y = df_daily_sales["daily_sales"]

split = int(len(df_daily_sales) * 0.8)
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = ETSModel(
    y_train,
    error="add",
    trend="add",
    damped_trend=True,
    seasonal="mul",
    seasonal_periods=seasonal_period,
)
res = model.fit()  # type: ignore[assignment]
s_params = res.params  # type: ignore[union-attr]
s_params.index = res.param_names  # type: ignore[union-attr]

y_pred_train = res.fittedvalues  # type: ignore[union-attr]
pred_test = res.get_prediction(start=len(y_train), end=len(y)-1)  # type: ignore[union-attr]

summary_frame = pred_test.summary_frame(alpha=0.05)  

y_pred_test = summary_frame["mean"]
lower = summary_frame["pi_lower"]
upper = summary_frame["pi_upper"]

rmse_exp_smoothing = np.sqrt(mean_squared_error(y_test, y_pred_test))

df_daily_sales["y_pred"] = pd.concat([y_pred_train, y_pred_test])

plt.figure(figsize=(14, 5))
sns.lineplot(
    data=df_daily_sales,
    x=df_daily_sales.index,
    y="y_pred",
    label="Holt-Winter's damped model",
)
sns.lineplot(
    data=df_daily_sales, x=df_daily_sales.index, y="daily_sales", label="Actual sales"
)
plt.axvline(x=df_daily_sales.index[split], color="red", label="Train/Test split")  # type: ignore[arg-type]

plt.fill_between(
    y_test.index,
    lower.to_numpy(),
    upper.to_numpy(),
    alpha=0.2,
    label="95% prediction interval",
)

plt.legend()
plt.tight_layout()
plt.show()
