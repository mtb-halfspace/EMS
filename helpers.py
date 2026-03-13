import pandas as pd
import numpy as np
import datetime as dt
from typing import Literal, Optional
from statsmodels.tsa.seasonal import STL

def make_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional time-related columns from the 'Time' column."""
    df["hour_of_day"] = df["Time"].apply(
        lambda t: t.hour if hasattr(t, "hour") else np.nan
    )

    time_of_day_conditions = [
        df["hour_of_day"] < 9,
        (df["hour_of_day"] >= 9) & (df["hour_of_day"] < 12),
        (df["hour_of_day"] >= 12) & (df["hour_of_day"] <= 15),
        df["hour_of_day"] > 15,
    ]

    time_of_day_cats = ["morning", "late_morning", "early_afternoon", "late_afternoon"]

    df["time_of_day"] = pd.Categorical(
        np.select(time_of_day_conditions, time_of_day_cats, default=None),
        categories=time_of_day_cats,
        ordered=True,
    )

    return df


def make_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional date-related columns from the 'Date' column."""
    df["month"] = df["Date"].dt.month

    season_conditions = [
        df["month"] < 3,
        (df["month"] >= 3) & (df["month"] < 6),
        (df["month"] >= 6) & (df["month"] < 9),
        (df["month"] >= 9) & (df["month"] < 12),
        df["month"] >= 12,
    ]

    df["day_of_week"] = df["Date"].dt.weekday

    season_bins = ["winter", "spring", "summer", "autumn", "winter"]
    season_cats = ["winter", "spring", "summer", "autumn"]

    df["season"] = pd.Categorical(
        np.select(season_conditions, season_bins, default=None),
        categories=season_cats,
        ordered=True,
    )

    return df


def make_transaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional transaction-level columns."""
    df["Avg Price per Item in Transaction"] = df.groupby(["Store No.", "Transaction No."])[
        "Net Price"
    ].transform("mean")
    return df


def make_daily_sales_dataframe(df: pd.DataFrame, retail_product: str) -> pd.DataFrame:
    """Create a daily sales dataframe for a specific retail product."""
    return (
        df.loc[df["Retail Product Name"] == retail_product]
        .groupby("Date")["Retail Product Name"]
        .size()
        .rename("daily_sales")
        .to_frame()
    )


def compute_benchmark_std(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    method: Literal["Mean", "Seasonal Mean", "Naive", "Seasonal Naive"],
    stl_res: Optional[STL] = None,
    seasonal_period: int = 7,
) -> pd.Series:
    if method == "Seasonal Mean" and stl_res is None:
        raise ValueError("stl_res must be provided for 'Seasonal Mean' method.")

    T = len(y_train)
    y_pred_train = y_pred.iloc[: -len(y_test)]

    residuals = y_train - y_pred_train
    future_step = np.arange(1, len(y_test) + 1)
    complete_seasons = np.floor((future_step - 1) / seasonal_period).astype(int)

    if method == "Mean":
        sigma = residuals.std(ddof=1) * np.sqrt(1 + 1 / T)

    elif method == "Seasonal Mean":
        sigma_hat = np.std(stl_res.resid, ddof=1)
        train_counts_per_day = y_train.groupby(y_train.index.dayofweek).size()
        train_counts_per_day = (
            y_test.index.to_series()
            .dt.dayofweek.map(train_counts_per_day)
            .astype(float)
            .values
        )
        sigma = sigma_hat * np.sqrt(1 + 1 / train_counts_per_day)

    elif method == "Naive":
        sigma = residuals.std(ddof=1) * np.sqrt(future_step)
    else:
        sigma = residuals.std(ddof=1) * np.sqrt(complete_seasons + 1)
    return pd.Series(sigma, index=y_test.index)


def build_store_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a per-store feature matrix and return (features, a_categories)."""
    txn_counts = df.groupby("Store No.")["Transaction No."].nunique().rename("n_transactions")
    item_counts = df.groupby("Store No.")["Quantity"].sum().rename("total_items")
    revenue = df.groupby("Store No.")["Paid Net Amount"].sum().rename("total_revenue")

    basket = (
        df.groupby(["Store No.", "Transaction No."])
        .agg(basket_items=("Quantity", "sum"), basket_value=("Paid Net Amount", "sum"))
    )
    basket_metrics = basket.groupby("Store No.").mean()

    tod_txn = (
        df.groupby(["Store No.", "time_of_day"])["Transaction No."]
        .nunique()
        .unstack(fill_value=0)
    )
    tod_shares = tod_txn.div(tod_txn.sum(axis=1), axis=0).add_prefix("tod_share_")

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    weekend_share = (
        df.drop_duplicates(subset=["Store No.", "Transaction No."])
        .groupby("Store No.")["is_weekend"]
        .mean()
        .rename("weekend_share")
    )

    abc = abc_classify(df, "Retail Product Name", "Paid Net Amount")
    a_categories = abc.loc[abc["ABC"] == "A", "Retail Product Name"].tolist()

    cat_rev = df[df["Retail Product Name"].isin(a_categories)].pivot_table(
        index="Store No.",
        columns="Retail Product Name",
        values="Paid Net Amount",
        aggfunc="sum",
        fill_value=0,
    )
    cat_shares = cat_rev.div(cat_rev.sum(axis=1), axis=0).add_prefix("cat_share_")

    promo_share = (
        df.assign(has_promo=df["Promotion No."].notna() & (df["Promotion No."] != 0))
        .groupby("Store No.")["has_promo"]
        .mean()
        .rename("promo_share")
    )

    discount_mean = df.groupby("Store No.")["Discount%"].mean().rename("mean_discount_pct")

    features = (
        pd.concat(
            [
                txn_counts,
                item_counts,
                revenue,
                basket_metrics,
                tod_shares,
                weekend_share,
                cat_shares,
                promo_share,
                discount_mean,
            ],
            axis=1,
        )
        .fillna(0)
    )
    return features, a_categories


def abc_classify(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    threshold_a: float = 0.80,
    threshold_b: float = 0.95,
) -> pd.DataFrame:
    """Classify items into A/B/C classes based on cumulative revenue share."""
    revenue = (
        df.groupby(group_col)[value_col]
        .sum()
        .reset_index(name="Revenue")
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = revenue["Revenue"].sum()
    revenue["CumulativeShare"] = revenue["Revenue"].cumsum() / total
    revenue["ABC"] = pd.cut(
        revenue["CumulativeShare"],
        bins=[0, threshold_a, threshold_b, float("inf")],
        labels=["A", "B", "C"],
        include_lowest=True,
    )
    revenue.loc[0, "ABC"] = "A"
    return revenue


def collapse_specialty_coffees(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse specialty coffee item numbers and descriptions into a single number and description."""
    specialty_coffee_items = [
        5100,
        5101,
        5110,
        5111,
        5112,
        5113,
        5120,
        5121,
        5122,
        5123,
        5130,
        5131,
        5140,
        5141,
        5142,
    ]
    specialty_coffee_item_description = "Specialkaffe"

    df.loc[df["Item No."].isin(specialty_coffee_items), "Item No."] = 5000
    df.loc[df["Item No."] == 5000, "Item Name"] = specialty_coffee_item_description

    return df


def unpack_frozenset_columns(
    df: pd.DataFrame, cols: list[str] = ["antecedents", "consequents"]
) -> pd.DataFrame:
    """Extract the single or multiple item numbers from frozenset columns."""

    def unpack_items(x):
        if isinstance(x, frozenset):
            x = sorted(x)
            return x[0] if len(x) == 1 else tuple(x)
        return pd.NA

    for col in cols:
        df[col] = df[col].apply(unpack_items).astype("Int64", errors="ignore")

    return df
