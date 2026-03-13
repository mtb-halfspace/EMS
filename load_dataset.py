import pandas as pd
from constants import (
    retail_product_mapping,
    filtered_retail_products_list,
    data_files,
)
from helpers import make_time_columns, make_date_columns, make_transaction_columns


def load_dataset(
    filter_retail_products: bool = True,
) -> pd.DataFrame:
    """Load the dataset from multiple Emmerys Excel files and combine them."""
    cols_transactions = [
        "Transaction No.",
        "Item No.",
        "Store No.",
        "Date",
        "Time",
        "Quantity",
        "Net Price",
        "Paid Net Amount",
        "Promotion No.",
        "Customer No.",
        "Periodic Disc. Group",
        "Disc. Amount From Std. Price",
        "Net amount Base",
        "Discount%",
    ]

    cols_skus = [
        "No.",
        "Description",
        "Item Category Code",
        "Retail Product Code",
    ]

    # Load SKUs once from the first file (they're the same across all stores)
    df_skus = pd.read_excel(
        io=data_files[0],
        sheet_name="1d. SKUs",
        header=9,
        usecols=cols_skus,
        engine="openpyxl",
    ).astype({"No.": "Int64"}, errors="ignore")

    df_skus = df_skus.rename(columns={"Description": "Item Name"})

    df_skus["Retail Product Name"] = df_skus["Retail Product Code"].map(
        retail_product_mapping
    )

    # Load transactions from all files and combine them
    dfs = []
    for file_path in data_files:
        df_transactions = pd.read_excel(
            io=file_path,
            sheet_name="1a. Sales",
            header=9,
            usecols=cols_transactions,
            engine="openpyxl",
        )
        
        # Merge with SKUs
        df_merged = df_transactions.merge(
            df_skus, left_on="Item No.", right_on="No.", how="inner"
        )
        dfs.append(df_merged)

    # Combine all store data
    df = pd.concat(dfs, ignore_index=True)

    # Keep only the date interval common to every store
    store_dates = df.groupby("Store No.")["Date"].agg(["min", "max"])
    common_start = store_dates["min"].max()
    common_end = store_dates["max"].min()
    df = df[(df["Date"] >= common_start) & (df["Date"] <= common_end)]

    df["Total Items in Transaction"] = df.groupby(["Store No.", "Transaction No."])["Item No."].transform("count")

    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="mixed",
        utc=True,
    )

    df = make_time_columns(df)
    df = make_date_columns(df)
    df = make_transaction_columns(df)

    if filter_retail_products:
        df = df[~df["Retail Product Name"].isin(filtered_retail_products_list)]

    return df
