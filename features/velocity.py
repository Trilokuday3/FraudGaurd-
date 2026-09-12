"""Velocity features: trailing transaction counts per customer, current transaction excluded."""

import pandas as pd


def build_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = (
        transactions[["transaction_id", "customer_id", "timestamp"]]
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )
    indexed = df.set_index("timestamp")

    counts_1h = (
        indexed.groupby("customer_id")["transaction_id"].rolling("1h", closed="left").count()
    )
    counts_24h = (
        indexed.groupby("customer_id")["transaction_id"].rolling("24h", closed="left").count()
    )

    df["txn_count_1h"] = counts_1h.fillna(0).to_numpy().astype(int)
    df["txn_count_24h"] = counts_24h.fillna(0).to_numpy().astype(int)

    return df.set_index("transaction_id")[["txn_count_1h", "txn_count_24h"]]
