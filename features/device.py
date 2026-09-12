"""Device features: novelty and tenure of the device used, and how many distinct
devices this customer has used before this transaction."""

import pandas as pd


def build_device_features(transactions: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    df = (
        transactions[["transaction_id", "customer_id", "device_id", "timestamp"]]
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )

    df["is_new_device"] = (df.groupby("device_id").cumcount() == 0).astype(int)

    device_first_seen = devices.set_index("device_id")["first_seen_at"]
    age_seconds = (df["timestamp"] - df["device_id"].map(device_first_seen)).dt.total_seconds()
    df["device_age_days"] = (age_seconds / 86400).clip(lower=0)

    first_device_use = (df.groupby(["customer_id", "device_id"]).cumcount() == 0).astype(int)
    distinct_devices_so_far = first_device_use.groupby(df["customer_id"]).cumsum()
    df["customer_device_count_so_far"] = (
        distinct_devices_so_far.groupby(df["customer_id"]).shift(1, fill_value=0).astype(int)
    )

    return df.set_index("transaction_id")[
        ["is_new_device", "device_age_days", "customer_device_count_so_far"]
    ]
