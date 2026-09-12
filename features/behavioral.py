"""Behavioral features: how does this transaction compare to the customer's own history."""

import pandas as pd

MIN_HISTORY_FOR_RATIO = 5
DEFAULT_RATIO = 1.0


def build_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions[["transaction_id", "customer_id", "timestamp", "amount"]].sort_values(
        ["customer_id", "timestamp"]
    )

    trailing_p95 = df.groupby("customer_id", group_keys=False)["amount"].apply(
        lambda s: s.expanding(min_periods=MIN_HISTORY_FOR_RATIO).quantile(0.95).shift(1)
    )

    df = df.assign(
        amount_vs_customer_p95=(df["amount"] / (trailing_p95 + 1e-6)).fillna(DEFAULT_RATIO)
    )
    return df.set_index("transaction_id")[["amount_vs_customer_p95"]]
