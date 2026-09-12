"""Transaction-family features: direct/derived from the row itself, no history needed."""

import pandas as pd

NIGHT_HOURS = frozenset({1, 2, 3, 4})


def build_transaction_features(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    home_country = customers.set_index("customer_id")["home_country"]

    out = transactions[["transaction_id", "amount", "payment_method", "timestamp"]].copy()
    out["hour_of_day"] = transactions["timestamp"].dt.hour
    out["is_night"] = out["hour_of_day"].isin(NIGHT_HOURS).astype(int)

    customer_home = transactions["customer_id"].map(home_country)
    out["is_cross_border"] = (transactions["country"].to_numpy() != customer_home).astype(int)

    return out.drop(columns=["timestamp"]).set_index("transaction_id")
