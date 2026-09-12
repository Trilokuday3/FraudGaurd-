"""Geo features: has this customer transacted in this country before, and does
the transaction's IP-derived country match its stated country."""

import pandas as pd


def build_geo_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = (
        transactions[["transaction_id", "customer_id", "country", "ip_country", "timestamp"]]
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )

    df["is_new_country_for_customer"] = (
        df.groupby(["customer_id", "country"]).cumcount() == 0
    ).astype(int)
    df["ip_country_mismatch"] = (df["country"] != df["ip_country"]).astype(int)

    return df.set_index("transaction_id")[
        ["is_new_country_for_customer", "ip_country_mismatch"]
    ]
