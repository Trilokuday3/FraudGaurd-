import pandas as pd
import pytest
from fraudguard_core.schemas import features_schema


def _valid_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TXN0000000001"],
            "customer_id": ["CUST0000001"],
            "merchant_id": ["MERC000001"],
            "timestamp": pd.to_datetime(["2026-01-01T12:00:00"]),
            "amount": [42.50],
            "payment_method": ["card"],
            "hour_of_day": [12],
            "is_night": [0],
            "is_cross_border": [0],
            "amount_vs_customer_p95": [1.0],
            "txn_count_1h": [0],
            "txn_count_24h": [0],
            "is_new_device": [1],
            "device_age_days": [0.0],
            "customer_device_count_so_far": [0],
            "merchant_fraud_rate_hist": [0.0],
            "is_new_country_for_customer": [1],
            "ip_country_mismatch": [0],
        }
    )


def test_features_schema_accepts_valid_row():
    features_schema.validate(_valid_row())


def test_features_schema_rejects_forbidden_column():
    df = _valid_row()
    df["fraud_label"] = [1]
    with pytest.raises(Exception):
        features_schema.validate(df)
