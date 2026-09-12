import pandas as pd
from fraudguard_core.schemas import features_schema

from features.build import build_feature_table


def _fixture():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C2"],
            "merchant_id": ["M1", "M1", "M2"],
            "device_id": ["D1", "D1", "D2"],
            "timestamp": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)],
            "amount": [10.0, 20.0, 30.0],
            "currency": ["USD", "USD", "USD"],
            "payment_method": ["card", "card", "wallet"],
            "country": ["US", "US", "GB"],
            "city": ["New York", "New York", "London"],
            "ip_country": ["US", "US", "GB"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "account_created_at": [t0 - pd.Timedelta(days=100)] * 2,
            "customer_segment": ["retail", "retail"],
            "home_country": ["US", "GB"],
            "home_city": ["New York", "London"],
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": ["D1", "D2"],
            "customer_id": ["C1", "C2"],
            "device_type": ["mobile", "mobile"],
            "first_seen_at": [t0 - pd.Timedelta(days=10), t0 - pd.Timedelta(days=5)],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "fraud_probability_true": [0.1, 0.1, 0.1],
            "fraud_label": [0, 0, 0],
            "confirmed_fraud_at": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    return transactions, customers, devices, ground_truth


def test_build_feature_table_matches_schema():
    transactions, customers, devices, ground_truth = _fixture()
    result = build_feature_table(transactions, customers, devices, ground_truth)
    features_schema.validate(result)
    assert len(result) == 3


def test_build_feature_table_is_deterministic():
    transactions, customers, devices, ground_truth = _fixture()
    r1 = build_feature_table(transactions, customers, devices, ground_truth)
    r2 = build_feature_table(transactions, customers, devices, ground_truth)
    pd.testing.assert_frame_equal(
        r1.sort_values("transaction_id").reset_index(drop=True),
        r2.sort_values("transaction_id").reset_index(drop=True),
    )
