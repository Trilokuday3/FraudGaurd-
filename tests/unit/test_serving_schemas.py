import pytest
from pydantic import ValidationError

from serving.schemas import FeatureRow


def _valid_payload():
    return {
        "transaction_id": "TXN0001",
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "timestamp": "2026-01-01T10:00:00",
        "amount": 100.0,
        "payment_method": "card",
        "hour_of_day": 10,
        "is_night": False,
        "is_cross_border": False,
        "amount_vs_customer_p95": 1.2,
        "txn_count_1h": 1,
        "txn_count_24h": 3,
        "is_new_device": False,
        "device_age_days": 30.0,
        "customer_device_count_so_far": 2,
        "merchant_fraud_rate_hist": 0.02,
        "is_new_country_for_customer": False,
        "ip_country_mismatch": False,
    }


def test_feature_row_accepts_valid_payload():
    row = FeatureRow(**_valid_payload())
    assert row.transaction_id == "TXN0001"
    assert row.amount == 100.0


def test_feature_row_rejects_missing_field():
    payload = _valid_payload()
    del payload["amount"]
    with pytest.raises(ValidationError):
        FeatureRow(**payload)


def test_feature_row_rejects_wrong_type():
    payload = _valid_payload()
    payload["amount"] = "not-a-number"
    with pytest.raises(ValidationError):
        FeatureRow(**payload)
