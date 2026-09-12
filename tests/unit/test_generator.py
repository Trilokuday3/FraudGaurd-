import pandas as pd
import pytest
from fraudguard_core.config import GeneratorSettings

from generator.generate import generate


@pytest.fixture
def small_config():
    return GeneratorSettings(
        seed=7,
        n_customers=200,
        n_merchants=50,
        avg_devices_per_customer=1.3,
        avg_txns_per_customer=20,
        target_fraud_rate=0.02,
        vintage_start="2025-03-01",
        vintage_end="2026-09-01",
    )


def _content_hash(df: pd.DataFrame) -> str:
    sorted_df = df.sort_values(list(df.columns)).reset_index(drop=True)
    return pd.util.hash_pandas_object(sorted_df, index=False).sum().__str__()


def test_generator_determinism(small_config):
    r1 = generate(small_config)
    r2 = generate(small_config)

    assert r1.meta["row_counts"] == r2.meta["row_counts"]
    for name in ("customers", "merchants", "devices", "transactions", "ground_truth"):
        df1, df2 = getattr(r1, name), getattr(r2, name)
        assert _content_hash(df1) == _content_hash(df2), f"{name} not deterministic"


def test_fraud_rate_within_tolerance():
    config = GeneratorSettings(
        seed=11,
        n_customers=3000,
        n_merchants=300,
        avg_txns_per_customer=25,
        target_fraud_rate=0.015,
    )
    result = generate(config)
    realised = result.meta["realised_fraud_rate"]
    assert (
        abs(realised - config.target_fraud_rate) < 0.01
    ), f"realised={realised:.4%} target={config.target_fraud_rate:.4%}"


def test_forbidden_columns_absent(small_config):
    result = generate(small_config)
    assert "base_fraud_rate" not in result.merchants.columns
    assert "fraud_label" not in result.transactions.columns
    assert "fraud_probability_true" not in result.transactions.columns
    assert "confirmed_fraud_at" not in result.transactions.columns


def test_referential_integrity(small_config):
    result = generate(small_config)

    customer_ids = set(result.customers["customer_id"])
    merchant_ids = set(result.merchants["merchant_id"])
    device_ids = set(result.devices["device_id"])
    txn_ids = set(result.transactions["transaction_id"])

    assert set(result.devices["customer_id"]) <= customer_ids
    assert set(result.transactions["customer_id"]) <= customer_ids
    assert set(result.transactions["merchant_id"]) <= merchant_ids
    assert set(result.transactions["device_id"]) <= device_ids
    assert set(result.ground_truth["transaction_id"]) == txn_ids


def test_device_ownership(small_config):
    result = generate(small_config)
    dev_owner = result.devices.set_index("device_id")["customer_id"]
    merged = result.transactions.assign(
        device_owner=result.transactions["device_id"].map(dev_owner)
    )
    assert (merged["device_owner"] == merged["customer_id"]).all()


def test_ground_truth_is_1to1_with_transactions(small_config):
    result = generate(small_config)
    assert len(result.ground_truth) == len(result.transactions)
    assert result.ground_truth["transaction_id"].is_unique
