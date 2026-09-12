"""Data-quality gate against the generated Parquet in ./data.

Run `python -m generator seed` first (or `make seed`). These tests read
committed output, they don't generate it -- keeps `make dq` fast and lets it
validate exactly what a real pipeline run produced.
"""

import os

import pandas as pd
import pytest
from fraudguard_core.schemas import ALL_SCHEMAS

DATA_DIR = os.environ.get("FRAUDGUARD_DATA_DIR", "./data")


def _load(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found -- run `python -m generator seed` first")
    return pd.read_parquet(path)


@pytest.mark.parametrize(
    "entity", ["customers", "merchants", "devices", "transactions", "ground_truth"]
)
def test_entity_schema(entity):
    df = _load(entity)
    ALL_SCHEMAS[entity].validate(df)


def test_devices_reference_valid_customers():
    customers = _load("customers")
    devices = _load("devices")
    orphans = set(devices["customer_id"]) - set(customers["customer_id"])
    assert not orphans, f"{len(orphans)} devices reference unknown customers"


def test_transactions_reference_valid_customers():
    customers = _load("customers")
    transactions = _load("transactions")
    orphans = set(transactions["customer_id"]) - set(customers["customer_id"])
    assert not orphans, f"{len(orphans)} transactions reference unknown customers"


def test_transactions_reference_valid_merchants():
    merchants = _load("merchants")
    transactions = _load("transactions")
    orphans = set(transactions["merchant_id"]) - set(merchants["merchant_id"])
    assert not orphans, f"{len(orphans)} transactions reference unknown merchants"


def test_transactions_reference_valid_devices_owned_by_same_customer():
    devices = _load("devices")
    transactions = _load("transactions")
    orphans = set(transactions["device_id"]) - set(devices["device_id"])
    assert not orphans, f"{len(orphans)} transactions reference unknown devices"

    dev_owner = devices.set_index("device_id")["customer_id"]
    merged = transactions.assign(device_owner=transactions["device_id"].map(dev_owner))
    mismatched = merged[merged["device_owner"] != merged["customer_id"]]
    assert (
        mismatched.empty
    ), f"{len(mismatched)} transactions used a device owned by a different customer"


def test_ground_truth_matches_transactions_1to1():
    transactions = _load("transactions")
    ground_truth = _load("ground_truth")
    assert set(ground_truth["transaction_id"]) == set(transactions["transaction_id"])
    assert ground_truth["transaction_id"].is_unique


def test_no_forbidden_leakage_columns():
    merchants = _load("merchants")
    transactions = _load("transactions")
    assert "base_fraud_rate" not in merchants.columns
    for col in ("fraud_label", "fraud_probability_true", "confirmed_fraud_at"):
        assert col not in transactions.columns
