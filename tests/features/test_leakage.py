"""Leakage + schema gate against the real generated ./data/features.parquet.

Run `make seed features` first (or `make seed && make features`). Mirrors
tests/dq/test_dq_suite.py: these tests validate committed output, they don't
regenerate it.
"""

import os

import pandas as pd
import pytest
from fraudguard_core.schemas import features_schema

DATA_DIR = os.environ.get("FRAUDGUARD_DATA_DIR", "./data")

FORBIDDEN_COLUMNS = (
    "fraud_label",
    "fraud_probability_true",
    "confirmed_fraud_at",
    "base_fraud_rate",
)


def _load_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "features.parquet")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found -- run `make seed features` first")
    return pd.read_parquet(path)


def test_features_schema():
    features_schema.validate(_load_features())


def test_no_forbidden_columns_in_features():
    df = _load_features()
    for col in FORBIDDEN_COLUMNS:
        assert col not in df.columns, f"forbidden column {col!r} leaked into features.parquet"


def test_every_transaction_has_a_feature_row():
    transactions = pd.read_parquet(os.path.join(DATA_DIR, "transactions.parquet"))
    features = _load_features()
    assert set(features["transaction_id"]) == set(transactions["transaction_id"])
