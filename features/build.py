"""Orchestrates all six feature families into one gold feature table."""

import os

import pandas as pd
from fraudguard_core.schemas import features_schema

from features.behavioral import build_behavioral_features
from features.device import build_device_features
from features.geo import build_geo_features
from features.merchant import build_merchant_features
from features.transaction import build_transaction_features
from features.velocity import build_velocity_features


def build_feature_table(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    devices: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    base = transactions[["transaction_id", "customer_id", "merchant_id", "timestamp"]].set_index(
        "transaction_id"
    )

    parts = [
        build_transaction_features(transactions, customers),
        build_behavioral_features(transactions),
        build_velocity_features(transactions),
        build_device_features(transactions, devices),
        build_merchant_features(transactions, ground_truth),
        build_geo_features(transactions),
    ]

    result = base
    for part in parts:
        result = result.join(part, how="left")

    result = result.reset_index()

    # pandas' `.dt.hour` accessor always returns int32 (a long-standing,
    # platform-independent pandas quirk), but features_schema declares
    # hour_of_day as int64 like the rest of the integer columns. Normalize
    # here rather than in features/transaction.py, since this module is the
    # first place the composed table is actually schema-validated.
    result["hour_of_day"] = result["hour_of_day"].astype("int64")

    features_schema.validate(result)
    return result


def write_features(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "features.parquet")
    df.to_parquet(path, index=False)
    return path
