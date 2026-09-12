"""Load the gold feature table + labels, prepare the model matrix, and time-split."""

import pandas as pd
from fraudguard_core.enums import PAYMENT_METHOD

ID_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "timestamp"]


def load_features_and_labels(data_dir: str = "./data") -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_parquet(f"{data_dir}/features.parquet")
    ground_truth = pd.read_parquet(f"{data_dir}/ground_truth.parquet")
    labels = features[["transaction_id"]].merge(
        ground_truth[["transaction_id", "fraud_label"]], on="transaction_id", how="left"
    )["fraud_label"]
    return features, labels


def prepare_model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    matrix = features.drop(columns=[c for c in ID_COLUMNS if c in features.columns])
    matrix = matrix.copy()
    matrix["payment_method"] = pd.Categorical(
        matrix["payment_method"], categories=sorted(PAYMENT_METHOD)
    )
    return pd.get_dummies(matrix, columns=["payment_method"])


def time_based_split(
    features: pd.DataFrame,
    labels: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    start = features["timestamp"].min()
    end = features["timestamp"].max()
    span = end - start
    train_cutoff = start + span * train_frac
    val_cutoff = start + span * (train_frac + val_frac)

    train_mask = features["timestamp"] < train_cutoff
    val_mask = (features["timestamp"] >= train_cutoff) & (features["timestamp"] < val_cutoff)
    test_mask = features["timestamp"] >= val_cutoff

    return (
        (features[train_mask], labels[train_mask]),
        (features[val_mask], labels[val_mask]),
        (features[test_mask], labels[test_mask]),
    )
