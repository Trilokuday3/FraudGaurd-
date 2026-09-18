import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split
from mlops.promote_model import promote_model
from mlops.rollback_model import rollback_model


def _tiny_data_dir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    n = 300
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    features = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:04d}" for i in range(n)],
            "customer_id": [f"CUST{i % 20:03d}" for i in range(n)],
            "merchant_id": [f"MERC{i % 10:03d}" for i in range(n)],
            "timestamp": timestamps,
            "amount": rng.lognormal(mean=3, sigma=1, size=n),
            "payment_method": rng.choice(["card", "wallet", "bank_transfer"], size=n),
            "hour_of_day": timestamps.hour,
            "is_night": (timestamps.hour < 5).astype(int),
            "is_cross_border": rng.integers(0, 2, size=n),
            "amount_vs_customer_p95": rng.uniform(0.5, 2.0, size=n),
            "txn_count_1h": rng.integers(0, 3, size=n),
            "txn_count_24h": rng.integers(0, 5, size=n),
            "is_new_device": rng.integers(0, 2, size=n),
            "device_age_days": rng.uniform(0, 100, size=n),
            "customer_device_count_so_far": rng.integers(0, 3, size=n),
            "merchant_fraud_rate_hist": rng.uniform(0, 0.1, size=n),
            "is_new_country_for_customer": rng.integers(0, 2, size=n),
            "ip_country_mismatch": rng.integers(0, 2, size=n),
        }
    )
    fraud_label = (rng.uniform(size=n) < 0.05).astype(int)
    guaranteed_fraud_idx = [10, 50, 100, 150, 200, 215, 225, 235, 245, 260, 270, 280, 290]
    fraud_label[guaranteed_fraud_idx] = 1
    ground_truth = pd.DataFrame(
        {
            "transaction_id": features["transaction_id"],
            "fraud_probability_true": rng.uniform(size=n),
            "fraud_label": fraud_label,
            "confirmed_fraud_at": pd.NaT,
        }
    )
    features.to_parquet(tmp_path / "features.parquet", index=False)
    ground_truth.to_parquet(tmp_path / "ground_truth.parquet", index=False)
    return str(tmp_path)


def _log_deployed_run(mlflow_dir, data_dir, experiment_name, run_name="candidate"):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment(experiment_name)

    features, labels = load_features_and_labels(data_dir)
    (train_X_raw, train_y), _, _ = time_based_split(features, labels)
    train_X = prepare_model_matrix(train_X_raw)
    model = LogisticRegression().fit(train_X, train_y)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            model, name="raw_deployed_model", serialization_format="cloudpickle"
        )
    return run_id


def test_rollback_model_swaps_champion_and_previous(tmp_path):
    data_dir = _tiny_data_dir(tmp_path / "data")
    mlflow_dir = str(tmp_path / "mlruns")
    store_dir = str(tmp_path / "store")
    thresholds_path = str(tmp_path / "thresholds.json")

    run_id_1 = _log_deployed_run(mlflow_dir, data_dir, "test-rollback-1", run_name="first")
    promote_model(
        run_id_1,
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-1",
    )

    run_id_2 = _log_deployed_run(mlflow_dir, data_dir, "test-rollback-1", run_name="second")
    promote_model(
        run_id_2,
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-1",
    )

    result = rollback_model(
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-1",
    )

    assert result["rolled_back_to_run_id"] == run_id_1
    assert result["previous_champion_run_id"] == run_id_2

    client = MlflowClient()
    champion = client.get_model_version_by_alias("test-rollback-model-1", "champion")
    previous = client.get_model_version_by_alias("test-rollback-model-1", "previous")
    assert champion.run_id == run_id_1
    assert previous.run_id == run_id_2

    with open(thresholds_path) as f:
        thresholds = json.load(f)
    assert thresholds["model_run_id"] == run_id_1


def test_rollback_model_is_reversible(tmp_path):
    """Rolling back twice returns to the original champion -- the alias
    swap is symmetric, not a one-way "forget the future" operation."""
    data_dir = _tiny_data_dir(tmp_path / "data")
    mlflow_dir = str(tmp_path / "mlruns")
    store_dir = str(tmp_path / "store")
    thresholds_path = str(tmp_path / "thresholds.json")

    run_id_1 = _log_deployed_run(mlflow_dir, data_dir, "test-rollback-2", run_name="first")
    promote_model(
        run_id_1,
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-2",
    )
    run_id_2 = _log_deployed_run(mlflow_dir, data_dir, "test-rollback-2", run_name="second")
    promote_model(
        run_id_2,
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-2",
    )

    rollback_model(
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-2",
    )
    second_result = rollback_model(
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-2",
    )

    assert second_result["rolled_back_to_run_id"] == run_id_2

    client = MlflowClient()
    champion = client.get_model_version_by_alias("test-rollback-model-2", "champion")
    assert champion.run_id == run_id_2


def test_rollback_model_raises_when_no_previous_alias_set(tmp_path):
    data_dir = _tiny_data_dir(tmp_path / "data")
    mlflow_dir = str(tmp_path / "mlruns")
    store_dir = str(tmp_path / "store")
    thresholds_path = str(tmp_path / "thresholds.json")

    run_id = _log_deployed_run(mlflow_dir, data_dir, "test-rollback-3", run_name="only")
    promote_model(
        run_id,
        mlflow_tracking_uri=mlflow_dir,
        data_dir=data_dir,
        model_store_dir=store_dir,
        thresholds_path=thresholds_path,
        registered_model_name="test-rollback-model-3",
    )

    with pytest.raises(ValueError, match="[Nn]othing to roll back"):
        rollback_model(
            mlflow_tracking_uri=mlflow_dir,
            data_dir=data_dir,
            model_store_dir=store_dir,
            thresholds_path=thresholds_path,
            registered_model_name="test-rollback-model-3",
        )
