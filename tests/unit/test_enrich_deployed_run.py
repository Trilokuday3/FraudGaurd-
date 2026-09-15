import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from ml.data import (
    load_features_and_labels,
    prepare_model_matrix,
    time_based_split,
)
from ml.enrich_deployed_run import enrich_deployed_run


def _tiny_data_dir(tmp_path):
    """Create a tiny fixture dataset with real fraud detection features."""
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


def test_enrich_deployed_run_logs_metric_and_artifact(tmp_path):
    """Test that enrich_deployed_run logs validation PR-AUC and SHAP background CSV."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    data_dir = _tiny_data_dir(tmp_path / "data")
    mlflow_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("test-enrich")

    # Load and prepare data using the real pipeline
    features, labels = load_features_and_labels(data_dir)
    (train_X_raw, train_y), _, _ = time_based_split(features, labels)
    train_X = prepare_model_matrix(train_X_raw)

    # Train model on prepared matrix so it has the correct columns
    model = LogisticRegression().fit(train_X, train_y)

    # Log the model to MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )

    # Call enrich_deployed_run
    result = enrich_deployed_run(
        run_id,
        data_dir=data_dir,
        mlflow_tracking_uri=mlflow_dir,
        sample_size=20,
        tmp_dir=str(tmp_path / "artifacts"),
    )

    # Verify result structure
    assert result["run_id"] == run_id
    assert 0.0 <= result["deployed_val_pr_auc"] <= 1.0
    assert result["shap_background_path"] is not None

    # Verify metric was logged
    client = MlflowClient()
    fetched_run = client.get_run(run_id)
    assert "deployed_val_pr_auc" in fetched_run.data.metrics

    # Verify artifact was logged
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "shap_background.csv" in artifacts
