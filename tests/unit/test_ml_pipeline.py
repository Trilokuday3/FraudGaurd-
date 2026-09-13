import os

import numpy as np
import pandas as pd
import pytest

from ml.__main__ import _run_training


@pytest.fixture
def tiny_data_dir(tmp_path):
    n = 300
    rng = np.random.default_rng(42)
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
    # Guarantee at least a few positive examples land in every chronological
    # split (train/val/test are contiguous thirds of this evenly-spaced
    # timestamp range, per ml.data.time_based_split). With only a ~5% base
    # fraud rate over 300 rows, seed 42 alone happens to put zero positives
    # in the 45-row validation slice, which makes
    # CalibratedClassifierCV(FrozenEstimator(...)).fit() blow up internally
    # (it runs cross_val_predict even when "frozen", and that requires both
    # classes to be present in y_val). This forces the split to always be
    # well-posed for calibration regardless of rng luck.
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


def test_run_training_end_to_end_on_tiny_fixture(tiny_data_dir, tmp_path):
    results = _run_training(data_dir=tiny_data_dir, mlflow_tracking_uri=str(tmp_path / "mlruns"))

    assert results["champion_name"] in {"xgboost", "lightgbm"}
    assert "pr_auc" in results["test_metrics"]
    assert "pr_auc" in results["baseline_test_metrics"]
    assert "shap_global_importance" in results
    assert "shap_local_example" in results
    assert "calibration_curve" in results
    # ARTIFACTS_DIR in ml/__main__.py is hardcoded to ./ml/artifacts (relative to
    # the process's working directory, not data_dir) -- this test writes real
    # files into the repo's gitignored ml/artifacts/ even though it trains on a
    # tiny fixture. Expected; no cleanup needed.
    assert os.path.exists("./ml/artifacts/calibration_curve.png")
    assert os.path.exists("./ml/artifacts/shap_global_importance.png")

    # baseline_test_metrics must reflect a CALIBRATED baseline, not raw
    # scores -- otherwise a calibrated deployed model gets compared against
    # an uncalibrated one, which can make an identical model look worse
    # than itself purely from calibration's effect on a rank-based metric.
    # When the baseline IS the deployed model, the two must be identical.
    if results["deployed_model_name"] == "baseline":
        assert results["test_metrics"]["pr_auc"] == results["baseline_test_metrics"]["pr_auc"]
        assert results["test_metrics"]["roc_auc"] == results["baseline_test_metrics"]["roc_auc"]
