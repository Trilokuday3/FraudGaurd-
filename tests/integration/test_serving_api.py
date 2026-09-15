import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from ml.calibration import calibrate
from ml.data import prepare_model_matrix


def _synthetic_raw_rows(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Raw (pre-prepare_model_matrix) feature rows, same shape as
    fraudguard_core.schemas.features_schema. The tiny model below is
    trained on these run through the REAL prepare_model_matrix function
    (not a hand-rolled column list), so its column set/order can never
    drift out of sync with what serving/app.py's _feature_row_to_model_input
    produces at request time -- confirmed via direct verification that
    prepare_model_matrix's actual output is
    ['amount', 'hour_of_day', 'is_night', 'is_cross_border',
    'amount_vs_customer_p95', 'txn_count_1h', 'txn_count_24h',
    'is_new_device', 'device_age_days', 'customer_device_count_so_far',
    'merchant_fraud_rate_hist', 'is_new_country_for_customer',
    'ip_country_mismatch', 'payment_method_bank_transfer',
    'payment_method_card', 'payment_method_wallet'] -- but this fixture
    doesn't hardcode that list at all, so it stays correct even if
    prepare_model_matrix's column order ever changes."""
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
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


def _valid_feature_row(**overrides):
    row = {
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
    row.update(overrides)
    return row


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("test-serving-api")

    rng = np.random.default_rng(0)
    raw_rows = _synthetic_raw_rows(200, rng)
    X = prepare_model_matrix(raw_rows)
    y = (X["amount_vs_customer_p95"] > 1.0).astype(int)

    raw_model = LogisticRegression(max_iter=1000).fit(X, y)
    model = calibrate(raw_model, X, y, method="isotonic")
    iso = IsolationForest(random_state=42).fit(X)
    background = X.sample(20, random_state=42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            raw_model, name="raw_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            iso, name="isolation_forest_model", serialization_format="cloudpickle"
        )
        bg_path = tmp_path / "shap_background.csv"
        background.to_csv(bg_path, index=False)
        mlflow.log_artifact(str(bg_path))
        mlflow.log_metric("deployed_val_pr_auc", 0.55)
        mlflow.log_metric("test_pr_auc", 0.60)
        mlflow.log_param("deployed_model", "baseline")

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({"t_review": 0.3, "t_block": 0.7, "model_run_id": run_id})
    )

    db_path = tmp_path / "decisions.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_dir)
    monkeypatch.setenv("MLFLOW_RUN_ID", run_id)
    monkeypatch.setenv("DECISION_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("THRESHOLDS_PATH", str(thresholds_path))

    # import after env vars are set, since serving.app builds module-level
    # state (loaded model, session factory) at import time
    import importlib

    import serving.app as app_module

    importlib.reload(app_module)

    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def test_health(app_client):
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_persists_and_returns_a_decision(app_client):
    response = app_client.post("/score", json=_valid_feature_row())
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TXN0001"
    assert body["decision"] in {"approve", "review", "block"}
    assert 0.0 <= body["model_score"] <= 1.0


def test_score_batch_returns_one_result_per_input(app_client):
    rows = [_valid_feature_row(transaction_id=f"TXN{i:04d}") for i in range(3)]
    response = app_client.post("/score/batch", json=rows)
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_investigation_lookup_matches_what_score_persisted(app_client):
    score_response = app_client.post("/score", json=_valid_feature_row(transaction_id="TXN9999"))
    scored = score_response.json()

    lookup_response = app_client.get("/investigations/TXN9999")
    assert lookup_response.status_code == 200
    investigated = lookup_response.json()

    assert investigated["model_score"] == scored["model_score"]
    assert investigated["decision"] == scored["decision"]
    assert investigated["triggered_rules"] == scored["triggered_rules"]


def test_investigation_lookup_404s_for_unknown_transaction(app_client):
    response = app_client.get("/investigations/DOES-NOT-EXIST")
    assert response.status_code == 404


def test_explain_returns_full_contributions(app_client):
    response = app_client.post("/explain", json=_valid_feature_row())
    assert response.status_code == 200
    contributions = response.json()["contributions"]
    assert len(contributions) > 0


def test_model_metadata(app_client):
    response = app_client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["t_review"] == 0.3
    assert body["t_block"] == 0.7
    assert body["deployed_model_name"] == "baseline"


def test_rules_force_block_regardless_of_model_score(app_client):
    row = _valid_feature_row(
        is_new_device=True, is_new_country_for_customer=True, ip_country_mismatch=True
    )
    response = app_client.post("/score", json=row)
    body = response.json()
    assert body["decision"] == "block"
    assert "device_location_takeover" in body["triggered_rules"]
    assert body["decision_source"] == "rule"


def test_scoring_is_deterministic_for_the_same_input(app_client):
    row = _valid_feature_row(transaction_id="TXN-DETERMINISTIC")
    first = app_client.post("/score", json=row).json()
    second = app_client.post("/score", json={**row, "transaction_id": "TXN-DETERMINISTIC-2"}).json()
    assert first["model_score"] == second["model_score"]
    assert first["decision"] == second["decision"]
