import numpy as np
import pandas as pd

from mlops.drift_report import generate_drift_report
from serving.models import Decision, make_session_factory


def _tiny_data_dir(tmp_path, amount_mean=3.0):
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
            "amount": rng.lognormal(mean=amount_mean, sigma=1, size=n),
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


def _seed_decisions_from_features(db_url, features_df):
    session_factory = make_session_factory(db_url)
    session = session_factory()
    try:
        for _, row in features_df.iterrows():
            feature_row = row.to_dict()
            feature_row["timestamp"] = feature_row["timestamp"].isoformat()
            for bool_col in [
                "is_night",
                "is_cross_border",
                "is_new_device",
                "is_new_country_for_customer",
                "ip_country_mismatch",
            ]:
                feature_row[bool_col] = bool(feature_row[bool_col])
            session.add(
                Decision(
                    transaction_id=feature_row["transaction_id"],
                    model_score=0.1,
                    decision="approve",
                    triggered_rules=[],
                    decision_source="model",
                    model_run_id="run123",
                    shap_top_features={},
                    feature_row=feature_row,
                )
            )
        session.commit()
    finally:
        session.close()


def test_generate_drift_report_produces_summary_and_html(tmp_path):
    data_dir = _tiny_data_dir(tmp_path / "data")
    db_url = f"sqlite:///{tmp_path}/test.db"

    features = pd.read_parquet(f"{data_dir}/features.parquet")
    _seed_decisions_from_features(db_url, features.sample(50, random_state=1))

    output_path = str(tmp_path / "reports" / "drift.html")
    result = generate_drift_report(data_dir=data_dir, db_url=db_url, output_path=output_path)

    assert result["checked_rows"] == 50
    assert "drifted_share" in result
    assert 0.0 <= result["drifted_share"] <= 1.0
    assert result["report_path"] == output_path
    import os

    assert os.path.exists(output_path)


def test_generate_drift_report_handles_no_scored_rows(tmp_path):
    data_dir = _tiny_data_dir(tmp_path / "data")
    db_url = f"sqlite:///{tmp_path}/empty.db"

    result = generate_drift_report(
        data_dir=data_dir,
        db_url=db_url,
        output_path=str(tmp_path / "reports" / "drift.html"),
    )

    assert result["checked_rows"] == 0
    assert result["report_path"] is None
