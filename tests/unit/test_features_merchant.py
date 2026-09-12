import pandas as pd

from features.merchant import build_merchant_features


def test_merchant_risk_respects_confirmation_delay():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C"],
            "merchant_id": ["M1", "M1", "M1"],
            "timestamp": [t0, t0 + pd.Timedelta(days=5), t0 + pd.Timedelta(days=15)],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C"],
            "fraud_label": [1, 0, 0],
            "confirmed_fraud_at": [t0 + pd.Timedelta(days=10), pd.NaT, pd.NaT],
        }
    )

    # min_history=0 isolates the confirmation-delay behavior from the
    # small-sample fallback, which is tested separately below.
    result = build_merchant_features(transactions, ground_truth, min_history=0)

    assert result.loc["B", "merchant_fraud_rate_hist"] == 0.0  # A's fraud not confirmed yet
    assert result.loc["C", "merchant_fraud_rate_hist"] == 0.5  # confirmed by now: 1 of 2 prior txns


def test_merchant_risk_falls_back_to_global_prior_below_min_history():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C", "D"],
            "merchant_id": ["M1", "M2", "M2", "M2"],
            "timestamp": [
                t0,
                t0 + pd.Timedelta(days=1),
                t0 + pd.Timedelta(days=2),
                t0 + pd.Timedelta(days=3),
            ],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C", "D"],
            "fraud_label": [1, 0, 0, 0],
            "confirmed_fraud_at": [t0 + pd.Timedelta(hours=1), pd.NaT, pd.NaT, pd.NaT],
        }
    )

    # M2 has fewer than 20 prior transactions -> falls back to the global rate.
    result = build_merchant_features(transactions, ground_truth, min_history=20)

    # By the time of D, the global population has 1 confirmed fraud (A) out of 3
    # prior transactions (A, B, C) -> global rate = 1/3.
    assert abs(result.loc["D", "merchant_fraud_rate_hist"] - (1 / 3)) < 1e-9
