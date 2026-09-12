import pandas as pd

from features.behavioral import build_behavioral_features


def test_behavioral_features_use_only_prior_history():
    base_time = pd.Timestamp("2026-01-01T00:00:00")
    amounts = [10, 10, 10, 10, 10, 100]  # TXN5 is the first row with 5 prior transactions
    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i}" for i in range(6)],
            "customer_id": ["C1"] * 6,
            "timestamp": [base_time + pd.Timedelta(hours=i) for i in range(6)],
            "amount": amounts,
        }
    )

    result = build_behavioral_features(transactions)

    assert result.loc["TXN0", "amount_vs_customer_p95"] == 1.0  # no history -> neutral default
    assert result.loc["TXN4", "amount_vs_customer_p95"] == 1.0  # only 4 prior txns -> still default
    assert result.loc["TXN5", "amount_vs_customer_p95"] > 5.0  # 5 prior txns of 10 -> ratio ~= 10
