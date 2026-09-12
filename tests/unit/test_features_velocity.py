import pandas as pd

from features.velocity import build_velocity_features


def test_velocity_counts_exclude_current_transaction():
    base_time = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "timestamp": [
                base_time,
                base_time + pd.Timedelta(minutes=30),
                base_time + pd.Timedelta(hours=2),
            ],
        }
    )

    result = build_velocity_features(transactions)

    assert result.loc["TXN1", "txn_count_1h"] == 0
    assert result.loc["TXN2", "txn_count_1h"] == 1
    assert result.loc["TXN3", "txn_count_1h"] == 0
    assert result.loc["TXN3", "txn_count_24h"] == 2
