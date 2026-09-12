import pandas as pd

from features.device import build_device_features


def test_device_features():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "device_id": ["D1", "D1", "D2"],
            "timestamp": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)],
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": ["D1", "D2"],
            "customer_id": ["C1", "C1"],
            "first_seen_at": [t0, t0 + pd.Timedelta(hours=2)],
        }
    )

    result = build_device_features(transactions, devices)

    assert result.loc["TXN1", "is_new_device"] == 1  # D1's first use
    assert result.loc["TXN2", "is_new_device"] == 0  # D1 used again
    assert result.loc["TXN3", "is_new_device"] == 1  # D2's first use
    assert result.loc["TXN1", "customer_device_count_so_far"] == 0  # no devices used yet
    assert result.loc["TXN2", "customer_device_count_so_far"] == 1  # D1 used before this txn
    assert result.loc["TXN3", "customer_device_count_so_far"] == 1  # still just D1 before this txn
    assert result.loc["TXN2", "device_age_days"] == 1 / 24  # 1 hour since D1.first_seen_at
