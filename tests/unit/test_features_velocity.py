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


def test_velocity_multiple_customers_interleaved():
    """Verify counts are computed per-customer without cross-customer bleed.

    Input DataFrame has INTERLEAVED rows (not pre-grouped by customer) to test
    that positional alignment between groupby().rolling() output and df is correct.
    """
    base_time = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["A1", "B1", "A2", "B2", "A3", "B3"],
            "customer_id": ["CA", "CB", "CA", "CB", "CA", "CB"],
            "timestamp": [
                base_time + pd.Timedelta(minutes=0),  # A1 00:00
                base_time + pd.Timedelta(minutes=15),  # B1 00:15
                base_time + pd.Timedelta(minutes=30),  # A2 00:30
                base_time + pd.Timedelta(minutes=45),  # B2 00:45
                base_time + pd.Timedelta(minutes=90),  # A3 01:30
                base_time + pd.Timedelta(hours=2),  # B3 02:00
            ],
        }
    )

    result = build_velocity_features(transactions)

    # Customer A: 1h window counts
    assert result.loc["A1", "txn_count_1h"] == 0  # First transaction
    assert result.loc["A2", "txn_count_1h"] == 1  # A1 is 30 min before
    assert result.loc["A3", "txn_count_1h"] == 1  # A2 is 60 min before (at boundary)

    # Customer B: 1h window counts
    assert result.loc["B1", "txn_count_1h"] == 0  # First transaction
    assert result.loc["B2", "txn_count_1h"] == 1  # B1 is 45 min before
    assert result.loc["B3", "txn_count_1h"] == 0  # B2 is 75 min before (outside window)

    # Customer A: 24h window counts
    assert result.loc["A1", "txn_count_24h"] == 0  # First transaction
    assert result.loc["A2", "txn_count_24h"] == 1  # A1 is within 24h
    assert result.loc["A3", "txn_count_24h"] == 2  # A1 and A2 both within 24h

    # Customer B: 24h window counts
    assert result.loc["B1", "txn_count_24h"] == 0  # First transaction
    assert result.loc["B2", "txn_count_24h"] == 1  # B1 is within 24h
    assert result.loc["B3", "txn_count_24h"] == 2  # B1 and B2 both within 24h
