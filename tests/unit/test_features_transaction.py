import pandas as pd

from features.transaction import build_transaction_features


def test_transaction_features_night_and_cross_border():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "amount": [10.0, 20.0, 30.0],
            "payment_method": ["card", "card", "wallet"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T02:00:00", "2026-01-01T14:00:00", "2026-01-01T14:30:00"]
            ),
            "country": ["US", "US", "GB"],
        }
    )
    customers = pd.DataFrame({"customer_id": ["C1"], "home_country": ["US"]})

    result = build_transaction_features(transactions, customers)

    assert list(result.index) == ["TXN1", "TXN2", "TXN3"]
    assert result.loc["TXN1", "is_night"] == 1
    assert result.loc["TXN2", "is_night"] == 0
    assert result.loc["TXN1", "hour_of_day"] == 2
    assert result.loc["TXN1", "is_cross_border"] == 0
    assert result.loc["TXN3", "is_cross_border"] == 1
