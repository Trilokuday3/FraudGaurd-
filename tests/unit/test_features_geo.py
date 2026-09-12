import pandas as pd

from features.geo import build_geo_features


def test_geo_features():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"]
            ),
            "country": ["US", "US", "GB"],
            "ip_country": ["US", "GB", "GB"],
        }
    )

    result = build_geo_features(transactions)

    assert result.loc["TXN1", "is_new_country_for_customer"] == 1
    assert result.loc["TXN2", "is_new_country_for_customer"] == 0
    assert result.loc["TXN3", "is_new_country_for_customer"] == 1
    assert result.loc["TXN1", "ip_country_mismatch"] == 0
    assert result.loc["TXN2", "ip_country_mismatch"] == 1
