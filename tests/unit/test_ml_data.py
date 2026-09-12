import pandas as pd

from ml.data import prepare_model_matrix, time_based_split


def test_time_based_split_respects_chronological_order():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="D")
    features = pd.DataFrame(
        {"transaction_id": [f"TXN{i}" for i in range(100)], "timestamp": timestamps}
    )
    labels = pd.Series([0] * 100)

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = time_based_split(
        features, labels, train_frac=0.70, val_frac=0.15
    )

    assert train_X["timestamp"].max() <= val_X["timestamp"].min()
    assert val_X["timestamp"].max() <= test_X["timestamp"].min()
    assert len(train_X) + len(val_X) + len(test_X) == 100
    assert len(train_y) == len(train_X)
    assert 60 <= len(train_X) <= 80
    assert 5 <= len(val_X) <= 25
    assert 5 <= len(test_X) <= 25


def test_prepare_model_matrix_drops_ids_and_encodes_payment_method():
    features = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2"],
            "customer_id": ["C1", "C2"],
            "merchant_id": ["M1", "M2"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "amount": [10.0, 20.0],
            "payment_method": ["card", "wallet"],
        }
    )

    matrix = prepare_model_matrix(features)

    assert "transaction_id" not in matrix.columns
    assert "customer_id" not in matrix.columns
    assert "merchant_id" not in matrix.columns
    assert "timestamp" not in matrix.columns
    assert "amount" in matrix.columns
    # all three known payment methods get a column even though only 2 appear here
    assert "payment_method_card" in matrix.columns
    assert "payment_method_wallet" in matrix.columns
    assert "payment_method_bank_transfer" in matrix.columns
    assert matrix.select_dtypes(exclude=["number", "bool"]).empty
