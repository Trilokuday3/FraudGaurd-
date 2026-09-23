from serving.app import SessionLocal
from serving.models import Decision
from streaming.spark_consumer import score_and_persist_row

_SAMPLE_ROW = {
    "transaction_id": "TXN_STREAM_TEST_0001",
    "customer_id": "CUST0001",
    "merchant_id": "MERC0001",
    "timestamp": "2026-01-01T12:00:00",
    "amount": 42.50,
    "payment_method": "card",
    "hour_of_day": 12,
    "is_night": False,
    "is_cross_border": False,
    "amount_vs_customer_p95": 0.5,
    "txn_count_1h": 1,
    "txn_count_24h": 3,
    "is_new_device": False,
    "device_age_days": 200.0,
    "customer_device_count_so_far": 2,
    "merchant_fraud_rate_hist": 0.01,
    "is_new_country_for_customer": False,
    "ip_country_mismatch": False,
}


def test_score_and_persist_row_writes_a_decision_row():
    score_and_persist_row(_SAMPLE_ROW)

    session = SessionLocal()
    try:
        record = (
            session.query(Decision)
            .filter(Decision.transaction_id == "TXN_STREAM_TEST_0001")
            .order_by(Decision.id.desc())
            .first()
        )
        assert record is not None
        assert record.feature_row["amount"] == 42.50
        assert record.decision in {"approve", "review", "block"}
    finally:
        session.close()
