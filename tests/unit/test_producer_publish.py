from unittest.mock import MagicMock

import pytest

from streaming.producer import publish_batch


def _producer_with_futures(n: int) -> tuple[MagicMock, list[MagicMock]]:
    futures = [MagicMock(name=f"future{i}") for i in range(n)]
    producer = MagicMock()
    producer.send.side_effect = futures
    return producer, futures


def test_publish_batch_sends_each_row_confirms_each_delivery_and_flushes():
    producer, futures = _producer_with_futures(2)
    rows = [{"transaction_id": "TXN1"}, {"transaction_id": "TXN2"}]

    publish_batch(producer, topic="fraudguard.transactions", rows=rows)

    assert producer.send.call_count == 2
    producer.send.assert_any_call("fraudguard.transactions", value=rows[0])
    producer.send.assert_any_call("fraudguard.transactions", value=rows[1])
    for future in futures:
        future.get.assert_called_once_with(timeout=30)
    producer.flush.assert_called_once()


def test_publish_batch_on_empty_rows_does_not_call_send():
    producer = MagicMock()
    publish_batch(producer, topic="fraudguard.transactions", rows=[])
    producer.send.assert_not_called()
    producer.flush.assert_called_once()


def test_publish_batch_raises_when_a_delivery_fails():
    # main() advances the stored cursor only after publish_batch returns,
    # so a raise here is what keeps the cursor from skipping lost rows.
    producer, futures = _producer_with_futures(3)
    futures[1].get.side_effect = RuntimeError("broker rejected record")

    with pytest.raises(RuntimeError, match="broker rejected record"):
        publish_batch(
            producer,
            topic="fraudguard.transactions",
            rows=[{"transaction_id": f"TXN{i}"} for i in range(3)],
        )


def test_main_does_not_advance_cursor_and_closes_producer_when_publish_fails(monkeypatch):
    import streaming.producer as producer_module

    producer, futures = _producer_with_futures(2)
    futures[0].get.side_effect = RuntimeError("delivery timed out")
    advance = MagicMock()
    monkeypatch.setattr(producer_module, "build_kafka_producer", lambda *a: producer)
    monkeypatch.setattr(producer_module, "advance_cursor", advance)
    monkeypatch.setattr(producer_module, "get_cursor", lambda session: 0)
    monkeypatch.setattr(producer_module, "load_sample_pool", lambda path: [{"id": 0}, {"id": 1}])
    monkeypatch.setenv("DECISION_DB_URL", "sqlite:///:memory:")
    monkeypatch.setenv("PRODUCER_BATCH_SIZE", "2")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "u")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "p")

    with pytest.raises(RuntimeError, match="delivery timed out"):
        producer_module.main()

    advance.assert_not_called()
    producer.close.assert_called_once()
