from unittest.mock import MagicMock

from streaming.producer import publish_batch


def test_publish_batch_sends_each_row_and_flushes():
    producer = MagicMock()
    rows = [{"transaction_id": "TXN1"}, {"transaction_id": "TXN2"}]

    publish_batch(producer, topic="fraudguard.transactions", rows=rows)

    assert producer.send.call_count == 2
    producer.send.assert_any_call("fraudguard.transactions", value=rows[0])
    producer.send.assert_any_call("fraudguard.transactions", value=rows[1])
    producer.flush.assert_called_once()


def test_publish_batch_on_empty_rows_does_not_call_send():
    producer = MagicMock()
    publish_batch(producer, topic="fraudguard.transactions", rows=[])
    producer.send.assert_not_called()
    producer.flush.assert_called_once()
