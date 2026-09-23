"""Publishes batches of the existing realistic sample transaction pool
(deploy/sample_transactions.json) onto Kafka, cursoring through it like
serving/replay_worker.py's itertools.cycle does today -- but as a
scheduled, stateless run (GitHub Actions) rather than a standing loop,
so the cursor is persisted in Postgres between runs (streaming.cursor_store)
instead of held in process memory."""

import json
import os
from typing import Any

from serving.models import make_session_factory
from streaming.cursor_store import advance_cursor, get_cursor


def load_sample_pool(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def select_batch(pool: list[dict], cursor: int, batch_size: int) -> tuple[list[dict], int]:
    """Return the next `batch_size` rows starting at `cursor`, wrapping
    around the pool, plus the cursor position to resume from next time."""
    n = len(pool)
    if n == 0:
        return [], cursor
    batch = [pool[(cursor + i) % n] for i in range(batch_size)]
    new_cursor = (cursor + batch_size) % n
    return batch, new_cursor


def build_kafka_producer(bootstrap_servers: str, username: str, password: str) -> Any:
    # SASL_PLAINTEXT, not SASL_SSL -- Task 6's broker has no TLS
    # termination in front of it (documented there as a follow-up, not
    # solved). Credentials are authenticated but not encrypted in
    # transit; matches infra/kafka-vm-setup.md's actual broker config.
    # kafka is imported lazily so CI (no kafka-python installed) can import this module.
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="PLAIN",
        sasl_plain_username=username,
        sasl_plain_password=password,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def publish_batch(producer: Any, topic: str, rows: list[dict]) -> None:
    """Send every row and wait for the broker to acknowledge each one.

    kafka-python's flush() does not raise per-record delivery errors, so
    each send()'s future is resolved explicitly: a failed or timed-out
    delivery raises here, before main() advances the stored cursor, so the
    cursor can never move past rows that never reached the topic."""
    futures = [producer.send(topic, value=row) for row in rows]
    for future in futures:
        future.get(timeout=30)
    producer.flush()


def main() -> None:
    pool_path = os.environ.get("SAMPLE_POOL_PATH", "deploy/sample_transactions.json")
    batch_size = int(os.environ.get("PRODUCER_BATCH_SIZE", "30"))
    topic = os.environ.get("KAFKA_TOPIC", "fraudguard.transactions")

    db_url = os.environ["DECISION_DB_URL"]
    bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    username = os.environ["KAFKA_SASL_USERNAME"]
    password = os.environ["KAFKA_SASL_PASSWORD"]

    session = make_session_factory(db_url)()
    try:
        cursor = get_cursor(session)
        pool = load_sample_pool(pool_path)
        batch, new_cursor = select_batch(pool, cursor, batch_size)

        producer = build_kafka_producer(bootstrap_servers, username, password)
        try:
            publish_batch(producer, topic, batch)
        finally:
            producer.close()

        advance_cursor(session, new_cursor)
        print(f"Published {len(batch)} rows, cursor {cursor} -> {new_cursor}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
