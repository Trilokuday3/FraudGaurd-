"""Spark Structured Streaming consumer for the fraudguard.transactions
Kafka topic. Spark's job here is orchestration and scale, not scoring:
score_and_persist_row below is a thin bridge to the existing, already-
tested serving.app._compute_score -- the same function /score and the
retired replay worker both already used. The Spark-specific wiring
(schema, foreachBatch, readStream) is added in the next task and is
verified locally against a real Kafka broker rather than unit tested
here, per the design spec (a full PySpark job is too heavy for CI)."""

from serving.app import SessionLocal, _compute_score
from serving.schemas import FeatureRow


def score_and_persist_row(raw: dict) -> None:
    row = FeatureRow(**raw)
    _response, record = _compute_score(row)
    session = SessionLocal()
    try:
        session.add(record)
        session.commit()
    finally:
        session.close()
