# streaming/spark_consumer.py
"""Spark Structured Streaming consumer for the fraudguard.transactions
Kafka topic. Spark's job here is orchestration and scale, not scoring:
score_rows below is a thin bridge to the existing, already-tested
serving.app._compute_score -- the same function /score and the retired
replay worker both already used.

Neither pyspark nor serving.app is imported at module import time:
pyspark isn't installed in CI (requirements-dev.txt only), and resolving
serving.app lazily means score_rows always uses serving.app's *current*
SessionLocal/_compute_score (tests reload that module in place)."""

import logging
import os
from collections.abc import Callable, Iterable

from pydantic import ValidationError

from serving.schemas import FeatureRow

logger = logging.getLogger(__name__)

# (field name, Spark type) in serving.schemas.FeatureRow's field order.
# The five boolean flags (is_night, is_cross_border, is_new_device,
# is_new_country_for_customer, ip_country_mismatch) are declared "integer",
# NOT boolean: deploy/sample_transactions.json -- what the producer
# publishes -- stores them as JSON 0/1, and Spark's JSON parser yields null
# for an integer under BooleanType, which would fail every row. FeatureRow
# (pydantic) coerces 0/1 to bool, so integers are the right wire type.
# tests/unit/test_spark_feature_schema.py pins this against both
# FeatureRow and the real sample pool.
FEATURE_SCHEMA_FIELDS: list[tuple[str, str]] = [
    ("transaction_id", "string"),
    ("customer_id", "string"),
    ("merchant_id", "string"),
    ("timestamp", "string"),
    ("amount", "double"),
    ("payment_method", "string"),
    ("hour_of_day", "integer"),
    ("is_night", "integer"),
    ("is_cross_border", "integer"),
    ("amount_vs_customer_p95", "double"),
    ("txn_count_1h", "integer"),
    ("txn_count_24h", "integer"),
    ("is_new_device", "integer"),
    ("device_age_days", "double"),
    ("customer_device_count_so_far", "integer"),
    ("merchant_fraud_rate_hist", "double"),
    ("is_new_country_for_customer", "integer"),
    ("ip_country_mismatch", "integer"),
]


def score_rows(
    rows: Iterable[dict],
    *,
    session_factory: Callable | None = None,
    compute_score: Callable | None = None,
    batch_id: int | None = None,
) -> tuple[int, int]:
    """Score every row and persist all resulting Decision records in ONE
    session with ONE commit. Returns (scored, skipped).

    A row that fails FeatureRow validation (e.g. a field Spark parsed as
    null) is logged and skipped rather than failing the whole batch, so a
    single poison message can't wedge the stream. Any other error rolls
    the session back and re-raises, so Spark retries the whole batch
    without having committed part of it.

    Delivery is at-least-once: if this commit succeeds but Spark then
    fails to write its own commit log for the batch, Spark replays the
    batch on the next run and its rows are scored and written again. That
    small duplicate window is accepted, not eliminated."""
    rows = list(rows)
    if not rows:
        return 0, 0

    if session_factory is None or compute_score is None:
        import serving.app

        session_factory = session_factory or serving.app.SessionLocal
        compute_score = compute_score or serving.app._compute_score

    session = session_factory()
    scored = skipped = 0
    try:
        for index, raw in enumerate(rows):
            try:
                row = FeatureRow(**raw)
            except ValidationError as exc:
                skipped += 1
                logger.warning(
                    "batch %s row %d (transaction_id=%r): skipping invalid row: %s",
                    batch_id,
                    index,
                    raw.get("transaction_id"),
                    exc,
                )
                continue
            _response, record = compute_score(row)
            session.add(record)
            scored += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return scored, skipped


def build_feature_schema():
    # Lazy import so the module imports without pyspark installed (CI).
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    spark_types = {"string": StringType, "double": DoubleType, "integer": IntegerType}
    return StructType(
        [StructField(name, spark_types[kind]()) for name, kind in FEATURE_SCHEMA_FIELDS]
    )


def process_batch(batch_df, batch_id: int) -> None:
    # collect() + asDict() rather than toPandas(): pandas would turn a null
    # numeric into NaN, which pydantic accepts for float fields; a plain
    # None fails FeatureRow validation and the row is skipped instead.
    rows = [r.asDict() for r in batch_df.collect()]
    scored, skipped = score_rows(rows, batch_id=batch_id)
    print(f"batch {batch_id}: scored {scored}, skipped {skipped}", flush=True)


def kafka_package_coordinate(pyspark_version: str) -> str:
    # Scala 2.12 matches the pinned pyspark 3.5.x in streaming/requirements.txt.
    return f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark_version}"


def build_spark_session():
    # Lazy import so the module imports without pyspark installed (CI).
    import pyspark
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("fraudguard-streaming-consumer")
        # Kafka source package built for the installed pyspark version, so
        # the two can't drift apart.
        .config("spark.jars.packages", kafka_package_coordinate(pyspark.__version__))
        .getOrCreate()
    )


def run(
    bootstrap_servers: str,
    topic: str,
    username: str,
    password: str,
    checkpoint_location: str,
) -> None:
    # Lazy import so the module imports without pyspark installed (CI).
    from pyspark.sql.functions import col, from_json

    spark = build_spark_session()
    jaas_config = (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="{username}" password="{password}";'
    )
    raw_stream = (
        # SASL_PLAINTEXT, matching Task 6's broker (no TLS termination
        # configured there) and streaming/producer.py's build_kafka_producer.
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("kafka.security.protocol", "SASL_PLAINTEXT")
        .option("kafka.sasl.mechanism", "PLAIN")
        .option("kafka.sasl.jaas.config", jaas_config)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )
    parsed = raw_stream.select(
        from_json(col("value").cast("string"), build_feature_schema()).alias("data")
    ).select("data.*")

    query = (
        parsed.writeStream.trigger(availableNow=True)
        .option("checkpointLocation", checkpoint_location)
        .foreachBatch(process_batch)
        .start()
    )
    query.awaitTermination()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(
        bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        topic=os.environ.get("KAFKA_TOPIC", "fraudguard.transactions"),
        username=os.environ["KAFKA_SASL_USERNAME"],
        password=os.environ["KAFKA_SASL_PASSWORD"],
        checkpoint_location=os.environ.get("SPARK_CHECKPOINT_DIR", "./spark-checkpoint"),
    )


if __name__ == "__main__":
    main()
