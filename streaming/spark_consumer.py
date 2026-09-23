# streaming/spark_consumer.py
"""Spark Structured Streaming consumer for the fraudguard.transactions
Kafka topic. Spark's job here is orchestration and scale, not scoring:
score_and_persist_row below is a thin bridge to the existing, already-
tested serving.app._compute_score -- the same function /score and the
retired replay worker both already used."""

import os

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


def build_feature_schema():
    # Lazy import so the module imports without pyspark installed (CI).
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("transaction_id", StringType()),
            StructField("customer_id", StringType()),
            StructField("merchant_id", StringType()),
            StructField("timestamp", StringType()),
            StructField("amount", DoubleType()),
            StructField("payment_method", StringType()),
            StructField("hour_of_day", IntegerType()),
            StructField("is_night", BooleanType()),
            StructField("is_cross_border", BooleanType()),
            StructField("amount_vs_customer_p95", DoubleType()),
            StructField("txn_count_1h", IntegerType()),
            StructField("txn_count_24h", IntegerType()),
            StructField("is_new_device", BooleanType()),
            StructField("device_age_days", DoubleType()),
            StructField("customer_device_count_so_far", IntegerType()),
            StructField("merchant_fraud_rate_hist", DoubleType()),
            StructField("is_new_country_for_customer", BooleanType()),
            StructField("ip_country_mismatch", BooleanType()),
        ]
    )


def process_batch(batch_df, batch_id: int) -> None:
    if batch_df.rdd.isEmpty():
        return
    for raw in batch_df.toPandas().to_dict(orient="records"):
        score_and_persist_row(raw)


def build_spark_session():
    # Lazy import so the module imports without pyspark installed (CI).
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("fraudguard-streaming-consumer")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
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
    run(
        bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        topic=os.environ.get("KAFKA_TOPIC", "fraudguard.transactions"),
        username=os.environ["KAFKA_SASL_USERNAME"],
        password=os.environ["KAFKA_SASL_PASSWORD"],
        checkpoint_location=os.environ.get("SPARK_CHECKPOINT_DIR", "./spark-checkpoint"),
    )


if __name__ == "__main__":
    main()
