"""Guards against drift between the Spark consumer's Kafka JSON schema,
serving.schemas.FeatureRow and the real sample pool the producer
publishes (deploy/sample_transactions.json). None of these tests need
pyspark except the last one, which is skipped when it isn't installed."""

from pathlib import Path

import pytest

from serving.schemas import FeatureRow
from streaming.producer import load_sample_pool
from streaming.spark_consumer import FEATURE_SCHEMA_FIELDS, kafka_package_coordinate

_POOL_PATH = Path(__file__).resolve().parents[2] / "deploy" / "sample_transactions.json"
_FEATURE_ROW_FIELDS = list(FeatureRow.model_fields)


def test_sample_pool_rows_contain_every_feature_row_field():
    pool = load_sample_pool(str(_POOL_PATH))
    assert pool, "sample pool is empty"
    for index, row in enumerate(pool):
        missing = set(_FEATURE_ROW_FIELDS) - set(row)
        assert not missing, f"row {index} is missing FeatureRow fields {sorted(missing)}"


def test_every_sample_pool_row_validates_as_a_feature_row():
    for row in load_sample_pool(str(_POOL_PATH)):
        FeatureRow(**row)


def test_spark_schema_field_names_and_order_match_feature_row():
    assert [name for name, _ in FEATURE_SCHEMA_FIELDS] == _FEATURE_ROW_FIELDS


def test_spark_schema_types_match_the_json_types_in_the_sample_pool():
    # Spark's JSON reader yields null when a value doesn't fit the declared
    # type (e.g. JSON 0/1 under BooleanType), which would make every row
    # fail FeatureRow validation. Check the declared type against the
    # actual JSON values the producer publishes.
    accepts = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "double": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    }
    pool = load_sample_pool(str(_POOL_PATH))
    for name, spark_type in FEATURE_SCHEMA_FIELDS:
        bad = [row[name] for row in pool if not accepts[spark_type](row[name])]
        assert not bad, f"{name} declared {spark_type} but pool has {bad[:3]!r}"


def test_boolean_feature_flags_are_declared_integer_not_boolean():
    declared = dict(FEATURE_SCHEMA_FIELDS)
    for flag in [
        "is_night",
        "is_cross_border",
        "is_new_device",
        "is_new_country_for_customer",
        "ip_country_mismatch",
    ]:
        assert declared[flag] == "integer"


def test_kafka_package_coordinate_tracks_the_given_pyspark_version():
    assert (
        kafka_package_coordinate("3.5.3")
        == "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"
    )


def test_build_feature_schema_matches_feature_row_when_pyspark_is_available():
    pytest.importorskip("pyspark")
    from pyspark.sql.types import IntegerType

    from streaming.spark_consumer import build_feature_schema

    schema = build_feature_schema()
    assert schema.fieldNames() == _FEATURE_ROW_FIELDS
    assert isinstance(schema["is_night"].dataType, IntegerType)
