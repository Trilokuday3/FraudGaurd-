# Live Kafka + Spark Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deployed app's in-process replay worker with a real,
live Kafka + Spark Structured Streaming pipeline: a scheduled GitHub
Actions job produces transactions from the existing sample pool onto a
self-hosted Kafka broker, and a Spark micro-batch job consumes them,
scores them through the existing (unchanged) scoring pipeline, and
writes `Decision` rows to the same Neon Postgres the API already reads.

**Architecture:** A VM hosts only a single-node, SASL-authenticated Kafka
broker (always on). A GitHub Actions workflow, scheduled every 5-10
minutes, runs two steps in one job: a producer (cursors through
`deploy/sample_transactions.json`, publishes a batch to Kafka) and a
Spark consumer (`trigger(availableNow=True)`, reads the topic, calls
`serving.app._compute_score` per row via `foreachBatch`, writes to
Postgres). Spark's checkpoint directory lives on the VM (rsynced down
before the job, up after) so offsets survive between ephemeral GitHub
Actions runs without reprocessing or losing messages.

**Tech Stack:** Apache Kafka (KRaft mode, Docker), PySpark Structured
Streaming, `kafka-python`, SQLAlchemy (cursor persistence, reusing the
existing `Decision`/Postgres setup), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-23-live-kafka-spark-streaming-design.md`

## Global Constraints

- Budget: $0/month — VM on a free-forever tier, GitHub Actions free
  minutes, no paid Kafka/Spark service.
- Spark must call the existing scoring pipeline (`serving.app._compute_score`)
  unchanged — no reimplementation of feature engineering or model
  scoring in Spark DataFrames/Spark ML.
- Data source is the existing 200-row `deploy/sample_transactions.json`
  pool, cursored through — no newly-generated synthetic customers.
- Kafka broker's public port must be SASL-authenticated — never an open,
  unauthenticated broker on the internet.
- No Alembic/migration tooling — new tables are created via the existing
  `Base.metadata.create_all()` call in `make_session_factory`, same
  pattern as `Decision` itself.

---

## Task 1: Stream cursor persistence

**Files:**
- Modify: `serving/models.py` (add `StreamCursor` model, same `Base` as `Decision`)
- Create: `streaming/__init__.py`
- Create: `streaming/cursor_store.py`
- Test: `tests/unit/test_cursor_store.py`

**Interfaces:**
- Produces: `StreamCursor` (SQLAlchemy model, table `stream_cursor`, columns `id: int`, `position: int`).
- Produces: `get_cursor(session: Session) -> int`, `advance_cursor(session: Session, new_position: int) -> None` in `streaming/cursor_store.py`.
- Consumes: `serving.models.make_session_factory` (existing, unchanged) — reusing it means `StreamCursor`'s table gets auto-created alongside `Decision`'s, no new migration step.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cursor_store.py
from serving.models import make_session_factory
from streaming.cursor_store import get_cursor, advance_cursor


def test_get_cursor_defaults_to_zero_when_no_row_exists():
    session = make_session_factory("sqlite:///:memory:")()
    assert get_cursor(session) == 0


def test_advance_cursor_then_get_cursor_round_trips():
    session = make_session_factory("sqlite:///:memory:")()
    advance_cursor(session, 42)
    assert get_cursor(session) == 42


def test_advance_cursor_twice_overwrites_not_duplicates():
    session = make_session_factory("sqlite:///:memory:")()
    advance_cursor(session, 10)
    advance_cursor(session, 20)
    assert get_cursor(session) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/unit/test_cursor_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'streaming'`

- [ ] **Step 3: Write minimal implementation**

Add to `serving/models.py`, right after the `Decision` class:

```python
class StreamCursor(Base):
    """Single-row table tracking how far the Kafka producer has replayed
    deploy/sample_transactions.json. Lives in the same Postgres as
    Decision so it's created by the same make_session_factory() call
    that already runs at every process boot -- no separate migration."""

    __tablename__ = "stream_cursor"

    id = Column(Integer, primary_key=True)
    position = Column(Integer, nullable=False, default=0)
```

Create `streaming/__init__.py` (empty).

Create `streaming/cursor_store.py`:

```python
"""Postgres-backed cursor tracking the producer's position in the
replayed sample pool. A single row (id=1) -- there is only ever one
producer for one pool."""

from sqlalchemy.orm import Session

from serving.models import StreamCursor

_CURSOR_ID = 1


def get_cursor(session: Session) -> int:
    row = session.get(StreamCursor, _CURSOR_ID)
    return row.position if row is not None else 0


def advance_cursor(session: Session, new_position: int) -> None:
    row = session.get(StreamCursor, _CURSOR_ID)
    if row is None:
        session.add(StreamCursor(id=_CURSOR_ID, position=new_position))
    else:
        row.position = new_position
    session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/unit/test_cursor_store.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add serving/models.py streaming/__init__.py streaming/cursor_store.py tests/unit/test_cursor_store.py
git commit -m "feat(streaming): add Postgres-backed cursor for the sample transaction pool"
```

---

## Task 2: Producer — pool loading and batch selection

**Files:**
- Modify: `streaming/producer.py` (new file)
- Test: `tests/unit/test_producer_batching.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_sample_pool(path: str) -> list[dict]`, `select_batch(pool: list[dict], cursor: int, batch_size: int) -> tuple[list[dict], int]` — later tasks (Task 3) import these from `streaming.producer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_producer_batching.py
from streaming.producer import select_batch


def test_select_batch_returns_batch_size_rows_and_advanced_cursor():
    pool = [{"id": i} for i in range(10)]
    batch, new_cursor = select_batch(pool, cursor=0, batch_size=3)
    assert batch == [{"id": 0}, {"id": 1}, {"id": 2}]
    assert new_cursor == 3


def test_select_batch_wraps_around_the_pool():
    pool = [{"id": i} for i in range(5)]
    batch, new_cursor = select_batch(pool, cursor=3, batch_size=4)
    assert batch == [{"id": 3}, {"id": 4}, {"id": 0}, {"id": 1}]
    assert new_cursor == 2


def test_select_batch_on_empty_pool_returns_empty_and_unchanged_cursor():
    batch, new_cursor = select_batch([], cursor=5, batch_size=3)
    assert batch == []
    assert new_cursor == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/unit/test_producer_batching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'streaming.producer'`

- [ ] **Step 3: Write minimal implementation**

```python
# streaming/producer.py
"""Publishes batches of the existing realistic sample transaction pool
(deploy/sample_transactions.json) onto Kafka, cursoring through it like
serving/replay_worker.py's itertools.cycle does today -- but as a
scheduled, stateless run (GitHub Actions) rather than a standing loop,
so the cursor is persisted in Postgres between runs (streaming.cursor_store)
instead of held in process memory."""

import json


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/unit/test_producer_batching.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add streaming/producer.py tests/unit/test_producer_batching.py
git commit -m "feat(streaming): add sample-pool batch selection for the Kafka producer"
```

---

## Task 3: Producer — Kafka publish and CLI entrypoint

**Files:**
- Modify: `streaming/producer.py`
- Create: `streaming/requirements.txt`
- Test: `tests/unit/test_producer_publish.py`

**Interfaces:**
- Consumes: `load_sample_pool`, `select_batch` (Task 2); `streaming.cursor_store.get_cursor`, `advance_cursor` (Task 1); `serving.models.make_session_factory` (existing).
- Produces: `build_kafka_producer(bootstrap_servers, username, password) -> KafkaProducer`, `publish_batch(producer, topic, rows) -> None`, `main() -> None` (CLI entrypoint, reads env vars) — Task 7's GitHub Actions workflow calls this as `python -m streaming.producer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_producer_publish.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/unit/test_producer_publish.py -v`
Expected: FAIL with `ImportError: cannot import name 'publish_batch'`

- [ ] **Step 3: Write minimal implementation**

Add `kafka-python` to a new, separate requirements file — Spark/Kafka
deps are only needed by the GitHub Actions streaming job, never by the
deployed Render API, so they don't belong in `deploy/requirements.txt`:

```
# streaming/requirements.txt
kafka-python>=2.0.2
pyspark>=3.5.0
```

Replace the full contents of `streaming/producer.py` with (this is the
complete file — it includes Task 2's `load_sample_pool`/`select_batch`
unchanged, plus this task's additions):

```python
# streaming/producer.py
"""Publishes batches of the existing realistic sample transaction pool
(deploy/sample_transactions.json) onto Kafka, cursoring through it like
serving/replay_worker.py's itertools.cycle does today -- but as a
scheduled, stateless run (GitHub Actions) rather than a standing loop,
so the cursor is persisted in Postgres between runs (streaming.cursor_store)
instead of held in process memory."""

import json
import os

from kafka import KafkaProducer

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


def build_kafka_producer(bootstrap_servers: str, username: str, password: str) -> KafkaProducer:
    # SASL_PLAINTEXT, not SASL_SSL -- Task 6's broker has no TLS
    # termination in front of it (documented there as a follow-up, not
    # solved). Credentials are authenticated but not encrypted in
    # transit; matches infra/kafka-vm-setup.md's actual broker config.
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="PLAIN",
        sasl_plain_username=username,
        sasl_plain_password=password,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def publish_batch(producer: KafkaProducer, topic: str, rows: list[dict]) -> None:
    for row in rows:
        producer.send(topic, value=row)
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
        publish_batch(producer, topic, batch)

        advance_cursor(session, new_cursor)
        print(f"Published {len(batch)} rows, cursor {cursor} -> {new_cursor}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/unit/test_producer_publish.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add streaming/producer.py streaming/requirements.txt tests/unit/test_producer_publish.py
git commit -m "feat(streaming): publish batches to Kafka and wire up the producer CLI"
```

---

## Task 4: Spark consumer — scoring bridge (testable without Spark)

**Files:**
- Create: `streaming/spark_consumer.py`
- Test: `tests/integration/test_spark_consumer_scoring.py`

**Interfaces:**
- Consumes: `serving.app._compute_score`, `serving.app.SessionLocal`, `serving.schemas.FeatureRow` (all existing, unchanged).
- Produces: `score_and_persist_row(raw: dict) -> None` — Task 5's Spark `foreachBatch` wiring calls this once per row; kept separate specifically so it's unit-testable without a running Spark session.

This is the one place the plan deliberately imports `serving.app` from
outside the FastAPI process — reusing `_compute_score` exactly as-is is
the whole point (see spec: Spark orchestrates, it does not reimplement
scoring). Importing it does trigger the same model-load-at-import-time
behavior the API itself has, so this test needs the same env vars local
dev already has in `.env` (`MLFLOW_RUN_ID`, `MLFLOW_TRACKING_URI`,
`THRESHOLDS_PATH`) — same requirement `tests/integration/test_serving_api.py`
already has, not a new one.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_spark_consumer_scoring.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/integration/test_spark_consumer_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'streaming.spark_consumer'`

- [ ] **Step 3: Write minimal implementation**

```python
# streaming/spark_consumer.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/integration/test_spark_consumer_scoring.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add streaming/spark_consumer.py tests/integration/test_spark_consumer_scoring.py
git commit -m "feat(streaming): bridge Spark micro-batches to the existing scoring pipeline"
```

---

## Task 5: Spark consumer — Structured Streaming wiring

**Files:**
- Modify: `streaming/spark_consumer.py`

**Interfaces:**
- Consumes: `score_and_persist_row` (Task 4).
- Produces: `FEATURE_SCHEMA` (Spark `StructType`), `process_batch(batch_df, batch_id) -> None`, `build_spark_session() -> SparkSession`, `run(bootstrap_servers, topic, username, password, checkpoint_location) -> None`, `main() -> None` — Task 7's workflow calls this as `python -m streaming.spark_consumer`.

Not unit tested (per spec: too heavy for CI). Verified manually against
a local Kafka broker — exact commands are in Task 6's local verification
step, after `docker-compose.kafka.yml` exists to test against.

- [ ] **Step 1: Replace `streaming/spark_consumer.py` with its complete, final contents**

This is the complete file — it includes Task 4's `score_and_persist_row`
unchanged, plus this task's additions:

```python
# streaming/spark_consumer.py
"""Spark Structured Streaming consumer for the fraudguard.transactions
Kafka topic. Spark's job here is orchestration and scale, not scoring:
score_and_persist_row below is a thin bridge to the existing, already-
tested serving.app._compute_score -- the same function /score and the
retired replay worker both already used."""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

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


FEATURE_SCHEMA = StructType(
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


def build_spark_session() -> SparkSession:
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
        from_json(col("value").cast("string"), FEATURE_SCHEMA).alias("data")
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
```

`startingOffsets=earliest` only matters the very first run (before a
checkpoint exists) — every run after that resumes from the checkpoint's
committed offsets regardless of this setting. The checkpoint directory
is what makes `trigger(availableNow=True)` safe to run every 5-10
minutes without reprocessing the whole topic each time; Task 6 makes
that directory persist on the VM between GitHub Actions runs.

- [ ] **Step 2: Commit**

```bash
git add streaming/spark_consumer.py
git commit -m "feat(streaming): wire Spark Structured Streaming to read Kafka and score each micro-batch"
```

---

## Task 6: Kafka broker + Spark checkpoint on a free VM (runbook)

**Files:**
- Create: `streaming/docker-compose.kafka.yml`
- Create: `infra/kafka-vm-setup.md`

This task is infrastructure provisioning, not code — like the
Neon/Render/Vercel account creation in `infra/deploy.md`, it's a manual
step only the account owner can perform. No automated test applies;
verification is the manual checklist at the end of this task.

- [ ] **Step 1: Write the Kafka broker compose file**

```yaml
# streaming/docker-compose.kafka.yml
# Single-node Kafka (KRaft mode, no Zookeeper) for the VM hosting the
# live streaming pipeline's broker. SASL_SSL is not used here (this
# compose file terminates plain SASL_PLAINTEXT internally); put a TLS
# terminator (e.g. a Caddy/nginx reverse proxy, or your VM host's own
# load balancer) in front of the exposed port before treating this as
# production-grade -- documented as a follow-up in infra/kafka-vm-setup.md,
# not solved here, to keep this compose file focused on Kafka itself.
services:
  kafka:
    image: apache/kafka:3.7.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: SASL_PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: SASL_PLAINTEXT://${VM_PUBLIC_HOST}:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: SASL_PLAINTEXT:SASL_PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_SASL_ENABLED_MECHANISMS: PLAIN
      KAFKA_SASL_MECHANISM_CONTROLLER_PROTOCOL: PLAIN
      KAFKA_INTER_BROKER_LISTENER_NAME: SASL_PLAINTEXT
      KAFKA_LISTENER_NAME_SASL_PLAINTEXT_PLAIN_SASL_JAAS_CONFIG: >-
        org.apache.kafka.common.security.plain.PlainLoginModule required
        username="${KAFKA_SASL_USERNAME}"
        password="${KAFKA_SASL_PASSWORD}"
        user_${KAFKA_SASL_USERNAME}="${KAFKA_SASL_PASSWORD}";
    volumes:
      - kafka-data:/var/lib/kafka/data

volumes:
  kafka-data:
```

- [ ] **Step 2: Write the VM setup runbook**

```markdown
# infra/kafka-vm-setup.md

Manual setup for the VM hosting the live streaming pipeline's Kafka
broker. Like infra/deploy.md's Neon/Render/Vercel steps, this is a
one-time manual process only you can do -- create the account, verify
current free-tier terms yourself (they change; this doc doesn't pin a
specific provider's exact numbers).

## 1. Provision the VM

Any perpetual (not trial) free compute tier works. At minimum: 1 GB RAM,
a public IP, Docker installable. Note the VM's public IP/hostname --
this becomes `VM_PUBLIC_HOST` below and `KAFKA_BOOTSTRAP_SERVERS`
(`<host>:9092`) in the GitHub Actions secrets (Task 7).

## 2. Install Docker

Follow your VM OS's standard Docker install instructions.

## 3. Choose SASL credentials

Pick a username/password for the Kafka broker's PLAIN SASL mechanism.
These become `KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD` -- set them
as GitHub Actions **secrets** (Task 7), never committed.

## 4. Start the broker

```
export VM_PUBLIC_HOST=<your VM's public IP or hostname>
export KAFKA_SASL_USERNAME=<chosen username>
export KAFKA_SASL_PASSWORD=<chosen password>
docker compose -f streaming/docker-compose.kafka.yml up -d
```

## 5. Firewall

Open port 9092 to the internet (GitHub Actions runners need to reach
it) -- but only 9092. Do not open the controller port (9093).

## 6. Create the topic

```
docker compose -f streaming/docker-compose.kafka.yml exec kafka \
  kafka-topics.sh --create --topic fraudguard.transactions \
  --bootstrap-server localhost:9092 \
  --command-config /dev/stdin <<< "security.protocol=SASL_PLAINTEXT
sasl.mechanism=PLAIN
sasl.jaas.config=org.apache.kafka.common.security.plain.PlainLoginModule required username=\"$KAFKA_SASL_USERNAME\" password=\"$KAFKA_SASL_PASSWORD\";"
```

## 7. Create the Spark checkpoint directory and SSH access

```
mkdir -p /opt/fraudguard/spark-checkpoint
```

Generate a dedicated SSH keypair for GitHub Actions to rsync this
directory (do not reuse your personal key):

```
ssh-keygen -t ed25519 -f fraudguard-streaming-deploy-key -N ""
cat fraudguard-streaming-deploy-key.pub >> ~/.ssh/authorized_keys
```

Add the **private** key's contents as a GitHub Actions secret
(`VM_SSH_PRIVATE_KEY`, Task 7). Never commit the private key.

## Verification checklist

- [ ] `docker compose -f streaming/docker-compose.kafka.yml ps` shows the broker running
- [ ] From your own machine: `nc -zv <VM_PUBLIC_HOST> 9092` succeeds (port reachable)
- [ ] A connection *without* the SASL credentials is rejected (confirms auth is actually enforced, not just configured)
- [ ] `ssh -i fraudguard-streaming-deploy-key deploy@<VM_PUBLIC_HOST> "ls /opt/fraudguard/spark-checkpoint"` succeeds
```

- [ ] **Step 3: Commit**

```bash
git add streaming/docker-compose.kafka.yml infra/kafka-vm-setup.md
git commit -m "docs(streaming): add the Kafka broker compose file and VM setup runbook"
```

---

## Task 7: GitHub Actions workflow wiring producer + Spark together

**Files:**
- Create: `.github/workflows/live-streaming.yml`

**Interfaces:**
- Consumes: `streaming.producer.main` (Task 3), `streaming.spark_consumer.main` (Task 5), the VM's SSH access and checkpoint directory (Task 6).

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/live-streaming.yml
name: Live streaming pipeline

on:
  schedule:
    - cron: "*/10 * * * *"
  workflow_dispatch: {}

jobs:
  produce-and-consume:
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: ./deploy/model_store
      MLFLOW_RUN_ID: 17d00e31654c4b9e8a9aa389f797d650
      THRESHOLDS_PATH: ./decision/thresholds.json
      KAFKA_TOPIC: fraudguard.transactions
    steps:
      - uses: actions/checkout@v4

      - name: Check required secrets are configured
        run: |
          if [ -z "${{ secrets.DEPLOYED_DECISION_DB_URL }}" ] || [ -z "${{ secrets.KAFKA_BOOTSTRAP_SERVERS }}" ]; then
            echo "Streaming secrets not configured yet -- skipping until infra/kafka-vm-setup.md is complete."
            exit 0
          fi

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"

      - run: pip install -r requirements-dev.txt -r streaming/requirements.txt

      - name: Run producer
        env:
          DECISION_DB_URL: ${{ secrets.DEPLOYED_DECISION_DB_URL }}
          KAFKA_BOOTSTRAP_SERVERS: ${{ secrets.KAFKA_BOOTSTRAP_SERVERS }}
          KAFKA_SASL_USERNAME: ${{ secrets.KAFKA_SASL_USERNAME }}
          KAFKA_SASL_PASSWORD: ${{ secrets.KAFKA_SASL_PASSWORD }}
        run: python -m streaming.producer

      - name: Fetch Spark checkpoint from the VM
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.VM_SSH_PRIVATE_KEY }}" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          ssh-keyscan -H "${{ secrets.VM_HOST }}" >> ~/.ssh/known_hosts
          rsync -avz -e "ssh -i ~/.ssh/deploy_key" \
            "deploy@${{ secrets.VM_HOST }}:/opt/fraudguard/spark-checkpoint/" \
            ./spark-checkpoint/

      - name: Run Spark consumer
        env:
          DECISION_DB_URL: ${{ secrets.DEPLOYED_DECISION_DB_URL }}
          KAFKA_BOOTSTRAP_SERVERS: ${{ secrets.KAFKA_BOOTSTRAP_SERVERS }}
          KAFKA_SASL_USERNAME: ${{ secrets.KAFKA_SASL_USERNAME }}
          KAFKA_SASL_PASSWORD: ${{ secrets.KAFKA_SASL_PASSWORD }}
          SPARK_CHECKPOINT_DIR: ./spark-checkpoint
        run: python -m streaming.spark_consumer

      - name: Push updated Spark checkpoint back to the VM
        if: always()
        run: |
          rsync -avz -e "ssh -i ~/.ssh/deploy_key" \
            ./spark-checkpoint/ \
            "deploy@${{ secrets.VM_HOST }}:/opt/fraudguard/spark-checkpoint/"
```

Required new GitHub secrets (document these in Task 8's `infra/deploy.md`
update): `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_SASL_USERNAME`,
`KAFKA_SASL_PASSWORD`, `VM_HOST`, `VM_SSH_PRIVATE_KEY`. Reuses the
existing `DEPLOYED_DECISION_DB_URL` secret already documented for
`mlops-monitor.yml`.

The checkpoint push step uses `if: always()` so a Spark failure mid-run
still saves whatever progress it made, rather than silently discarding
it and reprocessing from further back next time.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/live-streaming.yml
git commit -m "feat(streaming): schedule the producer + Spark consumer pipeline via GitHub Actions"
```

---

## Task 8: Retire the in-process replay worker, update docs

**Files:**
- Modify: `infra/deploy.md`
- Modify: `README.md`
- Modify: `.env.example` (document the new streaming-related vars for local testing)

No code change to `serving/app.py` — `ENABLE_REPLAY_WORKER` stays
supported (it's still useful for local dev / `docker-compose.yml`'s
existing local-only demo), it's just switched off on Render.

- [ ] **Step 1: Update `infra/deploy.md`**

Add a new section after the existing "Rollback" section:

```markdown
## Live streaming pipeline (Kafka + Spark)

As of `docs/superpowers/specs/2026-09-23-live-kafka-spark-streaming-design.md`,
the deployed app's "live" data comes from a real Kafka + Spark pipeline,
not the in-process replay worker. See `infra/kafka-vm-setup.md` for the
one-time VM/Kafka setup, and `docs/superpowers/plans/2026-09-23-live-kafka-spark-streaming.md`
for the full implementation.

**Turn off the old replay worker** on Render once the streaming pipeline
is verified working (Environment tab): set `ENABLE_REPLAY_WORKER=false`.
Running both simultaneously double-writes to the same `decisions` table.

**New GitHub Actions secrets required** (`.github/workflows/live-streaming.yml`):

| Secret | Value |
|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `<VM host>:9092` |
| `KAFKA_SASL_USERNAME` | chosen in `infra/kafka-vm-setup.md` step 3 |
| `KAFKA_SASL_PASSWORD` | chosen in `infra/kafka-vm-setup.md` step 3 |
| `VM_HOST` | the VM's public IP/hostname |
| `VM_SSH_PRIVATE_KEY` | the deploy key generated in `infra/kafka-vm-setup.md` step 7 |

Also requires `DEPLOYED_DECISION_DB_URL` (same Neon connection string
already used by `mlops-monitor.yml`, `+psycopg` scheme).
```

- [ ] **Step 2: Update `README.md`**

Find the section describing the deployed app's "live" data source (added
in sub-project 8) and update it to describe the Kafka + Spark pipeline
instead of the in-process replay worker, linking to
`docs/superpowers/specs/2026-09-23-live-kafka-spark-streaming-design.md`.

- [ ] **Step 3: Update `.env.example`**

```
# Live streaming pipeline (Kafka + Spark) -- only needed when running
# streaming/producer.py or streaming/spark_consumer.py locally against
# a test Kafka broker; not read by serving/app.py.
KAFKA_BOOTSTRAP_SERVERS=
KAFKA_SASL_USERNAME=
KAFKA_SASL_PASSWORD=
KAFKA_TOPIC=fraudguard.transactions
SPARK_CHECKPOINT_DIR=./spark-checkpoint
```

- [ ] **Step 4: Commit**

```bash
git add infra/deploy.md README.md .env.example
git commit -m "docs(streaming): document the live Kafka+Spark pipeline and retiring the replay worker"
```

---

## Task 9: Post-deploy acceptance

**Files:** none (verification only)

This task has no code changes — it's the checklist confirming the
pipeline actually works end-to-end on the real deployed infrastructure,
mirroring `2026-09-16-deployment-portfolio-design.md`'s own "Testing /
acceptance approach" section.

- [ ] **Step 1: Confirm the VM and Kafka broker are reachable**

Run `infra/kafka-vm-setup.md`'s verification checklist in full.

- [ ] **Step 2: Manually trigger the workflow once**

On GitHub: Actions tab → "Live streaming pipeline" → "Run workflow".
Confirm both steps (producer, Spark consumer) succeed.

- [ ] **Step 3: Confirm new decisions actually landed**

```
curl -s https://fraudgaurd-stza.onrender.com/decisions/stats
```

Note the `total` count, wait for the next scheduled run (or trigger
again manually), re-check — `total` should have grown by roughly the
producer's batch size.

- [ ] **Step 4: Confirm the checkpoint actually persists (no duplicate scoring)**

Run the workflow twice in a row with no new producer output in between
(temporarily set `PRODUCER_BATCH_SIZE=0` as a `workflow_dispatch` input,
or just check quickly before the next scheduled run) — `/decisions/stats`'s
`total` should **not** increase on a run with nothing new to consume.
If it does, the checkpoint isn't being restored correctly from the VM —
re-check Task 7's rsync steps before proceeding.

- [ ] **Step 5: Turn off the replay worker**

On Render: Environment tab → set `ENABLE_REPLAY_WORKER=false` → save
(auto-restarts). Confirm `/decisions/stats`'s `total` still grows purely
from the new pipeline over the following 10-20 minutes.

- [ ] **Step 6: Confirm the dashboard still renders correctly**

Open the deployed frontend, confirm the Dashboard, Live, and Monitoring
pages show growing data with no visible change in behavior — they only
ever read from Postgres, so which pipeline wrote a row should be
invisible to them.
