# Sub-project 5 (revised): Live Kafka + Spark Streaming — Design

**Date:** 2026-09-23
**Status:** Draft, pending review
**Roadmap:** `docs/superpowers/specs/2026-09-12-fraudguard-platform-roadmap.md` (sub-project 5)
**Supersedes:** the roadmap's original sub-project 5 scope ("local-only,
Docker Compose") and `2026-09-16-deployment-portfolio-design.md`'s replay
worker as the sole source of live deployed data.

## Purpose

The roadmap originally scoped Kafka + Spark Structured Streaming as
**local-only** — the deployed app was meant to keep using an in-process
replay worker instead, specifically because an always-on Kafka broker
"is not free-tier-shaped." That reasoning was sound and is not being
overturned lightly: it's re-confirmed by fresh research in this doc
(Upstash's managed Kafka offering has been discontinued since the
roadmap was written — its pricing and docs pages both 404 now — and no
other zero-cost managed Kafka option was found that's clearly
perpetual rather than trial-based).

The user explicitly wants this live on the deployed site anyway, having
been shown that tradeoff directly: a self-hosted VM is a real
operational risk (nothing restarts it the way Render manages the API
process; if it goes down, that part of the live demo breaks) that the
original plan deliberately avoided. This spec proceeds on that
informed, explicit basis — the risk is accepted, not overlooked.

## Constraints (confirmed with the user)

- **Budget: $0/month**, same hard constraint as the rest of the
  deployed stack. No trial credits, no "free for 30 days" managed
  Kafka.
- **Must be live on the actual deployed site**
  (`fraudgaurd-stza.onrender.com` / the Vercel frontend), not a
  separate local-only demo.
- **Spark reuses the existing scoring pipeline as-is** — it orchestrates
  and scales the stream, it does not reimplement feature engineering or
  model scoring in Spark DataFrames/Spark ML. The existing, already-
  tested scoring path stays the single source of truth.
- **Data source: the existing 200-row `deploy/sample_transactions.json`
  pool**, not freshly-minted synthetic customers. `generator/generate.py`
  simulates a whole customer population's coherent history at once
  (device age, prior transaction counts, calibrated fraud rate) — it has
  no "produce one new realistic transaction right now" mode. Regenerating
  a handful of brand-new random customers every cycle would make every
  transaction look artificially history-less (`is_new_device=1`,
  `device_age_days≈0` always) and risks skewing scores in a way that's
  misleading in a live demo. This sub-project is a data-*delivery*
  upgrade (replacing the in-process asyncio loop with a real Kafka+Spark
  pipeline), not a new data-*generation* problem.

## Architecture

```
[VM, always-on, free-forever tier — e.g. Oracle Cloud Always Free;
 exact provider verified by the user at signup, same as Neon/Render/
 Vercel's own account-creation steps]
   Kafka broker, single node, KRaft mode (no Zookeeper)
   SASL-authenticated, non-default port (never an open, unauthenticated
   broker reachable on the public internet)
   Topic: fraudguard.transactions

[GitHub Actions, scheduled every 5-10 min — same free pattern already
 used by keep-alive.yml / mlops-monitor.yml]
   1. Producer step (new, small Python + kafka-python):
      - reads deploy/sample_transactions.json
      - advances a persisted cursor through the 200-row pool (wraps
        around like the replay worker's itertools.cycle does today)
      - publishes the next batch of rows to fraudguard.transactions
   2. Spark step (new, PySpark):
      - Structured Streaming, trigger(availableNow=True) — a genuine
        micro-batch run against everything currently in the topic,
        not a standing cluster
      - foreachBatch hands each micro-batch to the existing scoring
        pipeline (the same logic serving/app.py's _compute_score already
        uses), completely unchanged
      - writes resulting Decision rows to Neon Postgres, same table the
        API already reads from
```

Both GitHub Actions steps run in the same scheduled workflow run rather
than as separate always-on services — the VM's only job is hosting the
Kafka broker, keeping its resource footprint minimal.

## Components

### 1. Kafka broker (VM, new)

Single-node, KRaft mode, in Docker. SASL auth required — this VM's port
is reachable from the public internet (GitHub Actions runners need to
reach it), so it is a real, if narrow, attack surface and gets treated
as one, not left open.

### 2. Producer (new, `streaming/producer.py` or similar)

Cursor-based replay of the existing sample pool, publishing to Kafka
instead of scoring in-process. The cursor persists between scheduled
runs (a small file or a row in Postgres — implementation detail for the
plan) so consecutive runs advance through the pool rather than each
starting over at row 0.

### 3. Spark consumer (new, `streaming/spark_consumer.py` or similar)

The only genuinely new *processing* code. Everything downstream of
"here is a micro-batch of rows" — feature validation, model scoring,
decision rules, the `Decision` row shape — is the existing, already-
tested pipeline, called via `foreachBatch`, not reimplemented.

### 4. Retiring the in-process replay worker

`ENABLE_REPLAY_WORKER` on Render gets turned off once this ships.
Running both simultaneously would double-write to the same `Decision`
table from two independent sources and produce confusing, inconsistent
demo data — not a real benefit, just noise.

### 5. New GitHub Actions workflow

A new scheduled workflow (mirroring `keep-alive.yml`'s cron pattern),
running the producer then Spark steps every 5-10 minutes. Visible run
history in the Actions tab is an explicit, deliberate benefit of this
choice over running Spark via the VM's own cron (which was considered
and rejected specifically to keep that visibility for interviews).

## Error handling / security

- **Kafka auth**: SASL, credentials stored as GitHub Actions secrets,
  never committed — same pattern as `DEPLOYED_DECISION_DB_URL`.
- **Missed cycles are acceptable**: if the VM is down or a scheduled run
  fails, that cycle is simply skipped — no data loss, the cursor and
  Kafka's own retention mean the next successful run picks up where the
  last one left off. This mirrors the replay worker's own existing
  tolerance for gaps (e.g. during Render's cold start).
- **GitHub Actions scheduling is not exact** — scheduled workflows can
  slip under GitHub's own load. This is an accepted, disclosed tradeoff
  of using Actions over VM-local cron (which would have been more
  punctual), explicitly chosen anyway for the visible run history.

## Testing

- **Producer cursor logic**: unit tested locally with the real
  publish call mocked out — no live Kafka needed for this part.
- **Spark job**: verified against a local Kafka broker (e.g. via
  `docker-compose`, similar to the existing local-only
  `docker-compose.yml`), not run in CI — a full PySpark job is too heavy
  for a per-push CI check, same reasoning that already keeps
  `tests/features`/`tests/dq` skippable-when-absent rather than CI-gated
  on real data.
- **Post-deploy acceptance**: confirm a scheduled workflow run actually
  produces new `Decision` rows in Neon (via `/decisions/stats`,
  the same endpoint already used to verify the replay worker), confirm
  the VM's Kafka broker rejects unauthenticated connections, confirm
  `ENABLE_REPLAY_WORKER=false` on Render with no regression to the
  Dashboard/Live/Monitoring pages (they only ever read from Postgres,
  never care which pipeline wrote a row).

## Risks / explicit decisions

- **VM is a real, personally-owned operational dependency** — the one
  thing Render/Neon/Vercel's managed platforms otherwise fully abstract
  away for this project. If it goes down, new decisions stop appearing
  until it's manually restarted. Accepted explicitly by the user after
  being shown this tradeoff directly, not a default that slipped
  through.
- **"Continuous" means every 5-10 minutes, not sub-second streaming.**
  A defensible, disclosed micro-batch cadence — worth being upfront
  about in an interview rather than implying true real-time latency.
- **Exact free VM provider and its current terms are unverified by this
  spec.** Same category of manual step as the Neon/Render/Vercel account
  creation in `infra/deploy.md` — the user signs up and confirms current
  specs/limits themselves; this spec does not commit to a specific
  provider's exact numbers, which change over time.
- **This revises, not extends, the roadmap's original sub-project 5.**
  The local-only Docker Compose demo the roadmap describes is
  superseded by this live version rather than built as a separate
  parallel thing — building both would be duplicated effort for the
  same underlying capability.

## Out of scope

- Reimplementing feature engineering or scoring natively in Spark
  DataFrames/Spark ML — explicitly rejected in favor of reusing the
  existing pipeline via `foreachBatch`.
- Minting genuinely new synthetic customers per cycle — the existing
  200-row sample pool is the data source; a richer/larger pool is a
  separate future decision, not this sub-project's job.
- Windowed velocity features computed in Spark (the roadmap's original
  sub-project 5 mentioned this as a possibility) — the existing
  pre-computed `feature_row` values from the sample pool are used as-is;
  adding real-time windowed aggregation is a separate, larger piece of
  work not undertaken here.
