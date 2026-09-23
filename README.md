# FraudGuard — Fraud Intelligence & Real-Time Risk Management Platform

A portfolio-grade fraud detection product: synthetic transaction data →
leakage-safe features → calibrated, explainable ML → a decision engine → a
FastAPI service → a Next.js web app → deployed, with a link on
[Trilokeshvenkatauday.github.io](https://trilokeshvenkatauday.github.io).

**Live demo:** [https://fraudgaurd.trilokeshvenkatauday.me/](https://fraudgaurd.trilokeshvenkatauday.me/)

Full design: `docs/superpowers/specs/2026-09-12-fraudguard-platform-roadmap.md`.

## Status

**Sub-project 1 (Data Foundation) — done.** See
`docs/superpowers/specs/2026-09-12-data-foundation-design.md` and
`docs/sub1-acceptance.md`.

**Sub-project 2 (EDA + Feature Engineering) — done.** See
`docs/superpowers/specs/2026-09-12-feature-engineering-design.md` and
`docs/sub2-acceptance.md`.

**Sub-project 3 (Modeling) — done.** See
`docs/superpowers/specs/2026-09-12-modeling-design.md`,
`docs/ml-acceptance.md`, and `ml/model_card.md`.

**Sub-project 4 (Decision Engine + API) — done.** See
`docs/superpowers/specs/2026-09-15-decision-engine-api-design.md` and
`docs/decision-engine-acceptance.md`.

**Sub-project 6 (Frontend) — done.** See
`docs/superpowers/specs/2026-09-15-frontend-design.md` and
`docs/frontend-acceptance.md`. A Next.js web app (`frontend/`) covering
Dashboard, Live Transactions, Investigations, Model Center, Threshold
Simulator, and Monitoring against the real decision engine API.
(Sub-project 5, Streaming, is local-only and deferred per the roadmap;
`scripts/replay_transactions.py` stands in for it locally so the
frontend's Live Transactions/Monitoring pages have a growing feed to
poll.)

**Live streaming pipeline (Kafka + Spark) — built, pending setup and
verification.** `streaming/` holds a producer, a Spark Structured Streaming
consumer and a cursor store, run on a 10-minute GitHub Actions schedule
against a Kafka broker on a VM (`streaming/docker-compose.kafka.yml`,
`infra/kafka-vm-setup.md`, `.github/workflows/live-streaming.yml`). The VM,
broker and secrets are not yet created and the pipeline has not been run
end-to-end, so the deployed app is not yet fed by it. See
`docs/superpowers/specs/2026-09-23-live-kafka-spark-streaming-design.md`.

**Sub-project 8 (Deployment + Portfolio Integration) — code/config/CI
complete, not yet live.** See
`docs/superpowers/specs/2026-09-16-deployment-portfolio-design.md`,
`docs/deployment-acceptance.md`, and `infra/deploy.md`. The repo is fully
ready to deploy at $0/month (Vercel + Render free Web Service + Neon free
Postgres) — an in-process replay worker keeps the deployed app's data
live without a paid background-worker service (the current data path; the
Kafka + Spark pipeline above is intended to replace it once verified), and a vendored,
pruned MLflow snapshot (`deploy/model_store/`) means the deployed API
never depends on a running MLflow server. What's left is the human
account-holder's own manual pass through `infra/deploy.md` — creating the
actual Render/Vercel/Neon accounts isn't something this automation does on
your behalf. **Live demo:** [https://fraudgaurd.trilokeshvenkatauday.me/](https://fraudgaurd.trilokeshvenkatauday.me/)

**Sub-project 7 (MLOps & Monitoring) — code complete, spec/runbook docs
pending.** `mlops/`: `promote_model.py` (MLflow Model Registry promotion,
gated on beating the current champion's validation PR-AUC),
`rollback_model.py` (reverts the registry's champion alias), `check_scored_dq.py`
(validates recently-scored transactions against the same schema
`tests/dq/` validates training data with), `drift_report.py` (Evidently
data-drift report comparing scored traffic to the training reference).
A local-only `Dockerfile` + `docker-compose.yml` (portfolio/reproducibility
artifact — not the deploy path; see Sub-project 8 below) and
`.github/workflows/mlops-monitor.yml` (scheduled DQ + drift check against
a real deployed database, gated on a `DEPLOYED_DECISION_DB_URL` secret
until one exists).

### Deployment

- **Frontend:** Vercel (free tier).
- **API + replay worker (current in-process fallback until the streaming
  cutover):** Render free **Web Service** — sleeps after ~15
  min idle, cold-starts in 30-60s on the next request. This is the
  accepted cost of staying at $0/month, not a bug; a ~$7/month instance
  would remove it if ever wanted later.
- **Database:** Neon free-tier Postgres — auto-wakes on the next query with
  no manual intervention (chosen over Supabase specifically because
  Supabase's free tier fully pauses a project after ~1 week of no
  traffic).
- Full step-by-step instructions: `infra/deploy.md`.

## Quickstart

```
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
python -m generator seed          # writes ./data/*.parquet
pytest tests/unit -v              # generator correctness + leakage guards
pytest tests/dq -v                # schema + referential-integrity gate on ./data
make features                     # writes ./data/features.parquet
pytest tests/features -v          # leakage + schema gate on the feature table
make train                        # trains baseline through champion candidates, selects the overall best performer, plus Isolation Forest
python -c "from ml.enrich_deployed_run import enrich_deployed_run; enrich_deployed_run('<run_id>')"  # persists shap_background + model_comparison.json onto the deployed run (required for /explain and /model/comparison)
cp .env.example .env               # then fill in MLFLOW_RUN_ID (see docs/decision-engine-acceptance.md)
make api                          # serves the decision engine (score/explain/investigate)
cd frontend && cp .env.example .env.local && npm install && npm run dev
# in another terminal, for a live-updating feed:
make replay
```

Local-only Docker (reproducibility/portfolio artifact, not the deploy path):

```
docker compose up --build   # reads MLFLOW_RUN_ID from your existing .env
# api on http://localhost:8000, backed by a real local Postgres container
```

`.env`'s deployment-only settings (all optional locally, defaulted off):
`DEPLOYED_FRONTEND_ORIGIN` (extra CORS origin allowed alongside localhost —
leave blank locally), `ENABLE_REPLAY_WORKER` (in-process background replay
loop, off by default so local dev doesn't get spammed with fake decisions
unless you opt in), `REPLAY_INTERVAL_SECONDS` (seconds between replayed
transactions when the worker is on, default `7.0`).

## Layout

```
libs/fraudguard_core/   shared schemas, config, value sets
generator/               synthetic data generator (customers, merchants, devices, transactions)
features/                leakage-safe feature engineering -> data/features.parquet
ml/                      modeling: data prep, training, calibration, SHAP, Isolation Forest, CLI
decision/                cost-sensitive threshold selection + rules engine -> decision/thresholds.json
serving/                 FastAPI decision engine service (score/explain/investigate)
mlops/                   MLflow registry promotion/rollback, scored-traffic DQ checks, Evidently drift reports (sub-project 7)
frontend/                Next.js web app: Dashboard, Live Transactions, Investigations, Model Center, Threshold Simulator, Monitoring -> consumes the serving/ API
scripts/                 local-dev + deploy helpers: replay_transactions.py (sub-project 5 streaming stand-in), vendor_model_store.py and generate_sample_transactions.py (deployment prep, sub-project 8)
deploy/                  committed deployment artifacts: model_store/ (vendored MLflow snapshot), requirements.txt (lean prod deps), sample_transactions.json (bundled replay data)
streaming/               live Kafka + Spark pipeline: producer, Spark consumer, cursor store, docker-compose.kafka.yml (built, pending setup; see docs/superpowers/specs/2026-09-23-live-kafka-spark-streaming-design.md)
infra/                   deploy.md: the real step-by-step deployment runbook; kafka-vm-setup.md: one-time Kafka VM setup
tests/unit/              generator + feature + modeling unit tests
tests/features/          leakage + schema gate on the feature table
tests/dq/                pandera data-quality gate against ./data
tests/integration/       end-to-end tests against the FastAPI serving app
docs/                    specs, data dictionary, generation-model writeup
```

See the roadmap doc for what's next: sub-project 7's spec/plan docs and
rollback runbook (code is done, see Status above), and going fully live
per `infra/deploy.md`.
