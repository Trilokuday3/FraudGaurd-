# FraudGuard — Fraud Intelligence & Real-Time Risk Management Platform

A portfolio-grade fraud detection product: synthetic transaction data →
leakage-safe features → calibrated, explainable ML → a decision engine → a
FastAPI service → a Next.js web app → deployed, with a link on
[Trilokeshvenkatauday.github.io](https://trilokeshvenkatauday.github.io).

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
cp .env.example .env               # then fill in MLFLOW_RUN_ID (see docs/decision-engine-acceptance.md)
make api                          # serves the decision engine (score/explain/investigate)
cd frontend && cp .env.example .env.local && npm install && npm run dev
# in another terminal, for a live-updating feed:
make replay
```

## Layout

```
libs/fraudguard_core/   shared schemas, config, value sets
generator/               synthetic data generator (customers, merchants, devices, transactions)
features/                leakage-safe feature engineering -> data/features.parquet
ml/                      modeling: data prep, training, calibration, SHAP, Isolation Forest, CLI
decision/                cost-sensitive threshold selection + rules engine -> decision/thresholds.json
serving/                 FastAPI decision engine service (score/explain/investigate)
frontend/                Next.js web app: Dashboard, Live Transactions, Investigations, Model Center, Threshold Simulator, Monitoring -> consumes the serving/ API
scripts/                 local-dev helpers, e.g. replay_transactions.py (stands in for sub-project 5's streaming pipeline by replaying data/features.parquet into /score)
tests/unit/              generator + feature + modeling unit tests
tests/features/          leakage + schema gate on the feature table
tests/dq/                pandera data-quality gate against ./data
tests/integration/       end-to-end tests against the FastAPI serving app
docs/                    specs, data dictionary, generation-model writeup
```

See the roadmap doc for what's next: EDA + feature engineering, modeling,
decision engine + API, streaming, frontend, MLOps, deployment.
