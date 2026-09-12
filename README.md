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

## Quickstart

```
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
python -m generator seed          # writes ./data/*.parquet
pytest tests/unit -v              # generator correctness + leakage guards
pytest tests/dq -v                # schema + referential-integrity gate on ./data
```

## Layout

```
libs/fraudguard_core/   shared schemas, config, value sets
generator/               synthetic data generator (customers, merchants, devices, transactions)
tests/unit/              generator correctness tests
tests/dq/                pandera data-quality gate against ./data
docs/                    specs, data dictionary, generation-model writeup
```

See the roadmap doc for what's next: EDA + feature engineering, modeling,
decision engine + API, streaming, frontend, MLOps, deployment.
