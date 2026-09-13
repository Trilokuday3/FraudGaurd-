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
```

## Layout

```
libs/fraudguard_core/   shared schemas, config, value sets
generator/               synthetic data generator (customers, merchants, devices, transactions)
features/                leakage-safe feature engineering -> data/features.parquet
ml/                      modeling: data prep, training, calibration, SHAP, Isolation Forest, CLI
tests/unit/              generator + feature + modeling unit tests
tests/features/          leakage + schema gate on the feature table
tests/dq/                pandera data-quality gate against ./data
docs/                    specs, data dictionary, generation-model writeup
```

See the roadmap doc for what's next: EDA + feature engineering, modeling,
decision engine + API, streaming, frontend, MLOps, deployment.
