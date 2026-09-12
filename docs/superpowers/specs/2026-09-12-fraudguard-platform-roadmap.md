# FraudGuard — Fraud Intelligence & Real-Time Risk Management Platform — Roadmap

**Date:** 2026-09-12
**Status:** Approved (design). Sub-project specs and plans produced one at a time.
**Source brief:** `FraudGuard_Fraud_Intelligence_Platform_Complete_Project_Guide.docx`
**Supersedes:** the loan-decisioning platform previously scoped at
`../Financial_Risk_Loan_Decisioning_Platform/docs/superpowers/specs/2026-09-06-platform-roadmap.md`.
That work is paused, not deleted — its `libs/riskcore` shell and specs stay in
git history/on disk in case any of it (schema patterns, DQ approach) gets reused
here.

## Goal

Portfolio / resume-proof build, actually deployed (not just `docker compose up`
on a laptop — the source brief's own goal is "a recruiter should be able to
open the platform"). Two deliverables:

1. **FraudGuard** — the fraud-detection product itself: data → features → model
   → explainability → decision engine → API → web app → monitoring.
2. **A live link** from the existing personal portfolio site
   (`Trilokeshvenkatauday.github.io`, tracked at the git root) into the
   deployed FraudGuard app, plus an updated project card there.

## Scope decision: the brief is a maximalist checklist, not a spec

The source doc is explicitly a "complete build guide" — 35 sections, 12
frontend modules, full streaming stack. Building all of it solo is not a
realistic portfolio timeline. This roadmap keeps every *concept* the brief
cares about (imbalanced classification, leakage-safe time-based validation,
SHAP, calibration, cost-sensitive thresholds, streaming concepts, MLOps,
monitoring, a real web app) but scopes the *surface area* down:

- **6 frontend modules**, not 12: Dashboard, Live Transactions, Investigations,
  Model Center, Threshold Simulator, Monitoring. Cut for v1: Customers,
  Merchants, Fraud Network (graph), Geo Intelligence, Experiment Lab, Alerts,
  Reports — each is a real feature, none is load-bearing for the story the
  brief itself describes in §29 (Portfolio Demo Script) or §30 (Resume Bullet
  Strategy). Reconsidered after v1 is deployed and demoable.
- **Streaming is simulated in production, real in dev.** Locally, Kafka +
  Spark Structured Streaming run in Docker Compose to legitimately demonstrate
  the streaming concepts (brief explicitly wants this — §33, "Demonstrate
  streaming concepts with Kafka and Spark Structured Streaming"). In the
  *deployed* environment, running a always-on Kafka broker is not
  free-tier-shaped; the deployed app instead runs a lightweight in-process
  replay worker that pushes transactions through the same scoring/decision
  path at a slow visible cadence, so "Live Transactions" is genuinely live,
  not hard-coded. This is exactly the brief's own rule 34: "Do not start with
  Kafka/Spark before establishing a sound offline ML baseline" — the offline
  baseline and the deployed product come first; the always-on streaming
  cluster is a local-only demonstration, documented as such.
- **Data is a purpose-built synthetic generator, not the IEEE-CIS Kaggle CSV.**
  The brief allows either ("IEEE-CIS ... or another appropriate transaction-
  fraud dataset") and explicitly wants "a realistic internal schema... rather
  than treating the raw CSV as the final database" — a generator gives that
  schema directly, plus determinism, no Kaggle-auth dependency, and reuses the
  exact latent-risk / leakage-guard pattern already proven in the loan
  project's sub-project 1. Same rationale as that project's data-foundation
  design.

## Approach (chosen)

- **pandas / pyarrow** for the generator and offline feature engineering
  (no Spark for v1 — added back only if the streaming sub-project needs it
  for windowed features; see below).
- **PostgreSQL** — transactional + analytical storage (schema per brief §5).
- **scikit-learn / XGBoost / LightGBM** — baseline → champion classifier.
  **Isolation Forest** — complementary anomaly score.
- **SHAP** — global + local explainability.
- **MLflow** — experiment tracking + model registry (local Postgres/SQLite
  backend + local artifact store; not deployed publicly, used for the repo's
  reproducibility story).
- **FastAPI** — model serving + decision engine + investigation/query
  endpoints.
- **Next.js / React** — FraudGuard web app (the brief's own frontend choice).
- **Kafka + Spark Structured Streaming** — local-only, Docker Compose,
  demonstrates the streaming path; deferrable-to-stub exactly like the loan
  project treated it, so descoping if it drags is a clean decision.
- **Docker Compose** for full local stack; **cloud deploy** on free/cheap
  tiers: Vercel (frontend), Render or Railway (FastAPI + worker), Neon or
  Supabase (Postgres). Picked in the deployment sub-project once the app
  exists — no infra spend before there's something to deploy.

## Target architecture

```
synthetic generator (customers, merchants, devices, transactions, fraud label)
        │
        ▼
raw Parquet (local / object storage)  ──►  data-quality gate (pandera)
        │
        ▼
EDA + point-in-time feature engineering (pandas)
        │
        ▼
gold feature table  ──►  training (sklearn / XGBoost / LightGBM / Isolation Forest)
        │                        │
        │                        ▼
        │                 MLflow tracking + registry
        │                        │
        ▼                        ▼
   ground_truth (fraud_label,   calibrated champion model + SHAP explainer
   never joined to features)              │
        │                                 ▼
        └──────────────────────►  Decision Engine (rules + model + thresholds)
                                          │
                                          ▼
                              FastAPI (score / batch / explain / investigate)
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                        PostgreSQL              Next.js FraudGuard UI
                    (predictions, alerts,   (Dashboard, Live Txns, Investigations,
                     investigations)         Model Center, Threshold Sim, Monitoring)

Local-only, Docker Compose: Kafka (topic: transactions) → Spark Structured
Streaming → windowed velocity features → same Decision Engine, to demonstrate
the streaming path end to end.
```

## Monorepo layout

```
FraudGuard/
├── docker-compose.yml         # postgres, mlflow, kafka+spark (local-only), api, frontend
├── .env.example
├── Makefile                   # up, down, seed, features, train, api, test, lint
├── data/                      # generated Parquet (gitignored)
├── libs/fraudguard_core/      # shared python pkg: schemas, config, value sets, pandera schemas
├── generator/                 # synthetic data generator (batch + stream producer)
├── features/                  # point-in-time behavioral/velocity/device/merchant/geo features
├── ml/                        # training/ evaluation/ explain/ calibration/ anomaly/ model_card.md
├── decision/                  # rules + policy engine (approve/review/block, cost thresholds)
├── serving/                   # FastAPI app: score, batch, explain, investigations, decision endpoints
├── streaming/                 # Kafka producer + Spark Structured Streaming job (local-only)
├── frontend/                  # Next.js app — 6 modules (see scope decision)
├── monitoring/                # drift reports (Evidently), data-quality schedule
├── infra/
│   └── deploy.md              # chosen cloud targets + steps, filled in at deployment sub-project
├── tests/                     # unit / integration / data-quality
└── docs/
    ├── superpowers/specs/     # design docs
    ├── ARCHITECTURE.md
    ├── DATA_DICTIONARY.md
    └── FEATURE_DEFINITIONS.md
```

## Sub-projects (build order)

Each gets its own spec → implementation → build cycle, same pattern as the
loan project. Effort is rough solo estimate.

| # | Sub-project | Key deliverables | Acceptance criteria | Effort |
|---|---|---|---|---|
| **1** | **Data Foundation** | `fraudguard_core` schemas; synthetic generator (customers, merchants, devices, transactions) with injected fraud signal; pandera DQ suite; data dictionary; leakage guards | `make seed dq` on a clean checkout yields DQ-passing Parquet; ≥500k transactions, realistic fraud prevalence (~0.5–3%); deterministic on `(seed, config)` | 4–6 days |
| **2** | **EDA + Feature Engineering** | Notebook-driven EDA (prevalence, amount dist, time/merchant/device/geo patterns); point-in-time transaction/behavioral/velocity/device/merchant/geo features; leakage tests | Feature table reproducible; leakage suite green; EDA writeup with ≥5 concrete business insights | 4–5 days |
| **3** | **Modeling** | Baseline (majority/logistic) → RF → XGBoost/LightGBM champion; probability calibration; Isolation Forest anomaly score; time-based train/val/test split; eval harness (PR-AUC primary, ROC-AUC, precision/recall/F1, calibration); SHAP global+local; model card | Time-aware held-out metrics checked in; champion beats baseline on PR-AUC; calibration curve + SHAP artifacts generated | 5–7 days |
| **4** | **Decision Engine + API** | Cost-sensitive threshold selection (approve/review/block); rules layer; FastAPI endpoints (score, batch score, explain, investigation lookup, model metadata); MLflow model loading; Postgres schema + decision log | All endpoints in OpenAPI; thresholds config-driven and justified by cost analysis; integration tests green | 4–5 days |
| **5** | **Streaming (local-only)** | Kafka producer replaying transactions; Spark Structured Streaming job computing windowed velocity features; wired into Decision Engine | `docker compose up` demonstrates event → streaming feature → score → decision end to end locally; documented as local-only, deferrable to stub if it drags | 3–4 days |
| **6** | **Frontend** | Next.js app: Dashboard, Live Transactions, Investigations (risk score, rules, SHAP), Model Center (baseline vs champion), Threshold Simulator, Monitoring | Every listed module wired to real API data (no hard-coded metrics, per brief §34); usable at phone width | 6–8 days |
| **7** | **MLOps & Monitoring** | MLflow registry promotion flow; Dockerized inference image; Evidently drift report (scheduled or on-demand); automated data-quality checks; rollback runbook | Drift report generated on real scored data; documented rollback to previous registered model | 3–4 days |
| **8** | **Deployment + Portfolio Integration** | Cloud deploy (Vercel + Render/Railway + Neon/Supabase); replay worker for deployed "live" transactions; `infra/deploy.md`; portfolio site (`index.html` at git root) updated with a FraudGuard project card + live link | Deployed URL loads Dashboard and Investigations with real, continuously-updating (replayed) data; portfolio site links to it; secrets via env vars, nothing committed | 4–6 days |

Total ≈ 6–7.5 weeks solo. Sub-projects 1–4 (offline ML core) come before
5–6 (streaming, frontend) before 7–8 (MLOps polish, deployment) — matches the
brief's own §34 warning against starting with Kafka/Spark before a sound
offline baseline exists.

## Cross-cutting conventions

- Shared code in `libs/fraudguard_core`, `pip install -e`-d by every service.
- Config via `pydantic-settings` + `.env`; no secrets in git; `.env.example` committed.
- Every generated/derived dataset is a pure function of `(seed, config)`.
- Every sub-project ships: code + unit tests + a short `docs/` writeup +
  updated `Makefile` targets.
- `PR-AUC` is the headline model metric throughout (brief §11 — never accuracy).
- Leakage discipline carried over verbatim from the loan project: every
  feature computed as-of the transaction/decision time; forbidden-column list
  asserted by a test; `fraud_label`/ground-truth kept out of the feature path.

## Open items carried into sub-project specs

- Exact fraud-prevalence target and cost matrix (false negative vs false
  positive vs review cost) — fixed in sub-project 1/4.
- Whether Isolation Forest output feeds the champion model as a feature or
  stays a separate displayed signal — fixed in sub-project 3.
- Final cloud provider choice per component (Render vs Railway vs Fly.io for
  the API) — fixed in sub-project 8, after checking then-current free-tier
  terms.
