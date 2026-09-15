# Sub-project 6 — Frontend — Design

**Date:** 2026-09-15
**Status:** Approved
**Parent:** `2026-09-12-fraudguard-platform-roadmap.md`
**Depends on:** sub-project 4 (Decision Engine + API) — every module reads
from `serving/app.py`'s existing and newly-added endpoints; no direct model
or database access from the frontend.
**Note:** sub-project 5 (Streaming, local-only) was explicitly skipped for
now per the roadmap's own "deferrable to stub if it drags" clause. Live
Transactions therefore gets its feed from a new minimal replay script
(below), not from a Kafka/Spark pipeline. Revisit only if a later sub-project
needs the local streaming demo restored.

## Purpose

Build the Next.js web app that makes FraudGuard demoable: the 6 roadmap
modules (Dashboard, Live Transactions, Investigations, Model Center,
Threshold Simulator, Monitoring), every one wired to real API data — no
hardcoded metrics (roadmap acceptance criterion, brief §34) — usable at
phone width.

**In scope:** `frontend/` (Next.js App Router app, all 6 modules), 3 small
new read-only endpoints on `serving/app.py` that the frontend needs and the
API doesn't yet expose, and `scripts/replay_transactions.py` (a minimal
local replay tool that gives Live Transactions a real, continuously-growing
feed to poll).

**Out of scope:** authentication/authorization (no login — this is a public
portfolio demo, not a multi-tenant product), the 7 cut frontend modules
(Customers, Merchants, Fraud Network, Geo Intelligence, Experiment Lab,
Alerts, Reports — per the roadmap's scope decision), real-time push
(WebSockets/SSE) — client-side polling is sufficient at this scale and
avoids a second transport to build/debug, cloud deployment (sub-project 8).

## Success criteria

1. Every one of the 6 modules renders data fetched from a real API call at
   page load or on an interval — no component hardcodes a metric, score, or
   chart value (roadmap acceptance criterion).
2. The app is usable at phone width (per roadmap acceptance criterion) —
   verified by manually resizing the browser to ~375px for each page during
   development, not just desktop-tested.
3. Live Transactions shows a feed that visibly grows over time when
   `scripts/replay_transactions.py` is running alongside `make api` — proving
   the "live" claim is real, not a static seed.
4. Investigations lets a user look up any transaction ID that has been
   scored and see its score, decision, triggered rules, and top-5 SHAP
   contribution breakdown, matching exactly what `/investigations/{id}`
   returns. (Not `/explain` — that endpoint recomputes SHAP from a supplied
   feature row, which the frontend never has for a past transaction; only
   the top-5 snapshot persisted at scoring time is available for lookback.)
5. Threshold Simulator's slider recomputes and displays cost against the
   precomputed `cost_curve` returned by `GET /model/metadata` without
   calling the backend on every drag — the curve is fetched once, then
   interpolated client-side.
6. Component tests exist for the components with real logic
   (`DecisionBadge`'s color mapping, `ThresholdSlider`'s cost lookup,
   `DataTable`'s pagination) and pass in CI-equivalent local run
   (`npm test`).

## New backend endpoints (`serving/app.py`)

All three are read-only, reuse `serving/models.py`'s existing `Decision`
table or the MLflow run object already loaded at startup, and follow the
same Pydantic-typed pattern as sub-project 4's endpoints — no new
infrastructure.

| Method | Path | Query params | Response |
|---|---|---|---|
| GET | `/decisions` | `decision` (optional filter: approve/review/block), `limit` (default 50, max 500), `before_id` (cursor, optional) | `{items: [DecisionRow, ...], next_cursor: int \| null}` — most recent first, `DecisionRow` mirrors the `Decision` model's columns |
| GET | `/decisions/stats` | `since_minutes` (optional, default: all-time) | `{total, approve_count, review_count, block_count, avg_score, buckets: [{minute, decision, count}, ...]}` — one row per `(minute, decision)` pair (missing pairs implicitly zero), so Monitoring can stack approve/review/block per minute |
| GET | `/model/comparison` | — | `{candidates: [{name, val_pr_auc, is_deployed}, ...], calibration_curve: [{mean_predicted, fraction_positive}, ...], shap_importances: [{feature, mean_abs_shap}, ...]}` — read from the deployed MLflow run's logged metrics/artifacts |

`GET /decisions` serves Live Transactions (polled), Investigations' browse
view, and raw rows for Dashboard/Monitoring. `GET /decisions/stats` serves
Dashboard's summary tiles and Monitoring's trend chart. `GET
/model/comparison` serves Model Center; `calibration_curve` and
`shap_importances` are read from artifacts/metrics already logged by
sub-project 3's training run and sub-project 4's `ml/enrich_deployed_run.py`
— no new computation, only a new read path.

`GET /model/metadata` (existing, sub-project 4) gains one additional field,
`cost_curve`, copied verbatim from `decision/thresholds.json`'s existing
`cost_curve` array — no new endpoint. This is the single read Threshold
Simulator uses (see below).

Pagination uses a simple `id`-based cursor (`before_id`) rather than
offset — `Decision.id` is an auto-incrementing primary key, so `before_id`
gives stable pagination even as new rows are inserted by the replay script
during a browsing session.

## Live Transactions data source: replay script

`scripts/replay_transactions.py`: reads `data/features.parquet` (already
gitignored, already the exact shape `/score` expects) in row order, and
`POST`s each row to `/score` at a fixed slow interval (default 2.5s,
configurable via `--interval`), stopping after `--count` rows or looping
back to the start if `--loop` is passed. Run via `make replay` in a second
terminal alongside `make api` during local development and demos. This is
explicitly the roadmap's documented local-dev stand-in for sub-project 5's
skipped streaming pipeline — not a claim that this is how the deployed
version will work (sub-project 8 designs the deployed replay worker
separately, per the roadmap's own streaming-substitution note).

## Frontend structure (`frontend/`, Next.js App Router + TypeScript + Tailwind)

```
frontend/
├── app/
│   ├── layout.tsx                        # shared nav shell, theme
│   ├── page.tsx                          # Dashboard (/)
│   ├── live/page.tsx                     # Live Transactions
│   ├── investigations/
│   │   ├── page.tsx                      # browse/search recent decisions
│   │   └── [transactionId]/page.tsx      # detail: score, rules, full SHAP
│   ├── model/page.tsx                    # Model Center
│   ├── simulator/page.tsx                # Threshold Simulator
│   └── monitoring/page.tsx               # Monitoring
├── lib/
│   ├── api.ts                            # typed fetch wrapper for serving/app.py
│   └── types.ts                          # TS types mirroring serving/schemas.py
├── components/
│   ├── StatTile.tsx
│   ├── DecisionBadge.tsx                 # approve/review/block color mapping
│   ├── DataTable.tsx                     # paginated table, used by Live Txns + Investigations
│   ├── ScoreGauge.tsx
│   ├── SHAPBarChart.tsx
│   └── ThresholdSlider.tsx
└── tests/                                # Vitest + React Testing Library
```

**Module → route → endpoints:**

| Module | Route | Endpoints used |
|---|---|---|
| Dashboard | `/` | `GET /decisions/stats`, `GET /model/metadata` |
| Live Transactions | `/live` | `GET /decisions` (polled every few seconds) |
| Investigations | `/investigations`, `/investigations/[id]` | `GET /decisions` (browse), `GET /investigations/{id}` (score, rules, top-5 SHAP) |
| Model Center | `/model` | `GET /model/comparison`, `GET /model/metadata` |
| Threshold Simulator | `/simulator` | `GET /model/metadata` (for `t_review`/`t_block`, and the new `cost_curve` field) |
| Monitoring | `/monitoring` | `GET /decisions/stats?since_minutes=...` |

**Data fetching**: Server Components fetch on page load for anything that
doesn't need to update live (Investigations detail, Model Center, Threshold
Simulator's curve — fetched once, then interpolated client-side per success
criterion 5). Live Transactions and Dashboard's summary tiles poll
`/decisions`/`/decisions/stats` client-side on a short interval via a small
`useEffect` + `setInterval` hook in `lib/api.ts` — no separate real-time
transport, matching the out-of-scope decision above.

## Design approach

Minimal fintech-dashboard aesthetic: dark-friendly neutral palette, one
accent color scale for risk severity (green/amber/red mapped to
approve/review/block via `DecisionBadge`), monospace for transaction
IDs/amounts. Tailwind utility classes directly in components — no separate
design-system package, appropriate for a fixed 6-module portfolio app rather
than a growing product.

Charts (cost curve, calibration curve, SHAP bars) use `recharts` — one
lightweight dependency covering all three chart types rather than hand-rolled
SVG.

## Testing approach

Proportionate to a portfolio-demo frontend sitting on an already
well-tested backend: component tests (Vitest + React Testing Library) only
for components with real logic — `DecisionBadge`'s color mapping,
`ThresholdSlider`'s client-side cost-curve lookup, `DataTable`'s pagination
— not exhaustive coverage of every presentational component. No E2E
framework; manual smoke-testing per page during development (including at
phone width, per success criterion 2), matching the discipline used for
sub-project 4's API.

## Component boundaries

| unit | does | consumed via | depends on |
|---|---|---|---|
| `serving/app.py` additions | 3 new read-only endpoints | HTTP, called by `frontend/lib/api.ts` | `serving/models.py`, MLflow run object already loaded at startup |
| `scripts/replay_transactions.py` | replays `data/features.parquet` rows into `/score` at a slow cadence | CLI, run via `make replay` | `data/features.parquet`, a running `serving/app.py` |
| `frontend/lib/api.ts` | typed fetch wrapper, polling hook | imported by every page | the API's OpenAPI-documented shapes (mirrored in `lib/types.ts`) |
| `frontend/components/*` | shared presentational + interactive units | imported by pages | `lib/api.ts`, `lib/types.ts` |
| `frontend/app/*/page.tsx` | one page per module | Next.js router | `components/*`, `lib/api.ts` |

## Deliverables checklist

- [ ] `serving/app.py` additions: `GET /decisions`, `GET /decisions/stats`, `GET /model/comparison` (+ Pydantic response models in `serving/schemas.py`)
- [ ] `tests/integration/test_serving_api.py` additions covering the 3 new endpoints
- [ ] `scripts/replay_transactions.py`
- [ ] `frontend/` — Next.js App Router app: `app/`, `lib/`, `components/`, `tests/`
- [ ] `Makefile` — `frontend` (dev server) and `replay` targets
- [ ] `docs/frontend-acceptance.md` — screenshots or description of each module against real running data, phone-width check noted
- [ ] `.gitignore` addition: `frontend/node_modules`, `frontend/.next`

## Risks / decisions

- **Polling, not WebSockets/SSE** — simpler, sufficient at this scale (a
  few-second interval on a low-traffic demo app), avoids a second transport
  to build/debug/deploy; revisit only if sub-project 8's deployed version
  shows polling is materially worse in production.
- **Replay script instead of sub-project 5's streaming pipeline** — sub-project
  5 was explicitly skipped for now; this keeps Live Transactions genuinely
  live without requiring Kafka/Spark locally. The script is intentionally
  minimal (one file, no queue) since its only job is to prove the frontend
  isn't hardcoded, not to be a production ingestion path.
- **No auth** — matches the roadmap's "a recruiter should be able to open
  the platform" goal; this is a public read-only demo, not a product with
  user accounts.
- **Threshold Simulator interpolates client-side from a precomputed curve**
  — `decision/thresholds.json`'s `cost_curve` (82 points, already generated
  by sub-project 4) is fetched once and interpolated in the browser as the
  slider moves, rather than calling the backend per drag event — keeps the
  interaction smooth and avoids re-running `compute_total_cost` server-side
  for every slider tick.
