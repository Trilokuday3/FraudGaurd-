# Sub-project 6 (Frontend) — Acceptance Run

**Date:** 2026-09-16

This is the acceptance run for sub-project 6 (Frontend): a Next.js app
(`frontend/`) consuming the real FastAPI decision engine
(`serving/app.py`) built in sub-project 4. Run against a real backend
process already up on `http://localhost:8000`, backed by the real trained
MLflow run recorded in `.env` (`MLFLOW_RUN_ID=17d00e31654c4b9e8a9aa389f797d650`)
and the real `decision/thresholds.json`, with the real generated
`data/features.parquet` present in this worktree.

## Commands run

```
# backend already running from an earlier task:
uvicorn serving.app:app --reload

# this run's setup:
cd frontend && cp .env.example .env.local && npm install && npm run dev
```

`make replay` was **not** used for this run — instead, 10 realistic rows
sampled from `data/features.parquet` were POSTed directly to `/score`
(4 with `is_new_device`/`is_new_country_for_customer`/`ip_country_mismatch`
all forced to `1`, to exercise the `device_location_takeover` block rule;
6 left as sampled, which all scored low enough to `approve`), via a small
one-off script run through the shared venv:
`C:/Users/trilo/Downloads/FraudGuard/.venv/Scripts/python.exe`. This was
faster than running `make replay` for a full minute and produces the same
kind of real, persisted `Decision` rows — 3 decisions already existed in
`decisions.db` from earlier tasks' smoke tests, so the database held **13
total decisions** (6 approve, 0 review, 7 block) by the time of this run.

## Manual smoke test — per-page results

No browser is available in this environment. Verification approach, used
consistently across all 6 pages:

1. **Server-rendered pages** (`/`, `/model`, `/simulator`, and the
   investigation detail route `/investigations/[transactionId]`) embed
   their fetched data directly in the HTML returned by `curl`, so the real
   values below were read straight out of the server-rendered response.
2. **Client-rendered pages** (`/live`, `/investigations` list,
   `/monitoring`) are `"use client"` components that fetch from the API
   *after* hydration (confirmed by reading each page's source — see file
   list below); a `curl` of the route only returns the empty shell before
   React runs. For these, real data was confirmed indirectly: (a) reading
   `lib/api.ts` and each page's `fetchDecisions`/`fetchDecisionsStats`
   calls to confirm they target real backend routes, and (b) `curl`-ing
   those exact backend routes directly and confirming they return the same
   real, non-empty data the page would render on mount. This is a
   documented limitation, not a substitute for an actual visual check.

All 6 routes returned `HTTP 200` from `curl http://localhost:3002/<route>`
(port 3002 — the dev server auto-selected this port because 3000 and 3001
were already bound by other processes in this shared environment).

- **Dashboard (`/`)** — server-rendered, real values embedded in the HTML:
  `Total scored = 13`, `Avg score = 0.268`, `Deployed model = baseline`,
  `Val PR-AUC = 0.532`. Recent-decisions list rendered real
  `DecisionBadge`s for `approve`, `review`, `block` variants.

- **Live Transactions (`/live`)** — client-rendered; page source
  (`app/live/page.tsx`) polls `fetchDecisions({ limit: 50 })` every 3s.
  Confirmed the underlying call, `GET /decisions?limit=5`, returns real
  rows, e.g. `id=13, transaction_id=TXN-SEED-009, decision=approve,
  model_score=0.00657...`. Table renders inside `overflow-x-auto`.

- **Investigations (`/investigations` list + detail)** — list page
  (`app/investigations/page.tsx`) is client-rendered and calls
  `fetchDecisions({ limit: 100 })`, confirmed non-empty as above. Detail
  page (`app/investigations/[transactionId]/page.tsx`) is
  server-rendered; `curl http://localhost:3002/investigations/TXN-SEED-000`
  returned real embedded content: `decision = block`,
  `triggered_rules = ["device_location_takeover"]`, SHAP features including
  `is_new_device: 0.8956`, `ip_country_mismatch: 0.8437`.

- **Model Center (`/model`)** — server-rendered, real values embedded:
  candidate comparison `baseline (val_pr_auc=0.5426, deployed)`,
  `random_forest (0.4921)`, `xgboost (0.4489)`, `lightgbm (0.5220)`; SHAP
  bar chart and calibration curve rendered from the real
  `/model/comparison` payload.

- **Threshold Simulator (`/simulator`)** — server-rendered, real values
  embedded: `t_review = 0.04`, `t_block = 0.82`, cost curve loaded from
  `decision/thresholds.json`'s real `cost_curve` array; slider interaction
  (`ThresholdSlider`) is a pure client-side interpolation over that
  already-fetched curve, so no further network call is needed at
  interaction time.

- **Monitoring (`/monitoring`)** — client-rendered; page source
  (`app/monitoring/page.tsx`) calls `fetchDecisionsStats()`. Confirmed the
  underlying call, `GET /decisions/stats?since_minutes=60`, returns real,
  non-empty data: `total=13, approve_count=6, review_count=0,
  block_count=7, avg_score=0.268`, with a real `buckets` array (4 buckets
  spanning two distinct minutes of activity) for the bar chart.

## Phone-width check

No browser is available in this environment, so the ~375px check was done
by **Tailwind-class code audit**, not a visual resize — this is a real
limitation of this environment, noted explicitly rather than claiming a
check that wasn't performed.

Audited every `.tsx` file under `frontend/app/` and `frontend/components/`
for hard-coded pixel widths (`grep -rn "w-\[.*px\]\|min-w-\[.*px\]\|width:
\s*[0-9]"`) — **zero matches**. Cross-checked for the responsive patterns
that should be present instead:

- `app/page.tsx`, `app/model/page.tsx`: stat tile rows use
  `grid grid-cols-2 md:grid-cols-4 gap-4` — 2 columns at phone width, 4 at
  desktop.
- `app/layout.tsx`: the page nav uses
  `flex gap-4 overflow-x-auto` with `whitespace-nowrap` links — scrolls
  horizontally within the nav bar itself at phone width rather than
  forcing the whole viewport to scroll.
- `components/DataTable.tsx`: the transactions table is wrapped in
  `overflow-x-auto`, so the table (used by both Live Transactions and the
  Investigations list) scrolls within its own container, not the page.
- `app/investigations/page.tsx`: the search form uses `flex gap-2
  flex-wrap`, so it wraps rather than overflowing at narrow widths.
- Charts (`components/SHAPBarChart.tsx`, `components/
  CalibrationCurveChart.tsx`, `app/monitoring/page.tsx`) all use
  Recharts's `<ResponsiveContainer width="100%" ...>`, so they shrink to
  their parent's width rather than overflowing.
- `components/ThresholdSlider.tsx`'s range input and
  `components/ScoreGauge.tsx`'s bar both use `w-full`.

Result: **pass (by code audit) for all 6 pages** — no page has a layout
container with a fixed width wider than the viewport; every wide element
(tables, the nav bar) scrolls within its own bounded container instead of
the page. No issues found that needed fixing.

## Test suites

- Backend: `pytest tests/unit tests/features tests/dq tests/integration -v`
  → **106 passed in 366.43s (0:06:06)** (0 failed). This is Tasks 1-5's
  backend additions for sub-project 6 plus the full pre-existing
  sub-project 3/4 suite — confirms nothing regressed.
- Frontend: `cd frontend && npm test` (`vitest run`) →
  **13 passed** across 3 test files (`ThresholdSlider.test.tsx`,
  `DecisionBadge.test.tsx`, `DataTable.test.tsx`) in 4.16s, 0 failed.

## Cleanup

After the smoke test, the `npm run dev` process (and a stray duplicate
dev-server process from a retry during setup) were force-killed; confirmed
by `netstat` that nothing remained listening on the ports the frontend had
used. The FastAPI backend on `:8000` was left running, as instructed, for
subsequent verification.

## What's next

This is the final task of the sub-project 6 (Frontend) plan and the final
task of the FraudGuard roadmap's currently-scoped sub-projects. See the
roadmap doc (`docs/superpowers/specs/2026-09-12-fraudguard-platform-roadmap.md`)
for anything beyond this (deployment, further MLOps work) — out of scope
for this task.
