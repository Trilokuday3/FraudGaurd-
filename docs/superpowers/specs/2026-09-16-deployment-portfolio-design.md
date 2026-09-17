# Sub-project 8: Deployment + Portfolio Integration — Design

**Date:** 2026-09-16
**Status:** Draft, pending review
**Roadmap:** `docs/superpowers/specs/2026-09-12-fraudguard-platform-roadmap.md` (sub-project 8)

## Purpose

Take the FraudGuard app (backend from sub-projects 1-4, frontend from
sub-project 6) from "runs on a laptop via `make api` + `npm run dev`" to a
live, publicly reachable deployment, and link it from the personal portfolio
site so a recruiter can open it. This is the roadmap's stated end goal — the
project isn't "resume-proof" until this ships.

Sub-projects 5 (local-only streaming) and 7 (MLOps/monitoring polish) remain
unbuilt; this sub-project does not depend on either and does not block them.

## Constraints (confirmed with the user)

- **Budget: $0/month.** Every piece must run on a genuinely free tier —
  no trial credits that expire, no "free for 30 days" offers.
- **Database: migrate off SQLite to hosted Postgres.** The roadmap's
  original target was PostgreSQL; sub-projects 4-6 built and tested against
  SQLite (`decisions.db`) as a local convenience. Deployment is the point
  where this gets corrected, and a hosted Postgres also avoids the
  ephemeral-filesystem risk most free app hosts have (a SQLite file on disk
  can be wiped on redeploy/restart).
- **Auto-deploy on push to `main`.** Standard CI/CD story; both chosen
  platforms (Vercel, Render) support this natively via their GitHub
  integration — no custom deploy pipeline required.

## Provider research (2026, as of this spec)

Free-tier terms change often, so this was checked fresh rather than assumed:

- **Railway**: no permanent free tier anymore (removed). Ruled out.
- **Fly.io**: free allowance is now a 2-hour trial only, not usable for a
  standing deployment. Ruled out.
- **Render**: has a real, ongoing free tier for **Web Services** — sleeps
  after ~15 min of no traffic, cold-starts in 30-60s on the next request.
  Render's separate **Background Worker** service type has *no* free
  instance type — it requires a paid instance starting at $7/month. This
  directly affects the replay-worker design below (see "Replay worker").
- **Neon** (Postgres): free tier auto-suspends compute after 5 min idle,
  but resumes automatically and quickly on the next query — no manual
  action ever required.
- **Supabase** (Postgres): free tier fully **pauses an entire project**
  after ~1 week of no activity, requiring a manual unpause from the
  dashboard before it serves traffic again.
- **Vercel**: free tier for Next.js apps has no equivalent sleep behavior
  relevant here; it's the natural fit for the existing `frontend/` app and
  needs no reconsideration.

**Decision:** Vercel (frontend) + Render free Web Service (API) + Neon free
Postgres (database). Neon over Supabase specifically because a portfolio
link that might sit unvisited between recruiter clicks must never need a
manual unpause to come back to life.

## Architecture

```
GitHub (main branch)
   │  push → auto-deploy (native git integration, no custom CI needed)
   ├──────────────► Vercel: frontend/ (Next.js)
   │                    NEXT_PUBLIC_API_BASE_URL → Render API URL
   │
   └──────────────► Render: serving/ (FastAPI, free Web Service)
                        │  in-process replay worker (see below)
                        │  loads deploy/model_artifacts/ from disk at boot
                        ▼
                   Neon (hosted Postgres, free tier)
                        DATABASE_URL

GitHub Actions (free on public repos):
  - existing pytest/npm test suites run on every push (CI visibility only;
    does not gate Vercel/Render's own auto-deploy)
  - scheduled workflow, every ~10 min: GET the deployed API's health/decisions
    endpoint — keeps the Render dyno warmer and nudges the replay worker
    forward even with nobody browsing
```

## Components

### 1. Replay worker (in-process, not a separate Render service)

Render's free tier has no cost-free way to run a standalone background
worker — only Web Services are free. So the replay loop is not a second
deployable; it's an `asyncio` task started from FastAPI's `lifespan`
context in `serving/app.py`, living inside the same process as the API.

- Every ~5-10s, pull the next row from a small bundled sample of realistic
  transactions (adapted from sub-project 6's `scripts/replay_transactions.py`
  — reusing its row-selection/shaping logic, but calling the scoring path
  as a direct in-process function call instead of an HTTP POST to itself).
- Score it through the same decision path as `/score`, persist the
  resulting `Decision` row to Postgres — real, growing data for the
  Dashboard, Live Transactions, and Monitoring pages.
- Because the free Render dyno sleeps after 15 min idle, the worker is
  only running while the dyno is awake: a visit wakes it (cold start,
  30-60s, shown as a loading state on first load), and from then on new
  transactions are genuinely live for the rest of that active period. This
  is a real, disclosed tradeoff of staying at $0/month, not a hidden gap —
  it will be documented in `infra/deploy.md` and `README.md`.

### 2. Model artifacts (vendored `mlruns/` tree, not a flat rewrite)

Per the roadmap, MLflow itself is never deployed publicly — it stays a
local reproducibility tool. But `serving/ml_loader.py` (`load_deployed_model`)
and two endpoints (`/model/metadata`, `/model/comparison`) don't just read a
few flat files: they call `MlflowClient()` at request time too —
`get_run`, `search_runs` across the whole experiment (to find the
baseline/random_forest/xgboost/lightgbm candidate runs for comparison), and
`download_artifacts`. Rewriting all of that to read from a flat directory
would be a real, risky code change touching request-time logic that's
already tested and working.

`MLFLOW_TRACKING_URI` is just a filesystem path (`./mlruns` by default) —
MLflow's local file-store backend needs no running server, only files on
disk in its own directory layout. So instead of a new loading mechanism,
this **vendors a pruned copy of the real `mlruns/` directory tree** — the
deployed run plus its 4 candidate sibling runs, all within the same
experiment — into a new committed path, `deploy/model_store/`, and points
`MLFLOW_TRACKING_URI` at that path in the deployed environment only.
Zero changes to the two endpoints (`/model/metadata`, `/model/comparison`).
One small, necessary change to `ml_loader.py`: MLflow's local file store
bakes an absolute path into each run's `meta.yaml` (and, separately, into
each "Logged Model" entity's own `meta.yaml` under
`<experiment_id>/models/m-<hash>/` — where `mlflow.sklearn.log_model`
actually stores model files) at creation time. A copy of the tree at a
different absolute path (a different worktree locally, or a different host
in production) still has the *original* machine's paths baked in, so
`runs:/<id>/...` artifact resolution silently fails until those paths are
rewritten to the copy's real location. `ml_loader.load_deployed_model` now
does this rewrite once, at boot, before loading — confirmed necessary and
sufficient by directly building and loading the real vendored snapshot,
not assumed. `decision/thresholds.json` keeps working exactly as it does
today (it's already a flat committed file, unaffected by this).

### 3. Database migration (schema only, no data)

`decisions.db` today holds only demo/smoke-test rows from local
development — nothing worth migrating. The deployed Postgres schema is
created fresh from the existing SQLAlchemy models (the same
`Base.metadata.create_all()` call already used locally), then the replay
worker populates it from empty. The only thing that changes between local
and deployed config is `DECISION_DB_URL` (SQLite connection string locally,
Neon connection string in production) — no new migration tooling
(Alembic, etc.) is introduced for this. (Confirmed via
`serving/models.py:39` — the engine already branches `connect_args` on
`db_url.startswith("sqlite")`, so a Postgres URL is already a supported
path; only a Postgres driver dependency, e.g. `psycopg[binary]`, needs
adding to `pyproject.toml`.)

### 4. Deploy mechanism (no Dockerfile needed for deployment)

Render deploys the API via its native Python runtime: build command
`pip install -r requirements.txt` (or the project's `pyproject.toml`
equivalent), start command `uvicorn serving.app:app --host 0.0.0.0 --port
$PORT`. This is separate from the roadmap's local-only `docker-compose.yml`
(used for the streaming demo in sub-project 5) — no container image is
built for this deployment.

### 5. CORS

`serving/app.py` already has `CORSMiddleware` (added in sub-project 6's
final fix wave) restricted to `http://localhost:\d+`. It needs the
deployed Vercel origin added, or the deployed frontend cannot reach the
deployed API at all — this is a concrete, required change, not a
hypothetical.

### 6. Secrets/env

Real values live only in each platform's dashboard, never committed.
`.env.example` stays the local template. New/changed variables for
deployment:

- `DECISION_DB_URL` (Render) — the existing env var name, just set to the
  Neon Postgres connection string in production instead of a SQLite path.
  No new var name is introduced; `serving/models.py` already branches on
  `db_url.startswith("sqlite")`, so this is a config-only change.
- `NEXT_PUBLIC_API_BASE_URL` (Vercel) — the deployed Render API's URL.
- `MLFLOW_TRACKING_URI` (Render) — repointed at the vendored
  `deploy/model_store/` path instead of local `./mlruns`. `MLFLOW_RUN_ID`
  stays the same run ID value it already is; `THRESHOLDS_PATH` stays
  `decision/thresholds.json` unchanged (already a flat committed file).

### 7. Portfolio site integration

`Personal/Portfolio/Trilokeshvenkatauday.github.io/index.html` (separate
git repo, GitHub Pages, remote `Trilokuday3/Trilokeshvenkatauday.github.io`)
has a `.projects-grid` of `.project-card` divs (title + bullet list + stack
chips) under `<section id="projects">` — none currently have an outbound
link. This adds a new `.project-card` for FraudGuard following the exact
same title/bullets/stack-chips structure as the existing three cards, plus
a small new "View Live →" link element (new CSS, styled to match the
site's existing cyan-accent dark theme) since no existing card has one to
copy from. Exact markup and copy is a plan-level detail, not a design-level
one — the site's structure is simple enough not to need further
architectural discussion here.

### 8. `infra/deploy.md`

Currently a placeholder per the roadmap's monorepo layout. This sub-project
writes it for real: env var list per platform, deploy order (Neon → Render
→ Vercel, since the API needs `DATABASE_URL` before it can boot, and the
frontend needs the API's live URL before it can be configured), and the
runbook for re-baking `deploy/model_artifacts/` if the model is ever
retrained (re-run `ml/enrich_deployed_run.py` locally, copy the new run's
artifacts into `deploy/model_artifacts/`, commit, push).

## Testing / acceptance approach

- Existing backend (`pytest`) and frontend (`npm test`) suites must stay
  green against the Postgres-shaped `DATABASE_URL` locally before deploying
  (verifies the SQLite→Postgres switch doesn't break anything at the ORM
  layer — SQLAlchemy abstracts most of this, but this gets verified, not
  assumed).
- A GitHub Actions workflow runs both suites on every push, for visible CI
  status (does not gate the platforms' own auto-deploy).
- Post-deploy acceptance: hit the deployed API's `/model/metadata` and
  `/decisions` directly, confirm the deployed frontend's Dashboard renders
  real (not hard-coded) data end-to-end, confirm the replay worker produces
  a new decision row within one active session, confirm CORS allows the
  real Vercel origin.
- Portfolio site: confirm the new card renders correctly at desktop and
  phone width (the site already has a responsive `.projects-grid` pattern
  used by the other three cards) and that "View Live →" opens the deployed
  frontend.

## Risks / explicit decisions

- **Cold start is real and disclosed.** The alternative (a paid Render
  instance, ~$7/mo) was offered to the user and explicitly declined in
  favor of staying at $0/month. Documented in `README.md` and
  `infra/deploy.md`, not hidden.
- **Replay worker liveness depends on dyno wake state**, not true 24/7
  streaming — this is a direct, disclosed consequence of the $0 constraint,
  mitigated (not fully solved) by the free GitHub Actions keep-alive ping.
- **No Alembic/migration tooling introduced.** Justified because there is
  no production data to migrate — schema is created fresh via
  `create_all()`. If a future sub-project needs real schema migrations,
  that's a new decision at that time, not made here.
- **Portfolio site is a separate git repo** (`Trilokeshvenkatauday.github.io`,
  not part of the `FraudGuard` monorepo) — this sub-project's plan will need
  its own task(s) operating in that second repo, called out explicitly so
  an implementer isn't confused about which repo's `git add`/`git commit`
  applies where.

## Out of scope (unchanged from roadmap)

- Sub-project 5 (Kafka/Spark streaming) — stays local-only, not deployed.
- Sub-project 7 (MLOps registry promotion, Evidently drift reports,
  rollback runbook) — separate sub-project, not built here.
- The dashboard redesign (sidebar nav, filter tabs, inline detail panel,
  persisted Analyst Notes) queued after sub-project 6 — separate follow-up,
  not part of this deployment work.
