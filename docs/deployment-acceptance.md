# Sub-project 8 (Deployment) — Acceptance Notes

**Date:** 2026-09-16

## Postgres compatibility (Task 7)

- Confirmed no test or production code path is SQLite-specific beyond
  `serving/models.py`'s existing `db_url.startswith("sqlite")` branch.
  `grep -rn "sqlite\|SQLite" tests/ serving/` found 5 matches, all
  either that same branch, a test fixture constructing a `sqlite:///...`
  URL (not asserting SQLite-specific behavior), or a comment. No hardcoded
  SQLite-only assertion anywhere.
- **Verified live against a real ephemeral Postgres 16 container**, not
  just inferred from the SQLAlchemy abstraction argument:
  `docker run --rm -d -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16`,
  then `DECISION_DB_URL=postgresql+psycopg://postgres:test@localhost:5433/postgres
  pytest tests/unit tests/integration -v` → **102 passed, 0 failed**
  (549.42s). Same pass count and same test names as the SQLite-backed run
  below — nothing Postgres-specific broke.

## Full test suite (Tasks 1-6, SQLite-backed)

`pytest tests/unit tests/features tests/dq tests/integration -v` →
**117 passed, 0 failed** (819.27s). This is 117 = 102 (the set also run
against Postgres above) + 15 (`tests/features` and `tests/dq`, which never
touch `DECISION_DB_URL` and so weren't re-run against Postgres — they
exercise the generator/feature pipeline, not the decision-log database).

This run covers every change from sub-project 8's Tasks 1-6: the vendored
MLflow model-store snapshot and its path-repair fix (Task 1), the lean
deploy requirements file (Task 2), settings-driven CORS (Task 3), the
bundled replay sample data (Task 4), the in-process replay worker (Task
5), and confirms the new GitHub Actions workflow YAML is valid (Task 6).

## Two real bugs found and fixed during this sub-project (not hypothetical)

1. **MLflow's file store bakes absolute paths into its metadata.** A plain
   `shutil.copytree` of `mlruns/` into `deploy/model_store/` looked
   correct but `mlflow.sklearn.load_model("runs:/<id>/<name>")` failed
   against the copy with `Failed to download artifacts from path
   '<name>', please ensure that the path is correct` — both each run's
   `meta.yaml` (`artifact_uri`) and each "Logged Model" entity's own
   `meta.yaml` under `<experiment_id>/models/m-<hash>/`
   (`artifact_location`) needed rewriting to the copy's real location.
   Fixed with an idempotent repair step in `serving/ml_loader.py`
   (`_repair_artifact_uris`), covered by a new regression test that
   relocates a real logged model's tracking directory and confirms it
   still loads.
2. **The replay worker's bundled sample data was plain dicts, but
   `_compute_score` expects a `FeatureRow`.** Booted cleanly (no
   import-time error) and logged nothing, but zero decisions ever
   appeared — the exception was silently swallowed inside
   `asyncio.create_task`'s fire-and-forget execution. Only surfaced by
   directly invoking `replay_worker_loop` with the real `_compute_score`
   outside of `create_task`, which is where the `AttributeError: 'dict'
   object has no attribute 'model_dump'` actually printed. Fixed by
   parsing each bundled row into `FeatureRow` in `_load_sample_transactions`;
   re-verified live against a running server afterward (`GET /decisions`
   showed real rows with real SHAP contributions and timestamps).

## Frontend (Task 9)

- `npm install` → 286 packages installed cleanly. **Flagged, not acted
  on:** npm reports a real security vulnerability in the pinned
  `next@14.2.5` (see npm's own advisory link in the install output).
  Upgrading it is out of scope for this deployment sub-project — it's an
  existing pin from sub-project 6, and bumping it risks the 6 already-built
  and reviewed pages. Left as a flagged follow-up, not silently fixed or
  silently ignored.
- `npm test` → **13 passed, 0 failed** (3 test files).
- `npm run build` → succeeds cleanly, all 9 routes compiled (a mix of
  static and server-rendered, matching sub-project 6's design).

## What's NOT verified here (a real, disclosed gap)

- No live Neon/Render/Vercel account exists in this environment, so the
  actual deployed-URL path (cold start behavior, real CORS against a real
  Vercel origin, the replay worker running under Render's real process
  lifecycle) has not been observed live. `infra/deploy.md` step 3
  explicitly tells the human operator to re-run
  `pytest tests/unit tests/integration -v` against the real Neon
  connection string before considering the deploy done — this doc's
  Docker-Postgres verification is strong evidence the ORM layer is fine,
  but it is not a substitute for that final real-account check.
- No browser is available in this environment, so the portfolio site's
  new FraudGuard card (Task 10) was verified by structural comparison
  against the existing three cards' markup, not a rendered screenshot —
  same disclosed limitation pattern as `docs/frontend-acceptance.md` from
  sub-project 6.

## Deployment checklist (mirrors `infra/deploy.md`)

- [ ] Create Neon project, get connection string
- [ ] Create Render Web Service, set env vars, deploy
- [ ] Run `pytest tests/unit tests/integration -v` against the real Neon URL
- [ ] Create Vercel project (root directory `frontend`), set env var, deploy
- [ ] Set `DEPLOYED_FRONTEND_ORIGIN` on Render to the real Vercel URL
- [ ] Set `DEPLOYED_API_URL` GitHub Actions repository variable to the real Render URL
- [ ] Confirm the Dashboard loads real data with no CORS errors
- [ ] Update the portfolio site's `href="PLACEHOLDER_DEPLOYED_URL"` to the real URL, commit, push
