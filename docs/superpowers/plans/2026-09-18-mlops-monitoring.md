# Sub-project 7 — MLOps & Monitoring — Implementation Plan

> **Retrospective plan.** All tasks below were implemented, tested, and
> merged (PR #6) before this document was written — the code came first,
> per explicit direction to ship the coding part immediately and defer
> docs. Checkboxes are marked complete against what was actually done, not
> as a forward-looking plan for an agentic worker to execute step-by-step
> (contrast with this repo's other plan docs, e.g.
> `2026-09-16-deployment-portfolio.md`, written before their code existed).

**Goal:** A real MLflow Model Registry promotion/rollback flow, automated
data-quality checks on live scored traffic (not just training data),
an Evidently drift report comparing scored traffic to the training
reference, a local-only Dockerized inference image, and a scheduled CI
job wiring the checks together — the roadmap's sub-project 7 deliverables.

**Architecture:** One new package, `mlops/`, of small CLI-able scripts
(promote/rollback/check/drift), each a thin orchestration layer over
already-existing, already-tested functions (`ml.enrich_deployed_run`,
`decision.select_thresholds`, `scripts.vendor_model_store`,
`fraudguard_core.schemas.features_schema`) rather than new modeling or
validation logic. One schema change (`Decision.feature_row`) is the only
change to existing sub-project 4 code, and it's additive/backward
compatible (defaults to `{}`).

**Tech Stack:** MLflow Model Registry (existing MLflow dependency, new API
surface), Evidently 0.7 (new dependency), Docker + Docker Compose (new,
local-only), GitHub Actions (existing pattern from sub-project 8, new
workflow file).

**Spec:** `docs/superpowers/specs/2026-09-18-mlops-monitoring-design.md`

## Global Constraints

- Promotion is always a deliberate, explicit `--run-id` invocation — never
  automated or triggered by a drift/DQ signal.
- The registry (`fraudguard-fraud-model`, aliases `champion`/`previous`)
  is the source of truth for which run *should* be deployed;
  `decision/thresholds.json` and `deploy/model_store/` stay as the serving
  layer's actual read path (sub-project 8, unchanged) — promotion/rollback
  re-sync them, they don't replace them.
- The scored-traffic DQ check reuses
  `fraudguard_core.schemas.features_schema` rather than defining a second
  schema for the same 17 columns.
- The Docker image is local-only and must never be described as sub-project
  8's deploy path (Render's native runtime).
- `DEPLOYED_DECISION_DB_URL` is a GitHub Actions *secret*, never a
  *variable* — it is a live database connection string.
- Conventional Commits, no `Co-Authored-By` / Claude attribution, per this
  repo's `CLAUDE.md`.

---

## Task 1: `Decision.feature_row` — persist the full feature row per decision

**Files:** `serving/models.py`, `serving/app.py`,
`tests/unit/test_serving_models.py`, `tests/integration/test_serving_api.py`

**Interfaces:** Adds `feature_row: Column(JSON, nullable=False, default=dict)`
to `Decision`; `serving/app.py`'s `_compute_score` populates it from
`row.model_dump(mode="json")`. Consumed by Task 4 (`check_scored_dq.py`)
and Task 5 (`drift_report.py`) — this is the one piece of shared plumbing
both depend on.

- [x] Added a failing test asserting `Decision.feature_row` round-trips
      through SQLite and defaults to `{}` when unset.
- [x] Added the column to `serving/models.py`.
- [x] Added a failing integration test asserting `/score` persists the
      full feature row (not just the top-5 SHAP snapshot).
- [x] Wired `feature_row=row.model_dump(mode="json")` into `_compute_score`'s
      `Decision(...)` construction.
- [x] Verified: `pytest tests/unit/test_serving_models.py tests/integration/test_serving_api.py -k feature_row` — both pass.
- [x] Committed as `ef47c99 feat(serving): persist the full feature row on every scored decision`.

---

## Task 2: `mlops/promote_model.py` — MLflow registry promotion

**Files:** `mlops/__init__.py`, `mlops/promote_model.py`,
`tests/unit/test_promote_model.py`

**Interfaces:** `promote_model(run_id, mlflow_tracking_uri, data_dir, model_store_dir, thresholds_path, registered_model_name, force) -> dict`.
Consumes `ml.enrich_deployed_run.enrich_deployed_run`,
`decision.select_thresholds.select_thresholds_for_run`,
`scripts.vendor_model_store.vendor_model_store` (all pre-existing,
unchanged). Exposes `REGISTERED_MODEL_NAME` and `_get_alias_version` for
Task 3 (`rollback_model.py`) to reuse.

- [x] Verified empirically, before writing any test, that MLflow's Model
      Registry (`register_model` + `set_registered_model_alias` +
      `get_model_version_by_alias`) actually works against this project's
      local file-store tracking backend — not assumed from the API docs,
      since older MLflow versions required a database-backed store for
      the registry.
- [x] Wrote 4 failing tests (first promotion has no previous; second
      promotion sets previous; refuses a worse candidate; `force=True`
      bypasses the refusal), each training a real tiny `LogisticRegression`
      via the real `ml.data` pipeline functions (not a hand-rolled column
      list).
- [x] Implemented `promote_model` and `_get_alias_version`.
- [x] Verified: `pytest tests/unit/test_promote_model.py -q` — 4 passed.
- [x] Real smoke test (not just the synthetic fixture): ran
      `promote_model` against this repo's actual deployed run and real
      `mlruns/`/`data/`, confirmed a real registry version was created and
      `thresholds.json`/`deploy/model_store/` were regenerated correctly,
      then cleaned up the scratch output paths.
- [x] Committed as `8191a30 feat(mlops): add MLflow registry promotion gated on validation PR-AUC`.

---

## Task 3: `mlops/rollback_model.py` — symmetric rollback

**Files:** `mlops/rollback_model.py`, `tests/unit/test_rollback_model.py`

**Interfaces:** `rollback_model(mlflow_tracking_uri, data_dir, model_store_dir, thresholds_path, registered_model_name) -> dict`.
Consumes `mlops.promote_model.{REGISTERED_MODEL_NAME, _get_alias_version}`,
`decision.select_thresholds.select_thresholds_for_run`,
`scripts.vendor_model_store.vendor_model_store`.

- [x] Wrote 3 failing tests: swaps champion/previous correctly; rolling
      back twice returns to the original champion (the symmetry property);
      raises `ValueError` with no `previous` alias set.
- [x] Implemented `rollback_model` as an alias swap (not a one-way "jump
      to previous and discard it") specifically so the second test's
      symmetry property holds.
- [x] Verified: `pytest tests/unit/test_rollback_model.py -q` — 3 passed.
- [x] Committed as `35ef13a feat(mlops): add a rollback script that reverts the registry's champion alias`.

---

## Task 4: `mlops/check_scored_dq.py` — DQ checks on scored traffic

**Files:** `mlops/check_scored_dq.py`, `tests/unit/test_check_scored_dq.py`

**Interfaces:** `check_scored_dq(db_url, limit) -> dict` (`checked_rows`,
`passed`, `failures`). Reuses `fraudguard_core.schemas.features_schema`
(sub-project 1) rather than a new schema. `main()` exits non-zero on
failure. `_load_recent_feature_rows` and `_BOOL_COLUMNS` are reused by
Task 5 (`drift_report.py`).

- [x] Wrote failing tests: passes on valid rows; reports failures (with
      the actual offending column names) on invalid rows; handles zero
      scored rows; ignores legacy rows with no `feature_row` set.
- [x] Implemented `check_scored_dq`, casting JSON booleans to the 0/1 ints
      `features_schema` expects, validating with `lazy=True` so every
      violation is reported, not just the first.
- [x] Verified: `pytest tests/unit/test_check_scored_dq.py -q` — 4 passed.
- [x] **Follow-up fix, found while wiring the scheduled CI job (Task 7):**
      `main()` printed "FAILED" but always exited 0 — a real gap, since a
      CI job running it would report green even on a genuine DQ failure.
      Added `raise SystemExit(1)` on `not result["passed"]`, plus two new
      tests (`test_main_exits_nonzero_when_dq_fails`,
      `test_main_exits_zero_when_dq_passes`) covering both exit codes.
- [x] Verified again: `pytest tests/unit/test_check_scored_dq.py -q` — 6 passed.
- [x] Committed as `c458a9b feat(mlops): validate scored traffic against the existing features DQ schema`
      and (the fix) `5ef6f23 fix(mlops): exit nonzero from check_scored_dq's CLI on DQ failure`.

---

## Task 5: `mlops/drift_report.py` — Evidently drift report

**Files:** `mlops/drift_report.py`, `tests/unit/test_drift_report.py`,
`requirements-dev.txt`

**Interfaces:** `generate_drift_report(data_dir, db_url, reference_sample_size, current_limit, output_path) -> dict`
(`checked_rows`, `drifted_columns_count`, `drifted_share`, `columns`,
`report_path`). Consumes `ml.data.load_features_and_labels` and Task 4's
`_load_recent_feature_rows`/`_BOOL_COLUMNS`.

- [x] Installed `evidently` and verified its real 0.7.x API directly
      (`from evidently import Report`; `from evidently.presets import DataDriftPreset`;
      `Report([DataDriftPreset()]).run(reference_data=..., current_data=...)`)
      against a throwaway script before writing any production code —
      Evidently's API changed substantially across major versions, so
      this wasn't assumed from memory.
- [x] Wrote 2 failing tests: produces a summary dict + a real HTML file
      given seeded scored rows; handles zero scored rows by skipping
      cleanly (`report_path: None`) rather than erroring.
- [x] Implemented `generate_drift_report` and `_summarize`, excluding
      identifier/timestamp columns from the comparison (drift on a unique
      `transaction_id` is meaningless).
- [x] Verified: `pytest tests/unit/test_drift_report.py -q` — 2 passed.
- [x] Real smoke test: scored 30 real rows through the live FastAPI app
      (not a mock), then ran `generate_drift_report` against that real
      data — produced a real ~4MB Evidently HTML report, confirmed via
      `os.path.exists`, then cleaned up.
- [x] Added `evidently>=0.7` to `requirements-dev.txt` and
      `mlops/reports/` to `.gitignore` (generated artifact).
- [x] Committed as `5028363 feat(mlops): add an Evidently drift report comparing scored traffic to training data`.

---

## Task 6: Local-only Dockerized inference image

**Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`

**Interfaces:** `docker compose up --build` — API on `localhost:8000`
against a real local Postgres container. Consumes sub-project 8's
`deploy/requirements.txt` and `deploy/model_store/` (no new artifact
generation).

- [x] Wrote the `Dockerfile`, reusing `deploy/requirements.txt` (lean,
      deploy-only deps — no jupyter/black/ruff/pytest) and copying
      `deploy/model_store/` in directly, so the image needs no MLflow
      server and no access to the full local `mlruns/` tree.
- [x] Wrote `docker-compose.yml`: the API service plus a real
      `postgres:16` service with a health check gate.
- [x] Verified `docker compose config` resolves the file correctly,
      including auto-loading `MLFLOW_RUN_ID` from the repo's existing
      `.env` with zero extra configuration — confirmed directly.
- [x] **Disclosed gap:** the actual `docker build`/`docker compose up`
      was **not** verified — Docker Desktop's daemon wasn't running in
      the development environment (CLI present, daemon down). Flagged
      explicitly rather than claimed working; the Dockerfile's `COPY`
      sources were individually confirmed to exist, but the full build
      (dependency resolution inside a fresh container image) is unverified.
- [x] Committed as `c4c6726 feat: add a local-only Dockerized inference image for reproducibility`.

---

## Task 7: Scheduled monitoring workflow

**Files:** `.github/workflows/mlops-monitor.yml`, `Makefile`

**Interfaces:** Daily cron + `workflow_dispatch`, gated on the
`DEPLOYED_DECISION_DB_URL` secret. `make dq-scored`, `make drift`,
`make promote RUN_ID=...`, `make rollback` for local invocation.

- [x] Wrote the workflow: seeds reference data (`generator seed` +
      `features build` — `data/` is gitignored, same regeneration
      `tests/dq`'s own CI gate already needs), runs the DQ check (fails
      the job on a real DQ failure, per Task 4's fix), then the drift
      report (runs even if DQ failed, via `!cancelled()`, but never fails
      the job itself — a report, not a gate), then uploads the HTML as a
      workflow artifact.
- [x] Verified the YAML is well-formed via `yaml.safe_load`.
- [x] Added the four `make` targets.
- [x] **Disclosed gap:** the workflow's actual scheduled run against a
      live database is unexercised — no real deploy exists yet (see
      sub-project 8's own disclosed gap in `docs/deployment-acceptance.md`).
      It no-ops cleanly in the meantime, same pattern as sub-project 8's
      `keep-alive.yml`.
- [x] Committed as `efb0d09 chore(ci): add scheduled DQ/drift monitoring workflow and mlops Makefile targets`.

---

## Definition of Done

- [x] All 7 tasks committed (9 commits total across two PRs: #6 for the
      code, this doc's own commit for the deferred spec/plan/runbook).
- [x] Full unit suite green: 99 passed (`pytest tests/unit -q`).
- [x] Full integration suite green (`pytest tests/integration -q`),
      including the new `feature_row` persistence test.
- [x] `ruff check` / `black --check` clean on every new/modified file.
- [x] Real, non-synthetic verification: promoted the actual trained model,
      scored real rows through the live app, and ran both the DQ check
      and the drift report against that real data.
- [ ] **What this plan does NOT produce:** a verified Docker build, or a
      scheduled-workflow run against real production data. Both require
      infrastructure (a running Docker daemon; an actual deployed
      database) that doesn't exist in the development environment or, in
      the database's case, doesn't exist at all yet.
