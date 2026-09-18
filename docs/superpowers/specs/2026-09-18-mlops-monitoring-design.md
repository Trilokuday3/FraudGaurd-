# Sub-project 7 — MLOps & Monitoring — Design

**Date:** 2026-09-18
**Status:** Approved
**Parent:** `2026-09-12-fraudguard-platform-roadmap.md`
**Depends on:** sub-project 3 (Modeling — MLflow runs to promote), sub-project 4
(Decision Engine + API — `serving/models.py`'s `Decision` table, `decision/select_thresholds.py`),
sub-project 8 (Deployment — `scripts/vendor_model_store.py`, `deploy/model_store/`,
`deploy/requirements.txt`; this sub-project's promotion/rollback scripts re-sync
those exact artifacts rather than introducing a second way to produce them).

## Purpose

Give the already-built decision engine an operational layer: a real way to
promote a newly-trained candidate to "deployed" with a quality gate (not a
manual `.env` edit), a way to undo that safely, a signal for when live
traffic's feature distribution stops looking like what the model was
trained on, and a signal for when live traffic itself stops looking like
valid data — plus a local, portfolio-facing way to run the API in a
container. Matches the roadmap's sub-project 7 deliverables: MLflow
registry promotion flow, Dockerized inference image, an Evidently drift
report, automated data-quality checks, and a rollback runbook.

**In scope:** a real MLflow Model Registry promotion flow (`mlops/promote_model.py`),
a symmetric rollback script (`mlops/rollback_model.py`), a scored-traffic DQ
check reusing the existing training-data schema (`mlops/check_scored_dq.py`),
an Evidently data-drift report comparing scored traffic to the training
reference (`mlops/drift_report.py`), a local-only Dockerfile/compose file,
a scheduled GitHub Actions job wiring the DQ + drift scripts into CI, and
this runbook.

**Out of scope:** a model-registry UI (MLflow's own `mlflow ui` already
covers this locally), automated retraining or auto-promotion (promotion is
always a deliberate `--run-id` invocation, never triggered by drift
crossing a threshold — matches this project's "no scheduled retraining"
scope throughout), and treating the Docker image as a second deploy target
(sub-project 8's Render-native deploy is the only one; see Risks below).

## Success criteria

1. A candidate MLflow run can be promoted to "deployed" via one command,
   and the promotion is refused (not silently allowed) if it doesn't at
   least match the current champion's validation PR-AUC, unless explicitly
   forced.
2. Rolling back is one command, and rolling back twice in a row returns to
   the exact state before the first rollback (roadmap acceptance
   criterion: "documented rollback to previous registered model").
3. The Evidently drift report is generated from real scored data (roadmap
   acceptance criterion: "Drift report generated on real scored data"),
   not a synthetic stand-in — verified directly against this repo's real
   trained model and real generated dataset, not only a test fixture.
4. The scored-traffic DQ check reuses the same schema
   (`fraudguard_core.schemas.features_schema`) sub-project 1's generator DQ
   suite already validates training data with, so there is exactly one
   definition of "a valid feature row" in the codebase, not two.
5. The local Docker image is clearly documented as a reproducibility/
   portfolio artifact, never conflated with sub-project 8's actual deploy
   path.

## MLflow Model Registry promotion (`mlops/promote_model.py`)

Registers runs under one registered model, `fraudguard-fraud-model`, using
**aliases** (`champion`, `previous`) rather than the legacy numbered-stage
API (MLflow's own recommended replacement). `promote_model(run_id, ...)`:

1. Runs `ml.enrich_deployed_run.enrich_deployed_run(run_id)` unconditionally
   (idempotent — see that function's own docstring) to compute/refresh
   `deployed_val_pr_auc` on the candidate run.
2. If a `champion` alias already exists, compares the candidate's
   `deployed_val_pr_auc` against the champion run's own metric of the same
   name; refuses with `ValueError` if the candidate is strictly worse,
   unless `force=True`.
3. Calls `mlflow.register_model(...)` to create a new registry version for
   the run, re-points `previous` at the outgoing champion (if any), and
   points `champion` at the new version.
4. Re-runs `decision.select_thresholds.select_thresholds_for_run` and
   `scripts.vendor_model_store.vendor_model_store` against the newly
   promoted run, so `decision/thresholds.json` and `deploy/model_store/` —
   the serving layer's actual read path, unchanged from sub-project 8 —
   stay in sync with whichever run the registry now calls champion.

The registry is the source of truth for *which run should be deployed*;
`thresholds.json`/`deploy/model_store/` remain the *mechanism* the serving
layer reads, kept in sync here rather than replaced — changing that
mechanism was explicitly out of scope (would have meant re-touching
sub-project 8's already-verified serving code for no behavioral gain).

## Rollback (`mlops/rollback_model.py`)

Swaps the `champion` and `previous` aliases (rather than jumping to
`previous` and discarding it), then re-runs the same two sync steps
promotion does, against whichever run is now champion. Swapping instead of
overwriting makes rollback symmetric: calling it twice in a row returns to
the state before the first call, which is also directly what the test
suite verifies — a real property, not just an implementation detail.
Raises `ValueError` if no `previous` alias exists yet (nothing to roll back
to — e.g. only one promotion has ever happened).

## Scored-traffic data quality (`mlops/check_scored_dq.py`)

Requires `Decision` (sub-project 4's decision-log table) to actually carry
the full feature row it scored, not just the persisted top-5 SHAP snapshot
it already had — added as a new `feature_row` JSON column, populated at
`/score` time. The check pulls the most recent N `Decision.feature_row`
values, casts the JSON booleans back to the 0/1 ints
`fraudguard_core.schemas.features_schema` checks for, and validates with
`lazy=True` so every violation is reported, not just the first. The CLI
(`python -m mlops.check_scored_dq`) exits non-zero on failure specifically
so a CI job running it actually fails when it should — printing "FAILED"
alone would not fail a workflow step.

## Drift report (`mlops/drift_report.py`)

Reference = a sample of `data/features.parquet` (the same training data
sub-project 3 trained on); current = the same recent `Decision.feature_row`
rows the DQ check reads. Identifier/timestamp columns
(`transaction_id`/`customer_id`/`merchant_id`/`timestamp`) are excluded
from the comparison — drift on a column that's unique per row by
construction isn't meaningful. Uses Evidently's `Report([DataDriftPreset()])`
(Evidently 0.7's current API — its Model Registry-era `Dashboard`/`Profile`
API from older tutorials no longer exists), saves an HTML artifact, and
returns a summary dict (drifted-column count/share, per-column drift
values) for scripting/CI use. If there's no scored traffic yet, skips
cleanly (returns `report_path: None`) rather than erroring — there is
nothing to compare against on a freshly-seeded database.

## Local Docker image

A root `Dockerfile` + `docker-compose.yml` (+ `.dockerignore`), reusing
sub-project 8's already-lean `deploy/requirements.txt` (no jupyter/black/
ruff/pytest) and already-vendored `deploy/model_store/` — no new artifact
generation, only a new way to run what already exists. `docker-compose.yml`
runs the API against a real local Postgres container and picks up
`MLFLOW_RUN_ID` from the repo's existing `.env` automatically (Docker
Compose's standard `.env`-file behavior — confirmed directly via
`docker compose config`, not assumed). `MLFLOW_RUN_ID`/`DECISION_DB_URL`
are deliberately not baked into the image itself: they have to match
whatever run `deploy/model_store/` currently holds and whichever database
the container should log to, both of which are per-environment.

## Scheduled monitoring (`.github/workflows/mlops-monitor.yml`)

A daily cron (+ manual `workflow_dispatch`) job: seeds reference data
(`generator seed` + `features build` — `data/` is gitignored, same
regeneration sub-project 1's own DQ gate already requires in CI), then runs
the DQ check and drift report against a real deployed database, gated on a
`DEPLOYED_DECISION_DB_URL` **secret** (repository *secret*, not a public
*variable* like sub-project 8's `DEPLOYED_API_URL` — this one is a live
database connection string with credentials in it). No-ops cleanly until
that secret exists, same pattern as sub-project 8's `keep-alive.yml`. The
drift step runs even if the DQ step fails (`!cancelled()`) since a drift
report is still useful context right when DQ has just flagged a problem;
the drift step itself never fails the job (it is a report, not a gate —
see success criterion 3's own framing).

## Testing approach

Unit tests train a real (tiny) `LogisticRegression` against a real, if
small, synthetic feature set for every promotion/rollback test — the same
"real pipeline functions, not a hand-rolled column list" discipline
`tests/integration/test_serving_api.py`'s fixture already established for
sub-project 4. Beyond the unit suite, promotion, scoring, the DQ check, and
the drift report were also run end-to-end against this repo's actual
trained model and actual generated dataset (not only synthetic fixtures) —
promoting the real deployed run through the registry, scoring 30 real rows
through the live FastAPI app, then running both checks against that real
data and confirming a real ~4MB Evidently HTML report is produced.

## Component boundaries

| unit | does | consumed via | depends on |
|---|---|---|---|
| `serving/models.py` (`Decision.feature_row`) | persists the full feature row per scored decision | populated in `serving/app.py`'s `_compute_score` | none new |
| `mlops/promote_model.py` | registry promotion gated on PR-AUC | CLI (`python -m mlops.promote_model --run-id ...`) or `make promote RUN_ID=...` | `ml.enrich_deployed_run`, `decision.select_thresholds`, `scripts.vendor_model_store` |
| `mlops/rollback_model.py` | swaps champion/previous aliases | CLI or `make rollback` | same three, plus `mlops.promote_model`'s alias helper |
| `mlops/check_scored_dq.py` | validates scored traffic | CLI, `make dq-scored`, or the scheduled workflow | `fraudguard_core.schemas.features_schema`, `Decision.feature_row` |
| `mlops/drift_report.py` | Evidently drift report | CLI, `make drift`, or the scheduled workflow | `ml.data.load_features_and_labels`, `Decision.feature_row` |
| `Dockerfile` / `docker-compose.yml` | local containerized API + Postgres | `docker compose up --build` | sub-project 8's `deploy/requirements.txt`, `deploy/model_store/` |
| `.github/workflows/mlops-monitor.yml` | scheduled DQ + drift in CI | GitHub Actions cron/dispatch | the two `mlops/` scripts above, a `DEPLOYED_DECISION_DB_URL` secret |

## Risks / decisions

- **Aliases, not stages** — MLflow's numbered-stage Model Registry API
  (`Staging`/`Production`) is deprecated in favor of aliases; using
  aliases here avoids building on a deprecated surface from day one.
- **One level of rollback history** — only `champion`/`previous` are
  tracked, not a longer promotion history. Matches the roadmap's own
  singular framing ("rollback to *previous* registered model"); a deeper
  history would need explicit design (and a use case) this project doesn't
  have yet.
- **No auto-promotion on drift** — drift/DQ checks are observability only;
  promotion always stays a deliberate, explicit action. Auto-promoting on
  a drift signal risks promoting a worse model in response to a data
  quality problem, not a real improvement — a materially different (and
  riskier) feature than what was asked for here.
- **Docker image is local-only, explicitly not a second deploy path** —
  sub-project 8 already made a considered, researched choice (Render's
  native Python runtime, no Dockerfile) for the actual deploy; this image
  exists purely to demonstrate containerization for portfolio purposes and
  to give a one-command way to run the full stack (API + Postgres) locally
  without installing Postgres directly. Conflating the two would undo
  sub-project 8's own reasoning for no reason.
- **`DEPLOYED_DECISION_DB_URL` as a secret, not a variable** — unlike
  `DEPLOYED_API_URL` (a public URL, fine as a variable), a database
  connection string carries credentials and must never be readable in
  workflow logs or by anyone with read access to the repo's variables.
