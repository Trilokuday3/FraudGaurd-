# Deployment Runbook

Sub-project 8's real, step-by-step deployment guide. Everything in this
repo is deploy-ready (code, config, CI) as of sub-project 8's commits —
this document is the manual part: creating real accounts and clicking
through each platform's dashboard. No task in the plan performs these
steps on your behalf; they're deliberately left for you.

**Cost: $0/month.** Every platform below is on its genuinely-free tier —
see `docs/superpowers/specs/2026-09-16-deployment-portfolio-design.md` for
the research behind each choice (Render over Railway/Fly.io, Neon over
Supabase).

## Order

**Neon → Render → Vercel.** The API needs `DECISION_DB_URL` before it can
boot, so the database has to exist first. The frontend needs the API's
live URL before `NEXT_PUBLIC_API_BASE_URL` can be set, so the API has to
exist before the frontend.

## 1. Neon (Postgres)

1. Create a free account/project at neon.tech.
2. Copy the connection string it gives you (starts `postgresql://...`).
   Render's SQLAlchemy driver needs the `+psycopg` dialect marker, so
   rewrite it as `postgresql+psycopg://...` before using it — same
   host/user/password/db, just the scheme changed.
3. Keep this string handy — it becomes `DECISION_DB_URL` in step 2 below.

Nothing else to configure: the deployed API creates its own schema on
first boot (`Base.metadata.create_all()`, same as local dev), so there's
no separate migration step.

## 2. Render (API)

1. Create a free account at render.com, connect the GitHub repo.
2. Create a new **Web Service** (not a "Background Worker" — that has no
   free instance type on Render; the replay worker runs in-process inside
   this Web Service instead, see the design spec).
3. Settings:
   - **Root directory:** `.` (repo root)
   - **Build command:** `pip install -r deploy/requirements.txt`
   - **Start command:** `uvicorn serving.app:app --host 0.0.0.0 --port $PORT`
4. Environment variables:

   | Var | Value |
   |---|---|
   | `DECISION_DB_URL` | the Neon connection string from step 1 (with `+psycopg`) |
   | `MLFLOW_TRACKING_URI` | `./deploy/model_store` |
   | `MLFLOW_RUN_ID` | `17d00e31654c4b9e8a9aa389f797d650` (the run currently deployed in this repo — check `.env`'s `MLFLOW_RUN_ID` locally if this has since changed) |
   | `THRESHOLDS_PATH` | `./decision/thresholds.json` |
   | `ENABLE_REPLAY_WORKER` | `true` |
   | `REPLAY_INTERVAL_SECONDS` | `7.0` |
   | `DEPLOYED_FRONTEND_ORIGIN` | *(leave blank for now — filled in at step 4 below, after the Vercel URL exists)* |

5. Deploy. **Expect a cold start**: Render's free tier sleeps a Web Service
   after ~15 minutes with no traffic, and the next request wakes it in
   30-60 seconds. This is the accepted cost of staying at $0/month (a
   ~$7/month Render instance would remove it — deliberately not chosen,
   see the design spec's Risks section). It is expected behavior, not a
   deploy failure — don't debug it as one.
6. Once it's up, hit `<your-render-url>/health` and `/model/metadata`
   directly to confirm the model loaded and the database connected.

## 3. Run the test suite once against the real Neon URL

This is the one verification this repo's own automation couldn't do for
you (no live Neon account existed while building sub-project 8 — see
`docs/deployment-acceptance.md` for exactly what *was* verified locally,
including an ephemeral Docker Postgres container where available). Before
trusting the deploy, run, from the repo root, with your real Neon URL:

```
FRAUDGUARD_TESTS_ALLOW_REMOTE_DB=1 DECISION_DB_URL=<your Neon connection string, with +psycopg> pytest tests/unit tests/integration -v
```

`FRAUDGUARD_TESTS_ALLOW_REMOTE_DB=1` is required: without it,
`tests/conftest.py` swaps any non-SQLite `DECISION_DB_URL` for a temporary
SQLite file. **Warning:** this run writes test rows into that database, so
point it at a throwaway Neon branch, never the live dashboard database.

Expect the same pass count as local SQLite runs. If anything fails here
that didn't fail locally, it's a real Postgres-specific issue to fix
before considering the deploy done.

## 4. Vercel (frontend)

1. Create a free account at vercel.com, import the GitHub repo.
2. Set the project's **root directory** to `frontend` (Vercel's project
   settings, not a build command — the repo root is not a valid Next.js
   project root).
3. Environment variable: `NEXT_PUBLIC_API_BASE_URL` = the Render URL from
   step 2.
4. Deploy. Vercel's own git integration handles auto-deploy on every push
   to `main` from here on — no additional CI wiring needed for that part.

## 5. Close the loop

1. Now that the Vercel URL exists, go back to Render's environment
   variables and set `DEPLOYED_FRONTEND_ORIGIN` to it (e.g.
   `https://fraudguard.vercel.app`, no trailing slash). Render restarts the
   process automatically on an environment variable change — no manual
   redeploy needed, and no code change either (`serving/config.py` reads
   this from the environment at boot).
2. In the GitHub repo's own settings (Settings → Secrets and variables →
   Actions → **Variables**, not Secrets — this is a public URL, not a
   secret), add `DEPLOYED_API_URL` = the Render URL. This is what
   `.github/workflows/keep-alive.yml` checks for; until it's set, that
   workflow no-ops cleanly on its schedule instead of failing.
3. Confirm the loop: open the Vercel URL in a browser, check the Dashboard
   loads real (non-empty) data, and that the browser's network tab shows
   no CORS errors calling the Render API.

## Re-baking after a model retrain

If the deployed model ever changes (a new `ml/__main__.py train` run
promoted to deployed):

1. Locally: `python -c "from ml.enrich_deployed_run import enrich_deployed_run; enrich_deployed_run('<new run id>')"` (persists `shap_background` + `model_comparison.json` onto the new run, same as the Quickstart already documents).
2. `python scripts/vendor_model_store.py --run-id <new run id>` — regenerates `deploy/model_store/` from the new run plus its candidate siblings.
3. Commit the regenerated `deploy/model_store/` and the updated
   `decision/thresholds.json` (re-run
   `decision.select_thresholds.select_thresholds_for_run` for the new run
   first, same as documented for local setup).
4. Update `MLFLOW_RUN_ID` on Render to the new run ID, push.

## Re-baking the sample transaction set

Rarely needed — only if `data/features.parquet`'s schema itself changes
(a new feature column, a renamed one). If it does:

```
python -m scripts.generate_sample_transactions
```

Note the `-m` invocation — running `python scripts/generate_sample_transactions.py`
directly fails with `ModuleNotFoundError: No module named 'serving'`
because it needs the repo root on `sys.path`, which only the `-m` form
(combined with the existing `scripts/__init__.py`) provides. Commit the
regenerated `deploy/sample_transactions.json`.

## Rollback

Both Render and Vercel keep every prior deploy browsable in their
dashboards with a one-click "redeploy this version" / "promote to
production" action. No custom rollback tooling is introduced for a project
at this scale — use the platform's own history.

## Live streaming pipeline (Kafka + Spark)

**Status: built, broker provisioned, pending the first end-to-end run.**
The code and workflow exist (see
`docs/superpowers/specs/2026-09-23-live-kafka-spark-streaming-design.md`;
that spec and its plan describe the original self-hosted-VM design, which
was replaced by Aiven for Kafka -- the connection details below are
current). The Kafka broker is an Aiven for Kafka service with the
`fraudguard.transactions` topic; a real mTLS connection to it from a
developer machine has been confirmed, but the workflow itself has not run.
Until it is verified, the deployed app's live data still comes from the
in-process replay worker (`ENABLE_REPLAY_WORKER=true` on Render, as
configured above). One-time setup is in `infra/kafka-aiven-setup.md`.

**GitHub Actions secrets required** (`.github/workflows/live-streaming.yml`):

| Secret | Value |
|---|---|
| `DEPLOYED_DECISION_DB_URL` | the same Neon string already used by `mlops-monitor.yml`, `postgresql+psycopg://` scheme |
| `KAFKA_BOOTSTRAP_SERVERS` | the Aiven service URI, `<host>:<port>` |
| `KAFKA_CA_CERT` | full contents of Aiven's `ca.pem` |
| `KAFKA_SERVICE_CERT` | full contents of `service.cert` (the access certificate) |
| `KAFKA_SERVICE_KEY` | full contents of `service.key` (the access key) |

The workflow is a visible no-op (it logs a "skipping" note) until **all
five** secrets are set.

**Acceptance checklist and cutover.** The pipeline is built but not yet
verified or live; these steps are how it gets verified. Judge acceptance
from the workflow run logs, not from the dashboard: every run publishes a
new batch (the producer always sends `PRODUCER_BATCH_SIZE` rows, default
30), and until cutover the replay worker keeps adding rows to the same
table. Do these in order; the replay worker stays on until the pipeline is
proven.

1. Run the workflow manually (Actions tab, "Live streaming pipeline", Run
   workflow). In its logs, the "Run producer" step must print
   `Published N rows, cursor A -> B`, and the "Run Spark consumer" step
   must print `batch <id>: scored N, skipped 0` (on the very first run
   there may be one such line per batch Spark splits the backlog into;
   the scored counts must add up to what was published).
2. Run it a second time. The Spark step must score exactly one new
   batch's worth -- the N rows that run's producer step published -- and
   nothing from the first run. Re-scoring the first run's rows would mean
   the checkpoint cache didn't restore (check the "Restore Spark checkpoint" step
   says "Cache restored").
3. Optionally, confirm `GET /decisions/stats` `total` on the Render API
   grew by at least N per run. It grows by more than N while the replay
   worker is still on, so this is a sanity check, not the acceptance test.
4. Only then, **turn off the old replay worker**: set
   `ENABLE_REPLAY_WORKER=false` on Render (Environment tab). Running both
   long-term double-writes to the same `decisions` table.
5. Confirm the dashboard pages (Dashboard, Live Transactions, Monitoring)
   still render, and that `total` keeps growing only as pipeline runs
   complete (by N per run).

**Known limitations:** the Spark checkpoint lives in the GitHub Actions
cache; if it is evicted, the next run re-reads the topic from its earliest
retained offset and re-scores those rows once (`decisions` has no unique
key on `transaction_id`, and the sample pool cycles anyway, so this is
harmless for demo data). "Continuous" means a 10-minute micro-batch cadence
(scheduled Actions runs), not per-event streaming, and scheduled runs can
be delayed or skipped by GitHub under load. The Aiven plan's topic limits
and retention should be confirmed in the Aiven console. See
`infra/kafka-aiven-setup.md`.
