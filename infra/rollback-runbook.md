# Rollback Runbook

Sub-project 7's real, step-by-step guide for reverting a bad promotion.
Companion to `infra/deploy.md` (the deploy runbook) — this document is
what to do *after* a deploy, if the currently-deployed model turns out to
be wrong.

## When to roll back

Any of these is a real signal, not just a vague "something feels off":

- `python -m mlops.check_scored_dq` (or the scheduled
  `.github/workflows/mlops-monitor.yml` run) reports `FAILED` — recently
  scored transactions no longer match the schema the model was trained
  against. This can mean an upstream bug in whatever produces feature
  rows, not necessarily the model itself — check the specific failing
  columns first (`--limit` shows more rows if the sample was too small to
  tell).
- `python -m mlops.drift_report` shows a high `drifted_share` on columns
  that plausibly explain a real behavior change (e.g. `amount`,
  `merchant_fraud_rate_hist`) — read the generated HTML report (per-column
  detail, not just the summary number) before deciding this is real drift
  and not sampling noise from a small `current_limit`.
- A promotion turns out, after real traffic, to score noticeably worse
  than the previous champion did — the validation PR-AUC gate at promotion
  time (`mlops/promote_model.py`) can't catch a model that generalizes
  worse than its own validation split suggested.

Promotion is never automated in response to any of these signals (see the
design spec's Risks section) — rollback is a human decision, and this
runbook is what to actually do once you've made it.

## Local rollback

```
python -m mlops.rollback_model
```

(equivalently: `make rollback`)

This swaps the registry's `champion`/`previous` aliases back and
regenerates `decision/thresholds.json` and `deploy/model_store/` to match
the restored champion. Raises `ValueError` if there's no `previous` alias
set yet — meaning only one promotion has ever happened, so there is
nothing to roll back *to*.

Confirm it worked:

```
python -c "
from mlflow.tracking import MlflowClient
import mlflow, os
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
mlflow.set_tracking_uri('./mlruns')
client = MlflowClient()
print(client.get_model_version_by_alias('fraudguard-fraud-model', 'champion').run_id)
"
```

This should print the run ID you expected to roll back to.

## Rolling back a real deployment

`decision/thresholds.json` and `deploy/model_store/` are what the deployed
Render service actually reads (see `infra/deploy.md`) — they're committed
to the repo, not regenerated on the server. Rolling back a live deployment
means:

1. Run the local rollback above. Note the run ID it prints
   (`rolled_back_to_run_id` if calling `rollback_model()` directly from
   Python, or read it back from the registry as shown above).
2. `git add decision/thresholds.json deploy/model_store/ && git commit -m "fix: roll back to the previous champion run"` —
   per this repo's `CLAUDE.md`, this commit is yours to make, not
   automated.
3. Push to `main`. Render redeploys automatically on push.
4. On Render's dashboard, update the `MLFLOW_RUN_ID` environment variable
   to the run ID from step 1, so it matches what `deploy/model_store/` now
   actually contains. Render restarts the process on an env var change.
5. Hit `<your-render-url>/model/metadata` and confirm `model_run_id`
   matches the run you rolled back to.
6. Re-run `python -m mlops.check_scored_dq` (with `DECISION_DB_URL`
   pointed at the real Neon database) once new traffic has been scored
   under the restored model, to confirm the original signal is gone.

## Rolling back a rollback

Rollback is symmetric — running `mlops.rollback_model` a second time
undoes the first rollback and returns to the model you rolled back *from*
(confirmed directly: `tests/unit/test_rollback_model.py::test_rollback_model_is_reversible`).
If you rolled back by mistake, just run it again rather than trying to
re-promote the original run from scratch.
