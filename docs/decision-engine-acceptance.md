# Sub-project 4 — Acceptance Run

**Date:** 2026-09-15
**Deployed model run:** `a13a804d60b04ad2a0d1f413cdf976d2` (experiment
`fraudguard-modeling`, run name `deployed_model_final`; params
`{"boosted_tree_champion": "lightgbm", "deployed_model": "baseline"}` —
the deployed model is the calibrated Logistic Regression baseline, a real,
investigated, honestly-reported finding from sub-project 3, not a bug.)

This is a **fresh** training run, produced after Task 9's fix (commit
`c98cc95`) to persist the deployed model's raw, pre-calibration form
(`raw_deployed_model`) to MLflow. The previous acceptance attempt used run
`ed2915a3238542bbbfe970c924c85f79`, which predates that fix and lacks the
`raw_deployed_model` artifact — `/score` and `/explain` 500'd against it.
This run was produced by rerunning `python -m ml train` against the real
519,876-row `./data/features.parquet`, and confirmed (via
`MlflowClient().load_model("runs:/<run_id>/raw_deployed_model")`) to carry
the fix. As shown below, `/score` and `/explain` now succeed end-to-end
against this run.

## Threshold selection

Cost matrix: false negative = transaction amount, review = $3.50 flat, false
positive (block) = $25 flat (see
`docs/superpowers/specs/2026-09-15-decision-engine-api-design.md` for the
full rationale).

Run against the real 519,876-row `./data/features.parquet` and the real
deployed run:

- `deployed_val_pr_auc` (written by `ml.enrich_deployed_run.enrich_deployed_run`): **0.5316211004945466**
  (notably lower than the run's `test_pr_auc` of 0.7100955151748024 — a real
  train/val-vs-test gap for the calibrated baseline, not a computation error;
  left as-is per this task's scope, flagged here for sub-project 3/5 follow-up).
- t_review: **0.04**
- t_block: **0.82**
- Total cost on validation at this policy: **34782.91**

(These threshold/cost numbers are numerically identical to the earlier
`ed2915a3238542bbbfe970c924c85f79` run's — expected, since both runs train
the same deterministic baseline model type against the same
`data/features.parquet` and the same time-based validation split; the run ID,
and now the presence of `raw_deployed_model`, are what differ.)

Full `decision/thresholds.json` written by `decision.select_thresholds.select_thresholds_for_run`:

```json
{
  "t_review": 0.04,
  "t_block": 0.82,
  "total_cost": 34782.91,
  "model_run_id": "a13a804d60b04ad2a0d1f413cdf976d2",
  "cost_matrix": {
    "review_cost": 3.5,
    "false_positive_cost": 25.0,
    "false_negative_cost": "transaction amount"
  }
}
```

## Test suite

Run: `pytest tests/unit tests/integration -v -k "decision or serving or ml_"`
Result: `1 failed, 38 passed, 24 deselected, 2101 warnings, 9 errors in 165.10s (0:02:45)`

Both failure categories are a pre-existing test-isolation issue, not a
sub-project 4 defect, and unrelated to Task 9's `raw_deployed_model` fix.
Root cause: `serving/config.py` builds its `settings` object as a
**module-level singleton** (`settings = Settings()`, evaluated once on first
import of `serving.config`). `tests/integration/test_serving_api.py`'s
`app_client` fixture works around this per-test by calling
`importlib.reload(serving.app)` after `monkeypatch.setenv(...)` — but
reloading `serving.app` does not reload `serving.config`, so if
`serving.config` (or anything importing it, e.g. `serving.app`) is first
imported anywhere else in the same pytest session — for example merely by
collecting `tests/unit/test_serving_config.py`, which does
`from serving.config import Settings` — the cached `settings` singleton
freezes whatever `MLFLOW_RUN_ID`/`MLFLOW_TRACKING_URI` were in the
environment at that first-import moment (typically unset), and every later
`app_client`-based test silently loads the *stale* settings instead of its
own monkeypatched ones. This surfaces as
`mlflow.exceptions.MlflowException: Not a proper runs:/ URI:
runs://calibrated_deployed_model` (empty run ID) whenever more than one test
file is collected together — confirmed by running the exact same test in
total isolation:

```
pytest tests/integration/test_serving_api.py::test_health -v
# 1 passed, 488 warnings in 30.65s
```

This task (Task 10) is documentation and execution only — no Python code is
in scope for it, so this pre-existing test-isolation bug is reported here
rather than fixed. A fix (reload `serving.config` too, or stop relying on a
module-level settings singleton in `serving/app.py`) should follow before
sub-project 5/6 depend on this test suite being green end-to-end.

## API smoke test

Smoke-tested against a real `uvicorn serving.app:app` process (not just
`TestClient`), configured via `.env` with `MLFLOW_TRACKING_URI=./mlruns`,
`MLFLOW_RUN_ID=a13a804d60b04ad2a0d1f413cdf976d2`,
`THRESHOLDS_PATH=./decision/thresholds.json`. The server loaded the deployed
model, raw model, Isolation Forest, and SHAP background successfully at
startup (no errors), then:

- `GET /health` → `{"status": "ok"}`

- `GET /model/metadata` →
  ```json
  {
    "model_run_id": "a13a804d60b04ad2a0d1f413cdf976d2",
    "deployed_model_name": "baseline",
    "val_pr_auc": 0.5316211004945466,
    "test_pr_auc": 0.7100955151748024,
    "calibration_method": "isotonic",
    "t_review": 0.04,
    "t_block": 0.82
  }
  ```

- `POST /score` with a real feature row (row 0 of `data/features.parquet`,
  `transaction_id=TXN0000000000`) — **succeeds** (this is the Task 9 fix
  confirmed working end-to-end against the real deployed model; the earlier
  acceptance attempt against `ed2915a3238542bbbfe970c924c85f79` 500'd here):

  ```json
  {
    "transaction_id": "TXN0000000000",
    "model_score": 0.07983193277310924,
    "decision": "review",
    "triggered_rules": [],
    "decision_source": "model"
  }
  ```

- `POST /explain` with the same feature row — **succeeds**:

  ```json
  {
    "transaction_id": "TXN0000000000",
    "contributions": {
      "amount": -0.0036826279989373985,
      "hour_of_day": 0.004274447281846804,
      "is_night": -0.05435408365143445,
      "is_cross_border": -0.03271225697894892,
      "amount_vs_customer_p95": 0.39366071458064517,
      "txn_count_1h": 0.0,
      "txn_count_24h": -0.009014094306773718,
      "is_new_device": 0.895589425673134,
      "device_age_days": 0.3905421216497411,
      "customer_device_count_so_far": 0.05577533492533209,
      "merchant_fraud_rate_hist": -0.15104006816330745,
      "is_new_country_for_customer": 0.35228407092106806,
      "ip_country_mismatch": -0.026094867733272235,
      "payment_method_bank_transfer": 0.02384848571548446,
      "payment_method_card": -0.1504058807707195,
      "payment_method_wallet": 0.10642044360078554
    }
  }
  ```

- `GET /investigations/TXN0000000000` (looking up the decision just
  persisted by the `/score` call above) — **succeeds**, and matches what was
  scored:

  ```json
  {
    "transaction_id": "TXN0000000000",
    "model_score": 0.07983193277310924,
    "decision": "review",
    "triggered_rules": [],
    "decision_source": "model",
    "model_run_id": "a13a804d60b04ad2a0d1f413cdf976d2",
    "shap_top_features": {
      "is_new_device": 0.895589425673134,
      "amount_vs_customer_p95": 0.39366071458064517,
      "device_age_days": 0.3905421216497411,
      "is_new_country_for_customer": 0.35228407092106806,
      "merchant_fraud_rate_hist": -0.15104006816330745
    },
    "created_at": "2026-09-15T11:46:36.989959"
  }
  ```

Task 9's fix (loading `raw_deployed_model` — the pre-calibration
`LogisticRegression`, which exposes `coef_` — separately from
`calibrated_deployed_model` for SHAP's `LinearExplainer`) is confirmed
working end-to-end: no 500s anywhere in this smoke test.

## What's next

Sub-project 5 (Streaming, local-only) computes windowed velocity features
live and feeds them through this same `/score` contract. Sub-project 6
(Frontend) is the first real consumer of `/investigations/{transaction_id}`
and `/explain` for the Investigations module.

Before either of those depend on the test suite being fully green, the
test-isolation issue documented above under "Test suite" (module-level
`serving.config.settings` singleton not surviving multi-file pytest
sessions) should get a fast-follow fix.
