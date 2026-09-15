# Sub-project 4 — Decision Engine + API — Design

**Date:** 2026-09-15
**Status:** Approved
**Parent:** `2026-09-12-fraudguard-platform-roadmap.md`
**Depends on:** sub-project 3 (Modeling) — serves its calibrated deployed model
(`ml.train.select_deployed_model`'s pick, currently Logistic Regression) and
`ml.explain.explain_prediction`
**Blocks:** sub-project 5 (Streaming) and sub-project 6 (Frontend) — both call
this API rather than the model directly

## Purpose

Turn the calibrated, explainable model from sub-project 3 into a decision a
product can act on: a cost-justified approve/review/block threshold policy, a
small hybrid rules layer, and a FastAPI service that scores transactions,
explains its decisions, and logs them for later investigation.

**In scope:** `decision/` (rules, cost-sensitive threshold selection),
`serving/` (FastAPI app, request/response schemas, decision log persistence,
model loading). A small addition to `ml/` to persist a SHAP background sample
as its own MLflow artifact (needed so `/explain` doesn't depend on the full,
gitignored `data/` directory being present at serving time).

**Out of scope:** real-time/streaming feature computation (sub-project 5 —
this API always receives an already-computed feature row, never a raw
transaction it must enrich itself), the frontend (sub-project 6), MLflow
Model Registry promotion/versioning workflow (sub-project 7 — this sub-project
loads one specific, config-named run, not "the current production model" via
registry stage), Postgres itself (SQLite now, with a schema that runs
unchanged on Postgres later).

## Success criteria

1. All endpoints are documented in the auto-generated OpenAPI spec (FastAPI's
   `/openapi.json` / `/docs`), with typed Pydantic request/response models —
   no untyped dict-in-dict-out endpoints.
2. `t_review`/`t_block` thresholds are config-driven (not hardcoded in
   endpoint code) and derived from an explicit, documented cost-sensitive
   grid search over the validation split — never test, matching sub-project
   3's split discipline.
3. Integration tests exercise every endpoint against a real (tiny, seeded)
   SQLite database and a real loaded model — not mocked scoring logic.
4. `/score` and `/score/batch` produce identical decisions for the same input
   whether called once or many times (deterministic given a fixed model run
   and threshold config) — asserted by a test.
5. Every scored transaction is persisted to the decision log exactly once,
   and `/investigations/{transaction_id}` returns exactly what was recorded
   at scoring time — asserted by a test that scores, then looks up, and
   compares.
6. `docs/decision-engine-acceptance.md` records a real run of the threshold
   selection script against sub-project 3's actual validation split, with the
   resulting cost curve and chosen thresholds — not fabricated numbers.

## Cost model and threshold derivation

Three-way decision, two thresholds, `t_review < t_block`, applied to the
calibrated fraud probability `p`:

- `p < t_review` → **approve**
- `t_review ≤ p < t_block` → **review**
- `p ≥ t_block` → **block**

**Cost matrix** (illustrative — there is no real business to survey, so this
is explicitly documented as a reasoned, plausible stand-in rather than a real
company's figures; the *methodology* is what's real):

| Outcome | Cost |
|---|---|
| Approve a legitimate transaction (true negative) | $0 |
| Block a legitimate transaction (false positive) | $25 (support/churn friction) |
| Review any transaction, regardless of true label | $3.50 (analyst labor, flat per case) |
| Approve a fraudulent transaction (false negative) | **the transaction's own `amount`** — a $30 fraud and a $3,000 fraud are not equally bad, so this cost is amount-aware rather than a flat figure |
| Block or review a fraudulent transaction (correctly caught) | $0 additional loss beyond the review cost already counted above |

**Selection procedure** (`decision/select_thresholds.py`):

1. Load the deployed model from the config-named MLflow run.
2. Recompute calibrated scores on the **validation** split (via
   `ml.data.time_based_split` + `ml.data.prepare_model_matrix`, same as
   training — never touches test).
3. Grid-search `t_review` and `t_block` over `[0.0, 1.0]` at a fixed step
   (e.g. 0.01), `t_review < t_block`, computing total realized cost per pair
   using the matrix above and the validation set's real `fraud_label` values
   and `amount` column.
4. Pick the pair minimizing total cost. Write `decision/thresholds.json`
   (`{"t_review": ..., "t_block": ..., "model_run_id": ..., "total_cost":
   ..., "cost_matrix": {...}}`) and a cost-curve summary into
   `docs/decision-engine-acceptance.md`.

The API loads `decision/thresholds.json` at startup — thresholds are never
hardcoded in endpoint code, satisfying success criterion 2.

## Rules layer

Evaluated first, using only columns already present in the feature row (no
new feature engineering). A rule can only ever *escalate* a decision (never
lower it below what the model/other rules would give), except the allowlist
rule, which is the one deliberate override in the other direction:

| Rule | Condition | Effect |
|---|---|---|
| Device/location takeover | `is_new_device AND is_new_country_for_customer AND ip_country_mismatch` | force **block** |
| Velocity spike | `txn_count_1h >= 5` | force **block** |
| High-risk merchant | `merchant_fraud_rate_hist > 0.15` | force at least **review** |
| Outsized amount, unfamiliar device | `amount_vs_customer_p95 > 3.0 AND is_new_device` | force at least **review** |
| Trusted allowlist | `customer_id` or `merchant_id` in a small static allowlist | force **approve**, overrides everything else |

Every transaction is always scored by the model, regardless of which rules
fire — scoring never short-circuits, so `model_score` in the decision log is
always populated, even for an allowlisted or rule-blocked transaction. Rule
evaluation happens independently and combines with the model's score as
follows: for the four escalation rules, the final decision is
`max(rule_decision, model_decision)` on the ordering approve < review <
block — a rule can only push the decision up, never down. The allowlist rule
is checked last and, if matched, unconditionally sets the final decision to
approve regardless of what the model or the other four rules produced.

Every triggered rule name is recorded in the decision log's
`triggered_rules` field (empty list if none fired), and `decision_source`
records whether the final decision came from `model`, `rule`, or
`rule_override` (the allowlist case).

## Feature input contract (avoiding training/serving skew)

The API's request schema mirrors `fraudguard_core.schemas.features_schema`
exactly (the same 18 columns sub-project 2's `features/build.py` produces) —
callers supply a fully-computed feature row, not a raw transaction. This is a
deliberate boundary: real-time feature computation is sub-project 5's job,
and this API is only ever "given a feature row, decide what to do with it."

Internally, the endpoint handler:
1. Validates the request against a Pydantic model matching `features_schema`.
2. Converts it to a single-row DataFrame and calls the *exact same*
   `ml.data.prepare_model_matrix` function sub-project 3's training pipeline
   uses — not a reimplementation — so the encoding (fixed payment-method
   category list, dropped ID columns) can never drift between training and
   serving.
3. Scores it with the loaded calibrated model, evaluates rules, combines per
   the rules-layer logic above, persists, and returns.

## Decision log schema (`serving/models.py`, SQLAlchemy)

Written to run unchanged on Postgres later — no Postgres-only column types.

```python
class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, index=True, nullable=False)
    model_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)  # "approve" | "review" | "block"
    triggered_rules = Column(JSON, nullable=False, default=list)
    decision_source = Column(String, nullable=False)  # "model" | "rule" | "rule_override"
    model_run_id = Column(String, nullable=False)
    shap_top_features = Column(JSON, nullable=False)  # {feature: contribution}, top 5
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
```

One row per scored transaction (`POST /score`/`/score/batch` each insert one
row per input). SQLite for local dev (`DECISION_DB_URL` defaults to a local
file); the same models/migrations run on Postgres by changing the connection
string only, once sub-project 8 stands up real infrastructure.

## API endpoints (`serving/`, FastAPI)

| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/score` | one feature row (features_schema shape) | `{transaction_id, model_score, decision, triggered_rules, decision_source}` |
| POST | `/score/batch` | list of feature rows | list of the same shape, one per input |
| POST | `/explain` | one feature row | `{transaction_id, contributions: {feature: value, ...}}` — full SHAP, recomputed on demand via `ml.explain.explain_prediction`, not the stored top-5 snapshot |
| GET | `/investigations/{transaction_id}` | — | the stored `Decision` row (404 if never scored) |
| GET | `/model/metadata` | — | `{model_run_id, deployed_model_name, val_pr_auc, test_pr_auc, calibration_method, t_review, t_block}` |
| GET | `/health` | — | `{"status": "ok"}` |

All request/response bodies are Pydantic models, so FastAPI's generated
OpenAPI spec documents every endpoint automatically (success criterion 1).

`/score` and `/score/batch` share one internal scoring function so their
behavior can never diverge; batch just loops it and does one bulk DB insert.

## Model loading and the SHAP background-sample gap

Sub-project 3's pipeline logs the calibrated model and Isolation Forest to
MLflow (`mlflow.sklearn.log_model(..., serialization_format="cloudpickle")`,
added in its final-review fix wave) but never persisted a SHAP background
sample as its own artifact — `ml/__main__.py`'s SHAP calls build one in
memory from `train_X`, which only exists during that run. `/explain` at
serving time has no `train_X` to sample from, and depending on the full
(gitignored, ~520k-row) `data/` directory being present on a deployed server
is exactly the kind of environment coupling this sub-project should avoid.

**Fix**: a small addition, `ml/save_shap_background.py` (or a flag on the
existing training run), that samples ~100 rows from `train_X` at the end of
`_run_training` and logs them as a CSV artifact on the same MLflow run
(`mlflow.log_artifact(..., artifact_path="shap_background.csv")`). The API
loads this artifact once at startup alongside the model, exactly the same
run ID as the model itself, so the two are always from the same training
pass.

Config (`serving/config.py`, `pydantic-settings` + `.env`):
- `MLFLOW_TRACKING_URI` (defaults to `./mlruns`, same as `ml/__main__.py`)
- `MLFLOW_RUN_ID` (the specific run to serve — no registry lookup)
- `DECISION_DB_URL` (defaults to a local SQLite file, e.g. `sqlite:///./decisions.db`)
- `THRESHOLDS_PATH` (defaults to `./decision/thresholds.json`)

## Component boundaries

| unit | does | consumed via | depends on |
|---|---|---|---|
| `decision/rules.py` | evaluates the 5 rules against a feature row | import, called by `serving/` | — |
| `decision/select_thresholds.py` | grid-searches the cost model on validation, writes `thresholds.json` | CLI script, run once per model run | `ml.data`, `ml.calibration`, MLflow |
| `decision/thresholds.json` | chosen `t_review`/`t_block` + cost breakdown | read by `serving/` at startup | output of the script above |
| `serving/config.py` | typed settings from env/`.env` | import | `pydantic-settings` |
| `serving/models.py` | SQLAlchemy `Decision` model + session/engine setup | import | SQLAlchemy |
| `serving/schemas.py` | Pydantic request/response models | import | `fraudguard_core.schemas` (mirrors `features_schema`) |
| `serving/ml_loader.py` | loads the model, Isolation Forest, SHAP background sample from the config-named MLflow run once at startup | import, FastAPI dependency | MLflow, `ml.explain` |
| `serving/app.py` | FastAPI app, the 6 endpoints | `uvicorn serving.app:app` | everything above |
| `ml/save_shap_background.py` | one-off: persist a background sample artifact for an already-trained run | CLI script | `ml/data.py`, MLflow |

## Deliverables checklist

- [ ] `decision/rules.py`, `decision/select_thresholds.py`, `decision/thresholds.json` (generated, not hand-written)
- [ ] `serving/config.py`, `serving/models.py`, `serving/schemas.py`, `serving/ml_loader.py`, `serving/app.py`
- [ ] `ml/save_shap_background.py`
- [ ] `requirements-dev.txt` additions: `fastapi`, `uvicorn`, `sqlalchemy`, `httpx` (FastAPI `TestClient` dependency)
- [ ] `tests/unit/test_decision_rules.py`, `tests/unit/test_select_thresholds.py`
- [ ] `tests/integration/test_serving_api.py` (real SQLite, real loaded model, FastAPI `TestClient`)
- [ ] `Makefile` — `api` target (`uvicorn serving.app:app --reload`)
- [ ] `docs/decision-engine-acceptance.md` — real threshold-selection run + cost curve
- [ ] `.gitignore` addition: local SQLite file (e.g. `decisions.db`)

## Risks / decisions

- **Illustrative cost matrix, real methodology** — approved design decision;
  the numbers are a reasoned stand-in, the grid-search-on-validation
  procedure is the actual portfolio-worthy content, and this is stated
  explicitly in the acceptance doc so it reads as rigorous rather than as
  fabricated precision.
- **SQLite now, Postgres-ready schema** — avoids standing up Docker
  Compose/Postgres before there's a deployment story that needs it
  (sub-project 8); revisit only if sub-project 5 (Streaming) needs a shared
  database sub-project 4's SQLite file can't provide.
- **Model loaded by config'd MLflow run ID, not registry stage** — registry
  promotion/versioning is explicitly sub-project 7's job; this sub-project's
  "deployed model" is whichever run ID is in `.env`, changed manually for now.
- **API never computes features itself** — keeps this sub-project's scope to
  decisioning; real-time feature computation belongs to sub-project 5, and
  duplicating point-in-time aggregate logic here would create two
  implementations of the same thing to keep in sync.
- **Rules can only escalate, except one explicit override** — chosen so the
  rules layer's behavior stays easy to reason about (a human reading the
  five rules can predict what they do to any decision) while still
  demonstrating that a rules layer can suppress false positives, not only
  add friction.
