# Sub-project 4 (Decision Engine + API) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn sub-project 3's calibrated deployed model into a decision a product can act on: cost-justified approve/review/block thresholds, a small hybrid rules layer, and a FastAPI service that scores, explains, and logs decisions.

**Architecture:** `decision/` owns rules + cost-sensitive threshold selection; `serving/` owns the FastAPI app, request/response schemas, decision-log persistence, and model loading; a small `ml/` addition persists a SHAP background sample and backfills a metric onto the deployed run so serving never depends on the full `data/` directory being present.

**Tech Stack:** FastAPI, SQLAlchemy (SQLite now, Postgres-portable schema), pydantic-settings, MLflow (loading an already-logged run from sub-project 3), numpy/pandas.

**Spec:** `docs/superpowers/specs/2026-09-15-decision-engine-api-design.md`

## Global Constraints

- Thresholds are config-driven (`decision/thresholds.json`), never hardcoded in endpoint code.
- Threshold selection grid-searches the **validation** split only — never test, matching sub-project 3's split discipline.
- The API never computes point-in-time features itself — it always receives an already-computed feature row matching `fraudguard_core.schemas.features_schema`'s columns.
- Feature-row-to-model-input conversion always goes through `ml.data.prepare_model_matrix` — never a reimplementation — so encoding can't drift between training and serving.
- Decision log schema uses no Postgres-only column types (must run unchanged on SQLite now and Postgres later).
- Rules can only escalate a decision (approve < review < block), except the allowlist rule, which unconditionally overrides to approve.
- Every scored transaction gets exactly one `Decision` row persisted.
- Line length 100, formatted with `ruff` + `black` (existing `pyproject.toml` config).
- Tests live under `tests/`; `pythonpath = ["."]` means `decision` and `serving` are top-level importable packages, same as `ml`/`features`/`generator`.
- **Environment note:** use `C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe` for every command. Do not create a new venv (this environment has previously hit a Windows Application Control DLL block on fresh native-extension installs).
- **Commit convention (per `CLAUDE.md`):** commit task-by-task as usual inside the isolated worktree, one logical commit per task step. Do **not** append `Co-Authored-By: Claude` or any Claude/Anthropic attribution to any commit message — this project's `CLAUDE.md` explicitly forbids it. Nothing gets pushed, merged, or made into a PR without the user's explicit go-ahead at the end (same as every prior sub-project).

---

### Task 1: `ml/enrich_deployed_run.py` — SHAP background persistence + val PR-AUC backfill

**Files:**
- Create: `ml/enrich_deployed_run.py`
- Test: `tests/unit/test_enrich_deployed_run.py`

**Interfaces:**
- Produces: `enrich_deployed_run(run_id: str, data_dir: str = "./data", mlflow_tracking_uri: str = "./mlruns", sample_size: int = 100, tmp_dir: str = "./ml/artifacts") -> dict` — loads the deployed model from the given MLflow run, computes its validation PR-AUC, logs it as metric `deployed_val_pr_auc` on that run, samples `sample_size` rows from the training split's model matrix, writes them to a CSV, and logs that CSV as an artifact (`shap_background.csv`) on the same run. Returns `{"run_id": ..., "deployed_val_pr_auc": ..., "shap_background_path": ...}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_enrich_deployed_run.py
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from ml.enrich_deployed_run import enrich_deployed_run


def _tiny_data_dir(tmp_path):
    n = 300
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    features = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:04d}" for i in range(n)],
            "customer_id": [f"CUST{i % 20:03d}" for i in range(n)],
            "merchant_id": [f"MERC{i % 10:03d}" for i in range(n)],
            "timestamp": timestamps,
            "amount": rng.lognormal(mean=3, sigma=1, size=n),
            "payment_method": rng.choice(["card", "wallet", "bank_transfer"], size=n),
            "hour_of_day": timestamps.hour,
            "is_night": (timestamps.hour < 5).astype(int),
            "is_cross_border": rng.integers(0, 2, size=n),
            "amount_vs_customer_p95": rng.uniform(0.5, 2.0, size=n),
            "txn_count_1h": rng.integers(0, 3, size=n),
            "txn_count_24h": rng.integers(0, 5, size=n),
            "is_new_device": rng.integers(0, 2, size=n),
            "device_age_days": rng.uniform(0, 100, size=n),
            "customer_device_count_so_far": rng.integers(0, 3, size=n),
            "merchant_fraud_rate_hist": rng.uniform(0, 0.1, size=n),
            "is_new_country_for_customer": rng.integers(0, 2, size=n),
            "ip_country_mismatch": rng.integers(0, 2, size=n),
        }
    )
    fraud_label = (rng.uniform(size=n) < 0.05).astype(int)
    guaranteed_fraud_idx = [10, 50, 100, 150, 200, 215, 225, 235, 245, 260, 270, 280, 290]
    fraud_label[guaranteed_fraud_idx] = 1
    ground_truth = pd.DataFrame(
        {
            "transaction_id": features["transaction_id"],
            "fraud_probability_true": rng.uniform(size=n),
            "fraud_label": fraud_label,
            "confirmed_fraud_at": pd.NaT,
        }
    )
    features.to_parquet(tmp_path / "features.parquet", index=False)
    ground_truth.to_parquet(tmp_path / "ground_truth.parquet", index=False)
    return str(tmp_path)


def test_enrich_deployed_run_logs_metric_and_artifact(tmp_path):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    data_dir = _tiny_data_dir(tmp_path / "data")
    mlflow_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("test-enrich")

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(model, name="calibrated_deployed_model", serialization_format="cloudpickle")

    result = enrich_deployed_run(
        run_id,
        data_dir=data_dir,
        mlflow_tracking_uri=mlflow_dir,
        sample_size=20,
        tmp_dir=str(tmp_path / "artifacts"),
    )

    assert result["run_id"] == run_id
    assert 0.0 <= result["deployed_val_pr_auc"] <= 1.0

    client = MlflowClient()
    fetched_run = client.get_run(run_id)
    assert "deployed_val_pr_auc" in fetched_run.data.metrics

    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "shap_background.csv" in artifacts
```

Note: this test logs a plain `LogisticRegression` (not one wrapped through `ml.data.prepare_model_matrix`'s column set) as the "deployed model" for the run, so it must be scored against a model matrix with the SAME columns (`a`, `b`, `c`) it was trained on — but `enrich_deployed_run` internally calls `ml.data.prepare_model_matrix` on the tiny fixture's real 17-feature schema, which won't match. **Fix this before implementing**: change the test's fixture model to be trained on the real `prepare_model_matrix(train_X_raw)` output instead of a synthetic 3-column `X`, i.e. build `train_X` from the tiny fixture first (via `ml.data.load_features_and_labels` + `time_based_split` + `prepare_model_matrix`), fit `LogisticRegression` on that, then log it. Reuse this real, schema-consistent train_X for both training and later validation scoring.

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_enrich_deployed_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.enrich_deployed_run'`

- [ ] **Step 3: Write the implementation**

```python
# ml/enrich_deployed_run.py
"""One-off enrichment of an already-completed deployed-model MLflow run:
persists a SHAP background sample as its own artifact (so /explain at
serving time doesn't need the full data/ directory present), and backfills
the deployed model's own validation PR-AUC as a metric on that run (so
serving's /model/metadata endpoint has one single source of truth -- the
run itself -- for everything it reports)."""

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split
from ml.evaluate import evaluate_predictions


def enrich_deployed_run(
    run_id: str,
    data_dir: str = "./data",
    mlflow_tracking_uri: str = "./mlruns",
    sample_size: int = 100,
    tmp_dir: str = "./ml/artifacts",
) -> dict:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    features, labels = load_features_and_labels(data_dir)
    (train_X_raw, _), (val_X_raw, val_y), _ = time_based_split(features, labels)
    train_X = prepare_model_matrix(train_X_raw)
    val_X = prepare_model_matrix(val_X_raw)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")
    val_pr_auc = evaluate_predictions(val_y, model.predict_proba(val_X)[:, 1])["pr_auc"]

    sample = train_X.sample(min(sample_size, len(train_X)), random_state=42)
    os.makedirs(tmp_dir, exist_ok=True)
    local_path = f"{tmp_dir}/shap_background.csv"
    sample.to_csv(local_path, index=False)

    client = MlflowClient()
    client.log_artifact(run_id, local_path)
    client.log_metric(run_id, "deployed_val_pr_auc", val_pr_auc)

    return {
        "run_id": run_id,
        "deployed_val_pr_auc": val_pr_auc,
        "shap_background_path": local_path,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_enrich_deployed_run.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/enrich_deployed_run.py tests/unit/test_enrich_deployed_run.py
git commit -m "feat(ml): persist SHAP background sample and backfill val PR-AUC on the deployed run"
```

---

### Task 2: `decision/rules.py` — hard rules + decision combination

**Files:**
- Create: `decision/__init__.py` (empty)
- Create: `decision/rules.py`
- Test: `tests/unit/test_decision_rules.py`

**Interfaces:**
- Produces: `evaluate_rules(feature_row: dict, allowlist_customer_ids: frozenset[str] = ALLOWLIST_CUSTOMER_IDS, allowlist_merchant_ids: frozenset[str] = ALLOWLIST_MERCHANT_IDS) -> tuple[str, list[str], bool]` — `(rule_decision, triggered_rule_names, is_allowlisted)`.
- Produces: `combine_decision(model_score: float, t_review: float, t_block: float, rule_decision: str, triggered_rules: list[str], is_allowlisted: bool) -> tuple[str, str]` — `(final_decision, decision_source)`, `decision_source` is `"model"`, `"rule"`, or `"rule_override"`.
- Produces: `DECISION_ORDER: dict[str, int]` — `{"approve": 0, "review": 1, "block": 2}`, used by both functions and by `serving/app.py` if it needs to reason about ordering.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_decision_rules.py
from decision.rules import combine_decision, evaluate_rules


def _base_row(**overrides):
    row = {
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "is_new_device": False,
        "is_new_country_for_customer": False,
        "ip_country_mismatch": False,
        "txn_count_1h": 0,
        "merchant_fraud_rate_hist": 0.01,
        "amount_vs_customer_p95": 1.0,
    }
    row.update(overrides)
    return row


def test_no_rules_trigger_on_a_clean_row():
    decision, triggered, allowlisted = evaluate_rules(_base_row())
    assert decision == "approve"
    assert triggered == []
    assert allowlisted is False


def test_device_location_takeover_forces_block():
    row = _base_row(
        is_new_device=True, is_new_country_for_customer=True, ip_country_mismatch=True
    )
    decision, triggered, _ = evaluate_rules(row)
    assert decision == "block"
    assert "device_location_takeover" in triggered


def test_velocity_spike_forces_block():
    decision, triggered, _ = evaluate_rules(_base_row(txn_count_1h=5))
    assert decision == "block"
    assert "velocity_spike" in triggered


def test_high_risk_merchant_forces_at_least_review():
    decision, triggered, _ = evaluate_rules(_base_row(merchant_fraud_rate_hist=0.2))
    assert decision == "review"
    assert "high_risk_merchant" in triggered


def test_outsized_amount_new_device_forces_at_least_review():
    decision, triggered, _ = evaluate_rules(
        _base_row(amount_vs_customer_p95=4.0, is_new_device=True)
    )
    assert decision == "review"
    assert "outsized_amount_new_device" in triggered


def test_multiple_escalation_rules_take_the_highest():
    row = _base_row(txn_count_1h=5, merchant_fraud_rate_hist=0.2)
    decision, triggered, _ = evaluate_rules(row)
    assert decision == "block"
    assert set(triggered) == {"velocity_spike", "high_risk_merchant"}


def test_allowlisted_customer_flagged_via_explicit_allowlist_param():
    row = _base_row(customer_id="CUST999", txn_count_1h=5)
    decision, triggered, allowlisted = evaluate_rules(
        row, allowlist_customer_ids=frozenset({"CUST999"})
    )
    assert allowlisted is True
    assert "trusted_allowlist" in triggered
    # the escalation rule still reports its own finding independently --
    # it's combine_decision's job to apply the allowlist override, not evaluate_rules's
    assert decision == "block"


def test_combine_decision_allowlist_overrides_everything():
    decision, source = combine_decision(
        model_score=0.99,
        t_review=0.3,
        t_block=0.7,
        rule_decision="block",
        triggered_rules=["velocity_spike", "trusted_allowlist"],
        is_allowlisted=True,
    )
    assert decision == "approve"
    assert source == "rule_override"


def test_combine_decision_rule_wins_over_lower_model_score():
    decision, source = combine_decision(
        model_score=0.1,
        t_review=0.3,
        t_block=0.7,
        rule_decision="block",
        triggered_rules=["velocity_spike"],
        is_allowlisted=False,
    )
    assert decision == "block"
    assert source == "rule"


def test_combine_decision_model_wins_when_no_rules_fire():
    decision, source = combine_decision(
        model_score=0.8,
        t_review=0.3,
        t_block=0.7,
        rule_decision="approve",
        triggered_rules=[],
        is_allowlisted=False,
    )
    assert decision == "block"
    assert source == "model"


def test_combine_decision_model_exceeds_rule_level():
    decision, source = combine_decision(
        model_score=0.9,
        t_review=0.3,
        t_block=0.7,
        rule_decision="review",
        triggered_rules=["high_risk_merchant"],
        is_allowlisted=False,
    )
    assert decision == "block"
    assert source == "model"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_decision_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decision'`

- [ ] **Step 3: Write the implementation**

```python
# decision/__init__.py
```

```python
# decision/rules.py
"""Hard rules that can escalate (or, for the allowlist, override) a
decision regardless of the model's score. Evaluated purely against columns
already present in the gold feature table -- no new feature engineering."""

DECISION_ORDER: dict[str, int] = {"approve": 0, "review": 1, "block": 2}

ALLOWLIST_CUSTOMER_IDS: frozenset[str] = frozenset()
ALLOWLIST_MERCHANT_IDS: frozenset[str] = frozenset()


def evaluate_rules(
    feature_row: dict,
    allowlist_customer_ids: frozenset[str] = ALLOWLIST_CUSTOMER_IDS,
    allowlist_merchant_ids: frozenset[str] = ALLOWLIST_MERCHANT_IDS,
) -> tuple[str, list[str], bool]:
    """Evaluate the escalation rules and the allowlist against one feature row.

    Returns
    -------
    tuple[str, list[str], bool]
        (rule_decision, triggered_rule_names, is_allowlisted) -- rule_decision
        is the highest escalation level any non-allowlist rule triggered
        ("approve" if none fired); triggered_rule_names lists every rule
        (including the allowlist, if matched) that fired; is_allowlisted is
        True if the allowlist rule matched.
    """
    triggered: list[str] = []
    decision = "approve"

    if (
        feature_row["is_new_device"]
        and feature_row["is_new_country_for_customer"]
        and feature_row["ip_country_mismatch"]
    ):
        triggered.append("device_location_takeover")
        decision = "block"

    if feature_row["txn_count_1h"] >= 5:
        triggered.append("velocity_spike")
        decision = "block"

    if feature_row["merchant_fraud_rate_hist"] > 0.15:
        triggered.append("high_risk_merchant")
        if DECISION_ORDER["review"] > DECISION_ORDER[decision]:
            decision = "review"

    if feature_row["amount_vs_customer_p95"] > 3.0 and feature_row["is_new_device"]:
        triggered.append("outsized_amount_new_device")
        if DECISION_ORDER["review"] > DECISION_ORDER[decision]:
            decision = "review"

    is_allowlisted = (
        feature_row["customer_id"] in allowlist_customer_ids
        or feature_row["merchant_id"] in allowlist_merchant_ids
    )
    if is_allowlisted:
        triggered.append("trusted_allowlist")

    return decision, triggered, is_allowlisted


def combine_decision(
    model_score: float,
    t_review: float,
    t_block: float,
    rule_decision: str,
    triggered_rules: list[str],
    is_allowlisted: bool,
) -> tuple[str, str]:
    """Combine the model's score-based decision with the rules layer.

    Returns
    -------
    tuple[str, str]
        (final_decision, decision_source) -- decision_source is "model",
        "rule", or "rule_override" (the allowlist case).
    """
    if is_allowlisted:
        return "approve", "rule_override"

    if model_score >= t_block:
        model_decision = "block"
    elif model_score >= t_review:
        model_decision = "review"
    else:
        model_decision = "approve"

    non_allowlist_rules = [r for r in triggered_rules if r != "trusted_allowlist"]
    if non_allowlist_rules and DECISION_ORDER[rule_decision] >= DECISION_ORDER[model_decision]:
        return rule_decision, "rule"
    return model_decision, "model"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_decision_rules.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add decision/__init__.py decision/rules.py tests/unit/test_decision_rules.py
git commit -m "feat(decision): add hard rules and model/rule decision combination"
```

---

### Task 3: `decision/select_thresholds.py` — cost-sensitive threshold selection

**Files:**
- Create: `decision/select_thresholds.py`
- Test: `tests/unit/test_select_thresholds.py`

**Interfaces:**
- Produces: `compute_total_cost(scores, labels, amounts, t_review: float, t_block: float, review_cost: float = REVIEW_COST, false_positive_cost: float = FALSE_POSITIVE_COST) -> float` — pure function.
- Produces: `grid_search_thresholds(scores, labels, amounts, step: float = 0.01) -> dict` — `{"t_review": ..., "t_block": ..., "total_cost": ...}`, pure function.
- Produces: `select_thresholds_for_run(run_id: str, data_dir: str = "./data", mlflow_tracking_uri: str = "./mlruns", step: float = 0.01, output_path: str = "./decision/thresholds.json") -> dict` — loads the model from the given run, scores the validation split, grid-searches, writes `output_path`, returns the written dict.
- Produces: `REVIEW_COST: float = 3.50`, `FALSE_POSITIVE_COST: float = 25.0` — the illustrative cost matrix constants (see spec).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_select_thresholds.py
import numpy as np

from decision.select_thresholds import compute_total_cost, grid_search_thresholds


def test_compute_total_cost_all_approve_charges_only_false_negatives():
    scores = np.array([0.1, 0.2, 0.1, 0.2])
    labels = np.array([0, 1, 0, 1])
    amounts = np.array([50.0, 100.0, 20.0, 200.0])

    cost = compute_total_cost(scores, labels, amounts, t_review=0.9, t_block=0.95)

    # both fraud rows (amounts 100, 200) are approved -> lose their full amount
    assert cost == 300.0


def test_compute_total_cost_all_block_charges_only_false_positives():
    scores = np.array([0.1, 0.2, 0.1, 0.2])
    labels = np.array([0, 1, 0, 1])
    amounts = np.array([50.0, 100.0, 20.0, 200.0])

    cost = compute_total_cost(scores, labels, amounts, t_review=0.0, t_block=0.0)

    # both legit rows (labels 0) get blocked -> false positive cost each
    assert cost == 50.0  # 2 * 25.0


def test_compute_total_cost_review_charges_flat_fee_regardless_of_label():
    scores = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    amounts = np.array([1000.0, 1000.0])

    cost = compute_total_cost(scores, labels, amounts, t_review=0.3, t_block=0.9)

    # both rows land in the review band -> flat review cost each, not the
    # amount, even for the fraud row
    assert cost == 7.0  # 2 * 3.50


def test_grid_search_finds_the_zero_cost_perfect_split():
    # fraud rows score high (0.95), legit rows score low (0.05) -- a perfect
    # split exists (approve everything below 0.10, block everything at or
    # above 0.95), which achieves zero total cost.
    scores = np.array([0.05, 0.95, 0.05, 0.95, 0.05])
    labels = np.array([0, 1, 0, 1, 0])
    amounts = np.array([40.0, 500.0, 30.0, 600.0, 20.0])

    best = grid_search_thresholds(scores, labels, amounts, step=0.05)

    assert best["total_cost"] == 0.0
    assert best["t_review"] > 0.05
    assert best["t_block"] <= 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_select_thresholds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decision.select_thresholds'`

- [ ] **Step 3: Write the implementation**

```python
# decision/select_thresholds.py
"""Cost-sensitive threshold selection: grid-search (t_review, t_block) over
the validation split to minimize total realized cost under the cost matrix
in docs/superpowers/specs/2026-09-15-decision-engine-api-design.md."""

import json
import os

import mlflow
import mlflow.sklearn
import numpy as np

from ml.data import load_features_and_labels, prepare_model_matrix, time_based_split

REVIEW_COST = 3.50
FALSE_POSITIVE_COST = 25.0


def compute_total_cost(
    scores,
    labels,
    amounts,
    t_review: float,
    t_block: float,
    review_cost: float = REVIEW_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    """Total realized cost of a (t_review, t_block) policy on labeled data.

    A false negative (approved fraud) costs the transaction's own amount;
    any review costs a flat fee regardless of the true label; a false
    positive (blocked legitimate transaction) costs a flat friction fee;
    correctly approved or correctly blocked/reviewed-and-caught fraud costs
    nothing beyond the review fee already counted.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    amounts = np.asarray(amounts)

    is_review = (scores >= t_review) & (scores < t_block)
    is_block = scores >= t_block
    is_approve = scores < t_review

    false_negative_cost = np.where(is_approve & (labels == 1), amounts, 0.0).sum()
    review_total_cost = np.where(is_review, review_cost, 0.0).sum()
    false_positive_total_cost = np.where(
        is_block & (labels == 0), false_positive_cost, 0.0
    ).sum()

    return float(false_negative_cost + review_total_cost + false_positive_total_cost)


def grid_search_thresholds(scores, labels, amounts, step: float = 0.01) -> dict:
    """Grid-search t_review < t_block over [0, 1] and return the minimum-cost pair."""
    candidates = np.arange(0.0, 1.0 + step, step)
    best = {"t_review": 0.0, "t_block": 1.0, "total_cost": float("inf")}

    for t_review in candidates:
        for t_block in candidates:
            if t_block <= t_review:
                continue
            cost = compute_total_cost(scores, labels, amounts, t_review, t_block)
            if cost < best["total_cost"]:
                best = {
                    "t_review": round(float(t_review), 4),
                    "t_block": round(float(t_block), 4),
                    "total_cost": cost,
                }

    return best


def select_thresholds_for_run(
    run_id: str,
    data_dir: str = "./data",
    mlflow_tracking_uri: str = "./mlruns",
    step: float = 0.01,
    output_path: str = "./decision/thresholds.json",
) -> dict:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")

    features, labels = load_features_and_labels(data_dir)
    (_, _), (val_X_raw, val_y), _ = time_based_split(features, labels)
    val_X = prepare_model_matrix(val_X_raw)
    val_scores = model.predict_proba(val_X)[:, 1]

    best = grid_search_thresholds(
        val_scores, val_y.to_numpy(), val_X_raw["amount"].to_numpy(), step=step
    )

    result = {
        **best,
        "model_run_id": run_id,
        "cost_matrix": {
            "review_cost": REVIEW_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": "transaction amount",
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_select_thresholds.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add decision/select_thresholds.py tests/unit/test_select_thresholds.py
git commit -m "feat(decision): add cost-sensitive threshold grid search"
```

---

### Task 4: `serving/config.py` — typed settings

**Files:**
- Create: `serving/__init__.py` (empty)
- Create: `serving/config.py`
- Test: `tests/unit/test_serving_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings `BaseSettings`) with fields `mlflow_tracking_uri: str = "./mlruns"`, `mlflow_run_id: str = ""`, `decision_db_url: str = "sqlite:///./decisions.db"`, `thresholds_path: str = "./decision/thresholds.json"`.
- Produces: `settings: Settings` — a module-level instance, imported by `serving/app.py` and `serving/ml_loader.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_serving_config.py
from serving.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.mlflow_tracking_uri == "./mlruns"
    assert s.decision_db_url == "sqlite:///./decisions.db"
    assert s.thresholds_path == "./decision/thresholds.json"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc123")
    s = Settings(_env_file=None)
    assert s.mlflow_run_id == "abc123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_serving_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serving'`

- [ ] **Step 3: Write the implementation**

```python
# serving/__init__.py
```

```python
# serving/config.py
"""Typed settings for the serving app, loaded from env vars / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    mlflow_tracking_uri: str = "./mlruns"
    mlflow_run_id: str = ""
    decision_db_url: str = "sqlite:///./decisions.db"
    thresholds_path: str = "./decision/thresholds.json"


settings = Settings()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_serving_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add serving/__init__.py serving/config.py tests/unit/test_serving_config.py
git commit -m "feat(serving): add typed settings for the API"
```

---

### Task 5: `serving/models.py` — decision log persistence

**Files:**
- Create: `serving/models.py`
- Test: `tests/unit/test_serving_models.py`

**Interfaces:**
- Produces: `Decision` (SQLAlchemy declarative model) — columns `id, transaction_id, model_score, decision, triggered_rules (JSON), decision_source, model_run_id, shap_top_features (JSON), created_at`.
- Produces: `make_session_factory(db_url: str) -> sessionmaker` — creates the engine, creates tables if absent, returns a session factory.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_serving_models.py
from serving.models import Decision, make_session_factory


def test_decision_round_trips_through_sqlite(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    session_factory = make_session_factory(db_url)
    session = session_factory()

    record = Decision(
        transaction_id="TXN0001",
        model_score=0.42,
        decision="review",
        triggered_rules=["high_risk_merchant"],
        decision_source="rule",
        model_run_id="run123",
        shap_top_features={"amount": 0.1},
    )
    session.add(record)
    session.commit()

    fetched = session.query(Decision).filter_by(transaction_id="TXN0001").one()
    assert fetched.model_score == 0.42
    assert fetched.triggered_rules == ["high_risk_merchant"]
    assert fetched.decision_source == "rule"
    assert fetched.created_at is not None
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_serving_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serving.models'`

- [ ] **Step 3: Write the implementation**

```python
# serving/models.py
"""SQLAlchemy decision log model -- written to run unchanged on Postgres
later, so no Postgres-only column types."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, index=True, nullable=False)
    model_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    triggered_rules = Column(JSON, nullable=False, default=list)
    decision_source = Column(String, nullable=False)
    model_run_id = Column(String, nullable=False)
    shap_top_features = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


def make_session_factory(db_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_serving_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add serving/models.py tests/unit/test_serving_models.py
git commit -m "feat(serving): add SQLAlchemy decision log model"
```

---

### Task 6: `serving/schemas.py` — request/response models

**Files:**
- Create: `serving/schemas.py`
- Test: `tests/unit/test_serving_schemas.py`

**Interfaces:**
- Produces: `FeatureRow` (Pydantic `BaseModel`) — mirrors `fraudguard_core.schemas.features_schema`'s columns exactly: `transaction_id, customer_id, merchant_id, timestamp, amount, payment_method, hour_of_day, is_night, is_cross_border, amount_vs_customer_p95, txn_count_1h, txn_count_24h, is_new_device, device_age_days, customer_device_count_so_far, merchant_fraud_rate_hist, is_new_country_for_customer, ip_country_mismatch`.
- Produces: `ScoreResponse`, `ExplainResponse`, `InvestigationResponse`, `ModelMetadataResponse` (all Pydantic `BaseModel`s, see field lists in implementation below).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_serving_schemas.py
import pytest
from pydantic import ValidationError

from serving.schemas import FeatureRow


def _valid_payload():
    return {
        "transaction_id": "TXN0001",
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "timestamp": "2026-01-01T10:00:00",
        "amount": 100.0,
        "payment_method": "card",
        "hour_of_day": 10,
        "is_night": False,
        "is_cross_border": False,
        "amount_vs_customer_p95": 1.2,
        "txn_count_1h": 1,
        "txn_count_24h": 3,
        "is_new_device": False,
        "device_age_days": 30.0,
        "customer_device_count_so_far": 2,
        "merchant_fraud_rate_hist": 0.02,
        "is_new_country_for_customer": False,
        "ip_country_mismatch": False,
    }


def test_feature_row_accepts_valid_payload():
    row = FeatureRow(**_valid_payload())
    assert row.transaction_id == "TXN0001"
    assert row.amount == 100.0


def test_feature_row_rejects_missing_field():
    payload = _valid_payload()
    del payload["amount"]
    with pytest.raises(ValidationError):
        FeatureRow(**payload)


def test_feature_row_rejects_wrong_type():
    payload = _valid_payload()
    payload["amount"] = "not-a-number"
    with pytest.raises(ValidationError):
        FeatureRow(**payload)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_serving_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serving.schemas'`

- [ ] **Step 3: Write the implementation**

```python
# serving/schemas.py
"""Pydantic request/response models. FeatureRow mirrors
fraudguard_core.schemas.features_schema exactly, since /score accepts an
already-computed feature row, not a raw transaction."""

from datetime import datetime

from pydantic import BaseModel


class FeatureRow(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: str
    timestamp: datetime
    amount: float
    payment_method: str
    hour_of_day: int
    is_night: bool
    is_cross_border: bool
    amount_vs_customer_p95: float
    txn_count_1h: int
    txn_count_24h: int
    is_new_device: bool
    device_age_days: float
    customer_device_count_so_far: int
    merchant_fraud_rate_hist: float
    is_new_country_for_customer: bool
    ip_country_mismatch: bool


class ScoreResponse(BaseModel):
    transaction_id: str
    model_score: float
    decision: str
    triggered_rules: list[str]
    decision_source: str


class ExplainResponse(BaseModel):
    transaction_id: str
    contributions: dict[str, float]


class InvestigationResponse(BaseModel):
    transaction_id: str
    model_score: float
    decision: str
    triggered_rules: list[str]
    decision_source: str
    model_run_id: str
    shap_top_features: dict[str, float]
    created_at: datetime


class ModelMetadataResponse(BaseModel):
    model_run_id: str
    deployed_model_name: str
    val_pr_auc: float
    test_pr_auc: float
    calibration_method: str
    t_review: float
    t_block: float
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_serving_schemas.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add serving/schemas.py tests/unit/test_serving_schemas.py
git commit -m "feat(serving): add request/response schemas"
```

---

### Task 7: `serving/ml_loader.py` — load model, Isolation Forest, SHAP background

**Files:**
- Create: `serving/ml_loader.py`
- Test: `tests/unit/test_ml_loader.py`

**Interfaces:**
- Produces: `LoadedModel` (dataclass) — `model, isolation_forest, shap_background: pd.DataFrame, run_id: str`.
- Produces: `load_deployed_model(run_id: str, mlflow_tracking_uri: str = "./mlruns") -> LoadedModel`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_ml_loader.py
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from serving.ml_loader import load_deployed_model


def test_load_deployed_model_round_trips(tmp_path):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("test-loader")

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    iso = IsolationForest(random_state=42).fit(X)
    background = X.sample(10, random_state=42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            iso, name="isolation_forest_model", serialization_format="cloudpickle"
        )
        bg_path = tmp_path / "shap_background.csv"
        background.to_csv(bg_path, index=False)
        mlflow.log_artifact(str(bg_path))

    loaded = load_deployed_model(run_id, mlflow_tracking_uri=mlflow_dir)

    assert loaded.run_id == run_id
    assert loaded.model.predict_proba(X)[:, 1].shape == (50,)
    assert loaded.isolation_forest.score_samples(X).shape == (50,)
    assert len(loaded.shap_background) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_ml_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serving.ml_loader'`

- [ ] **Step 3: Write the implementation**

```python
# serving/ml_loader.py
"""Loads the deployed model, Isolation Forest, and SHAP background sample
from one config-named MLflow run, once, at app startup."""

import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


@dataclass
class LoadedModel:
    model: object
    isolation_forest: object
    shap_background: pd.DataFrame
    run_id: str


def load_deployed_model(run_id: str, mlflow_tracking_uri: str = "./mlruns") -> LoadedModel:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")
    isolation_forest = mlflow.sklearn.load_model(f"runs:/{run_id}/isolation_forest_model")

    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "shap_background.csv")
    shap_background = pd.read_csv(local_path)

    return LoadedModel(
        model=model,
        isolation_forest=isolation_forest,
        shap_background=shap_background,
        run_id=run_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_ml_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add serving/ml_loader.py tests/unit/test_ml_loader.py
git commit -m "feat(serving): load deployed model, isolation forest, and SHAP background from MLflow"
```

---

### Task 8: `serving/app.py` — the FastAPI app (all 6 endpoints) + dependencies/Makefile/.gitignore

**Files:**
- Create: `serving/app.py`
- Test: `tests/integration/__init__.py` (empty — new test directory), `tests/integration/test_serving_api.py`
- Modify: `requirements-dev.txt` (add `fastapi`, `uvicorn`, `sqlalchemy`, `httpx`)
- Modify: `Makefile` (add `api` target)
- Modify: `.gitignore` (add local SQLite file)

**Interfaces:**
- Consumes: every function/class from Tasks 1-7 (`decision.rules`, `decision.select_thresholds`'s `REVIEW_COST`/`FALSE_POSITIVE_COST` not needed here, `serving.config.settings`, `serving.models.Decision`/`make_session_factory`, `serving.schemas.*`, `serving.ml_loader.load_deployed_model`, `ml.data.prepare_model_matrix`, `ml.explain.explain_prediction`).
- Produces: `app: FastAPI` with `POST /score`, `POST /score/batch`, `POST /explain`, `GET /investigations/{transaction_id}`, `GET /model/metadata`, `GET /health`.

- [ ] **Step 1: Add dependencies**

Add to `requirements-dev.txt`:

```
fastapi>=0.110
uvicorn>=0.29
sqlalchemy>=2.0
httpx>=0.27
```

Add to `.gitignore` (near the existing `mlruns/` line):

```
decisions.db
```

- [ ] **Step 2: Write the failing integration test**

This test needs a real, tiny, fully-trained-and-logged model plus real thresholds and a temp SQLite DB — it exercises the whole app through FastAPI's `TestClient`, not mocks.

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_serving_api.py
import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from ml.data import prepare_model_matrix


def _synthetic_raw_rows(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Raw (pre-prepare_model_matrix) feature rows, same shape as
    fraudguard_core.schemas.features_schema. The tiny model below is
    trained on these run through the REAL prepare_model_matrix function
    (not a hand-rolled column list), so its column set/order can never
    drift out of sync with what serving/app.py's _feature_row_to_model_input
    produces at request time -- confirmed via direct verification that
    prepare_model_matrix's actual output is
    ['amount', 'hour_of_day', 'is_night', 'is_cross_border',
    'amount_vs_customer_p95', 'txn_count_1h', 'txn_count_24h',
    'is_new_device', 'device_age_days', 'customer_device_count_so_far',
    'merchant_fraud_rate_hist', 'is_new_country_for_customer',
    'ip_country_mismatch', 'payment_method_bank_transfer',
    'payment_method_card', 'payment_method_wallet'] -- but this fixture
    doesn't hardcode that list at all, so it stays correct even if
    prepare_model_matrix's column order ever changes."""
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:04d}" for i in range(n)],
            "customer_id": [f"CUST{i % 20:03d}" for i in range(n)],
            "merchant_id": [f"MERC{i % 10:03d}" for i in range(n)],
            "timestamp": timestamps,
            "amount": rng.lognormal(mean=3, sigma=1, size=n),
            "payment_method": rng.choice(["card", "wallet", "bank_transfer"], size=n),
            "hour_of_day": timestamps.hour,
            "is_night": (timestamps.hour < 5).astype(int),
            "is_cross_border": rng.integers(0, 2, size=n),
            "amount_vs_customer_p95": rng.uniform(0.5, 2.0, size=n),
            "txn_count_1h": rng.integers(0, 3, size=n),
            "txn_count_24h": rng.integers(0, 5, size=n),
            "is_new_device": rng.integers(0, 2, size=n),
            "device_age_days": rng.uniform(0, 100, size=n),
            "customer_device_count_so_far": rng.integers(0, 3, size=n),
            "merchant_fraud_rate_hist": rng.uniform(0, 0.1, size=n),
            "is_new_country_for_customer": rng.integers(0, 2, size=n),
            "ip_country_mismatch": rng.integers(0, 2, size=n),
        }
    )


def _valid_feature_row(**overrides):
    row = {
        "transaction_id": "TXN0001",
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "timestamp": "2026-01-01T10:00:00",
        "amount": 100.0,
        "payment_method": "card",
        "hour_of_day": 10,
        "is_night": False,
        "is_cross_border": False,
        "amount_vs_customer_p95": 1.2,
        "txn_count_1h": 1,
        "txn_count_24h": 3,
        "is_new_device": False,
        "device_age_days": 30.0,
        "customer_device_count_so_far": 2,
        "merchant_fraud_rate_hist": 0.02,
        "is_new_country_for_customer": False,
        "ip_country_mismatch": False,
    }
    row.update(overrides)
    return row


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("test-serving-api")

    rng = np.random.default_rng(0)
    raw_rows = _synthetic_raw_rows(200, rng)
    X = prepare_model_matrix(raw_rows)
    y = (X["amount_vs_customer_p95"] > 1.0).astype(int)

    model = LogisticRegression(max_iter=1000).fit(X, y)
    iso = IsolationForest(random_state=42).fit(X)
    background = X.sample(20, random_state=42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            iso, name="isolation_forest_model", serialization_format="cloudpickle"
        )
        bg_path = tmp_path / "shap_background.csv"
        background.to_csv(bg_path, index=False)
        mlflow.log_artifact(str(bg_path))
        mlflow.log_metric("deployed_val_pr_auc", 0.55)
        mlflow.log_metric("test_pr_auc", 0.60)
        mlflow.log_param("deployed_model", "baseline")

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({"t_review": 0.3, "t_block": 0.7, "model_run_id": run_id})
    )

    db_path = tmp_path / "decisions.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_dir)
    monkeypatch.setenv("MLFLOW_RUN_ID", run_id)
    monkeypatch.setenv("DECISION_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("THRESHOLDS_PATH", str(thresholds_path))

    # import after env vars are set, since serving.app builds module-level
    # state (loaded model, session factory) at import time
    import importlib

    import serving.app as app_module

    importlib.reload(app_module)

    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def test_health(app_client):
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_persists_and_returns_a_decision(app_client):
    response = app_client.post("/score", json=_valid_feature_row())
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TXN0001"
    assert body["decision"] in {"approve", "review", "block"}
    assert 0.0 <= body["model_score"] <= 1.0


def test_score_batch_returns_one_result_per_input(app_client):
    rows = [_valid_feature_row(transaction_id=f"TXN{i:04d}") for i in range(3)]
    response = app_client.post("/score/batch", json=rows)
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_investigation_lookup_matches_what_score_persisted(app_client):
    score_response = app_client.post("/score", json=_valid_feature_row(transaction_id="TXN9999"))
    scored = score_response.json()

    lookup_response = app_client.get("/investigations/TXN9999")
    assert lookup_response.status_code == 200
    investigated = lookup_response.json()

    assert investigated["model_score"] == scored["model_score"]
    assert investigated["decision"] == scored["decision"]
    assert investigated["triggered_rules"] == scored["triggered_rules"]


def test_investigation_lookup_404s_for_unknown_transaction(app_client):
    response = app_client.get("/investigations/DOES-NOT-EXIST")
    assert response.status_code == 404


def test_explain_returns_full_contributions(app_client):
    response = app_client.post("/explain", json=_valid_feature_row())
    assert response.status_code == 200
    contributions = response.json()["contributions"]
    assert len(contributions) > 0


def test_model_metadata(app_client):
    response = app_client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["t_review"] == 0.3
    assert body["t_block"] == 0.7
    assert body["deployed_model_name"] == "baseline"


def test_rules_force_block_regardless_of_model_score(app_client):
    row = _valid_feature_row(
        is_new_device=True, is_new_country_for_customer=True, ip_country_mismatch=True
    )
    response = app_client.post("/score", json=row)
    body = response.json()
    assert body["decision"] == "block"
    assert "device_location_takeover" in body["triggered_rules"]
    assert body["decision_source"] == "rule"


def test_scoring_is_deterministic_for_the_same_input(app_client):
    row = _valid_feature_row(transaction_id="TXN-DETERMINISTIC")
    first = app_client.post("/score", json=row).json()
    second = app_client.post("/score", json={**row, "transaction_id": "TXN-DETERMINISTIC-2"}).json()
    assert first["model_score"] == second["model_score"]
    assert first["decision"] == second["decision"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/integration/test_serving_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serving.app'`

- [ ] **Step 4: Write the implementation**

```python
# serving/app.py
"""FastAPI app: score, batch score, explain, investigation lookup, model
metadata, health."""

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from decision.rules import combine_decision, evaluate_rules
from ml.data import prepare_model_matrix
from ml.explain import explain_prediction
from serving.config import settings
from serving.ml_loader import load_deployed_model
from serving.models import Decision, make_session_factory
from serving.schemas import (
    ExplainResponse,
    FeatureRow,
    InvestigationResponse,
    ModelMetadataResponse,
    ScoreResponse,
)

app = FastAPI(title="FraudGuard Decision Engine API")

_loaded = load_deployed_model(settings.mlflow_run_id, settings.mlflow_tracking_uri)
with open(settings.thresholds_path) as _f:
    _thresholds = json.load(_f)

SessionLocal = make_session_factory(settings.decision_db_url)


def _feature_row_to_model_input(row: FeatureRow) -> pd.DataFrame:
    raw = pd.DataFrame([row.model_dump()])
    return prepare_model_matrix(raw)


def _score_one(row: FeatureRow) -> ScoreResponse:
    model_input = _feature_row_to_model_input(row)
    model_score = float(_loaded.model.predict_proba(model_input)[:, 1][0])

    rule_decision, triggered_rules, is_allowlisted = evaluate_rules(row.model_dump())
    final_decision, decision_source = combine_decision(
        model_score,
        _thresholds["t_review"],
        _thresholds["t_block"],
        rule_decision,
        triggered_rules,
        is_allowlisted,
    )

    contributions = explain_prediction(
        _loaded.model, model_input, background=_loaded.shap_background
    )
    top5 = dict(
        sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    )

    session = SessionLocal()
    try:
        record = Decision(
            transaction_id=row.transaction_id,
            model_score=model_score,
            decision=final_decision,
            triggered_rules=triggered_rules,
            decision_source=decision_source,
            model_run_id=_loaded.run_id,
            shap_top_features=top5,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

    return ScoreResponse(
        transaction_id=row.transaction_id,
        model_score=model_score,
        decision=final_decision,
        triggered_rules=triggered_rules,
        decision_source=decision_source,
    )


@app.post("/score", response_model=ScoreResponse)
def score(row: FeatureRow) -> ScoreResponse:
    return _score_one(row)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(rows: list[FeatureRow]) -> list[ScoreResponse]:
    return [_score_one(row) for row in rows]


@app.post("/explain", response_model=ExplainResponse)
def explain(row: FeatureRow) -> ExplainResponse:
    model_input = _feature_row_to_model_input(row)
    contributions = explain_prediction(
        _loaded.model, model_input, background=_loaded.shap_background
    )
    return ExplainResponse(transaction_id=row.transaction_id, contributions=contributions)


@app.get("/investigations/{transaction_id}", response_model=InvestigationResponse)
def get_investigation(transaction_id: str) -> InvestigationResponse:
    session = SessionLocal()
    try:
        record = (
            session.query(Decision)
            .filter_by(transaction_id=transaction_id)
            .order_by(Decision.created_at.desc())
            .first()
        )
        if record is None:
            raise HTTPException(status_code=404, detail="transaction not found")
        return InvestigationResponse(
            transaction_id=record.transaction_id,
            model_score=record.model_score,
            decision=record.decision,
            triggered_rules=record.triggered_rules,
            decision_source=record.decision_source,
            model_run_id=record.model_run_id,
            shap_top_features=record.shap_top_features,
            created_at=record.created_at,
        )
    finally:
        session.close()


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    client = MlflowClient()
    run = client.get_run(_loaded.run_id)
    return ModelMetadataResponse(
        model_run_id=_loaded.run_id,
        deployed_model_name=run.data.params.get("deployed_model", "unknown"),
        val_pr_auc=float(run.data.metrics.get("deployed_val_pr_auc", 0.0)),
        test_pr_auc=float(run.data.metrics.get("test_pr_auc", 0.0)),
        calibration_method="isotonic",
        t_review=_thresholds["t_review"],
        t_block=_thresholds["t_block"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Add the `api` Makefile target**

Modify `Makefile`:

```makefile
.PHONY: venv seed dq test lint features train api

api:
	uvicorn serving.app:app --reload
```

(Add `api` to the existing `.PHONY` line, add the target after the existing `train:` target.)

- [ ] **Step 6: Run test to verify it passes**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/integration/test_serving_api.py -v`
Expected: PASS (10 tests).

- [ ] **Step 7: Commit**

```bash
git add serving/app.py tests/integration/__init__.py tests/integration/test_serving_api.py requirements-dev.txt Makefile .gitignore
git commit -m "feat(serving): add FastAPI app with score, explain, investigation, and metadata endpoints"
```

---

### Task 9: Fix `/score`/`/explain` 500 errors — persist and load the raw (pre-calibration) model for SHAP

**Context:** Running Task 9's original real-run smoke test (before this fix was inserted) surfaced a genuine bug: `POST /score` and `POST /explain` both 500 against the real deployed model with `shap.InvalidModelError`. Root cause, confirmed by direct investigation: the deployed model MLflow logs is `CalibratedClassifierCV(FrozenEstimator(...))` — a calibration wrapper, not the raw estimator. `ml/explain.py`'s `_model_shap_values` detects linear-vs-tree via `hasattr(model, "coef_")`, which is `False` on the wrapper (it doesn't proxy the attribute), so it falls through to `TreeExplainer`, which then fails because the wrapper isn't a tree model either. `ml/__main__.py` (sub-project 3, already merged) computes SHAP using the RAW `deployed_model` (before calibration) internally, but only ever persists the CALIBRATED wrapper to MLflow (`calibrated_deployed_model`) — the raw model was never saved as its own artifact, so sub-project 4 has no way to load it. This slipped through both sub-project 3's and sub-project 8's test suites because every SHAP-related test fixture used a bare, never-calibrated model — not the shape production actually deploys.

**Files:**
- Modify: `ml/__main__.py` (sub-project 3's orchestration — add one more `mlflow.sklearn.log_model` call)
- Modify: `serving/ml_loader.py` (load the new artifact)
- Modify: `serving/app.py` (use the raw model for SHAP, the calibrated model for scoring — unchanged)
- Modify: `tests/integration/test_serving_api.py` (fix the fixture to actually calibrate its model, closing the test gap that let this ship)
- Test: extend `tests/unit/test_ml_loader.py`

**Interfaces:**
- Modifies: `serving.ml_loader.LoadedModel` gains a new field `raw_model: object` (the pre-calibration estimator, used only for SHAP).
- Modifies: `serving.ml_loader.load_deployed_model` additionally loads `runs:/{run_id}/raw_deployed_model`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_ml_loader.py -- extend the existing test, don't duplicate the fixture setup
```

Modify the existing `test_load_deployed_model_round_trips` test: after logging `calibrated_deployed_model` and `isolation_forest_model` as it already does, ALSO log the raw (pre-calibration) `model` under a new artifact name:

```python
        mlflow.sklearn.log_model(
            model, name="raw_deployed_model", serialization_format="cloudpickle"
        )
```

(Add this call inside the same `with mlflow.start_run() as run:` block, alongside the two existing `log_model` calls — before the `mlflow.log_artifact(str(bg_path))` line.)

Add a new assertion after the existing ones:

```python
    assert loaded.raw_model.predict_proba(X)[:, 1].shape == (50,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_ml_loader.py -v`
Expected: FAIL with `AttributeError: 'LoadedModel' object has no attribute 'raw_model'`

- [ ] **Step 3: Fix `serving/ml_loader.py`**

```python
# serving/ml_loader.py -- full replacement
"""Loads the deployed model, Isolation Forest, and SHAP background sample
from one config-named MLflow run, once, at app startup.

Loads two versions of the deployed model: the calibrated wrapper
(CalibratedClassifierCV) for actual scoring, and the raw pre-calibration
estimator for SHAP -- shap.TreeExplainer/LinearExplainer both need direct
access to a real linear or tree model's internals, which a calibration
wrapper does not expose (confirmed via shap.InvalidModelError when tried
directly against the wrapper)."""

import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


@dataclass
class LoadedModel:
    model: object
    raw_model: object
    isolation_forest: object
    shap_background: pd.DataFrame
    run_id: str


def load_deployed_model(run_id: str, mlflow_tracking_uri: str = "./mlruns") -> LoadedModel:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/calibrated_deployed_model")
    raw_model = mlflow.sklearn.load_model(f"runs:/{run_id}/raw_deployed_model")
    isolation_forest = mlflow.sklearn.load_model(f"runs:/{run_id}/isolation_forest_model")

    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "shap_background.csv")
    shap_background = pd.read_csv(local_path)

    return LoadedModel(
        model=model,
        raw_model=raw_model,
        isolation_forest=isolation_forest,
        shap_background=shap_background,
        run_id=run_id,
    )
```

- [ ] **Step 4: Fix `ml/__main__.py` — persist the raw model too**

In `ml/__main__.py`, find the `with mlflow.start_run(run_name="deployed_model_final"):` block. Immediately after the existing `mlflow.sklearn.log_model(calibrated_deployed, name="calibrated_deployed_model", serialization_format="cloudpickle")` call, add:

```python
        mlflow.sklearn.log_model(
            deployed_model, name="raw_deployed_model", serialization_format="cloudpickle"
        )
```

(`deployed_model` is already in scope at that point in the function — it's the variable `select_deployed_model` returned, used earlier for the SHAP computation in the same function.)

- [ ] **Step 5: Fix `serving/app.py` — use the raw model for SHAP**

In `serving/app.py`, change both `explain_prediction` call sites from `_loaded.model` to `_loaded.raw_model`:

```python
    # inside _score_one:
    contributions = explain_prediction(
        _loaded.raw_model, model_input, background=_loaded.shap_background
    )
```

```python
    # inside the /explain endpoint:
    contributions = explain_prediction(
        _loaded.raw_model, model_input, background=_loaded.shap_background
    )
```

(Scoring itself — `_loaded.model.predict_proba(...)` — stays unchanged; only the two `explain_prediction` calls switch to `_loaded.raw_model`.)

- [ ] **Step 6: Fix the integration test fixture to actually calibrate its model**

In `tests/integration/test_serving_api.py`'s `app_client` fixture, the model logged as `calibrated_deployed_model` must actually BE a `CalibratedClassifierCV`, matching real production shape — this is what let the bug ship undetected. Replace:

```python
    model = LogisticRegression(max_iter=1000).fit(X, y)
```

with:

```python
    from ml.calibration import calibrate

    raw_model = LogisticRegression(max_iter=1000).fit(X, y)
    model = calibrate(raw_model, X, y, method="isotonic")
```

(`calibrate` is already imported/available via `ml.calibration` — add the import at the top of the file with the other imports rather than inline, matching this file's existing style.) Then update the `mlflow.start_run()` block to log BOTH:

```python
        mlflow.sklearn.log_model(
            model, name="calibrated_deployed_model", serialization_format="cloudpickle"
        )
        mlflow.sklearn.log_model(
            raw_model, name="raw_deployed_model", serialization_format="cloudpickle"
        )
```

(replacing the single existing `calibrated_deployed_model` log call — keep the `isolation_forest_model` log call as-is).

- [ ] **Step 7: Run all affected tests to verify the fix**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m pytest tests/unit/test_ml_loader.py tests/integration/test_serving_api.py -v`
Expected: PASS, all green. The `/explain`/`/score` integration tests now exercise a genuinely-calibrated model, closing the gap that let this bug ship.

- [ ] **Step 8: Run ruff/black**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m ruff check ml/__main__.py serving/ml_loader.py serving/app.py tests/unit/test_ml_loader.py tests/integration/test_serving_api.py` and `black --check` on the same files. Fix with `ruff check --fix` / `black` if either fails.

- [ ] **Step 9: Commit**

```bash
git add ml/__main__.py serving/ml_loader.py serving/app.py tests/unit/test_ml_loader.py tests/integration/test_serving_api.py
git commit -m "fix(serving): persist and load the raw pre-calibration model for SHAP, fixing /score and /explain 500s"
```

---

### Task 10: End-to-end run + acceptance doc

**Files:**
- Create: `docs/decision-engine-acceptance.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: a FRESH real training run from `python -m ml train` (sub-project 3's orchestration, now fixed by Task 9 to also persist `raw_deployed_model`) — the run used for Task 9's original acceptance attempt (`ed2915a3238542bbbfe970c924c85f79`) predates that fix and does NOT have the `raw_deployed_model` artifact, so it cannot be reused here. A new run is required.

- [ ] **Step 1: Run the real training pipeline again to get a run with the Task 9 fix applied**

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -m ml train` (uses the real `./data/features.parquet`, 519,876 rows — takes several minutes; this worktree already has this file, confirm with `ls data/features.parquet` first and regenerate via `python -m generator seed && python -m features build` only if missing).

Then find the new run's ID:

Run: `"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -c "from mlflow.tracking import MlflowClient; import mlflow; mlflow.set_tracking_uri('./mlruns'); c = MlflowClient(); exp = c.get_experiment_by_name('fraudguard-modeling'); runs = c.search_runs([exp.experiment_id], filter_string=\"tags.mlflow.runName = 'deployed_model_final'\", order_by=['start_time DESC'], max_results=1); print(runs[0].info.run_id)"`

Record the printed run ID — call it `RUN_ID` below. Confirm it differs from `ed2915a3238542bbbfe970c924c85f79` (the pre-fix run) and that `MlflowClient().list_artifacts(RUN_ID)` includes `raw_deployed_model` this time.

- [ ] **Step 2: Enrich the run and select thresholds for real**

```bash
"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -c "from ml.enrich_deployed_run import enrich_deployed_run; print(enrich_deployed_run('RUN_ID'))"
"C:\Users\trilo\Downloads\FraudGuard\.venv\Scripts\python.exe" -c "from decision.select_thresholds import select_thresholds_for_run; print(select_thresholds_for_run('RUN_ID'))"
```

(Replace `RUN_ID` with the value from Step 1.) This uses the real `./data/features.parquet` (must be present — regenerate via `python -m generator seed && python -m features build` if this worktree doesn't have it). Confirm `decision/thresholds.json` was written with real, non-placeholder numbers.

- [ ] **Step 3: Start the API and smoke-test it manually**

Set `.env` (or export env vars) with `MLFLOW_RUN_ID=RUN_ID`, then run `make api` (or `uvicorn serving.app:app`), and hit `/health`, `/model/metadata`, and one `/score` call with a real feature row (e.g. copy one row's values from `data/features.parquet`) using `curl` or the interactive docs at `/docs`. Confirm all three respond sensibly with real numbers, not errors.

- [ ] **Step 4: Write `docs/decision-engine-acceptance.md`**

Using the real values from Steps 1-3 (no placeholders in the committed file):

```markdown
# Sub-project 4 — Acceptance Run

**Date:** [fill in actual run date]
**Deployed model run:** [RUN_ID from Step 1]

## Threshold selection

Cost matrix: false negative = transaction amount, review = $3.50 flat, false
positive (block) = $25 flat (see
`docs/superpowers/specs/2026-09-15-decision-engine-api-design.md` for the
full rationale).

- t_review: [value from decision/thresholds.json]
- t_block: [value from decision/thresholds.json]
- Total cost on validation at this policy: [total_cost value]

## Test suite

Run: `pytest tests/unit tests/integration -v -k "decision or serving or ml_"`
Result: [fill in the pytest summary line]

## API smoke test

- `GET /health` → `{"status": "ok"}`
- `GET /model/metadata` → [paste the real JSON response]
- `POST /score` with a real feature row → [paste the real JSON response]

## What's next

Sub-project 5 (Streaming, local-only) computes windowed velocity features
live and feeds them through this same `/score` contract. Sub-project 6
(Frontend) is the first real consumer of `/investigations/{transaction_id}`
and `/explain` for the Investigations module.
```

- [ ] **Step 5: Update `README.md` status**

Modify the `## Status` section of `README.md`, adding after the sub-project 3 entry:

```markdown
**Sub-project 4 (Decision Engine + API) — done.** See
`docs/superpowers/specs/2026-09-15-decision-engine-api-design.md` and
`docs/decision-engine-acceptance.md`.
```

Also update the `## Quickstart` code block, adding after the existing `make train` line:

```
make api                          # serves the decision engine (score/explain/investigate)
```

- [ ] **Step 6: Commit**

```bash
git add decision/thresholds.json docs/decision-engine-acceptance.md README.md
git commit -m "docs: record sub-project 4 acceptance run"
```
