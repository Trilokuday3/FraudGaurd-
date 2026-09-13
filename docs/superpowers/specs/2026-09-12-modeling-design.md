# Sub-project 3 — Modeling — Design

**Date:** 2026-09-12
**Status:** Approved
**Parent:** `2026-09-12-fraudguard-platform-roadmap.md`
**Depends on:** sub-project 2 (EDA + Feature Engineering) — trains on `data/features.parquet`
**Blocks:** sub-project 4 (Decision Engine + API) — serves this sub-project's calibrated champion model and SHAP explainer

## Amendment (2026-09-13)

Running the real pipeline against the full dataset showed the plain
Logistic Regression baseline beating every tree-based candidate (Random
Forest, XGBoost, LightGBM) on both validation and test PR-AUC, consistently
across all three tree algorithms — investigated and confirmed not a bug.
Per an explicit decision, this project deploys whichever trained candidate
is genuinely best by validation PR-AUC (`ml.train.select_deployed_model`),
including the baseline — not restricted to a boosted-tree-only comparison.
Everywhere below that says "champion" describes the *original* design
(still present in the code as `ml.train.select_champion`, an informational
comparison between XGBoost and LightGBM only); the model actually
calibrated, evaluated on test, explained, and intended for sub-project 4 to
serve is whatever `select_deployed_model` returns — see
`ml/model_card.md` and `docs/ml-acceptance.md` for this project's actual
result. SHAP explainability uses `TreeExplainer` for tree-based candidates
and `shap.LinearExplainer` for a linear one (`ml/explain.py`), not only
`TreeExplainer` as originally written below.

## Purpose

Turn the leakage-safe gold feature table into a calibrated, explainable fraud classifier that actually earns its precision, evaluated the way a real deployment would be evaluated (time-aware, PR-AUC first), plus a complementary anomaly signal and the documentation artifacts a decision engine (and a recruiter) can trust.

**In scope:** `ml/` package — data loading/splitting, baseline/RF/champion training, calibration, Isolation Forest, eval harness, SHAP, model card, MLflow tracking.
**Out of scope:** the decision engine and its cost-sensitive thresholds (sub-project 4 — this sub-project produces calibrated probabilities and explanations, not approve/review/block decisions), the API (sub-project 4), anything requiring Spark/Kafka (sub-project 5).

## Success criteria

1. `make train` on a clean checkout with `data/features.parquet` present produces a time-aware held-out (test-set) evaluation report, checked into `docs/ml-acceptance.md`, with no manual steps.
2. The champion model beats the logistic-regression baseline on **test-set PR-AUC** (the roadmap's non-negotiable: PR-AUC is the headline metric throughout, never accuracy).
3. A calibration curve and both global and local SHAP artifacts are generated and saved to `ml/artifacts/`.
4. The train/val/test split is strictly time-ordered — no transaction in val or test has a `timestamp` earlier than any transaction in train, and no transaction in test is earlier than any in val. Asserted by a test.
5. Every training run (baseline, RF, champion candidates, final champion) is logged to a local MLflow tracking store (params, metrics, model artifact) — reproducible from the recorded run.
6. `ml/model_card.md` documents the champion's architecture, training window, metrics, calibration approach, and known limitations.

## Why time-based split, not random split

Sub-project 2's entire design (`docs/superpowers/specs/2026-09-12-feature-engineering-design.md`) exists to make every feature computable "as of" the transaction being scored. A random train/test split would let the model implicitly see future information through correlated rows (e.g., two transactions from the same customer 10 minutes apart, one in train and one in test, sharing near-identical velocity/behavioral features) — silently undermining the leakage discipline sub-project 2 worked to establish, and inflating test metrics in a way that wouldn't survive contact with a real deployment. A strict time split is the only split that tests what actually matters: can this model generalize to transactions that happen *after* everything it was trained on.

## Split

Using the full `timestamp` range in `data/features.parquet` (2025-03-01 to 2026-09-01, ~18 months):

- **Train:** first ~70% of the date range (chronologically earliest transactions)
- **Validation:** next ~15% (model selection, threshold/calibration tuning)
- **Test:** final ~15% (touched exactly once, for the final reported metrics)

Split is computed on **date boundaries**, not row-count quantiles (a pure row-count split could be skewed by the generator's diurnal/volume patterns) — i.e., pick the two cutoff dates that are 70% and 85% of the way through the date range, then partition rows by `timestamp` against those cutoffs.

## Class imbalance handling

`class_weight='balanced'` (scikit-learn: logistic regression, random forest) / `scale_pos_weight = n_negative / n_positive` (XGBoost/LightGBM), computed from the **training fold only**. No resampling (SMOTE, undersampling): reweighting the loss trains on the real distribution without synthesizing fraud patterns that don't exist in the data, and avoids the classic SMOTE-before-split leakage bug entirely by construction (there's no resampling step to accidentally fit before splitting).

## Models & training pipeline

| Stage | Model | Config | Purpose |
|---|---|---|---|
| Baseline | Logistic Regression | `class_weight='balanced'`, default regularization | Floor every later model must beat on PR-AUC |
| Candidate | Random Forest | `class_weight='balanced'`, modest `n_estimators` (e.g. 200) | Middle checkpoint, nonlinear baseline |
| Champion | XGBoost **or** LightGBM | `scale_pos_weight` set from train-fold ratio; whichever scores higher val-set PR-AUC wins | Final model, the one calibration/SHAP/model card describe |
| Complementary | Isolation Forest | Trained unsupervised on the full feature set (no labels) | Anomaly score, computed and stored alongside predictions but **never used as a champion input feature** (per approved design decision — keeps SHAP explanations fully attributable to real, non-model-derived features, and demonstrates supervised + unsupervised detection as two distinct signals) |

All four models are trained inside `ml/train.py`, each run (params + metrics + serialized model) logged to MLflow via a local file-based tracking URI (`./mlruns`, already gitignored per sub-project 1's `.gitignore`).

## Calibration

The champion's raw scores are recalibrated (isotonic regression, or Platt/sigmoid if isotonic overfits on the validation set — chosen by comparing calibration curves on val, not test) so that a "0.3" from the model means roughly "30% of transactions scored this vary end up fraudulent." This calibration step is essential for sub-project 4's cost-sensitive thresholds to be meaningful (an approve/review/block cutoff only makes sense against a calibrated probability). Calibration is fit on validation data only, evaluated on test.

## Evaluation harness

Computed on **validation** (during model selection) and **test** (once, final):

- **PR-AUC** — headline metric, reported first everywhere
- ROC-AUC — secondary, reported for completeness (brief's own §11 warns against leaning on it given the imbalance)
- Precision / recall / F1 at a small fixed set of threshold points (e.g. 0.3, 0.5, 0.7) — illustrative, not the final operating threshold (that's sub-project 4's cost-sensitive analysis)
- Calibration curve (predicted probability vs. observed fraud rate, binned) — saved as a plot artifact

All metrics and artifacts are saved to `ml/artifacts/` (gitignored, like `data/` and `mlruns/`) and the final test-set numbers are transcribed into `docs/ml-acceptance.md`, mirroring sub-projects 1 and 2's acceptance-doc pattern.

## Explainability

SHAP `TreeExplainer` against the champion (XGBoost/LightGBM both support it natively and efficiently, unlike `KernelExplainer`):

- **Global:** mean absolute SHAP value per feature, computed on a representative sample of the test set (a few thousand rows — full-scale SHAP on 500k+ rows is unnecessary for a global summary and meaningfully slower), saved as a bar-chart artifact.
- **Local:** a function, `explain_prediction(model, explainer, feature_row) -> dict`, producing per-feature SHAP contributions for one transaction — this is the function sub-project 4's `/explain` API endpoint will call directly.

## Model card

`ml/model_card.md`: architecture (which of XGBoost/LightGBM won and why), training data window and volume, class balance handling, calibration method, full val/test metrics table, top global SHAP features with brief interpretation, and explicitly stated limitations (e.g., synthetic data, ~18-month window, no adversarial/concept-drift testing yet — that's sub-project 7's job).

## Component boundaries

| unit | does | consumed via | depends on |
|---|---|---|---|
| `ml/data.py` | loads `data/features.parquet` + `ground_truth`, joins on `transaction_id`, computes the time-based split | import | `fraudguard_core` (schema import for validation) |
| `ml/train.py` | trains baseline → RF → champion → Isolation Forest, logs to MLflow | CLI `python -m ml train` | `ml/data.py` |
| `ml/calibration.py` | fits/applies probability calibration | import | — |
| `ml/evaluate.py` | PR-AUC/ROC-AUC/precision/recall/F1/calibration-curve computation | import, `python -m ml evaluate` | — |
| `ml/explain.py` | SHAP global summary + `explain_prediction()` | import, consumed directly by sub-project 4 | champion model artifact |
| `ml/model_card.md` | human-readable model documentation | read by humans / linked from frontend | — |

## Deliverables checklist

- [ ] `ml/data.py`, `ml/train.py`, `ml/calibration.py`, `ml/evaluate.py`, `ml/explain.py`, `ml/__main__.py`
- [ ] `requirements-dev.txt` additions: `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `mlflow`
- [ ] `Makefile` — `train` target
- [ ] `tests/unit/test_ml_split.py` (time-order invariant), `tests/unit/test_ml_evaluate.py` (metric functions on known fixtures)
- [ ] `ml/model_card.md`
- [ ] `docs/ml-acceptance.md` recorded after a real training run
- [ ] `.gitignore` additions: `ml/artifacts/`, `mlruns/` (mlruns already present; artifacts/ new)

## Risks / decisions

- **Isolation Forest stays separate, not a champion feature** — approved design decision; keeps SHAP explanations attributable to real features only and gives the portfolio a "two complementary detection paradigms" story rather than one opaque blended score.
- **No resampling (SMOTE)** — `class_weight`/`scale_pos_weight` avoids a well-known class of leakage bugs (resampling fit before the split) and doesn't synthesize unrealistic fraud rows; revisit only if PR-AUC is disappointingly low after the champion is trained.
- **Date-boundary split, not row-count split** — protects against the generator's non-uniform transaction volume over time skewing what "70%" means.
- **Local file-based MLflow, no server** — matches the roadmap's stated approach ("not deployed publicly, used for the repo's reproducibility story"); revisit only if sub-project 8's deployment needs a shared registry.
