# Sub-project 3 — Acceptance Run

**Date:** 2026-09-13
**Command:** `python -m generator seed && python -m features build && python -m ml train`

## Result

- Deployed model: baseline (overall best of baseline/RF/XGBoost/LightGBM by validation PR-AUC — `ml.train.select_deployed_model`)
- Boosted-tree champion (XGBoost vs LightGBM only, for reference): lightgbm
- Baseline test PR-AUC: 0.7236 (raw/uncalibrated `baseline.predict_proba` on test)
- Deployed model test PR-AUC: 0.7101 (isotonic-**calibrated** deployed-model scores on test)
- **Deployed model beats baseline on PR-AUC: NO, by -0.0135 (-1.9%)** — the
  deployed model *is* the baseline model in this run, so this is not two
  different models competing; it is the same fitted Logistic Regression
  evaluated two ways: `test_metrics` after isotonic calibration (fit on
  validation, applied to test) vs. `baseline_test_metrics` on raw
  uncalibrated scores. Isotonic calibration is a monotonic step function;
  applying a validation-fit step function to unseen test data collapses some
  previously-distinct scores into shared bins (ties), which can slightly
  change a rank-based metric like PR-AUC. That is the source of this small
  gap, not a case of a tree model or a different baseline outperforming the
  deployed model — see `ml/model_card.md`'s "Metrics" section note for detail.
  ROC-AUC shows the same small direction (0.9463 deployed vs 0.9466 raw
  baseline). This was investigated, not assumed: the same underlying model
  object supplies both numbers in this run.
- Row counts: 325,830 / 86,135 / 107,911 (train/val/test)

**Note on this run's result:** the deployed model was the logistic-regression
baseline — every tree-based candidate (Random Forest, XGBoost, LightGBM)
underperformed it on validation PR-AUC (0.5426 baseline vs. 0.4921 Random
Forest, 0.4489 XGBoost, 0.5220 LightGBM), consistently across all three tree
algorithms. This was investigated as a possible bug before being accepted as
a finding: the orchestration pipeline (`ml/__main__.py`) was independently
reviewed end-to-end and confirmed correct — calibration applied before final
evaluation, SHAP explaining the right model, Isolation Forest never seeing
labels — so the result reflects the actual data, not a wiring defect. See
`ml/model_card.md`'s "Why the baseline won here" section for the working
theory (a few near-linear/monotonic SHAP-dominant features, `device_age_days`
roughly 1.6x the next-ranked feature in this run). No hyperparameter search
was run to try to close the gap; this was a deliberate scope decision to
report the honest result rather than chase a specific algorithm family. Full
context in `.superpowers/sdd/2026-09-12-modeling/progress.md` (Task 8/9
entries).

## Test suite

Run: `pytest tests/unit -v -k ml_`
Result: `21 passed, 19 deselected, 133 warnings in 7.25s`

## Artifacts

- `ml/artifacts/results.json` — full metrics, SHAP importances, per-threshold precision/recall/F1
- `ml/artifacts/calibration_curve.png` — predicted probability vs. observed fraud rate
- `ml/artifacts/shap_global_importance.png` — top-15 features by mean |SHAP value|
- `./mlruns/` — every training run (baseline, RF, XGBoost, LightGBM, deployed_model_final), gitignored
- `ml/model_card.md` — human-readable summary

## What's next

Sub-project 4 (Decision Engine + API) consumes the calibrated deployed model
and `ml/explain.py`'s `explain_prediction()` to build cost-sensitive
approve/review/block thresholds and a FastAPI serving layer.
