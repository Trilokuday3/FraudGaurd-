# Sub-project 3 — Acceptance Run

**Date:** 2026-09-13
**Command:** `python -m generator seed && python -m features build && python -m ml train`

## Result

- Deployed model: baseline (overall best of baseline/RF/XGBoost/LightGBM by validation PR-AUC — `ml.train.select_deployed_model`)
- Boosted-tree champion (XGBoost vs LightGBM only, for reference): lightgbm
- Baseline test PR-AUC (calibrated the same way as the deployed model): 0.7101
- Deployed model test PR-AUC: 0.7101
- **Deployed model beats baseline on PR-AUC: TIE (identical), as expected —
  the deployed model *is* the baseline model in this run**, and both numbers
  come from the exact same calibrated model object
  (`ml/__main__.py` reuses `calibrated_deployed` for the baseline comparison
  whenever `deployed_model_name == "baseline"`, rather than recomputing a
  separately-calibrated baseline). An earlier version of this run compared
  the calibrated deployed model against the baseline's *raw, uncalibrated*
  scores, which produced a spurious ~1.9% PR-AUC gap purely from isotonic
  calibration's tie-collapsing effect on a rank-based metric — that
  inconsistency was found and fixed (`ml/__main__.py`, commit `a6ec725`)
  before this acceptance run.
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
Result: `21 passed, 19 deselected, 134 warnings in 9.67s`

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
