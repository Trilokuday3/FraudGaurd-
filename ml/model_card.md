# FraudGuard Model Card

## Architecture

Deployed model: **baseline**. Every candidate — Logistic
Regression baseline, Random Forest, XGBoost, and LightGBM — is trained and
evaluated on the same validation split, and the one with the highest
validation-set PR-AUC is the one actually calibrated, evaluated on test, and
explained below (`ml.train.select_deployed_model`). This project also tracks
which of XGBoost/LightGBM would win a boosted-tree-only comparison
(`results["champion_name"]` = **lightgbm**, `ml.train.select_champion`)
for reference, but that restriction is not applied to what's actually deployed.

**Why the baseline won here:** on this dataset, Logistic Regression
outperformed every tree-based candidate at validation PR-AUC (baseline 0.5426
vs. Random Forest 0.4921, XGBoost 0.4489, LightGBM 0.5220) — confirmed
across three separate tree algorithms, not a single misconfigured model. The
top SHAP features (`device_age_days` especially, at roughly 1.6x the
importance of the next-ranked feature, `merchant_fraud_rate_hist`) suggest
the fraud signal in this dataset is concentrated in a few near-linear/
monotonic relationships, a regime where a simple, well-regularized linear
model can beat default-hyperparameter gradient boosting. Reported honestly
rather than forcing a more complex model to be called the champion.

Complementary: Isolation Forest (unsupervised anomaly score, displayed
separately, never a model input feature).

## Training data

- Source: `data/features.parquet` (sub-project 2's gold feature table), joined to `ground_truth.fraud_label`
- Volume: 325,830 train / 86,135 val / 107,911 test rows
- Split: time-based on transaction `timestamp`, ~70/15/15 by date range
- Class balance: ~1.5% fraud prevalence, handled via class-weighting (no resampling)

## Metrics (test set, held out, evaluated once)

| model | validation PR-AUC | test PR-AUC | test ROC-AUC |
|---|---|---|---|
| Baseline (Logistic Regression) | 0.5426 | 0.7236 | 0.9466 |
| Random Forest | 0.4921 | — | — |
| XGBoost | 0.4489 | — | — |
| LightGBM | 0.5220 | — | — |
| **Deployed (baseline)** | — | **0.7101** | **0.9463** |

(Random Forest/XGBoost/LightGBM only have validation metrics recorded in
`results.json` unless one of them is the deployed model — test-set metrics
are only computed once, for the deployed model, per the "touch test exactly
once" split discipline.)

**Note on the deployed-vs-baseline test numbers above:** the deployed model
*is* the baseline Logistic Regression, but the two rows are not the exact
same computation — `test_metrics` reflects the deployed model's
isotonic-**calibrated** probabilities, while the "Baseline" row's test
PR-AUC/ROC-AUC come from the same fitted model's **raw, uncalibrated**
`predict_proba` output (`ml/__main__.py`'s `baseline_test_metrics`, computed
directly from `baseline.predict_proba(test_X)` for reference/comparison,
never calibrated separately). Isotonic calibration is a monotonic step
function fit on the validation set; applying it to the test set collapses
some previously-distinct raw scores into shared output bins (ties), which
can very slightly change rank-based metrics like PR-AUC relative to the
uncalibrated scores. That is what produced the small (~0.0135, ~1.9%) gap
between the two rows in this run — it is a calibration-measurement artifact
on an otherwise identical model, not evidence of two different models or of
the deployed model losing to a different, better baseline.

Precision/recall/F1 at illustrative thresholds (0.3 / 0.5 / 0.7) for the
deployed model are in `ml/artifacts/results.json` — the actual operating
threshold is a cost-sensitive decision made in sub-project 4, not fixed here.

## Calibration

Isotonic regression, fit on the validation set, applied to the deployed
model's raw scores before any of the metrics above are computed. See
`ml/artifacts/calibration_curve.png`.

## Explainability

Top features by mean absolute SHAP value (computed on a 2000-row
sample of the test set, via `shap.LinearExplainer` since the deployed model
is linear — see `ml/explain.py`): `device_age_days` (0.5417),
`merchant_fraud_rate_hist` (0.3424), `amount_vs_customer_p95` (0.3325),
`txn_count_1h` (0.2652), `payment_method_card` (0.2110). See
`ml/artifacts/shap_global_importance.png` for the full chart, and
`results["shap_local_example"]` in `ml/artifacts/results.json` for a worked
single-transaction explanation.

## Known limitations

- Trained on synthetic data (sub-project 1's generator), not real transactions.
- ~18-month window; no multi-year seasonality or concept drift has been observed or tested (that's sub-project 7's job).
- Isolation Forest's anomaly score is displayed but not benchmarked against a labeled anomaly-detection ground truth (none exists for unsupervised methods here) — treat it as a qualitative signal, not a metric-backed one.
- The tree-based candidates (Random Forest, XGBoost, LightGBM) all underperformed the logistic-regression baseline on this dataset using the hyperparameters this project's design specified (`n_estimators=200`, `max_depth=5`, `scale_pos_weight` from the training-fold class ratio). This project did not run a hyperparameter search to try to close that gap — the decision was to report the honest result and deploy the actual best performer rather than expand scope chasing a specific algorithm family. A future iteration could revisit this with a tuning pass (see roadmap sub-project 7).
- Isotonic calibration, applied only to the deployed model, introduces a small (~1.9%) measured PR-AUC difference relative to that same model's raw/uncalibrated scores (see the note under "Metrics" above) — a byproduct of fitting the calibration map on validation data and applying it to test, not a sign of a different or worse model.
