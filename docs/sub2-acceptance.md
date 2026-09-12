# Sub-project 2 — Acceptance Run

**Date:** 2026-09-12
**Command:** `make seed && make features && pytest tests/unit tests/dq tests/features -v`

## Output

- Feature table row count: 519,876 (must equal the transaction count from `make seed`)
- Test summary: 33 passed

## Leakage checks

`tests/features/test_leakage.py` confirms no `fraud_label`, `fraud_probability_true`,
`confirmed_fraud_at`, or `base_fraud_rate` column reaches `data/features.parquet`,
and every transaction has exactly one feature row.

## Point-in-time correctness

`tests/unit/test_features_merchant.py::test_merchant_risk_respects_confirmation_delay`
and `test_merchant_risk_falls_back_to_global_prior_below_min_history` cover the
one feature (`merchant_fraud_rate_hist`) that reads `ground_truth`, proving it
respects `confirmed_fraud_at` rather than transaction `timestamp`.

## EDA

See `docs/EDA_INSIGHTS.md` for the 6 business insights from `notebooks/01-eda.ipynb`.

## What's next

Sub-project 3 (Modeling) trains a baseline → champion classifier on
`data/features.parquet`, using a time-based train/val/test split (the
`timestamp` column carried through from Task 8's `build_feature_table`
makes this possible without rejoining raw transactions).
