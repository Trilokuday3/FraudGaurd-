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

## Known deviations from spec

- The spec (`docs/superpowers/specs/2026-09-12-feature-engineering-design.md`)
  named `tests/features/test_correctness.py` as the home for per-feature
  correctness tests. That file was never created; the equivalent tests were
  written as `tests/unit/test_features_*.py`, one per feature family, which
  is where TDD naturally put them alongside each feature module. Coverage is
  equivalent — every correctness case the spec called for exists — just
  organized per-feature-file rather than in one combined file.
- The spec's determinism criterion (`test_features_determinism`: same input
  Parquet ⇒ identical output content hash) was instead satisfied via an
  in-process fixture comparison, `test_build_feature_table_is_deterministic`
  in `tests/unit/test_features_build.py`, which builds the feature table
  twice from the same fixture and asserts frame equality, rather than a
  content-hash test run against the full real `data/features.parquet`. This
  proves the same property (determinism of `build_feature_table`) without
  depending on a generated data file being present.

## What's next

Sub-project 3 (Modeling) trains a baseline → champion classifier on
`data/features.parquet`, using a time-based train/val/test split (the
`timestamp` column carried through from Task 8's `build_feature_table`
makes this possible without rejoining raw transactions).
