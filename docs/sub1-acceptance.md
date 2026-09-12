# Sub-project 1 — Acceptance Run

**Date:** 2026-09-12
**Command:** `python -m generator seed` (default config, seed=42), then `pytest tests/unit tests/dq -v`

## Volumes

| entity | count | target |
|---|---|---|
| customers | 20,000 | ≥ 20,000 ✅ |
| merchants | 2,000 | ≥ 2,000 ✅ |
| devices | 31,481 | ≥ 25,000 ✅ |
| transactions | 519,876 | ≥ 500,000 ✅ |

## Fraud rate

Target 1.5000%, realised **1.5067%** — 0.0067pp off, within the ±0.3pp
tolerance. `b0 = -6.339` (see `docs/data-generation-model.md`).

## Determinism

`tests/unit/test_generator.py::test_generator_determinism` — two runs of the
same `(seed, config)` produce identical row counts and identical content
hashes for every entity. ✅

## Data quality

`pytest tests/unit tests/dq -v` — **17/17 passed**:
- 6 unit tests: determinism, fraud-rate tolerance, forbidden-column absence,
  referential integrity, device ownership, ground-truth 1:1.
- 11 DQ tests: pandera schema validation for all 5 entities, cross-entity FK
  checks (devices→customers, transactions→customers/merchants/devices with
  ownership check), ground-truth 1:1 match, forbidden-leakage-column check.

No manual fixes were needed to the data itself; one deprecation warning
(`pandera` top-level import) was fixed in `fraudguard_core/schemas.py`
(`from pandera.pandas import ...`) before the final clean run.

## What's next

Sub-project 2 (EDA + Feature Engineering) reads `./data/*.parquet`, builds
point-in-time transaction/behavioral/velocity/device/merchant/geo features
following the same causal-computation discipline the generator itself uses,
and produces the leakage-tested feature table sub-project 3 trains on.
