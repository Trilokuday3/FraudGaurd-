# Sub-project 2 — EDA + Feature Engineering — Design

**Date:** 2026-09-12
**Status:** Approved
**Parent:** `2026-09-12-fraudguard-platform-roadmap.md`
**Depends on:** sub-project 1 (Data Foundation) — reads `./data/*.parquet`
**Blocks:** sub-project 3 (Modeling) — trains on the gold feature table produced here

## Purpose

Turn the raw synthetic entities into (a) documented insight about the data
and (b) a leakage-safe, point-in-time **gold feature table** that
sub-project 3 can train on without ever touching `fraud_label` directly.

**In scope:** EDA notebook + written insights, `features/` package (six
feature families below), pandera schema for the feature table, leakage +
correctness tests.
**Out of scope:** modeling (sub-project 3), anything requiring Spark/Kafka
(sub-project 5 — this sub-project is pandas-only, consistent with the
roadmap's "no Spark for v1" decision).

## Success criteria

1. `make features` on a clean checkout with seeded data (`make seed`)
   produces `data/features.parquet` with no manual steps.
2. Output passes a pandera schema (`strict=True`) and the leakage test suite:
   no `fraud_label`, `fraud_probability_true`, `confirmed_fraud_at`, or
   `base_fraud_rate` reachable in the feature table or its build path.
3. Deterministic: same input Parquet ⇒ identical feature table (content
   hash), same pattern as `test_generator_determinism`.
4. `merchant_fraud_rate_hist` respects **label-confirmation time**, not just
   transaction time — a fraud confirmed *after* the scored transaction must
   not affect that transaction's feature value. Covered by a dedicated test.
5. `notebooks/01-eda.ipynb` runs top-to-bottom on `make seed` output and
   states **≥ 5 concrete business insights**, also captured in
   `docs/EDA_INSIGHTS.md`.

## Why this is harder than "join some columns"

Sub-project 1's data dictionary and `docs/data-generation-model.md` already
flag the trap: `merchants.base_fraud_rate` is never serialized, and
`ground_truth` is deliberately kept separate. A naive feature ("this
merchant's fraud rate") computed by joining all of `ground_truth` regardless
of time would leak the future into the past and hand sub-project 3 a
trivially high, worthless PR-AUC — the same failure mode the generator's
latent-risk design was built to avoid on the label side. This sub-project
has to earn a real merchant-risk signal from **historical, already-confirmed**
fraud only.

## Feature families

All computed from `transactions.parquet` (+ `customers`, `merchants`,
`devices` for joins) sorted by `(customer_id / merchant_id, timestamp)`.
`ground_truth` is read only by the merchant-risk feature, and only through
the point-in-time filter described below — never merged wholesale.

### 1. Transaction (direct/derived from the row)
| feature | definition |
|---|---|
| `amount` | pass-through |
| `payment_method` | pass-through |
| `hour_of_day` | `timestamp.hour` |
| `is_night` | `1` if local hour in [1, 5) |
| `is_cross_border` | `country != customer.home_country` |

### 2. Behavioral
| feature | definition |
|---|---|
| `amount_vs_customer_p95` | `amount / customer's trailing p95(amount)`, computed via `groupby(customer_id).expanding().quantile(0.95)` **shifted by one row** so the current transaction is excluded. Customers with < 5 prior transactions get a neutral default (`1.0`), same convention as the generator. |

### 3. Velocity
| feature | definition |
|---|---|
| `txn_count_1h` | per-customer `rolling('1h', closed='left')` count on a `timestamp`-indexed, sorted series — `closed='left'` excludes the current transaction, vectorized equivalent of the generator's trailing-window loop |
| `txn_count_24h` | same, 24h window |

### 4. Device
| feature | definition |
|---|---|
| `is_new_device` | `1` on a device's first-ever transaction (`groupby(device_id).cumcount() == 0`) |
| `device_age_days` | `(timestamp - device.first_seen_at).days`, clipped at 0 |
| `customer_device_count_so_far` | count of distinct devices this customer has used strictly before this transaction |

### 5. Merchant (the tricky one)
| feature | definition |
|---|---|
| `merchant_fraud_rate_hist` | For merchant `m` at transaction time `t`: `(# of m's past transactions with fraud_label=1 AND confirmed_fraud_at ≤ t) / (# of m's past transactions with timestamp < t)`. Merchants with < 20 qualifying past transactions get a global-prior default (overall historical fraud rate at that point in time) instead of a noisy small-sample rate. |

Implementation: per merchant (~2,000 groups), build a sorted array of that
merchant's past transaction timestamps (denominator) and a sorted array of
`confirmed_fraud_at` values for its confirmed frauds (numerator), then use
`np.searchsorted` per transaction — avoids a global merge of
`transactions × ground_truth` and keeps the "as of t" semantics explicit and
testable.

### 6. Geo
| feature | definition |
|---|---|
| `is_new_country_for_customer` | `1` on the first transaction where this customer used this `country` (`groupby([customer_id, country]).cumcount() == 0`) |
| `ip_country_mismatch` | pass-through: `ip_country != country` |

## Notebook (`notebooks/01-eda.ipynb`)

Joins `ground_truth` for analysis only (never inside `features/`). Sections:
fraud prevalence; amount distribution overall vs. fraud/non-fraud; hour-of-day
and merchant-category fraud rates; new-device / new-country / ip-mismatch
prevalence vs. fraud; cross-border patterns. Markdown cells state ≥5 concrete,
numbered business insights (e.g. "new-device transactions are Nx more likely
to be fraud"), mirrored into `docs/EDA_INSIGHTS.md`.

## Leakage & correctness tests (`tests/features/`)

Mirrors `tests/dq`'s style (load committed Parquet, don't regenerate).

| test | asserts |
|---|---|
| `test_no_forbidden_columns_in_features` | none of `fraud_label`, `fraud_probability_true`, `confirmed_fraud_at`, `base_fraud_rate` appear in `features.parquet` |
| `test_merchant_risk_respects_confirmation_delay` | synthetic fixture: a fraud confirmed after the scored transaction's `timestamp` must not move that transaction's `merchant_fraud_rate_hist`; confirming it before does |
| `test_velocity_excludes_current_transaction` | a customer's first-ever transaction has `txn_count_1h == 0`, `txn_count_24h == 0` |
| `test_amount_ratio_excludes_current_transaction` | a customer's first transaction gets the neutral default, not a ratio against itself |
| `test_features_determinism` | same input Parquet ⇒ identical output content hash |
| `test_features_schema` | pandera `features_schema`, `strict=True`, passes on `data/features.parquet` |

## Component boundaries

| unit | does | consumed via | depends on |
|---|---|---|---|
| `features/transaction.py`, `behavioral.py`, `velocity.py`, `device.py`, `merchant.py`, `geo.py` | one feature family each, pure function `(raw dataframes) -> DataFrame` | import, composed by `build.py` | `fraudguard_core` |
| `features/build.py` | orchestrates all six families, joins to `transaction_id`, writes `data/features.parquet` | import | family modules |
| `features/__main__.py` | CLI: `python -m features build` | Makefile `features` target | `build.py` |
| `libs/fraudguard_core.schemas.features_schema` | pandera schema for the gold table | import | — |
| `tests/features/` | leakage + correctness | `pytest tests/features` | `fraudguard_core`, `data/features.parquet` |

## Deliverables checklist

- [ ] `features/` package: `transaction.py`, `behavioral.py`, `velocity.py`, `device.py`, `merchant.py`, `geo.py`, `build.py`, `__main__.py`
- [ ] `libs/fraudguard_core/src/fraudguard_core/schemas.py` — add `features_schema`
- [ ] `Makefile` — add `features` target
- [ ] `notebooks/01-eda.ipynb`
- [ ] `docs/EDA_INSIGHTS.md`
- [ ] `docs/FEATURE_DEFINITIONS.md` (this spec's family tables, kept in sync with code)
- [ ] `tests/features/test_leakage.py`, `tests/features/test_correctness.py`
- [ ] `docs/sub2-acceptance.md` recorded after a clean-checkout run

## Risks / decisions

- **`np.searchsorted` per merchant, not a global merge-then-filter** —
  chosen because the "as of t" semantics for `merchant_fraud_rate_hist` are
  otherwise easy to get subtly wrong (off-by-one on `≤` vs `<`, or
  accidentally using `timestamp` instead of `confirmed_fraud_at` for the
  numerator). Isolating it to ~2,000 small per-merchant arrays keeps it fast
  and keeps the point-in-time logic in one reviewable place.
- **No customer-level fraud-risk-rate feature in v1** — the roadmap's family
  list (transaction/behavioral/velocity/device/merchant/geo) doesn't call for
  one, and a customer-level historical fraud rate would be extremely sparse
  at 1.5% prevalence with ~25 transactions/customer. Revisit only if
  sub-project 3's error analysis specifically wants it.
- **Small-sample default for merchant risk (< 20 qualifying transactions)** —
  without it, new merchants would get noisy 0%/100% rates from 1-2
  transactions; falling back to the global historical prior is the standard
  fix and matches how a real system would Bayesian-smooth a sparse rate.
