# Sub-project 1 — Data Foundation — Design

**Date:** 2026-09-12
**Status:** Approved, implemented same session
**Parent:** `2026-09-12-fraudguard-platform-roadmap.md`
**Depends on:** nothing (first sub-project)
**Blocks:** sub-project 2 (EDA + Feature Engineering) and everything downstream

## Purpose

Produce a realistic, reproducible synthetic transaction-fraud dataset —
customers, merchants, devices, transactions — with an injected fraud signal
that later features can actually learn from, plus a data-quality gate. Mirrors
the loan-platform's data-foundation approach (proven pattern), retargeted at
fraud.

**In scope:** data model, synthetic generator, pandera DQ suite, data
dictionary, generation-model writeup, unit tests.
**Out of scope:** feature engineering (sub-project 2), modeling (sub-project
3), streaming ingestion (sub-project 5 — generator has a `--stream` mode stub
but Kafka wiring happens later).

## Success criteria

1. `make seed dq` (or `python -m generator seed && pytest tests/dq`) completes
   with no manual steps beyond a Python env.
2. Default volume: **≥ 20,000 customers**, **≥ 2,000 merchants**,
   **≥ 25,000 devices**, **≥ 500,000 transactions**.
3. Realised `fraud_label` rate within **±0.3pp** of the configured target
   (default 1.5% — realistic for card-not-present fraud, more imbalanced than
   IEEE-CIS's ~3.5% so the modeling sub-project has to earn PR-AUC honestly).
4. Every pandera schema check and every cross-entity referential-integrity
   check passes.
5. Deterministic: identical `(seed, config)` ⇒ identical row counts and
   identical content hash per entity.
6. No post-outcome leakage column is reachable from the features path:
   `fraud_label` and `confirmed_fraud_at` live only in a separate
   `ground_truth` table keyed by `transaction_id`, never joined into what
   sub-project 2 reads as the feature base.

## Data model

Four entities generated here (`alerts` and `model_predictions` from the
brief's §5 schema are created later, by the Decision Engine sub-project, once
there's a model to produce them).

### `customers`
| column | type | notes |
|---|---|---|
| customer_id | string PK | `CUST%07d` |
| account_created_at | timestamp | 1 day – 6 years before generation end |
| customer_segment | cat | retail / premium / business |
| home_country | cat | ISO-ish 2-letter, weighted (majority one country, long tail) |
| home_city | string | drawn from a per-country city list |
| risk_appetite | float | latent, not exposed to features — drives behavioral variance only |

### `merchants`
| column | type | notes |
|---|---|---|
| merchant_id | string PK | `MERC%06d` |
| merchant_category | cat | grocery / electronics / travel / gaming / crypto / fashion / dining / other |
| country | cat | same space as `customers.home_country` |
| created_at | timestamp | merchant onboarding date |
| base_fraud_rate | float | latent per-category+country prior — drives label generation, not exposed as a feature (must be *estimated* downstream from history, not read directly) |

### `devices`
| column | type | notes |
|---|---|---|
| device_id | string PK | `DEV%07d` |
| customer_id | string FK → customers | a device belongs to one customer for v1 (device sharing/reuse across customers deferred — real fraud signal, noted as a future extension) |
| device_type | cat | mobile / desktop / tablet |
| first_seen_at | timestamp | ≥ `customers.account_created_at` |

### `transactions`
| column | type | notes |
|---|---|---|
| transaction_id | string PK | `TXN%010d` |
| customer_id | string FK → customers | |
| merchant_id | string FK → merchants | |
| device_id | string FK → devices, must belong to `customer_id` | |
| timestamp | timestamp | across the vintage window, non-uniform (diurnal pattern per customer timezone proxy) |
| amount | float | lognormal per customer_segment, occasional high-amount outliers |
| currency | cat | fixed small set, mostly one currency |
| payment_method | cat | card / wallet / bank_transfer |
| country | cat | usually `customers.home_country`; a minority cross-border |
| city | string | usually `customers.home_city` |
| ip_country | cat | usually equals `country`; occasional mismatch (proxy/VPN signal) |

`ground_truth` (separate table, not joined into features):
| column | type | notes |
|---|---|---|
| transaction_id | string FK → transactions, unique | |
| fraud_probability_true | float | model of latent risk, for calibration checks later |
| fraud_label | int 0/1 | Bernoulli draw — **the target** |
| confirmed_fraud_at | timestamp? | non-null iff `fraud_label=1`; 0–14 days after `timestamp`, models real label delay per brief §3 |

## Synthetic generation model

Generation order: `merchants → customers → devices → transactions (+ ground_truth)`.

For each transaction, a latent risk score `z`:

```
z = b0
  + w1 · new_device_for_customer                 # device.first_seen_at within 1h of txn (proxy for "never used before")
  + w2 · new_country_for_customer                 # txn.country != customer.home_country, first time
  + w3 · ip_country_mismatch                       # ip_country != country
  + w4 · zscore(amount_vs_customer_p95)            # amount / customer's trailing 30-day p95 amount (computed causally, see below)
  + w5 · zscore(velocity_1h)                       # count of this customer's txns in the trailing 1h, causal
  + w6 · merchant.base_fraud_rate_zscore           # latent merchant risk (category × country prior)
  + w7 · night_hour_indicator                      # local hour in [1,5)
  − w8 · zscore(log(customer_account_age_days))    # newer accounts riskier
  + ε,  ε ~ Normal(0, σ)

fraud_probability_true = sigmoid(z)
fraud_label ~ Bernoulli(fraud_probability_true)
```

`b0` solved by bisection so realised `fraud_label` rate ≈ `target_fraud_rate`
(default 1.5%). Weights `w1..w8`, `σ`, `b0` written to
`docs/data-generation-model.md` and `generation_meta.json` alongside output.

**Causality note:** `amount_vs_customer_p95` and `velocity_1h` are computed
using a single chronological pass per customer during generation (each
transaction only sees that customer's *prior* transactions), so the same
point-in-time discipline the label-generation model requires is also what
sub-project 2's features must replicate — the generator is a worked example
of the leakage rule, not just an assertion of it.

### Leakage guards (defined here, enforced downstream)

- `ground_truth.fraud_label` and `confirmed_fraud_at` are never written into
  `transactions` or any feature-adjacent table.
- `merchant.base_fraud_rate` is a generator-internal latent variable — it is
  **not** a column exposed on `merchants`; sub-project 2 must estimate a
  merchant risk feature from historical transaction/label data the same way a
  real system would, or the model would trivially leak the ground-truth prior.
- Forbidden-in-features list (asserted by a test): `ground_truth.*`,
  `merchants.base_fraud_rate` (internal only, not even serialized to the
  merchants Parquet output).

### Configuration (`generator/config.py`)

| key | default | meaning |
|---|---|---|
| seed | 42 | global RNG seed |
| n_customers | 20_000 | |
| n_merchants | 2_000 | |
| avg_devices_per_customer | 1.3 | |
| avg_txns_per_customer | 25 | ⇒ ~500k transactions |
| target_fraud_rate | 0.015 | ±0.3pp |
| vintage_start / vintage_end | 2025-03-01 / 2026-09-01 | |
| output_dir | ./data | local Parquet; swappable to object storage later via same `fraudguard_core.config` pattern as the loan project |
| chunk_customers | 2_000 | batches of customers processed per pass, keeps memory bounded |

Volume scales linearly with `n_customers`; defaults are laptop-friendly
(~150–250 MB Parquet, generates in well under 2 minutes on pandas alone — no
Spark needed at this scale, unlike the loan project's larger volumes).

## Data-quality suite (`tests/dq/`, pandera)

Per-entity `DataFrameSchema`: column presence, dtype, nullability, value-set
membership, numeric ranges (`amount > 0`, `fraud_probability_true` in [0,1]),
PK uniqueness.

Cross-entity checks (pytest):
- every `devices.customer_id` ∈ `customers.customer_id`
- every `transactions.customer_id` ∈ `customers.customer_id`
- every `transactions.merchant_id` ∈ `merchants.merchant_id`
- every `transactions.device_id` ∈ `devices.device_id` **and** that device's
  `customer_id` matches the transaction's `customer_id`
- every `ground_truth.transaction_id` ∈ `transactions.transaction_id`, 1:1

`make dq` / `pytest tests/dq` runs the whole suite, exits non-zero on failure.

## Component boundaries

| unit | does | consumed via | depends on |
|---|---|---|---|
| `libs/fraudguard_core.schemas` | canonical column lists, dtypes, value sets, pandera schemas | import | — |
| `libs/fraudguard_core.config` | `pydantic-settings` config, `.env` loading | import | — |
| `generator` | `(seed, config) → Parquet` (customers, merchants, devices, transactions, ground_truth) | CLI `python -m generator seed` | `fraudguard_core` |
| `tests/dq` | assert generated data quality | `pytest tests/dq` | `fraudguard_core` |

## Tests

| test | asserts |
|---|---|
| `test_generator_determinism` | same `(seed, config)` ⇒ identical row counts + content hash per entity |
| `test_fraud_rate_tolerance` | realised fraud rate within ±0.3pp of target |
| `test_forbidden_columns_absent` | `merchants` Parquet has no `base_fraud_rate` column; no `ground_truth` column reachable from `transactions` |
| `test_referential_integrity` | injected orphan FK row is caught by the DQ suite |
| `test_device_ownership` | every transaction's device belongs to that transaction's customer |
| `dq` pandera suite | all per-entity + cross-entity schemas pass |

Unit tests use a tiny in-memory config (`n_customers=200`); the default-volume
run is exercised by `make seed dq`, not by unit tests.

## Deliverables checklist

- [x] `libs/fraudguard_core/` package (`schemas`, `config`, `enums`) with `pyproject.toml`, `pip install -e`
- [x] `generator/` package + `__main__.py` CLI (`seed`), `config.py`, `generation_meta.json` output
- [x] `tests/unit/` suite + `tests/dq/` pandera suite
- [x] `docs/DATA_DICTIONARY.md`
- [x] `docs/data-generation-model.md` (weights, σ, b0, label logic, leakage guards)
- [x] `.gitignore` for `data/`
- [x] Acceptance run recorded: volumes + realised fraud rate + DQ pass in `docs/sub1-acceptance.md`

## Risks / decisions

- **No Spark, no Kafka at this stage** — deliberate: default volume fits
  comfortably in pandas memory; Spark/Kafka are introduced only in the
  streaming sub-project (5), where they demonstrate a real concept
  (windowed streaming features) rather than being infrastructure for its own
  sake — directly following the brief's own §34 warning.
- **Single-customer device ownership** — real fraud systems care about device
  *reuse across customers* (device fingerprint linked to multiple accounts) as
  a strong signal. Deferred to keep sub-project 1 shippable in days not weeks;
  flagged here so sub-project 2/6 (Fraud Network module, if it's ever added
  back in scope) knows it needs richer device-sharing data first.
