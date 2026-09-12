# Feature Definitions — Sub-project 2

Source of truth for what each column in `data/features.parquet` means and how
it's computed. Schema enforced by `fraudguard_core.schemas.features_schema`;
computation lives in `features/`. Every feature is causal: computed using
only information available strictly before the transaction being scored.

## Transaction (`features/transaction.py`)
| feature | definition |
|---|---|
| `amount` | pass-through |
| `payment_method` | pass-through |
| `hour_of_day` | `timestamp.hour` |
| `is_night` | `1` if local hour in `{1, 2, 3, 4}` |
| `is_cross_border` | `1` if `country != customer.home_country` |

## Behavioral (`features/behavioral.py`)
| feature | definition |
|---|---|
| `amount_vs_customer_p95` | `amount / customer's trailing p95(amount)`, computed over strictly prior transactions only. Customers with fewer than 5 prior transactions get the neutral default `1.0`. |

## Velocity (`features/velocity.py`)
| feature | definition |
|---|---|
| `txn_count_1h` | count of this customer's transactions in the trailing 1 hour, current transaction excluded |
| `txn_count_24h` | same, trailing 24 hours |

## Device (`features/device.py`)
| feature | definition |
|---|---|
| `is_new_device` | `1` on a device's first-ever transaction |
| `device_age_days` | days since `device.first_seen_at`, clipped at 0 |
| `customer_device_count_so_far` | count of distinct devices this customer used strictly before this transaction |

## Merchant (`features/merchant.py`)
| feature | definition |
|---|---|
| `merchant_fraud_rate_hist` | this merchant's historical fraud rate, using only past transactions whose fraud was **confirmed** (`confirmed_fraud_at <= t`) by the current transaction's time — not merely committed by then. Merchants with fewer than 20 qualifying prior transactions fall back to the global historical fraud rate as of the same instant. |

## Geo (`features/geo.py`)
| feature | definition |
|---|---|
| `is_new_country_for_customer` | `1` on the first transaction where this customer used this `country` |
| `ip_country_mismatch` | `1` if `ip_country != country` |

## Leakage guards (enforced by `tests/features/test_leakage.py`)

- `fraud_label`, `fraud_probability_true`, `confirmed_fraud_at`, and
  `base_fraud_rate` never appear in `features.parquet`.
- `merchant_fraud_rate_hist` is the only feature that reads `ground_truth`,
  and only through the confirmation-time filter above — see
  `docs/superpowers/specs/2026-09-12-feature-engineering-design.md` for why
  this needs its own dedicated test (`test_merchant_risk_respects_confirmation_delay`
  in `tests/unit/test_features_merchant.py`).
