# Sub-project 2 (EDA + Feature Engineering) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe, point-in-time gold feature table (`data/features.parquet`) from the sub-project-1 synthetic data, plus an EDA notebook documenting ≥5 business insights.

**Architecture:** Six independent, pure feature-family functions (`features/transaction.py`, `behavioral.py`, `velocity.py`, `device.py`, `merchant.py`, `geo.py`), each `(raw dataframes) -> DataFrame` indexed by `transaction_id`. `features/build.py` joins them into one gold table validated against a new pandera schema. A CLI (`python -m features build`) and `make features` target wire it to disk. Tests are pure in-memory unit tests per family (fast, no dependency on generated data) plus one integration suite that validates the real committed `data/features.parquet`.

**Tech Stack:** pandas, numpy, pandera (existing project stack — no new production dependencies). Jupyter + matplotlib added as dev-only dependencies for the EDA notebook.

**Spec:** `docs/superpowers/specs/2026-09-12-feature-engineering-design.md`

## Global Constraints

- pandas only — no Spark/Kafka in this sub-project (roadmap decision, carried into the spec).
- Every feature is computed causally: only information available strictly before the transaction being scored. `merchant_fraud_rate_hist` specifically must respect `confirmed_fraud_at`, not transaction `timestamp`.
- Forbidden columns must never appear in `data/features.parquet` or any `features/` module's output: `fraud_label`, `fraud_probability_true`, `confirmed_fraud_at`, `base_fraud_rate`.
- Shared/canonical schema code lives in `libs/fraudguard_core`; `features/` imports it, never redefines value sets.
- Line length 100, formatted with `ruff` + `black` (existing `pyproject.toml` config).
- Tests live under `tests/`; `pythonpath = ["."]` means `features` and `generator` are both imported as top-level packages — no `src/` layout for them.
- New/derived Parquet output goes in `./data` (gitignored), never committed.

---

### Task 1: `features_schema` in `fraudguard_core`

**Files:**
- Modify: `libs/fraudguard_core/src/fraudguard_core/schemas.py`
- Test: `tests/unit/test_features_schema.py`

**Interfaces:**
- Consumes: `fraudguard_core.enums.PAYMENT_METHOD` (already exists)
- Produces: `fraudguard_core.schemas.features_schema` (a `pandera.pandas.DataFrameSchema`, `strict=True`) — every later task's output is validated against this. Exact columns: `transaction_id, customer_id, merchant_id, timestamp, amount, payment_method, hour_of_day, is_night, is_cross_border, amount_vs_customer_p95, txn_count_1h, txn_count_24h, is_new_device, device_age_days, customer_device_count_so_far, merchant_fraud_rate_hist, is_new_country_for_customer, ip_country_mismatch`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_schema.py
import pandas as pd
import pytest
from fraudguard_core.schemas import features_schema


def _valid_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TXN0000000001"],
            "customer_id": ["CUST0000001"],
            "merchant_id": ["MERC000001"],
            "timestamp": pd.to_datetime(["2026-01-01T12:00:00"]),
            "amount": [42.50],
            "payment_method": ["card"],
            "hour_of_day": [12],
            "is_night": [0],
            "is_cross_border": [0],
            "amount_vs_customer_p95": [1.0],
            "txn_count_1h": [0],
            "txn_count_24h": [0],
            "is_new_device": [1],
            "device_age_days": [0.0],
            "customer_device_count_so_far": [0],
            "merchant_fraud_rate_hist": [0.0],
            "is_new_country_for_customer": [1],
            "ip_country_mismatch": [0],
        }
    )


def test_features_schema_accepts_valid_row():
    features_schema.validate(_valid_row())


def test_features_schema_rejects_forbidden_column():
    df = _valid_row()
    df["fraud_label"] = [1]
    with pytest.raises(Exception):
        features_schema.validate(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'features_schema'`

- [ ] **Step 3: Add `features_schema` to `schemas.py`**

Add to `libs/fraudguard_core/src/fraudguard_core/schemas.py`, after `ground_truth_schema` and before `ALL_SCHEMAS`:

```python
from fraudguard_core.enums import PAYMENT_METHOD  # add to existing import line if not present

features_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, unique=True, checks=Check.str_startswith("TXN")),
        "customer_id": Column(str),
        "merchant_id": Column(str),
        "timestamp": Column("datetime64[ns]"),
        "amount": Column(float, checks=Check.gt(0)),
        "payment_method": Column(str, checks=Check.isin(PAYMENT_METHOD)),
        "hour_of_day": Column(int, checks=Check.in_range(0, 23)),
        "is_night": Column(int, checks=Check.isin({0, 1})),
        "is_cross_border": Column(int, checks=Check.isin({0, 1})),
        "amount_vs_customer_p95": Column(float, checks=Check.gt(0)),
        "txn_count_1h": Column(int, checks=Check.ge(0)),
        "txn_count_24h": Column(int, checks=Check.ge(0)),
        "is_new_device": Column(int, checks=Check.isin({0, 1})),
        "device_age_days": Column(float, checks=Check.ge(0)),
        "customer_device_count_so_far": Column(int, checks=Check.ge(0)),
        "merchant_fraud_rate_hist": Column(float, checks=Check.in_range(0, 1)),
        "is_new_country_for_customer": Column(int, checks=Check.isin({0, 1})),
        "ip_country_mismatch": Column(int, checks=Check.isin({0, 1})),
    },
    strict=True,
)
```

(`PAYMENT_METHOD` is already imported at the top of the file for `transactions_schema` — just reuse that import, don't duplicate it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_schema.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add libs/fraudguard_core/src/fraudguard_core/schemas.py tests/unit/test_features_schema.py
git commit -m "feat(fraudguard_core): add features_schema for the sub-project 2 gold table"
```

---

### Task 2: `features/transaction.py`

**Files:**
- Create: `features/__init__.py` (empty)
- Create: `features/transaction.py`
- Test: `tests/unit/test_features_transaction.py`

**Interfaces:**
- Consumes: raw `transactions` DataFrame (columns per `fraudguard_core.schemas.transactions_schema`: `transaction_id, customer_id, merchant_id, device_id, timestamp, amount, currency, payment_method, country, city, ip_country`) and raw `customers` DataFrame (`customer_id, account_created_at, customer_segment, home_country, home_city`).
- Produces: `build_transaction_features(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame` indexed by `transaction_id`, columns `amount, payment_method, hour_of_day, is_night, is_cross_border`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_transaction.py
import pandas as pd

from features.transaction import build_transaction_features


def test_transaction_features_night_and_cross_border():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "amount": [10.0, 20.0, 30.0],
            "payment_method": ["card", "card", "wallet"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T02:00:00", "2026-01-01T14:00:00", "2026-01-01T14:30:00"]
            ),
            "country": ["US", "US", "GB"],
        }
    )
    customers = pd.DataFrame({"customer_id": ["C1"], "home_country": ["US"]})

    result = build_transaction_features(transactions, customers)

    assert list(result.index) == ["TXN1", "TXN2", "TXN3"]
    assert result.loc["TXN1", "is_night"] == 1
    assert result.loc["TXN2", "is_night"] == 0
    assert result.loc["TXN1", "hour_of_day"] == 2
    assert result.loc["TXN1", "is_cross_border"] == 0
    assert result.loc["TXN3", "is_cross_border"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_transaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features'`

- [ ] **Step 3: Write the implementation**

```python
# features/__init__.py
```

```python
# features/transaction.py
"""Transaction-family features: direct/derived from the row itself, no history needed."""

import pandas as pd

NIGHT_HOURS = frozenset({1, 2, 3, 4})


def build_transaction_features(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    home_country = customers.set_index("customer_id")["home_country"]

    out = transactions[["transaction_id", "amount", "payment_method", "timestamp"]].copy()
    out["hour_of_day"] = transactions["timestamp"].dt.hour
    out["is_night"] = out["hour_of_day"].isin(NIGHT_HOURS).astype(int)

    customer_home = transactions["customer_id"].map(home_country)
    out["is_cross_border"] = (transactions["country"].to_numpy() != customer_home).astype(int)

    return out.drop(columns=["timestamp"]).set_index("transaction_id")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_transaction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/__init__.py features/transaction.py tests/unit/test_features_transaction.py
git commit -m "feat(features): add transaction-family features"
```

---

### Task 3: `features/behavioral.py`

**Files:**
- Create: `features/behavioral.py`
- Test: `tests/unit/test_features_behavioral.py`

**Interfaces:**
- Consumes: raw `transactions` DataFrame (`transaction_id, customer_id, timestamp, amount`).
- Produces: `build_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame` indexed by `transaction_id`, column `amount_vs_customer_p95`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_behavioral.py
import pandas as pd

from features.behavioral import build_behavioral_features


def test_behavioral_features_use_only_prior_history():
    base_time = pd.Timestamp("2026-01-01T00:00:00")
    amounts = [10, 10, 10, 10, 10, 100]  # TXN5 is the first row with 5 prior transactions
    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i}" for i in range(6)],
            "customer_id": ["C1"] * 6,
            "timestamp": [base_time + pd.Timedelta(hours=i) for i in range(6)],
            "amount": amounts,
        }
    )

    result = build_behavioral_features(transactions)

    assert result.loc["TXN0", "amount_vs_customer_p95"] == 1.0  # no history -> neutral default
    assert result.loc["TXN4", "amount_vs_customer_p95"] == 1.0  # only 4 prior txns -> still default
    assert result.loc["TXN5", "amount_vs_customer_p95"] > 5.0  # 5 prior txns of 10 -> ratio ~= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_behavioral.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.behavioral'`

- [ ] **Step 3: Write the implementation**

```python
# features/behavioral.py
"""Behavioral features: how does this transaction compare to the customer's own history."""

import pandas as pd

MIN_HISTORY_FOR_RATIO = 5
DEFAULT_RATIO = 1.0


def build_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions[["transaction_id", "customer_id", "timestamp", "amount"]].sort_values(
        ["customer_id", "timestamp"]
    )

    trailing_p95 = df.groupby("customer_id", group_keys=False)["amount"].apply(
        lambda s: s.expanding(min_periods=MIN_HISTORY_FOR_RATIO).quantile(0.95).shift(1)
    )

    df = df.assign(
        amount_vs_customer_p95=(df["amount"] / (trailing_p95 + 1e-6)).fillna(DEFAULT_RATIO)
    )
    return df.set_index("transaction_id")[["amount_vs_customer_p95"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_behavioral.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/behavioral.py tests/unit/test_features_behavioral.py
git commit -m "feat(features): add behavioral (amount-vs-history) feature"
```

---

### Task 4: `features/velocity.py`

**Files:**
- Create: `features/velocity.py`
- Test: `tests/unit/test_features_velocity.py`

**Interfaces:**
- Consumes: raw `transactions` DataFrame (`transaction_id, customer_id, timestamp`).
- Produces: `build_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame` indexed by `transaction_id`, columns `txn_count_1h, txn_count_24h`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_velocity.py
import pandas as pd

from features.velocity import build_velocity_features


def test_velocity_counts_exclude_current_transaction():
    base_time = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "timestamp": [
                base_time,
                base_time + pd.Timedelta(minutes=30),
                base_time + pd.Timedelta(hours=2),
            ],
        }
    )

    result = build_velocity_features(transactions)

    assert result.loc["TXN1", "txn_count_1h"] == 0
    assert result.loc["TXN2", "txn_count_1h"] == 1
    assert result.loc["TXN3", "txn_count_1h"] == 0
    assert result.loc["TXN3", "txn_count_24h"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_velocity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.velocity'`

- [ ] **Step 3: Write the implementation**

```python
# features/velocity.py
"""Velocity features: trailing transaction counts per customer, current transaction excluded."""

import pandas as pd


def build_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = (
        transactions[["transaction_id", "customer_id", "timestamp"]]
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )
    indexed = df.set_index("timestamp")

    counts_1h = (
        indexed.groupby("customer_id")["transaction_id"].rolling("1h", closed="left").count()
    )
    counts_24h = (
        indexed.groupby("customer_id")["transaction_id"].rolling("24h", closed="left").count()
    )

    df["txn_count_1h"] = counts_1h.to_numpy().astype(int)
    df["txn_count_24h"] = counts_24h.to_numpy().astype(int)

    return df.set_index("transaction_id")[["txn_count_1h", "txn_count_24h"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_velocity.py -v`
Expected: PASS. If it fails on the count values (e.g. off-by-one), print `counts_1h`/`counts_24h` and `df` side by side to check alignment — the fix is almost always `closed="left"` vs default, not the groupby/rolling structure itself.

- [ ] **Step 5: Commit**

```bash
git add features/velocity.py tests/unit/test_features_velocity.py
git commit -m "feat(features): add velocity (trailing transaction count) features"
```

---

### Task 5: `features/device.py`

**Files:**
- Create: `features/device.py`
- Test: `tests/unit/test_features_device.py`

**Interfaces:**
- Consumes: raw `transactions` (`transaction_id, customer_id, device_id, timestamp`) and raw `devices` (`device_id, customer_id, device_type, first_seen_at`).
- Produces: `build_device_features(transactions: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame` indexed by `transaction_id`, columns `is_new_device, device_age_days, customer_device_count_so_far`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_device.py
import pandas as pd

from features.device import build_device_features


def test_device_features():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "device_id": ["D1", "D1", "D2"],
            "timestamp": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)],
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": ["D1", "D2"],
            "customer_id": ["C1", "C1"],
            "first_seen_at": [t0, t0 + pd.Timedelta(hours=2)],
        }
    )

    result = build_device_features(transactions, devices)

    assert result.loc["TXN1", "is_new_device"] == 1  # D1's first use
    assert result.loc["TXN2", "is_new_device"] == 0  # D1 used again
    assert result.loc["TXN3", "is_new_device"] == 1  # D2's first use
    assert result.loc["TXN1", "customer_device_count_so_far"] == 0  # no devices used yet
    assert result.loc["TXN2", "customer_device_count_so_far"] == 1  # D1 used before this txn
    assert result.loc["TXN3", "customer_device_count_so_far"] == 1  # still just D1 before this txn
    assert result.loc["TXN2", "device_age_days"] == 1 / 24  # 1 hour since D1.first_seen_at
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.device'`

- [ ] **Step 3: Write the implementation**

```python
# features/device.py
"""Device features: novelty and tenure of the device used, and how many distinct
devices this customer has used before this transaction."""

import pandas as pd


def build_device_features(transactions: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    df = (
        transactions[["transaction_id", "customer_id", "device_id", "timestamp"]]
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )

    df["is_new_device"] = (df.groupby("device_id").cumcount() == 0).astype(int)

    device_first_seen = devices.set_index("device_id")["first_seen_at"]
    age_seconds = (df["timestamp"] - df["device_id"].map(device_first_seen)).dt.total_seconds()
    df["device_age_days"] = (age_seconds / 86400).clip(lower=0)

    first_device_use = (df.groupby(["customer_id", "device_id"]).cumcount() == 0).astype(int)
    distinct_devices_so_far = first_device_use.groupby(df["customer_id"]).cumsum()
    df["customer_device_count_so_far"] = (
        distinct_devices_so_far.groupby(df["customer_id"]).shift(1, fill_value=0).astype(int)
    )

    return df.set_index("transaction_id")[
        ["is_new_device", "device_age_days", "customer_device_count_so_far"]
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_device.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/device.py tests/unit/test_features_device.py
git commit -m "feat(features): add device novelty/tenure features"
```

---

### Task 6: `features/merchant.py` (label-delay-aware merchant risk)

**Files:**
- Create: `features/merchant.py`
- Test: `tests/unit/test_features_merchant.py`

**Interfaces:**
- Consumes: raw `transactions` (`transaction_id, merchant_id, timestamp`) and raw `ground_truth` (`transaction_id, fraud_label, confirmed_fraud_at`).
- Produces: `build_merchant_features(transactions: pd.DataFrame, ground_truth: pd.DataFrame, min_history: int = 20) -> pd.DataFrame` indexed by `transaction_id`, column `merchant_fraud_rate_hist`.

This is the feature the spec calls out as the trap: it must use `confirmed_fraud_at`, not `timestamp`, to decide whether a past fraud is "known yet." A merchant with fewer than `min_history` prior transactions falls back to the global historical fraud rate as of that same instant (same causal rule, just population-wide) instead of a noisy small-sample rate.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_merchant.py
import pandas as pd

from features.merchant import build_merchant_features


def test_merchant_risk_respects_confirmation_delay():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C"],
            "merchant_id": ["M1", "M1", "M1"],
            "timestamp": [t0, t0 + pd.Timedelta(days=5), t0 + pd.Timedelta(days=15)],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C"],
            "fraud_label": [1, 0, 0],
            "confirmed_fraud_at": [t0 + pd.Timedelta(days=10), pd.NaT, pd.NaT],
        }
    )

    # min_history=0 isolates the confirmation-delay behavior from the
    # small-sample fallback, which is tested separately below.
    result = build_merchant_features(transactions, ground_truth, min_history=0)

    assert result.loc["B", "merchant_fraud_rate_hist"] == 0.0  # A's fraud not confirmed yet
    assert result.loc["C", "merchant_fraud_rate_hist"] == 0.5  # confirmed by now: 1 of 2 prior txns


def test_merchant_risk_falls_back_to_global_prior_below_min_history():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C", "D"],
            "merchant_id": ["M1", "M2", "M2", "M2"],
            "timestamp": [
                t0,
                t0 + pd.Timedelta(days=1),
                t0 + pd.Timedelta(days=2),
                t0 + pd.Timedelta(days=3),
            ],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "C", "D"],
            "fraud_label": [1, 0, 0, 0],
            "confirmed_fraud_at": [t0 + pd.Timedelta(hours=1), pd.NaT, pd.NaT, pd.NaT],
        }
    )

    # M2 has fewer than 20 prior transactions -> falls back to the global rate.
    result = build_merchant_features(transactions, ground_truth, min_history=20)

    # By the time of D, the global population has 1 confirmed fraud (A) out of 3
    # prior transactions (A, B, C) -> global rate = 1/3.
    assert abs(result.loc["D", "merchant_fraud_rate_hist"] - (1 / 3)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_merchant.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.merchant'`

- [ ] **Step 3: Write the implementation**

```python
# features/merchant.py
"""Merchant risk: historical fraud rate for this merchant, computed as-of the
transaction's own timestamp using only frauds that were CONFIRMED by then
(confirmed_fraud_at <= t), not merely committed by then. Falls back to the
global historical rate when the merchant has too little history to trust."""

import numpy as np
import pandas as pd

MIN_MERCHANT_HISTORY = 20


def build_merchant_features(
    transactions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    min_history: int = MIN_MERCHANT_HISTORY,
) -> pd.DataFrame:
    txn_merchant = transactions.set_index("transaction_id")["merchant_id"]
    fraud = ground_truth[ground_truth["fraud_label"] == 1].copy()
    fraud["merchant_id"] = fraud["transaction_id"].map(txn_merchant)

    all_txn_times = np.sort(transactions["timestamp"].to_numpy())
    all_fraud_times = np.sort(fraud["confirmed_fraud_at"].to_numpy())

    df = transactions[["transaction_id", "merchant_id", "timestamp"]].reset_index(drop=True)
    rates = np.empty(len(df))

    for merchant_id, group in df.groupby("merchant_id"):
        idx = group.index.to_numpy()
        order = np.argsort(group["timestamp"].to_numpy())
        sorted_times = group["timestamp"].to_numpy()[order]

        merchant_fraud_times = np.sort(
            fraud.loc[fraud["merchant_id"] == merchant_id, "confirmed_fraud_at"].to_numpy()
        )

        denom = np.searchsorted(sorted_times, sorted_times, side="left")
        numer = np.searchsorted(merchant_fraud_times, sorted_times, side="right")
        merchant_rate = numer / np.maximum(denom, 1)

        global_denom = np.searchsorted(all_txn_times, sorted_times, side="left")
        global_numer = np.searchsorted(all_fraud_times, sorted_times, side="right")
        global_rate = global_numer / np.maximum(global_denom, 1)

        rate = np.where(denom >= min_history, merchant_rate, global_rate)
        rates[idx[order]] = rate

    result = df.copy()
    result["merchant_fraud_rate_hist"] = rates
    return result.set_index("transaction_id")[["merchant_fraud_rate_hist"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_merchant.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add features/merchant.py tests/unit/test_features_merchant.py
git commit -m "feat(features): add label-delay-aware merchant risk feature"
```

---

### Task 7: `features/geo.py`

**Files:**
- Create: `features/geo.py`
- Test: `tests/unit/test_features_geo.py`

**Interfaces:**
- Consumes: raw `transactions` (`transaction_id, customer_id, country, ip_country, timestamp`).
- Produces: `build_geo_features(transactions: pd.DataFrame) -> pd.DataFrame` indexed by `transaction_id`, columns `is_new_country_for_customer, ip_country_mismatch`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_geo.py
import pandas as pd

from features.geo import build_geo_features


def test_geo_features():
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C1"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"]
            ),
            "country": ["US", "US", "GB"],
            "ip_country": ["US", "GB", "GB"],
        }
    )

    result = build_geo_features(transactions)

    assert result.loc["TXN1", "is_new_country_for_customer"] == 1
    assert result.loc["TXN2", "is_new_country_for_customer"] == 0
    assert result.loc["TXN3", "is_new_country_for_customer"] == 1
    assert result.loc["TXN1", "ip_country_mismatch"] == 0
    assert result.loc["TXN2", "ip_country_mismatch"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_geo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.geo'`

- [ ] **Step 3: Write the implementation**

```python
# features/geo.py
"""Geo features: has this customer transacted in this country before, and does
the transaction's IP-derived country match its stated country."""

import pandas as pd


def build_geo_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = (
        transactions[["transaction_id", "customer_id", "country", "ip_country", "timestamp"]]
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )

    df["is_new_country_for_customer"] = (
        df.groupby(["customer_id", "country"]).cumcount() == 0
    ).astype(int)
    df["ip_country_mismatch"] = (df["country"] != df["ip_country"]).astype(int)

    return df.set_index("transaction_id")[
        ["is_new_country_for_customer", "ip_country_mismatch"]
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_geo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add features/geo.py tests/unit/test_features_geo.py
git commit -m "feat(features): add geo (new-country, ip-mismatch) features"
```

---

### Task 8: `features/build.py`, CLI, and `make features`

**Files:**
- Create: `features/build.py`
- Create: `features/__main__.py`
- Modify: `Makefile`
- Test: `tests/unit/test_features_build.py`

**Interfaces:**
- Consumes: all six `build_*_features` functions from Tasks 2-7, and `fraudguard_core.schemas.features_schema` from Task 1.
- Produces: `build_feature_table(transactions, customers, devices, ground_truth) -> pd.DataFrame` (flat, `transaction_id` as a column, validated against `features_schema`) and `write_features(df: pd.DataFrame, output_dir: str) -> str` (returns the written path). CLI: `python -m features build [--data-dir DIR] [--output-dir DIR]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features_build.py
import pandas as pd

from features.build import build_feature_table
from fraudguard_core.schemas import features_schema


def _fixture():
    t0 = pd.Timestamp("2026-01-01T00:00:00")
    transactions = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "customer_id": ["C1", "C1", "C2"],
            "merchant_id": ["M1", "M1", "M2"],
            "device_id": ["D1", "D1", "D2"],
            "timestamp": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)],
            "amount": [10.0, 20.0, 30.0],
            "currency": ["USD", "USD", "USD"],
            "payment_method": ["card", "card", "wallet"],
            "country": ["US", "US", "GB"],
            "city": ["New York", "New York", "London"],
            "ip_country": ["US", "US", "GB"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "account_created_at": [t0 - pd.Timedelta(days=100)] * 2,
            "customer_segment": ["retail", "retail"],
            "home_country": ["US", "GB"],
            "home_city": ["New York", "London"],
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": ["D1", "D2"],
            "customer_id": ["C1", "C2"],
            "device_type": ["mobile", "mobile"],
            "first_seen_at": [t0 - pd.Timedelta(days=10), t0 - pd.Timedelta(days=5)],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2", "TXN3"],
            "fraud_probability_true": [0.1, 0.1, 0.1],
            "fraud_label": [0, 0, 0],
            "confirmed_fraud_at": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    return transactions, customers, devices, ground_truth


def test_build_feature_table_matches_schema():
    transactions, customers, devices, ground_truth = _fixture()
    result = build_feature_table(transactions, customers, devices, ground_truth)
    features_schema.validate(result)
    assert len(result) == 3


def test_build_feature_table_is_deterministic():
    transactions, customers, devices, ground_truth = _fixture()
    r1 = build_feature_table(transactions, customers, devices, ground_truth)
    r2 = build_feature_table(transactions, customers, devices, ground_truth)
    pd.testing.assert_frame_equal(
        r1.sort_values("transaction_id").reset_index(drop=True),
        r2.sort_values("transaction_id").reset_index(drop=True),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.build'`

- [ ] **Step 3: Write the implementation**

```python
# features/build.py
"""Orchestrates all six feature families into one gold feature table."""

import os

import pandas as pd

from fraudguard_core.schemas import features_schema

from features.behavioral import build_behavioral_features
from features.device import build_device_features
from features.geo import build_geo_features
from features.merchant import build_merchant_features
from features.transaction import build_transaction_features
from features.velocity import build_velocity_features


def build_feature_table(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    devices: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    base = transactions[["transaction_id", "customer_id", "merchant_id", "timestamp"]].set_index(
        "transaction_id"
    )

    parts = [
        build_transaction_features(transactions, customers),
        build_behavioral_features(transactions),
        build_velocity_features(transactions),
        build_device_features(transactions, devices),
        build_merchant_features(transactions, ground_truth),
        build_geo_features(transactions),
    ]

    result = base
    for part in parts:
        result = result.join(part, how="left")

    result = result.reset_index()
    features_schema.validate(result)
    return result


def write_features(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "features.parquet")
    df.to_parquet(path, index=False)
    return path
```

```python
# features/__main__.py
"""CLI: python -m features build [--data-dir DIR] [--output-dir DIR]"""

import argparse
import os

import pandas as pd

from features.build import build_feature_table, write_features


def _load(data_dir: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="features")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="Build the gold feature table")
    build_cmd.add_argument("--data-dir", type=str, default="./data")
    build_cmd.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.command == "build":
        output_dir = args.output_dir or args.data_dir
        transactions = _load(args.data_dir, "transactions")
        customers = _load(args.data_dir, "customers")
        devices = _load(args.data_dir, "devices")
        ground_truth = _load(args.data_dir, "ground_truth")

        table = build_feature_table(transactions, customers, devices, ground_truth)
        path = write_features(table, output_dir)
        print(f"Wrote {len(table)} rows to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features_build.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the `features` Makefile target**

Modify `Makefile`:

```makefile
.PHONY: venv seed dq test lint features

features:
	python -m features build
```

(Add `features` to the existing `.PHONY` line rather than duplicating it, and add the new `features:` target after the existing `dq:` target.)

- [ ] **Step 6: Verify the CLI works end to end against real seeded data**

Run: `make seed` (if `./data/*.parquet` isn't already present), then `make features`
Expected: prints `Wrote <N> rows to ./data/features.parquet` where N matches the transaction count from `make seed`'s output (~500k+ by default)

- [ ] **Step 7: Commit**

```bash
git add features/build.py features/__main__.py Makefile tests/unit/test_features_build.py
git commit -m "feat(features): add build orchestration, CLI, and make features target"
```

---

### Task 9: Real-data leakage suite + `docs/FEATURE_DEFINITIONS.md`

**Files:**
- Create: `tests/features/__init__.py` (empty)
- Create: `tests/features/test_leakage.py`
- Create: `docs/FEATURE_DEFINITIONS.md`

**Interfaces:**
- Consumes: `fraudguard_core.schemas.features_schema` (Task 1) and the committed `./data/features.parquet` produced by `make features` (Task 8).
- Produces: nothing consumed by later tasks — this is the acceptance-facing leakage gate, mirroring `tests/dq/test_dq_suite.py`'s pattern of loading real generated output rather than regenerating it in-test.

- [ ] **Step 1: Write the test**

```python
# tests/features/__init__.py
```

```python
# tests/features/test_leakage.py
"""Leakage + schema gate against the real generated ./data/features.parquet.

Run `make seed features` first (or `make seed && make features`). Mirrors
tests/dq/test_dq_suite.py: these tests validate committed output, they don't
regenerate it.
"""

import os

import pandas as pd
import pytest
from fraudguard_core.schemas import features_schema

DATA_DIR = os.environ.get("FRAUDGUARD_DATA_DIR", "./data")

FORBIDDEN_COLUMNS = (
    "fraud_label",
    "fraud_probability_true",
    "confirmed_fraud_at",
    "base_fraud_rate",
)


def _load_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "features.parquet")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found -- run `make seed features` first")
    return pd.read_parquet(path)


def test_features_schema():
    features_schema.validate(_load_features())


def test_no_forbidden_columns_in_features():
    df = _load_features()
    for col in FORBIDDEN_COLUMNS:
        assert col not in df.columns, f"forbidden column {col!r} leaked into features.parquet"


def test_every_transaction_has_a_feature_row():
    transactions = pd.read_parquet(os.path.join(DATA_DIR, "transactions.parquet"))
    features = _load_features()
    assert set(features["transaction_id"]) == set(transactions["transaction_id"])
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/features -v`
Expected: PASS if `make seed features` has already been run in this checkout (it should have been, from Task 8 Step 6); otherwise SKIPPED with the message above — either outcome is correct at this point, just confirm it's not an ERROR (which would mean an import/syntax problem).

- [ ] **Step 3: Write `docs/FEATURE_DEFINITIONS.md`**

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add tests/features/__init__.py tests/features/test_leakage.py docs/FEATURE_DEFINITIONS.md
git commit -m "test(features): add real-data leakage gate and feature definitions doc"
```

---

### Task 10: EDA notebook + `docs/EDA_INSIGHTS.md`

**Files:**
- Modify: `requirements-dev.txt`
- Create: `notebooks/01-eda.ipynb`
- Create: `docs/EDA_INSIGHTS.md`

**Interfaces:**
- Consumes: `./data/{transactions,customers,merchants,devices,ground_truth}.parquet` (read directly with `pd.read_parquet`, joined only inside the notebook — never inside `features/`).
- Produces: nothing consumed by other tasks; this is a documentation/analysis deliverable required by the spec's success criterion 5.

- [ ] **Step 1: Add notebook dependencies**

Add to `requirements-dev.txt`:

```
jupyter>=1.0
nbconvert>=7.0
matplotlib>=3.8
```

Run: `.venv/Scripts/pip install -r requirements-dev.txt`
Expected: installs without error

- [ ] **Step 2: Create `notebooks/01-eda.ipynb`**

Create the file with exactly this content (valid nbformat 4 JSON):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# FraudGuard — Sub-project 2 EDA\n",
    "\n",
    "Explores the raw synthetic data from sub-project 1 (`../data/*.parquet`).\n",
    "`ground_truth` is joined here **for analysis only** — the `features/` package\n",
    "never does this join outside the point-in-time merchant-risk feature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "DATA_DIR = \"../data\"\n",
    "\n",
    "transactions = pd.read_parquet(f\"{DATA_DIR}/transactions.parquet\")\n",
    "customers = pd.read_parquet(f\"{DATA_DIR}/customers.parquet\")\n",
    "merchants = pd.read_parquet(f\"{DATA_DIR}/merchants.parquet\")\n",
    "devices = pd.read_parquet(f\"{DATA_DIR}/devices.parquet\")\n",
    "ground_truth = pd.read_parquet(f\"{DATA_DIR}/ground_truth.parquet\")\n",
    "\n",
    "df = transactions.merge(ground_truth[[\"transaction_id\", \"fraud_label\"]], on=\"transaction_id\")\n",
    "df = df.merge(merchants[[\"merchant_id\", \"merchant_category\"]], on=\"merchant_id\")\n",
    "df = df.merge(customers[[\"customer_id\", \"home_country\"]], on=\"customer_id\")\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Fraud prevalence"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "prevalence = df[\"fraud_label\"].mean()\n",
    "print(f\"Fraud prevalence: {prevalence:.4%} ({df['fraud_label'].sum():,} of {len(df):,} transactions)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Amount distribution: fraud vs. legitimate"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "df[df[\"fraud_label\"] == 0][\"amount\"].clip(upper=1000).hist(bins=50, alpha=0.6, label=\"legit\", ax=ax)\n",
    "df[df[\"fraud_label\"] == 1][\"amount\"].clip(upper=1000).hist(bins=50, alpha=0.6, label=\"fraud\", ax=ax)\n",
    "ax.set_xlabel(\"amount (clipped at 1000)\")\n",
    "ax.legend()\n",
    "plt.show()\n",
    "\n",
    "df.groupby(\"fraud_label\")[\"amount\"].median()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Fraud rate by hour of day"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"hour\"] = df[\"timestamp\"].dt.hour\n",
    "hourly_rate = df.groupby(\"hour\")[\"fraud_label\"].mean().sort_values(ascending=False)\n",
    "hourly_rate.head(6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Fraud rate by merchant category"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "category_rate = df.groupby(\"merchant_category\")[\"fraud_label\"].mean().sort_values(ascending=False)\n",
    "category_rate"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## New-device and IP-mismatch signals"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "device_first_seen = devices.set_index(\"device_id\")[\"first_seen_at\"]\n",
    "df[\"is_new_device_txn\"] = (\n",
    "    (df[\"timestamp\"] - df[\"device_id\"].map(device_first_seen)).dt.total_seconds().abs() < 3600\n",
    ").astype(int)\n",
    "df[\"ip_mismatch\"] = (df[\"country\"] != df[\"ip_country\"]).astype(int)\n",
    "\n",
    "for col in [\"is_new_device_txn\", \"ip_mismatch\"]:\n",
    "    flagged = df[df[col] == 1][\"fraud_label\"].mean()\n",
    "    not_flagged = df[df[col] == 0][\"fraud_label\"].mean()\n",
    "    print(f\"{col}: flagged={flagged:.4%} vs not_flagged={not_flagged:.4%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Cross-border transactions"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"is_cross_border\"] = (df[\"country\"] != df[\"home_country\"]).astype(int)\n",
    "df.groupby(\"is_cross_border\")[\"fraud_label\"].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary: business insights\n",
    "\n",
    "Fill this cell in after running every cell above, using the actual printed\n",
    "numbers (do not leave the placeholders below unfilled):\n",
    "\n",
    "1. Overall fraud prevalence is **_% (_ of _ transactions).\n",
    "2. Fraud transactions have a **_x** higher/lower median amount than legitimate ones ($__ vs $__).\n",
    "3. The riskiest hour of day is **__:00**, at **_%** fraud rate vs the overall **_%**.\n",
    "4. The riskiest merchant category is **__**, at **_%** fraud rate.\n",
    "5. New-device transactions are **_x** more likely to be fraud than established-device transactions (**_%** vs **_%**).\n",
    "6. Cross-border transactions are **_x** more likely to be fraud than domestic ones (**_%** vs **_%**)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Run the notebook top to bottom**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/01-eda.ipynb`
Expected: completes with no errors, and `notebooks/01-eda.ipynb` is rewritten in place with real outputs

- [ ] **Step 4: Fill in the summary cell with real numbers**

Open the executed notebook, read the printed output of each cell, and edit the final markdown cell's placeholders (`_%`, `$__`, `__:00`, etc.) with the actual observed values. Save the notebook again.

- [ ] **Step 5: Write `docs/EDA_INSIGHTS.md`**

Copy the final, filled-in "Summary: business insights" numbered list from the notebook into a new file:

```markdown
# EDA Insights — Sub-project 2

Numbered business insights from `notebooks/01-eda.ipynb`, copied here so
they're readable without opening the notebook. Regenerate by re-running the
notebook top to bottom against a fresh `make seed` and updating both files.

[paste the same numbered list you wrote into the notebook's final cell here]
```

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt notebooks/01-eda.ipynb docs/EDA_INSIGHTS.md
git commit -m "docs(eda): add sub-project 2 EDA notebook and business insights"
```

---

### Task 11: End-to-end run + `docs/sub2-acceptance.md`

**Files:**
- Create: `docs/sub2-acceptance.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: every prior task's deliverable, run together as the full pipeline.

- [ ] **Step 1: Run the full pipeline from a clean state**

Run: `make seed && make features && pytest tests/unit tests/dq tests/features -v`
Expected: all tests pass (no skips in `tests/features`, since `data/features.parquet` now exists); note the printed row count from `make features` and the full pytest summary line (e.g. `41 passed`)

- [ ] **Step 2: Write `docs/sub2-acceptance.md`**

```markdown
# Sub-project 2 — Acceptance Run

**Date:** [fill in actual run date]
**Command:** `make seed && make features && pytest tests/unit tests/dq tests/features -v`

## Output

- Feature table row count: [fill in from `make features` output] (must equal the transaction count from `make seed`)
- Test summary: [fill in the final pytest summary line, e.g. "41 passed"]

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
```

- [ ] **Step 3: Update `README.md` status**

Modify the `## Status` section of `README.md`:

```markdown
## Status

**Sub-project 1 (Data Foundation) — done.** See
`docs/superpowers/specs/2026-09-12-data-foundation-design.md` and
`docs/sub1-acceptance.md`.

**Sub-project 2 (EDA + Feature Engineering) — done.** See
`docs/superpowers/specs/2026-09-12-feature-engineering-design.md` and
`docs/sub2-acceptance.md`.
```

Also update the `## Quickstart` code block to add the two new commands after the existing `pytest tests/dq -v` line:

```
make features                     # writes ./data/features.parquet
pytest tests/features -v          # leakage + schema gate on the feature table
```

- [ ] **Step 4: Commit**

```bash
git add docs/sub2-acceptance.md README.md
git commit -m "docs: record sub-project 2 acceptance run"
```
