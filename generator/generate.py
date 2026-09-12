"""Core generation logic.

Generation order: merchants -> customers -> (per customer, chronologically)
devices + transactions -> population-level z-score pass -> ground_truth.

The per-customer transaction loop is intentionally causal: amount_vs_p95 and
velocity_1h for a given transaction only ever look at that same customer's
*earlier* transactions. This is the same point-in-time discipline
sub-project 2's real features must follow -- the generator is a worked
example of the leakage rule, not just an assertion of it. See
docs/superpowers/specs/2026-09-12-data-foundation-design.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fraudguard_core.config import GeneratorSettings
from fraudguard_core.enums import (
    CITIES_BY_COUNTRY,
    COUNTRIES,
    COUNTRY_WEIGHTS,
    MERCHANT_CATEGORY,
    MERCHANT_CATEGORY_RISK_MULTIPLIER,
)

COUNTRY_LIST = sorted(COUNTRIES)
COUNTRY_CURRENCY = {
    "US": "USD",
    "CA": "USD",
    "AU": "USD",
    "SG": "USD",
    "GB": "GBP",
    "IN": "INR",
    "DE": "EUR",
}
SEGMENT_WEIGHTS = {"retail": 0.7, "premium": 0.2, "business": 0.1}
SEGMENT_AMOUNT_MU = {"retail": 3.5, "premium": 4.3, "business": 4.8}
DEVICE_TYPE_WEIGHTS = {"mobile": 0.6, "desktop": 0.3, "tablet": 0.1}
PAYMENT_METHOD_WEIGHTS = {"card": 0.7, "wallet": 0.2, "bank_transfer": 0.1}

# Latent-risk model weights (see design doc). Written to generation_meta.json
# so every run is auditable.
WEIGHTS = {
    "w1_new_device": 1.1,
    "w2_new_country": 1.3,
    "w3_ip_mismatch": 0.9,
    "w4_amount_ratio": 0.8,
    "w5_velocity_1h": 0.7,
    "w6_merchant_risk": 1.0,
    "w7_night_hour": 0.5,
    "w8_account_age": 0.6,
    "sigma_noise": 0.6,
}


@dataclass
class GenerationResult:
    customers: pd.DataFrame
    merchants: pd.DataFrame
    devices: pd.DataFrame
    transactions: pd.DataFrame
    ground_truth: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _country_probs() -> np.ndarray:
    p = np.array([COUNTRY_WEIGHTS[c] for c in COUNTRY_LIST])
    return p / p.sum()


def _generate_merchants(rng: np.random.Generator, n_m: int, vintage_start: pd.Timestamp):
    merchant_ids = [f"MERC{idx:06d}" for idx in range(n_m)]
    categories = rng.choice(sorted(MERCHANT_CATEGORY), size=n_m)
    countries = rng.choice(COUNTRY_LIST, size=n_m, p=_country_probs())
    created_at = vintage_start - pd.to_timedelta(rng.integers(30, 365 * 3, size=n_m), unit="D")

    category_mult = np.array([MERCHANT_CATEGORY_RISK_MULTIPLIER[c] for c in categories])
    noise = rng.lognormal(mean=0.0, sigma=0.3, size=n_m)
    base_fraud_rate = 0.01 * category_mult * noise  # internal only, never serialized

    merchants_df = pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "merchant_category": categories,
            "country": countries,
            "created_at": created_at,
        }
    )
    return merchants_df, base_fraud_rate


def _generate_customers(rng: np.random.Generator, n_c: int, vintage_start, vintage_end):
    customer_ids = [f"CUST{idx:07d}" for idx in range(n_c)]
    segments = rng.choice(list(SEGMENT_WEIGHTS), size=n_c, p=list(SEGMENT_WEIGHTS.values()))
    home_countries = rng.choice(COUNTRY_LIST, size=n_c, p=_country_probs())
    home_cities = [rng.choice(CITIES_BY_COUNTRY[c]) for c in home_countries]

    earliest = vintage_start - pd.Timedelta(days=365 * 6)
    span_days = max((vintage_end - earliest).days, 1)
    created_days = rng.integers(0, span_days, size=n_c)
    account_created_at = earliest + pd.to_timedelta(created_days, unit="D")

    risk_appetite = rng.lognormal(mean=0.0, sigma=0.4, size=n_c)  # internal spending scale

    customers_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "account_created_at": account_created_at,
            "customer_segment": segments,
            "home_country": home_countries,
            "home_city": home_cities,
        }
    )
    return customers_df, risk_appetite


def _generate_transactions_and_devices(
    rng: np.random.Generator,
    customers_df: pd.DataFrame,
    risk_appetite: np.ndarray,
    merchants_df: pd.DataFrame,
    merchant_base_fraud_rate: np.ndarray,
    config: GeneratorSettings,
    vintage_start: pd.Timestamp,
    vintage_end: pd.Timestamp,
):
    n_m = len(merchants_df)
    merchant_ids = merchants_df["merchant_id"].to_numpy()

    device_rows: list[dict] = []
    txn_rows: list[dict] = []
    raw_components: list[dict] = []
    dev_counter = 0
    txn_counter = 0

    for i, row in enumerate(customers_df.itertuples(index=False)):
        cust_id = row.customer_id
        seg = row.customer_segment
        home_c = row.home_country
        acc_created = row.account_created_at
        appetite = risk_appetite[i]

        n_dev = max(1, int(rng.poisson(config.avg_devices_per_customer)))
        dev_ids_for_cust = [f"DEV{dev_counter + j:07d}" for j in range(n_dev)]
        dev_counter += n_dev

        n_txn = max(1, int(rng.poisson(config.avg_txns_per_customer)))
        start = max(acc_created, vintage_start)
        if start >= vintage_end:
            continue
        span_seconds = (vintage_end - start).total_seconds()
        offsets = np.sort(rng.random(n_txn) * span_seconds)
        timestamps = [start + pd.Timedelta(seconds=float(s)) for s in offsets]

        device_first_seen: list[pd.Timestamp | None] = [None] * n_dev
        amounts_hist: list[float] = []
        recent_times: list[pd.Timestamp] = []
        seen_countries = {home_c}

        for t in timestamps:
            dev_idx = (
                0 if n_dev == 1 else (0 if rng.random() < 0.85 else int(rng.integers(1, n_dev)))
            )
            is_new_device = device_first_seen[dev_idx] is None
            if is_new_device:
                device_first_seen[dev_idx] = t

            m_idx = int(rng.integers(0, n_m))

            is_new_country = False
            if rng.random() < 0.03:
                candidates = [c for c in COUNTRY_LIST if c != home_c]
                country = rng.choice(candidates)
                if country not in seen_countries:
                    is_new_country = True
                seen_countries.add(country)
            else:
                country = home_c

            ip_mismatch = 1 if rng.random() < 0.02 else 0
            ip_country = country if not ip_mismatch else rng.choice(COUNTRY_LIST)

            amount = float(rng.lognormal(mean=SEGMENT_AMOUNT_MU[seg], sigma=0.7)) * appetite
            if rng.random() < 0.01:
                amount *= float(rng.uniform(5, 15))

            if len(amounts_hist) >= 5:
                p95 = float(np.percentile(amounts_hist, 95))
                amount_ratio = amount / (p95 + 1e-6)
            else:
                amount_ratio = 1.0
            amounts_hist.append(amount)

            cutoff = t - pd.Timedelta(hours=1)
            recent_times = [rt for rt in recent_times if rt >= cutoff]
            velocity_1h = len(recent_times)
            recent_times.append(t)

            night = 1 if t.hour in (1, 2, 3, 4) else 0
            account_age_days = max((t - acc_created).days, 0)
            merchant_risk = merchant_base_fraud_rate[m_idx]

            txn_id = f"TXN{txn_counter:010d}"
            txn_counter += 1
            city = rng.choice(CITIES_BY_COUNTRY[country])

            txn_rows.append(
                {
                    "transaction_id": txn_id,
                    "customer_id": cust_id,
                    "merchant_id": merchant_ids[m_idx],
                    "device_id": dev_ids_for_cust[dev_idx],
                    "timestamp": t,
                    "amount": round(amount, 2),
                    "currency": COUNTRY_CURRENCY[country],
                    "payment_method": rng.choice(
                        list(PAYMENT_METHOD_WEIGHTS), p=list(PAYMENT_METHOD_WEIGHTS.values())
                    ),
                    "country": country,
                    "city": city,
                    "ip_country": ip_country,
                }
            )
            raw_components.append(
                {
                    "transaction_id": txn_id,
                    "new_device": int(is_new_device),
                    "new_country": int(is_new_country),
                    "ip_mismatch": ip_mismatch,
                    "amount_ratio": amount_ratio,
                    "velocity_1h": velocity_1h,
                    "merchant_risk": merchant_risk,
                    "night": night,
                    "account_age_days": account_age_days,
                    "timestamp": t,
                }
            )

        for d_idx in range(n_dev):
            first_seen = device_first_seen[d_idx] or acc_created
            device_rows.append(
                {
                    "device_id": dev_ids_for_cust[d_idx],
                    "customer_id": cust_id,
                    "device_type": rng.choice(
                        list(DEVICE_TYPE_WEIGHTS), p=list(DEVICE_TYPE_WEIGHTS.values())
                    ),
                    "first_seen_at": first_seen,
                }
            )

    transactions_df = pd.DataFrame(txn_rows)
    devices_df = pd.DataFrame(device_rows)
    components_df = pd.DataFrame(raw_components)
    return transactions_df, devices_df, components_df


def _solve_b0_and_label(
    rng: np.random.Generator, components_df: pd.DataFrame, target_rate: float, sigma: float
) -> tuple[pd.DataFrame, float]:
    log_age = np.log1p(components_df["account_age_days"].to_numpy())

    def zscore(x: np.ndarray) -> np.ndarray:
        mu, sd = x.mean(), x.std()
        sd = sd if sd > 1e-9 else 1.0
        return (x - mu) / sd

    z_amount = zscore(components_df["amount_ratio"].to_numpy())
    z_velocity = zscore(components_df["velocity_1h"].to_numpy())
    z_merchant = zscore(components_df["merchant_risk"].to_numpy())
    z_age = zscore(log_age)

    linear = (
        WEIGHTS["w1_new_device"] * components_df["new_device"].to_numpy()
        + WEIGHTS["w2_new_country"] * components_df["new_country"].to_numpy()
        + WEIGHTS["w3_ip_mismatch"] * components_df["ip_mismatch"].to_numpy()
        + WEIGHTS["w4_amount_ratio"] * z_amount
        + WEIGHTS["w5_velocity_1h"] * z_velocity
        + WEIGHTS["w6_merchant_risk"] * z_merchant
        + WEIGHTS["w7_night_hour"] * components_df["night"].to_numpy()
        - WEIGHTS["w8_account_age"] * z_age
    )
    noise = rng.normal(0, sigma, size=len(linear))
    linear_plus_noise = linear + noise

    def mean_fraud_rate(b0: float) -> float:
        z = b0 + linear_plus_noise
        return float(np.mean(1.0 / (1.0 + np.exp(-z))))

    lo, hi = -15.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if mean_fraud_rate(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    b0 = (lo + hi) / 2

    z_final = b0 + linear_plus_noise
    fraud_probability_true = 1.0 / (1.0 + np.exp(-z_final))
    fraud_label = rng.binomial(1, fraud_probability_true)

    confirmed_fraud_at = pd.Series(pd.NaT, index=components_df.index, dtype="datetime64[ns]")
    fraud_idx = np.where(fraud_label == 1)[0]
    if len(fraud_idx) > 0:
        delay_days = rng.integers(0, 15, size=len(fraud_idx))
        confirmed_fraud_at.iloc[fraud_idx] = [
            components_df["timestamp"].iloc[idx] + pd.Timedelta(days=int(d))
            for idx, d in zip(fraud_idx, delay_days)
        ]

    ground_truth_df = pd.DataFrame(
        {
            "transaction_id": components_df["transaction_id"].to_numpy(),
            "fraud_probability_true": fraud_probability_true,
            "fraud_label": fraud_label.astype(int),
            "confirmed_fraud_at": confirmed_fraud_at.to_numpy(),
        }
    )
    realised_rate = float(ground_truth_df["fraud_label"].mean())
    return ground_truth_df, b0, realised_rate


def generate(config: GeneratorSettings) -> GenerationResult:
    rng = np.random.default_rng(config.seed)
    vintage_start = pd.Timestamp(config.vintage_start)
    vintage_end = pd.Timestamp(config.vintage_end)

    merchants_df, merchant_base_fraud_rate = _generate_merchants(
        rng, config.n_merchants, vintage_start
    )
    customers_df, risk_appetite = _generate_customers(
        rng, config.n_customers, vintage_start, vintage_end
    )

    transactions_df, devices_df, components_df = _generate_transactions_and_devices(
        rng,
        customers_df,
        risk_appetite,
        merchants_df,
        merchant_base_fraud_rate,
        config,
        vintage_start,
        vintage_end,
    )

    ground_truth_df, b0, realised_rate = _solve_b0_and_label(
        rng, components_df, config.target_fraud_rate, WEIGHTS["sigma_noise"]
    )

    meta = {
        "seed": config.seed,
        "config": config.model_dump(),
        "weights": WEIGHTS,
        "b0": b0,
        "target_fraud_rate": config.target_fraud_rate,
        "realised_fraud_rate": realised_rate,
        "row_counts": {
            "customers": len(customers_df),
            "merchants": len(merchants_df),
            "devices": len(devices_df),
            "transactions": len(transactions_df),
        },
    }

    return GenerationResult(
        customers=customers_df,
        merchants=merchants_df,
        devices=devices_df,
        transactions=transactions_df,
        ground_truth=ground_truth_df,
        meta=meta,
    )


def write_output(result: GenerationResult, output_dir: str) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    result.customers.to_parquet(os.path.join(output_dir, "customers.parquet"), index=False)
    result.merchants.to_parquet(os.path.join(output_dir, "merchants.parquet"), index=False)
    result.devices.to_parquet(os.path.join(output_dir, "devices.parquet"), index=False)
    result.transactions.to_parquet(os.path.join(output_dir, "transactions.parquet"), index=False)
    result.ground_truth.to_parquet(os.path.join(output_dir, "ground_truth.parquet"), index=False)
    with open(os.path.join(output_dir, "generation_meta.json"), "w") as f:
        json.dump(result.meta, f, indent=2, default=str)
