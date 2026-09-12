# Data Generation Model — Sub-project 1

How `generator/generate.py` decides which transactions are fraud. Kept
separate from the data dictionary because this is process/methodology, not
schema.

## Why a latent-risk model instead of hand-picked "if X then fraud" rules

A rule-based label (e.g. "flag if amount > $500 and new device") would let
sub-project 3's model simply re-discover the rule and score a suspicious
100% PR-AUC — worthless as a portfolio signal, because it wouldn't
demonstrate anything about handling noisy, imperfectly-separable real fraud.
Instead, each transaction gets a **latent risk score** built from several
weighted signals plus Gaussian noise, then a **probability**, then a
**Bernoulli draw** — so even a "high risk" transaction is usually legitimate,
and the champion model in sub-project 3 has to actually earn its precision by
combining weak signals, the same way a real fraud model does.

## The formula

For transaction `i`:

```
z_i = b0
    + w1 · new_device_i          (1 if this is the first-ever use of this device)
    + w2 · new_country_i         (1 if this is the customer's first transaction in this country)
    + w3 · ip_mismatch_i         (1 if ip_country != country)
    + w4 · zscore(amount_ratio)_i    (amount ÷ this customer's trailing p95 amount, computed causally)
    + w5 · zscore(velocity_1h)_i     (count of this customer's txns in the trailing 1h, computed causally)
    + w6 · zscore(merchant_risk)_i   (latent per-merchant fraud propensity, never exposed as a column)
    + w7 · night_hour_i          (1 if local hour is 1–4am)
    − w8 · zscore(log(account_age_days + 1))_i
    + ε_i,   ε_i ~ Normal(0, σ)

fraud_probability_true_i = sigmoid(z_i)
fraud_label_i ~ Bernoulli(fraud_probability_true_i)
```

`zscore(x)` standardizes `x` across the whole generated population (mean 0,
std 1) so the weights below are comparable to each other regardless of each
raw signal's native scale.

## Values used for the checked-in dataset (seed=42, default config)

| weight | value | signal |
|---|---|---|
| w1 (new device) | 1.1 | strongest single binary signal — matches real card-fraud patterns (new-device fraud is a well-known strong indicator) |
| w2 (new country) | 1.3 | strongest overall — geography mismatch is one of the highest-precision fraud signals in practice |
| w3 (ip mismatch) | 0.9 | proxy/VPN indicator |
| w4 (amount ratio) | 0.8 | spending-pattern deviation |
| w5 (velocity 1h) | 0.7 | card-testing / rapid-fire pattern |
| w6 (merchant risk) | 1.0 | category+country merchant prior (crypto/gaming/electronics weighted higher — see `enums.MERCHANT_CATEGORY_RISK_MULTIPLIER`) |
| w7 (night hour) | 0.5 | weak standalone signal, realistic — most fraud does *not* happen at night, this just nudges it slightly |
| w8 (account age, subtracted) | 0.6 | newer accounts are riskier |
| σ (noise) | 0.6 | irreducible noise — keeps the problem genuinely hard, not perfectly separable |
| b0 (intercept) | **-6.339** | solved by bisection so the population mean of `fraud_probability_true` hits the target rate |

**Realised on the checked-in run:** target fraud rate 1.5000%, realised
1.5067% (519,876 transactions, 20,000 customers, 2,000 merchants, 31,481
devices) — within the ±0.3pp success criterion.

## Causal computation of `amount_ratio` and `velocity_1h`

Both are computed inside a single **chronological pass per customer**: when
generating transaction `k` for a customer, only transactions `1..k-1` for
that same customer are visible. `amount_ratio` divides the current amount by
that customer's trailing 95th-percentile amount *so far*; `velocity_1h`
counts that customer's own transactions in the hour *before* the current
one. Neither ever looks at a transaction that hasn't "happened yet" in
generated time.

This matters beyond generating a realistic label: it is the **same
point-in-time discipline sub-project 2's real feature-engineering code must
follow**. The generator is a worked, testable example of the leakage rule
(`tests/unit/test_generator.py` asserts determinism and referential
integrity; the DQ suite asserts no forbidden columns leak into the feature
path), not just a policy stated in a doc.

## Leakage guards

- `merchant_risk` (the per-merchant latent fraud propensity) is used to
  *generate* the label but is **never serialized** to `merchants.parquet`.
  Any merchant-risk feature sub-project 2 wants must be estimated from
  historical `ground_truth`-joined data, exactly as a real system would have
  to learn it from confirmed-fraud history rather than reading a hidden
  oracle column.
- `fraud_probability_true`, `fraud_label`, and `confirmed_fraud_at` live only
  in `ground_truth.parquet`, joined to `transactions` by `transaction_id` and
  never merged into anything that looks like a feature table.
- `confirmed_fraud_at` models real label delay (0–14 days after the
  transaction) specifically so sub-project 2/3 have to reckon with "the label
  wasn't known yet at scoring time" the way the brief's §3 requires.

## Reproducing this

```
python -m generator seed             # writes ./data/*.parquet + generation_meta.json
pytest tests/unit -v                 # determinism, referential integrity, leakage guards
pytest tests/dq -v                   # schema + cross-entity validation on ./data
```

Every run's exact weights, seed, config, and realised rate are written to
`data/generation_meta.json` (gitignored — regenerate rather than diff).
