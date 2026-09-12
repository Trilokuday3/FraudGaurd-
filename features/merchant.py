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

    # A transaction's own fraud confirmation must never count toward its own
    # rate. This only matters when the confirmation delay is zero
    # (confirmed_fraud_at == timestamp); the generator never produces a
    # negative delay, so no other row can be affected by this correction.
    fraud_confirmed_by_txn = fraud.set_index("transaction_id")["confirmed_fraud_at"]
    own_confirmation = fraud_confirmed_by_txn.reindex(df["transaction_id"]).to_numpy()
    self_leak = own_confirmation <= df["timestamp"].to_numpy()

    rates = np.empty(len(df))

    for merchant_id, group in df.groupby("merchant_id"):
        idx = group.index.to_numpy()
        order = np.argsort(group["timestamp"].to_numpy())
        sorted_times = group["timestamp"].to_numpy()[order]

        sorted_self_leak = self_leak[idx][order]

        merchant_fraud_times = np.sort(
            fraud.loc[fraud["merchant_id"] == merchant_id, "confirmed_fraud_at"].to_numpy()
        )

        denom = np.searchsorted(sorted_times, sorted_times, side="left")
        numer = np.searchsorted(merchant_fraud_times, sorted_times, side="right") - sorted_self_leak
        merchant_rate = numer / np.maximum(denom, 1)

        global_denom = np.searchsorted(all_txn_times, sorted_times, side="left")
        global_numer = (
            np.searchsorted(all_fraud_times, sorted_times, side="right") - sorted_self_leak
        )
        global_rate = global_numer / np.maximum(global_denom, 1)

        rate = np.where(denom >= min_history, merchant_rate, global_rate)
        # Defensive guard: an exact-timestamp tie between a transaction and a
        # fraud confirmation could in principle push a rate fractionally
        # outside [0, 1] due to the self-leak subtraction above. Unreachable
        # on real data today (max observed rate is 0.636) but clip anyway so
        # a future edge case fails as a clean value, not an obscure
        # downstream pandera range-check error.
        rate = np.clip(rate, 0.0, 1.0)
        rates[idx[order]] = rate

    result = df.copy()
    result["merchant_fraud_rate_hist"] = rates
    return result.set_index("transaction_id")[["merchant_fraud_rate_hist"]]
