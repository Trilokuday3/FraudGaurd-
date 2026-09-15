"""Hard rules that can escalate (or, for the allowlist, override) a
decision regardless of the model's score. Evaluated purely against columns
already present in the gold feature table -- no new feature engineering."""

DECISION_ORDER: dict[str, int] = {"approve": 0, "review": 1, "block": 2}

# Illustrative example entries so the allowlist override path is reachable
# and demonstrable in the deployed service (this project has no real
# customer/merchant list to draw from) — same spirit as the illustrative
# cost matrix in decision/select_thresholds.py. Hyphenated, non-numeric
# suffixes so these can never collide with the generator's zero-padded
# numeric ID formats (customer_id: CUST{idx:07d}, merchant_id:
# MERC{idx:06d}), regardless of how many digits those formats use.
ALLOWLIST_CUSTOMER_IDS: frozenset[str] = frozenset({"CUST-EXAMPLE-ALLOWLISTED"})
ALLOWLIST_MERCHANT_IDS: frozenset[str] = frozenset({"MERC-EXAMPLE-ALLOWLISTED"})


def evaluate_rules(
    feature_row: dict,
    allowlist_customer_ids: frozenset[str] = ALLOWLIST_CUSTOMER_IDS,
    allowlist_merchant_ids: frozenset[str] = ALLOWLIST_MERCHANT_IDS,
) -> tuple[str, list[str], bool]:
    """Evaluate the escalation rules and the allowlist against one feature row.

    Returns
    -------
    tuple[str, list[str], bool]
        (rule_decision, triggered_rule_names, is_allowlisted) -- rule_decision
        is the highest escalation level any non-allowlist rule triggered
        ("approve" if none fired); triggered_rule_names lists every rule
        (including the allowlist, if matched) that fired; is_allowlisted is
        True if the allowlist rule matched.
    """
    triggered: list[str] = []
    decision = "approve"

    if (
        feature_row["is_new_device"]
        and feature_row["is_new_country_for_customer"]
        and feature_row["ip_country_mismatch"]
    ):
        triggered.append("device_location_takeover")
        decision = "block"

    if feature_row["txn_count_1h"] >= 5:
        triggered.append("velocity_spike")
        decision = "block"

    if feature_row["merchant_fraud_rate_hist"] > 0.15:
        triggered.append("high_risk_merchant")
        if DECISION_ORDER["review"] > DECISION_ORDER[decision]:
            decision = "review"

    if feature_row["amount_vs_customer_p95"] > 3.0 and feature_row["is_new_device"]:
        triggered.append("outsized_amount_new_device")
        if DECISION_ORDER["review"] > DECISION_ORDER[decision]:
            decision = "review"

    is_allowlisted = (
        feature_row["customer_id"] in allowlist_customer_ids
        or feature_row["merchant_id"] in allowlist_merchant_ids
    )
    if is_allowlisted:
        triggered.append("trusted_allowlist")

    return decision, triggered, is_allowlisted


def combine_decision(
    model_score: float,
    t_review: float,
    t_block: float,
    rule_decision: str,
    triggered_rules: list[str],
    is_allowlisted: bool,
) -> tuple[str, str]:
    """Combine the model's score-based decision with the rules layer.

    Returns
    -------
    tuple[str, str]
        (final_decision, decision_source) -- decision_source is "model",
        "rule", or "rule_override" (the allowlist case).
    """
    if is_allowlisted:
        return "approve", "rule_override"

    if model_score >= t_block:
        model_decision = "block"
    elif model_score >= t_review:
        model_decision = "review"
    else:
        model_decision = "approve"

    non_allowlist_rules = [r for r in triggered_rules if r != "trusted_allowlist"]
    if non_allowlist_rules and DECISION_ORDER[rule_decision] >= DECISION_ORDER[model_decision]:
        return rule_decision, "rule"
    return model_decision, "model"
