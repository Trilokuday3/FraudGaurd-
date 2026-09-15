from decision.rules import combine_decision, evaluate_rules


def _base_row(**overrides):
    row = {
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "is_new_device": False,
        "is_new_country_for_customer": False,
        "ip_country_mismatch": False,
        "txn_count_1h": 0,
        "merchant_fraud_rate_hist": 0.01,
        "amount_vs_customer_p95": 1.0,
    }
    row.update(overrides)
    return row


def test_no_rules_trigger_on_a_clean_row():
    decision, triggered, allowlisted = evaluate_rules(_base_row())
    assert decision == "approve"
    assert triggered == []
    assert allowlisted is False


def test_device_location_takeover_forces_block():
    row = _base_row(is_new_device=True, is_new_country_for_customer=True, ip_country_mismatch=True)
    decision, triggered, _ = evaluate_rules(row)
    assert decision == "block"
    assert "device_location_takeover" in triggered


def test_velocity_spike_forces_block():
    decision, triggered, _ = evaluate_rules(_base_row(txn_count_1h=5))
    assert decision == "block"
    assert "velocity_spike" in triggered


def test_high_risk_merchant_forces_at_least_review():
    decision, triggered, _ = evaluate_rules(_base_row(merchant_fraud_rate_hist=0.2))
    assert decision == "review"
    assert "high_risk_merchant" in triggered


def test_outsized_amount_new_device_forces_at_least_review():
    decision, triggered, _ = evaluate_rules(
        _base_row(amount_vs_customer_p95=4.0, is_new_device=True)
    )
    assert decision == "review"
    assert "outsized_amount_new_device" in triggered


def test_multiple_escalation_rules_take_the_highest():
    row = _base_row(txn_count_1h=5, merchant_fraud_rate_hist=0.2)
    decision, triggered, _ = evaluate_rules(row)
    assert decision == "block"
    assert set(triggered) == {"velocity_spike", "high_risk_merchant"}


def test_allowlisted_customer_flagged_via_explicit_allowlist_param():
    row = _base_row(customer_id="CUST999", txn_count_1h=5)
    decision, triggered, allowlisted = evaluate_rules(
        row, allowlist_customer_ids=frozenset({"CUST999"})
    )
    assert allowlisted is True
    assert "trusted_allowlist" in triggered
    # the escalation rule still reports its own finding independently --
    # it's combine_decision's job to apply the allowlist override, not evaluate_rules's
    assert decision == "block"


def test_default_allowlist_flags_the_illustrative_example_customer():
    row = _base_row(customer_id="CUST-EXAMPLE-ALLOWLISTED")
    _decision, triggered, allowlisted = evaluate_rules(row)
    assert allowlisted is True
    assert "trusted_allowlist" in triggered


def test_combine_decision_allowlist_overrides_everything():
    decision, source = combine_decision(
        model_score=0.99,
        t_review=0.3,
        t_block=0.7,
        rule_decision="block",
        triggered_rules=["velocity_spike", "trusted_allowlist"],
        is_allowlisted=True,
    )
    assert decision == "approve"
    assert source == "rule_override"


def test_combine_decision_rule_wins_over_lower_model_score():
    decision, source = combine_decision(
        model_score=0.1,
        t_review=0.3,
        t_block=0.7,
        rule_decision="block",
        triggered_rules=["velocity_spike"],
        is_allowlisted=False,
    )
    assert decision == "block"
    assert source == "rule"


def test_combine_decision_model_wins_when_no_rules_fire():
    decision, source = combine_decision(
        model_score=0.8,
        t_review=0.3,
        t_block=0.7,
        rule_decision="approve",
        triggered_rules=[],
        is_allowlisted=False,
    )
    assert decision == "block"
    assert source == "model"


def test_combine_decision_model_exceeds_rule_level():
    decision, source = combine_decision(
        model_score=0.9,
        t_review=0.3,
        t_block=0.7,
        rule_decision="review",
        triggered_rules=["high_risk_merchant"],
        is_allowlisted=False,
    )
    assert decision == "block"
    assert source == "model"
