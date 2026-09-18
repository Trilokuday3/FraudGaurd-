import pytest

from mlops.check_scored_dq import check_scored_dq, main
from serving.models import Decision, make_session_factory


def _valid_feature_row(**overrides):
    row = {
        "transaction_id": "TXN0001",
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "timestamp": "2026-01-01T10:00:00",
        "amount": 100.0,
        "payment_method": "card",
        "hour_of_day": 10,
        "is_night": 0,
        "is_cross_border": 0,
        "amount_vs_customer_p95": 1.0,
        "txn_count_1h": 1,
        "txn_count_24h": 2,
        "is_new_device": 0,
        "device_age_days": 30.0,
        "customer_device_count_so_far": 1,
        "merchant_fraud_rate_hist": 0.01,
        "is_new_country_for_customer": 0,
        "ip_country_mismatch": 0,
    }
    row.update(overrides)
    return row


def _seed_decisions(db_url, feature_rows):
    session_factory = make_session_factory(db_url)
    session = session_factory()
    try:
        for i, fr in enumerate(feature_rows):
            session.add(
                Decision(
                    transaction_id=fr["transaction_id"],
                    model_score=0.1,
                    decision="approve",
                    triggered_rules=[],
                    decision_source="model",
                    model_run_id="run123",
                    shap_top_features={},
                    feature_row=fr,
                )
            )
        session.commit()
    finally:
        session.close()


def test_check_scored_dq_passes_on_valid_rows(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    _seed_decisions(
        db_url,
        [_valid_feature_row(transaction_id=f"TXN{i:04d}") for i in range(5)],
    )

    result = check_scored_dq(db_url=db_url)

    assert result["checked_rows"] == 5
    assert result["passed"] is True
    assert result["failures"] == []


def test_check_scored_dq_reports_failures_on_invalid_rows(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    _seed_decisions(
        db_url,
        [
            _valid_feature_row(transaction_id="TXN0001"),
            _valid_feature_row(transaction_id="TXN0002", amount=-50.0),
            _valid_feature_row(transaction_id="TXN0003", payment_method="crypto"),
        ],
    )

    result = check_scored_dq(db_url=db_url)

    assert result["checked_rows"] == 3
    assert result["passed"] is False
    assert len(result["failures"]) >= 2
    failed_columns = {f["column"] for f in result["failures"]}
    assert "amount" in failed_columns
    assert "payment_method" in failed_columns


def test_check_scored_dq_handles_no_scored_rows(tmp_path):
    db_url = f"sqlite:///{tmp_path}/empty.db"

    result = check_scored_dq(db_url=db_url)

    assert result["checked_rows"] == 0
    assert result["passed"] is True
    assert result["failures"] == []


def test_check_scored_dq_ignores_rows_with_no_feature_row(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    session_factory = make_session_factory(db_url)
    session = session_factory()
    try:
        session.add(
            Decision(
                transaction_id="TXN-LEGACY",
                model_score=0.1,
                decision="approve",
                triggered_rules=[],
                decision_source="model",
                model_run_id="run123",
                shap_top_features={},
            )
        )
        session.commit()
    finally:
        session.close()

    result = check_scored_dq(db_url=db_url)

    assert result["checked_rows"] == 0
    assert result["passed"] is True


def test_main_exits_nonzero_when_dq_fails(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/test.db"
    _seed_decisions(db_url, [_valid_feature_row(amount=-50.0)])

    import mlops.check_scored_dq as module

    monkeypatch.setattr(module.settings, "decision_db_url", db_url)
    monkeypatch.setattr("sys.argv", ["check_scored_dq"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_exits_zero_when_dq_passes(tmp_path, monkeypatch, capsys):
    db_url = f"sqlite:///{tmp_path}/test.db"
    _seed_decisions(db_url, [_valid_feature_row()])

    import mlops.check_scored_dq as module

    monkeypatch.setattr(module.settings, "decision_db_url", db_url)
    monkeypatch.setattr("sys.argv", ["check_scored_dq"])

    main()  # should not raise

    assert "PASSED" in capsys.readouterr().out
