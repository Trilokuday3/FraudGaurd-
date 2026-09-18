from serving.models import Decision, make_session_factory


def test_decision_round_trips_through_sqlite(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    session_factory = make_session_factory(db_url)
    session = session_factory()

    record = Decision(
        transaction_id="TXN0001",
        model_score=0.42,
        decision="review",
        triggered_rules=["high_risk_merchant"],
        decision_source="rule",
        model_run_id="run123",
        shap_top_features={"amount": 0.1},
        feature_row={"transaction_id": "TXN0001", "amount": 100.0},
    )
    session.add(record)
    session.commit()

    fetched = session.query(Decision).filter_by(transaction_id="TXN0001").one()
    assert fetched.model_score == 0.42
    assert fetched.triggered_rules == ["high_risk_merchant"]
    assert fetched.decision_source == "rule"
    assert fetched.created_at is not None
    assert fetched.feature_row == {"transaction_id": "TXN0001", "amount": 100.0}
    session.close()


def test_decision_feature_row_defaults_to_empty_dict(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    session_factory = make_session_factory(db_url)
    session = session_factory()

    record = Decision(
        transaction_id="TXN0002",
        model_score=0.1,
        decision="approve",
        triggered_rules=[],
        decision_source="model",
        model_run_id="run123",
        shap_top_features={},
    )
    session.add(record)
    session.commit()

    fetched = session.query(Decision).filter_by(transaction_id="TXN0002").one()
    assert fetched.feature_row == {}
    session.close()
