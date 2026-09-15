import pytest
from pydantic import ValidationError

from serving.schemas import FeatureRow


def _valid_payload():
    return {
        "transaction_id": "TXN0001",
        "customer_id": "CUST001",
        "merchant_id": "MERC001",
        "timestamp": "2026-01-01T10:00:00",
        "amount": 100.0,
        "payment_method": "card",
        "hour_of_day": 10,
        "is_night": False,
        "is_cross_border": False,
        "amount_vs_customer_p95": 1.2,
        "txn_count_1h": 1,
        "txn_count_24h": 3,
        "is_new_device": False,
        "device_age_days": 30.0,
        "customer_device_count_so_far": 2,
        "merchant_fraud_rate_hist": 0.02,
        "is_new_country_for_customer": False,
        "ip_country_mismatch": False,
    }


def test_feature_row_accepts_valid_payload():
    row = FeatureRow(**_valid_payload())
    assert row.transaction_id == "TXN0001"
    assert row.amount == 100.0


def test_feature_row_rejects_missing_field():
    payload = _valid_payload()
    del payload["amount"]
    with pytest.raises(ValidationError):
        FeatureRow(**payload)


def test_feature_row_rejects_wrong_type():
    payload = _valid_payload()
    payload["amount"] = "not-a-number"
    with pytest.raises(ValidationError):
        FeatureRow(**payload)


from datetime import datetime

from serving.schemas import (
    CalibrationPoint,
    CostCurvePoint,
    DecisionRow,
    DecisionsListResponse,
    DecisionsStatsBucket,
    DecisionsStatsResponse,
    ModelComparisonCandidate,
    ModelComparisonResponse,
    ModelMetadataResponse,
    ShapImportance,
)


def test_decisions_list_response_accepts_valid_payload():
    row = DecisionRow(
        id=1,
        transaction_id="TXN0001",
        model_score=0.42,
        decision="review",
        triggered_rules=["velocity_spike"],
        decision_source="rule",
        model_run_id="run123",
        shap_top_features={"amount": 0.1},
        created_at=datetime(2026, 1, 1, 10, 0, 0),
    )
    response = DecisionsListResponse(items=[row], next_cursor=1)
    assert response.items[0].transaction_id == "TXN0001"
    assert response.next_cursor == 1


def test_decisions_stats_response_accepts_valid_payload():
    bucket = DecisionsStatsBucket(minute="2026-01-01T10:00", decision="approve", count=3)
    response = DecisionsStatsResponse(
        total=10, approve_count=7, review_count=2, block_count=1, avg_score=0.2, buckets=[bucket]
    )
    assert response.total == 10
    assert response.buckets[0].count == 3


def test_model_metadata_response_includes_cost_curve():
    response = ModelMetadataResponse(
        model_run_id="run123",
        deployed_model_name="baseline",
        val_pr_auc=0.5,
        test_pr_auc=0.5,
        calibration_method="isotonic",
        t_review=0.3,
        t_block=0.7,
        cost_curve=[CostCurvePoint(t_review=0.3, cost=100.0)],
    )
    assert response.cost_curve[0].t_review == 0.3


def test_model_comparison_response_accepts_valid_payload():
    response = ModelComparisonResponse(
        candidates=[ModelComparisonCandidate(name="baseline", val_pr_auc=0.5, is_deployed=True)],
        calibration_curve=[CalibrationPoint(mean_predicted=0.1, fraction_positive=0.05)],
        shap_importances=[ShapImportance(feature="amount", mean_abs_shap=0.2)],
    )
    assert response.candidates[0].is_deployed is True
    assert response.shap_importances[0].feature == "amount"
