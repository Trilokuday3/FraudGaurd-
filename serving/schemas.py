"""Pydantic request/response models. FeatureRow mirrors
fraudguard_core.schemas.features_schema exactly, since /score accepts an
already-computed feature row, not a raw transaction."""

from datetime import datetime

from pydantic import BaseModel


class FeatureRow(BaseModel):
    transaction_id: str
    customer_id: str
    merchant_id: str
    timestamp: datetime
    amount: float
    payment_method: str
    hour_of_day: int
    is_night: bool
    is_cross_border: bool
    amount_vs_customer_p95: float
    txn_count_1h: int
    txn_count_24h: int
    is_new_device: bool
    device_age_days: float
    customer_device_count_so_far: int
    merchant_fraud_rate_hist: float
    is_new_country_for_customer: bool
    ip_country_mismatch: bool


class ScoreResponse(BaseModel):
    transaction_id: str
    model_score: float
    decision: str
    triggered_rules: list[str]
    decision_source: str


class ExplainResponse(BaseModel):
    transaction_id: str
    contributions: dict[str, float]


class InvestigationResponse(BaseModel):
    transaction_id: str
    model_score: float
    decision: str
    triggered_rules: list[str]
    decision_source: str
    model_run_id: str
    shap_top_features: dict[str, float]
    created_at: datetime


class ModelMetadataResponse(BaseModel):
    model_run_id: str
    deployed_model_name: str
    val_pr_auc: float
    test_pr_auc: float
    calibration_method: str
    t_review: float
    t_block: float
