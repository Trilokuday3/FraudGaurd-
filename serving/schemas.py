"""Pydantic request/response models. FeatureRow mirrors
fraudguard_core.schemas.features_schema exactly, since /score accepts an
already-computed feature row, not a raw transaction."""

from datetime import UTC, datetime

from pydantic import BaseModel, field_serializer


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


def _ensure_utc_offset(value: datetime) -> str:
    """Serialize a datetime with an explicit UTC offset even if the value
    itself is naive.

    `Decision.created_at` (serving/models.py) is written as an
    offset-aware UTC datetime, but SQLAlchemy's `DateTime` column type
    strips the offset on the SQLite round-trip (Postgres, the eventual
    target, preserves it) -- so by the time it reaches this serializer it
    may be naive-but-actually-UTC. Without this, the emitted ISO string has
    no offset and the frontend's `new Date(...)` parses it as the viewer's
    *local* time instead of UTC, shifting every displayed timestamp by the
    viewer's UTC offset.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


class InvestigationResponse(BaseModel):
    transaction_id: str
    model_score: float
    decision: str
    triggered_rules: list[str]
    decision_source: str
    model_run_id: str
    shap_top_features: dict[str, float]
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        return _ensure_utc_offset(value)


class DecisionRow(BaseModel):
    id: int
    transaction_id: str
    model_score: float
    decision: str
    triggered_rules: list[str]
    decision_source: str
    model_run_id: str
    shap_top_features: dict[str, float]
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        return _ensure_utc_offset(value)


class DecisionsListResponse(BaseModel):
    items: list[DecisionRow]
    next_cursor: int | None


class DecisionsStatsBucket(BaseModel):
    minute: str
    decision: str
    count: int


class DecisionsStatsResponse(BaseModel):
    total: int
    approve_count: int
    review_count: int
    block_count: int
    avg_score: float
    buckets: list[DecisionsStatsBucket]


class CostCurvePoint(BaseModel):
    t_review: float
    cost: float


class ModelComparisonCandidate(BaseModel):
    name: str
    val_pr_auc: float
    is_deployed: bool


class CalibrationPoint(BaseModel):
    mean_predicted: float
    fraction_positive: float


class ShapImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class ModelMetadataResponse(BaseModel):
    model_run_id: str
    deployed_model_name: str
    val_pr_auc: float
    test_pr_auc: float
    calibration_method: str
    t_review: float
    t_block: float
    cost_curve: list[CostCurvePoint]


class ModelComparisonResponse(BaseModel):
    candidates: list[ModelComparisonCandidate]
    calibration_curve: list[CalibrationPoint]
    shap_importances: list[ShapImportance]
