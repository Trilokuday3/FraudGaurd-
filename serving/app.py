"""FastAPI app: score, batch score, explain, investigation lookup, model
metadata, health."""

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from decision.rules import combine_decision, evaluate_rules
from ml.data import prepare_model_matrix
from ml.explain import explain_prediction
from serving.config import settings
from serving.ml_loader import load_deployed_model
from serving.models import Decision, make_session_factory
from serving.schemas import (
    ExplainResponse,
    FeatureRow,
    InvestigationResponse,
    ModelMetadataResponse,
    ScoreResponse,
)

app = FastAPI(title="FraudGuard Decision Engine API")

_loaded = load_deployed_model(settings.mlflow_run_id, settings.mlflow_tracking_uri)
with open(settings.thresholds_path) as _f:
    _thresholds = json.load(_f)

SessionLocal = make_session_factory(settings.decision_db_url)


def _feature_row_to_model_input(row: FeatureRow) -> pd.DataFrame:
    raw = pd.DataFrame([row.model_dump()])
    return prepare_model_matrix(raw)


def _score_one(row: FeatureRow) -> ScoreResponse:
    model_input = _feature_row_to_model_input(row)
    model_score = float(_loaded.model.predict_proba(model_input)[:, 1][0])

    rule_decision, triggered_rules, is_allowlisted = evaluate_rules(row.model_dump())
    final_decision, decision_source = combine_decision(
        model_score,
        _thresholds["t_review"],
        _thresholds["t_block"],
        rule_decision,
        triggered_rules,
        is_allowlisted,
    )

    contributions = explain_prediction(
        _loaded.model, model_input, background=_loaded.shap_background
    )
    top5 = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])

    session = SessionLocal()
    try:
        record = Decision(
            transaction_id=row.transaction_id,
            model_score=model_score,
            decision=final_decision,
            triggered_rules=triggered_rules,
            decision_source=decision_source,
            model_run_id=_loaded.run_id,
            shap_top_features=top5,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

    return ScoreResponse(
        transaction_id=row.transaction_id,
        model_score=model_score,
        decision=final_decision,
        triggered_rules=triggered_rules,
        decision_source=decision_source,
    )


@app.post("/score", response_model=ScoreResponse)
def score(row: FeatureRow) -> ScoreResponse:
    return _score_one(row)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(rows: list[FeatureRow]) -> list[ScoreResponse]:
    return [_score_one(row) for row in rows]


@app.post("/explain", response_model=ExplainResponse)
def explain(row: FeatureRow) -> ExplainResponse:
    model_input = _feature_row_to_model_input(row)
    contributions = explain_prediction(
        _loaded.model, model_input, background=_loaded.shap_background
    )
    return ExplainResponse(transaction_id=row.transaction_id, contributions=contributions)


@app.get("/investigations/{transaction_id}", response_model=InvestigationResponse)
def get_investigation(transaction_id: str) -> InvestigationResponse:
    session = SessionLocal()
    try:
        record = (
            session.query(Decision)
            .filter_by(transaction_id=transaction_id)
            .order_by(Decision.created_at.desc())
            .first()
        )
        if record is None:
            raise HTTPException(status_code=404, detail="transaction not found")
        return InvestigationResponse(
            transaction_id=record.transaction_id,
            model_score=record.model_score,
            decision=record.decision,
            triggered_rules=record.triggered_rules,
            decision_source=record.decision_source,
            model_run_id=record.model_run_id,
            shap_top_features=record.shap_top_features,
            created_at=record.created_at,
        )
    finally:
        session.close()


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    client = MlflowClient()
    run = client.get_run(_loaded.run_id)
    return ModelMetadataResponse(
        model_run_id=_loaded.run_id,
        deployed_model_name=run.data.params.get("deployed_model", "unknown"),
        val_pr_auc=float(run.data.metrics.get("deployed_val_pr_auc", 0.0)),
        test_pr_auc=float(run.data.metrics.get("test_pr_auc", 0.0)),
        calibration_method="isotonic",
        t_review=_thresholds["t_review"],
        t_block=_thresholds["t_block"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
