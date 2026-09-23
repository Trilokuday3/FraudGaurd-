"""FastAPI app: score, batch score, explain, investigation lookup, model
metadata, health."""

import asyncio
import json
import tempfile
import warnings
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient

from decision.rules import combine_decision, evaluate_rules
from ml.data import prepare_model_matrix
from ml.explain import explain_prediction
from serving.config import settings
from serving.ml_loader import load_deployed_model
from serving.models import Decision, make_session_factory
from serving.replay_worker import replay_worker_loop
from serving.schemas import (
    CalibrationPoint,
    CostCurvePoint,
    DecisionRow,
    DecisionsListResponse,
    DecisionsStatsBucket,
    DecisionsStatsResponse,
    ExplainResponse,
    FeatureRow,
    InvestigationResponse,
    ModelComparisonCandidate,
    ModelComparisonResponse,
    ModelMetadataResponse,
    ScoreResponse,
    ShapImportance,
)

_SAMPLE_TRANSACTIONS_PATH = Path(__file__).resolve().parent.parent / "deploy" / "sample_transactions.json"


def _load_sample_transactions() -> list[FeatureRow]:
    # `_compute_score` (below) expects a `FeatureRow` instance -- it calls
    # `.model_dump()` on what it's given, exactly like FastAPI's request
    # parsing already hands it for `/score`. The bundled JSON file holds
    # plain dicts, so each row is parsed into `FeatureRow` here rather than
    # handed to the replay loop raw -- confirmed necessary by directly
    # invoking replay_worker_loop against a plain dict, which raised
    # `AttributeError: 'dict' object has no attribute 'model_dump'`; inside
    # a fire-and-forget `asyncio.create_task`, that exception would have
    # been silently swallowed instead of surfacing.
    with open(_SAMPLE_TRANSACTIONS_PATH) as f:
        raw_rows = json.load(f)
    return [FeatureRow(**row) for row in raw_rows]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # `_compute_score` and `SessionLocal` are defined further down in this
    # module (after `app` is constructed), but this closure only resolves
    # those names when it actually runs -- at process startup, well after
    # the whole module has finished importing -- so the textual order here
    # doesn't matter, only that both exist by then.
    task = None
    if settings.enable_replay_worker:
        rows = _load_sample_transactions()
        task = asyncio.create_task(
            replay_worker_loop(
                compute_score=_compute_score,
                session_factory=SessionLocal,
                rows=rows,
                interval_seconds=settings.replay_interval_seconds,
            )
        )
    yield
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="FraudGuard Decision Engine API", lifespan=lifespan)


# Local-dev-scoped default: the frontend (`frontend/`) runs client-rendered
# pages (Live Transactions, Monitoring, Investigations) that fetch this API
# directly from the browser, and Next's dev server doesn't always land on
# the same port (3000, 3002, ... depending on what's free). A regex
# matching any localhost port keeps that working without hardcoding one.
# In production, `settings.deployed_frontend_origin` adds the real deployed
# Vercel URL -- set once as an env var after that URL is known (Render
# restarts the process on an env var change, so this takes effect on next
# boot, not via any runtime mutation).
def _build_allow_origins(deployed_frontend_origin: str) -> list[str]:
    return [deployed_frontend_origin] if deployed_frontend_origin else []


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_origins=_build_allow_origins(settings.deployed_frontend_origin),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CANDIDATE_RUN_NAMES = ["baseline", "random_forest", "xgboost", "lightgbm"]


def _load_json_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        local_path = client.download_artifacts(run_id, artifact_path, tmp)
        with open(local_path) as f:
            return json.load(f)


_loaded = load_deployed_model(settings.mlflow_run_id, settings.mlflow_tracking_uri)
with open(settings.thresholds_path) as _f:
    _thresholds = json.load(_f)

if _thresholds.get("model_run_id") != settings.mlflow_run_id:
    warnings.warn(
        f"thresholds.json was computed for run {_thresholds.get('model_run_id')!r} "
        f"but the API is configured to load run {settings.mlflow_run_id!r} — "
        "re-run decision.select_thresholds.select_thresholds_for_run for the "
        "currently-configured run before trusting these thresholds.",
        stacklevel=1,
    )

SessionLocal = make_session_factory(settings.decision_db_url)


def _feature_row_to_model_input(row: FeatureRow) -> pd.DataFrame:
    raw = pd.DataFrame([row.model_dump()])
    return prepare_model_matrix(raw)


def _compute_score(row: FeatureRow) -> tuple[ScoreResponse, Decision]:
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
        _loaded.raw_model, model_input, background=_loaded.shap_background
    )
    top5 = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])

    record = Decision(
        transaction_id=row.transaction_id,
        model_score=model_score,
        decision=final_decision,
        triggered_rules=triggered_rules,
        decision_source=decision_source,
        model_run_id=_loaded.run_id,
        shap_top_features=top5,
        feature_row=row.model_dump(mode="json"),
    )
    response = ScoreResponse(
        transaction_id=row.transaction_id,
        model_score=model_score,
        decision=final_decision,
        triggered_rules=triggered_rules,
        decision_source=decision_source,
    )
    return response, record


def _score_one(row: FeatureRow) -> ScoreResponse:
    response, record = _compute_score(row)
    session = SessionLocal()
    try:
        session.add(record)
        session.commit()
    finally:
        session.close()
    return response


@app.post("/score", response_model=ScoreResponse)
def score(row: FeatureRow) -> ScoreResponse:
    return _score_one(row)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(rows: list[FeatureRow]) -> list[ScoreResponse]:
    results = [_compute_score(row) for row in rows]
    responses = [response for response, _ in results]
    records = [record for _, record in results]

    session = SessionLocal()
    try:
        session.add_all(records)
        session.commit()
    finally:
        session.close()

    return responses


def _decision_to_row(record: Decision) -> DecisionRow:
    return DecisionRow(
        id=record.id,
        transaction_id=record.transaction_id,
        model_score=record.model_score,
        decision=record.decision,
        triggered_rules=record.triggered_rules,
        decision_source=record.decision_source,
        model_run_id=record.model_run_id,
        shap_top_features=record.shap_top_features,
        feature_row=record.feature_row,
        created_at=record.created_at,
    )


@app.get("/decisions", response_model=DecisionsListResponse)
def list_decisions(
    decision: str | None = None,
    limit: int = 50,
    before_id: int | None = None,
) -> DecisionsListResponse:
    limit = min(max(limit, 1), 500)
    session = SessionLocal()
    try:
        query = session.query(Decision)
        if decision is not None:
            query = query.filter(Decision.decision == decision)
        if before_id is not None:
            query = query.filter(Decision.id < before_id)
        rows = query.order_by(Decision.id.desc()).limit(limit + 1).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = rows[-1].id if has_more and rows else None
        return DecisionsListResponse(
            items=[_decision_to_row(r) for r in rows], next_cursor=next_cursor
        )
    finally:
        session.close()


@app.get("/decisions/stats", response_model=DecisionsStatsResponse)
def decisions_stats(since_minutes: int | None = None) -> DecisionsStatsResponse:
    session = SessionLocal()
    try:
        query = session.query(Decision)
        if since_minutes is not None:
            cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)
            query = query.filter(Decision.created_at >= cutoff)
        rows = query.all()

        total = len(rows)
        approve_count = sum(1 for r in rows if r.decision == "approve")
        review_count = sum(1 for r in rows if r.decision == "review")
        block_count = sum(1 for r in rows if r.decision == "block")
        avg_score = (sum(r.model_score for r in rows) / total) if total else 0.0

        bucket_counts: dict[tuple[str, str], int] = {}
        for r in rows:
            minute_key = r.created_at.strftime("%Y-%m-%dT%H:%M")
            key = (minute_key, r.decision)
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
        buckets = [
            DecisionsStatsBucket(minute=minute, decision=dec, count=count)
            for (minute, dec), count in sorted(bucket_counts.items())
        ]

        return DecisionsStatsResponse(
            total=total,
            approve_count=approve_count,
            review_count=review_count,
            block_count=block_count,
            avg_score=avg_score,
            buckets=buckets,
        )
    finally:
        session.close()


@app.post("/explain", response_model=ExplainResponse)
def explain(row: FeatureRow) -> ExplainResponse:
    model_input = _feature_row_to_model_input(row)
    contributions = explain_prediction(
        _loaded.raw_model, model_input, background=_loaded.shap_background
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
            feature_row=record.feature_row,
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
        cost_curve=[CostCurvePoint(**pt) for pt in _thresholds.get("cost_curve", [])],
    )


@app.get("/model/comparison", response_model=ModelComparisonResponse)
def model_comparison() -> ModelComparisonResponse:
    client = MlflowClient()
    deployed_run = client.get_run(_loaded.run_id)
    deployed_name = deployed_run.data.params.get("deployed_model", "unknown")

    # Search within the deployed run's own experiment rather than a
    # hardcoded experiment name: in production this is always
    # "fraudguard-modeling" (ml/__main__.py logs the deployed run and all
    # candidate runs into that one experiment), and this also makes the
    # lookup work correctly against whatever experiment name a test/other
    # environment's fixture happens to use.
    candidates = []
    for name in _CANDIDATE_RUN_NAMES:
        matches = client.search_runs(
            experiment_ids=[deployed_run.info.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if matches:
            run = matches[0]
            candidates.append(
                ModelComparisonCandidate(
                    name=name,
                    val_pr_auc=float(run.data.metrics.get("val_pr_auc", 0.0)),
                    is_deployed=(name == deployed_name),
                )
            )

    try:
        comparison_artifact = _load_json_artifact(client, _loaded.run_id, "model_comparison.json")
    except Exception as exc:
        # Covers both MLflow's own artifact-not-found errors and the plain
        # OSError/FileNotFoundError raised while opening the downloaded file
        # -- either way it means this run predates ml/enrich_deployed_run.py
        # writing model_comparison.json (or was never enriched).
        raise HTTPException(
            status_code=503,
            detail=(
                "model_comparison.json artifact not found for the deployed run "
                f"({_loaded.run_id!r}). Re-run enrichment for this run before "
                "calling /model/comparison: "
                "python -c \"from ml.enrich_deployed_run import enrich_deployed_run; "
                f"enrich_deployed_run({_loaded.run_id!r})\""
            ),
        ) from exc
    calibration_curve = [
        CalibrationPoint(mean_predicted=p, fraction_positive=t)
        for p, t in zip(
            comparison_artifact["calibration_curve"]["prob_pred"],
            comparison_artifact["calibration_curve"]["prob_true"],
        )
    ]
    shap_importances = [
        ShapImportance(feature=k, mean_abs_shap=v)
        for k, v in comparison_artifact["shap_importances"].items()
    ]

    return ModelComparisonResponse(
        candidates=candidates,
        calibration_curve=calibration_curve,
        shap_importances=shap_importances,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
