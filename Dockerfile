# Inference image for the FraudGuard API. Originally built for local
# `docker compose up` only, but Render auto-detects this Dockerfile and
# builds with it regardless of what infra/deploy.md's Render section says
# -- Render's environment type is fixed at service creation and can't be
# switched to native Python after the fact, so this image is also the real
# Render deploy path now. Keep both use cases working.
#
# Reuses deploy/requirements.txt (sub-project 8's lean, deploy-only
# dependency set -- no jupyter/black/ruff/pytest) and deploy/model_store/
# (sub-project 8's vendored, path-repairing MLflow snapshot), so this image
# needs no MLflow server and no access to the full local mlruns/ tree.
FROM python:3.11-slim

WORKDIR /app

COPY libs/ libs/
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

COPY serving/ serving/
COPY decision/ decision/
COPY ml/ ml/
COPY mlops/ mlops/
COPY deploy/model_store/ deploy/model_store/
COPY deploy/sample_transactions.json deploy/sample_transactions.json

ENV MLFLOW_TRACKING_URI=./deploy/model_store
ENV THRESHOLDS_PATH=./decision/thresholds.json
# MLFLOW_RUN_ID and DECISION_DB_URL are intentionally not set here -- they
# must match whatever run deploy/model_store/ actually contains and
# whichever database this container should log to. docker-compose.yml
# supplies both (the run ID from .env, already required for local dev; the
# DB URL pointed at the compose-managed Postgres service).

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
