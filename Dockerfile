# Local-only inference image for the FraudGuard API -- a reproducibility /
# portfolio artifact demonstrating containerization, NOT the deploy path.
# Sub-project 8 deliberately deploys via Render's native Python runtime
# (see infra/deploy.md); this image is for `docker compose up` on a laptop.
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

ENV MLFLOW_TRACKING_URI=./deploy/model_store
ENV THRESHOLDS_PATH=./decision/thresholds.json
# MLFLOW_RUN_ID and DECISION_DB_URL are intentionally not set here -- they
# must match whatever run deploy/model_store/ actually contains and
# whichever database this container should log to. docker-compose.yml
# supplies both (the run ID from .env, already required for local dev; the
# DB URL pointed at the compose-managed Postgres service).

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
