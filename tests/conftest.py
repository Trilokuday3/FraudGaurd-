import os
import tempfile
from pathlib import Path

import pytest

# Never let a test run against a non-sqlite decision DB. serving.app builds
# its SessionLocal from DECISION_DB_URL at import time, and several test
# modules (tests/integration/*, tests/unit/test_serving_config.py) import
# it at collection time -- with a developer's .env pointing at a real
# Postgres (e.g. the deployed Neon DB), that would read/write production
# data. This root conftest is loaded before any test module is imported,
# so forcing the env var here wins over .env (pydantic-settings prefers
# real env vars over the dotenv file). CI already sets a sqlite URL, so
# this is a no-op there.
if not os.environ.get("DECISION_DB_URL", "").startswith("sqlite"):
    _test_db_dir = Path(tempfile.mkdtemp(prefix="fraudguard-tests-"))
    os.environ["DECISION_DB_URL"] = f"sqlite:///{(_test_db_dir / 'decisions.db').as_posix()}"


@pytest.fixture(autouse=True)
def _isolate_mlflow_run_id_env(monkeypatch):
    """Strip MLFLOW_RUN_ID from the real OS environment for every test.

    mlflow's fluent API (mlflow.start_run() with no args) auto-resumes
    whatever run MLFLOW_RUN_ID points to. CI and the deployed Render
    environment both set this as a real process env var to pin the
    serving app to a specific model run -- without this guard, any test
    that calls mlflow.start_run() to log a throwaway test model would
    instead try to resume that unrelated run id and fail with
    "Run '<id>' not found" against its own tmp tracking dir.
    """
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
