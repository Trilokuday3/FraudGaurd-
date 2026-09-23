import os
import tempfile

import pytest

from tests.db_guard import resolve_forced_db_url

# By default, never let a test run against a non-sqlite decision DB.
# serving.app builds its SessionLocal from DECISION_DB_URL at import time,
# and several test modules (tests/integration/*,
# tests/unit/test_serving_config.py) import it at collection time -- with a
# developer's .env pointing at a real Postgres (e.g. the deployed Neon DB),
# that would read/write production data. This root conftest is loaded
# before any test module is imported, so forcing the env var here wins over
# .env (pydantic-settings prefers real env vars over the dotenv file). CI
# already sets a sqlite URL, so this is a no-op there.
#
# Explicit opt-in: FRAUDGUARD_TESTS_ALLOW_REMOTE_DB=1 disables the guard
# entirely and leaves the caller's DECISION_DB_URL (or, if unset, .env's)
# in effect -- only for the deliberate Postgres verification run in
# infra/deploy.md, against a throwaway database. See tests/db_guard.py.
_forced_db_url = resolve_forced_db_url(
    os.environ, lambda: tempfile.mkdtemp(prefix="fraudguard-tests-")
)
if _forced_db_url is not None:
    os.environ["DECISION_DB_URL"] = _forced_db_url


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
