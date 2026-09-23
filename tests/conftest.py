import pytest


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
