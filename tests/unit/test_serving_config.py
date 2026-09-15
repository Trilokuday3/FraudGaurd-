from serving.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.mlflow_tracking_uri == "./mlruns"
    assert s.decision_db_url == "sqlite:///./decisions.db"
    assert s.thresholds_path == "./decision/thresholds.json"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc123")
    s = Settings(_env_file=None)
    assert s.mlflow_run_id == "abc123"
