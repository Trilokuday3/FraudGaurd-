from serving.app import _build_allow_origins
from serving.config import Settings

_ENV_KEYS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_RUN_ID",
    "DECISION_DB_URL",
    "THRESHOLDS_PATH",
    "DEPLOYED_FRONTEND_ORIGIN",
    "ENABLE_REPLAY_WORKER",
    "REPLAY_INTERVAL_SECONDS",
]


def test_settings_defaults(monkeypatch):
    # Guard against real OS env leakage from other tests in the same pytest
    # process -- mlflow.set_tracking_uri() mutates os.environ as a side
    # effect, and Settings(_env_file=None) still reads real env vars (only
    # the .env FILE source is disabled, not the environment-variable one).
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.mlflow_tracking_uri == "./mlruns"
    assert s.decision_db_url == "sqlite:///./decisions.db"
    assert s.thresholds_path == "./decision/thresholds.json"
    assert s.deployed_frontend_origin == ""
    assert s.enable_replay_worker is False
    assert s.replay_interval_seconds == 7.0


def test_build_allow_origins_empty_when_unset():
    assert _build_allow_origins("") == []


def test_build_allow_origins_includes_configured_origin():
    assert _build_allow_origins("https://fraudguard.vercel.app") == [
        "https://fraudguard.vercel.app"
    ]


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc123")
    s = Settings(_env_file=None)
    assert s.mlflow_run_id == "abc123"
