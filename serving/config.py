"""Typed settings for the serving app, loaded from env vars / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    mlflow_tracking_uri: str = "./mlruns"
    mlflow_run_id: str = ""
    decision_db_url: str = "sqlite:///./decisions.db"
    thresholds_path: str = "./decision/thresholds.json"
    deployed_frontend_origin: str = ""
    enable_replay_worker: bool = False
    replay_interval_seconds: float = 7.0


settings = Settings()
