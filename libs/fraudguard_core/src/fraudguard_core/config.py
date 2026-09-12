"""Typed configuration, populated from environment variables.

`pydantic-settings` reads each field from an env var built as
``<env_prefix><FIELD_NAME>`` (upper-cased). Defaults here match a clean local
checkout with no `.env` — everything runs against `./data` with no external
services required for sub-project 1.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneratorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEN_", extra="ignore")

    seed: int = 42
    n_customers: int = 20_000
    n_merchants: int = 2_000
    avg_devices_per_customer: float = 1.3
    avg_txns_per_customer: float = 26.0
    target_fraud_rate: float = 0.015
    vintage_start: str = "2025-03-01"
    vintage_end: str = "2026-09-01"
    output_dir: str = "./data"
    chunk_customers: int = 2_000


@lru_cache
def get_generator_settings() -> GeneratorSettings:
    """Cached accessor. Tests that override env must call ``.cache_clear()``."""
    return GeneratorSettings()
