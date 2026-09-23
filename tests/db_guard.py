"""Decides whether the test suite must override DECISION_DB_URL (applied
by tests/conftest.py before any test module can import serving.app).
Kept as a pure function so it is unit-testable without any database."""

from collections.abc import Callable, Mapping
from pathlib import Path

ALLOW_REMOTE_DB_ENV = "FRAUDGUARD_TESTS_ALLOW_REMOTE_DB"


def resolve_forced_db_url(env: Mapping[str, str], make_tmp_dir: Callable[[], Path]) -> str | None:
    """Return the sqlite URL to force, or None to leave DECISION_DB_URL alone.

    Left alone when it is already sqlite, or when the caller explicitly
    opted in with FRAUDGUARD_TESTS_ALLOW_REMOTE_DB=1 (the deliberate
    Postgres verification run in infra/deploy.md). Anything else --
    including a Postgres URL picked up from a developer's .env -- is
    replaced with a sqlite file in a fresh temp dir."""
    if env.get(ALLOW_REMOTE_DB_ENV) == "1":
        return None
    if env.get("DECISION_DB_URL", "").startswith("sqlite"):
        return None
    return f"sqlite:///{(Path(make_tmp_dir()) / 'decisions.db').as_posix()}"
