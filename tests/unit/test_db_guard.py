"""Pure-logic tests for the test-suite DB guard (tests/db_guard.py), which
tests/conftest.py applies before any test module can import serving.app.
No database is touched here."""

from pathlib import Path

from tests.db_guard import ALLOW_REMOTE_DB_ENV, resolve_forced_db_url


def _tmp_dir(tmp_path):
    return lambda: tmp_path


def test_non_sqlite_url_is_replaced_with_a_temp_sqlite_file(tmp_path):
    env = {"DECISION_DB_URL": "postgresql+psycopg://u:p@host/db"}
    forced = resolve_forced_db_url(env, _tmp_dir(tmp_path))
    assert forced == f"sqlite:///{(Path(tmp_path) / 'decisions.db').as_posix()}"


def test_missing_url_is_replaced_with_a_temp_sqlite_file(tmp_path):
    assert resolve_forced_db_url({}, _tmp_dir(tmp_path)).startswith("sqlite:///")


def test_existing_sqlite_url_is_left_alone(tmp_path):
    env = {"DECISION_DB_URL": "sqlite:///./decisions.db"}
    assert resolve_forced_db_url(env, _tmp_dir(tmp_path)) is None


def test_explicit_opt_in_leaves_a_remote_url_alone(tmp_path):
    env = {"DECISION_DB_URL": "postgresql+psycopg://u:p@host/db", ALLOW_REMOTE_DB_ENV: "1"}
    assert resolve_forced_db_url(env, _tmp_dir(tmp_path)) is None


def test_opt_in_requires_exactly_1(tmp_path):
    for value in ["", "0", "true", "yes"]:
        env = {"DECISION_DB_URL": "postgresql+psycopg://u:p@host/db", ALLOW_REMOTE_DB_ENV: value}
        assert resolve_forced_db_url(env, _tmp_dir(tmp_path)) is not None
