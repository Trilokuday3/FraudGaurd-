"""Tests for streaming.spark_consumer.score_rows, the pure-Python bridge
between a Spark micro-batch and serving.app._compute_score. No Spark
needed: score_rows takes plain dicts, exactly what process_batch hands it.

Order-independence: tests/integration/test_serving_api.py's app_client
fixture importlib.reload()s serving.app in place (swapping in a tiny
synthetic model and a tmp sqlite SessionLocal). score_rows resolves
serving.app's SessionLocal/_compute_score at call time, and these tests
never assert on model-specific output, so they pass whichever version of
serving.app's module state is current. Where a test needs a DB, it uses
its own per-test sqlite file (tmp_path) or deletes its rows afterwards.
tests/conftest.py guarantees DECISION_DB_URL is sqlite before
serving.app can be imported."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from serving.models import Decision, make_session_factory
from streaming.producer import load_sample_pool
from streaming.spark_consumer import score_rows

_POOL_PATH = Path(__file__).resolve().parents[2] / "deploy" / "sample_transactions.json"


def _real_sample_rows(n: int) -> list[dict]:
    return [dict(row) for row in load_sample_pool(str(_POOL_PATH))[:n]]


@pytest.fixture
def session_factory(tmp_path):
    return make_session_factory(f"sqlite:///{(tmp_path / 'stream.db').as_posix()}")


def _decisions(session_factory, transaction_ids):
    session = session_factory()
    try:
        return (
            session.query(Decision)
            .filter(Decision.transaction_id.in_(transaction_ids))
            .all()
        )
    finally:
        session.close()


def test_score_rows_scores_a_real_sample_row(session_factory):
    row = _real_sample_rows(1)[0]

    scored, skipped = score_rows([row], session_factory=session_factory)

    assert (scored, skipped) == (1, 0)
    [record] = _decisions(session_factory, [row["transaction_id"]])
    assert record.feature_row["amount"] == row["amount"]
    assert record.decision in {"approve", "review", "block"}


def test_score_rows_skips_and_counts_an_invalid_row(session_factory, caplog):
    good = _real_sample_rows(2)
    bad = dict(good[0], transaction_id="TXN_STREAM_BAD", amount=None, is_night=None)

    with caplog.at_level("WARNING", logger="streaming.spark_consumer"):
        scored, skipped = score_rows([good[0], bad, good[1]], session_factory=session_factory, batch_id=7)

    assert (scored, skipped) == (2, 1)
    ids = [r["transaction_id"] for r in good] + ["TXN_STREAM_BAD"]
    persisted = {d.transaction_id for d in _decisions(session_factory, ids)}
    assert persisted == {good[0]["transaction_id"], good[1]["transaction_id"]}
    assert "TXN_STREAM_BAD" in caplog.text
    assert "batch 7" in caplog.text


def test_score_rows_adds_every_record_to_one_session_and_commits_once():
    session = MagicMock()
    factory = MagicMock(return_value=session)

    scored, skipped = score_rows(_real_sample_rows(3), session_factory=factory)

    assert (scored, skipped) == (3, 0)
    factory.assert_called_once()
    added = [c.args[0] for c in session.add.call_args_list] + [
        r for c in session.add_all.call_args_list for r in c.args[0]
    ]
    assert len(added) == 3
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    session.close.assert_called_once()


def test_score_rows_rolls_back_and_raises_on_a_commit_failure():
    session = MagicMock()
    session.commit.side_effect = RuntimeError("db down")
    factory = MagicMock(return_value=session)

    with pytest.raises(RuntimeError, match="db down"):
        score_rows(_real_sample_rows(2), session_factory=factory)

    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_score_rows_on_empty_batch_touches_no_database():
    factory = MagicMock()
    assert score_rows([], session_factory=factory) == (0, 0)
    factory.assert_not_called()


@pytest.fixture
def default_db_transaction_id():
    """A unique id for a row written through serving.app's own
    SessionLocal (the default path), deleted again on teardown."""
    transaction_id = f"TXN_STREAM_TEST_{uuid.uuid4().hex}"
    yield transaction_id
    import serving.app

    session = serving.app.SessionLocal()
    try:
        session.query(Decision).filter(Decision.transaction_id == transaction_id).delete()
        session.commit()
    finally:
        session.close()


def test_score_rows_defaults_to_serving_apps_session(default_db_transaction_id):
    import serving.app

    assert str(serving.app.SessionLocal.kw["bind"].url).startswith("sqlite")
    row = dict(_real_sample_rows(1)[0], transaction_id=default_db_transaction_id)

    assert score_rows([row]) == (1, 0)

    [record] = _decisions(serving.app.SessionLocal, [default_db_transaction_id])
    assert record.decision in {"approve", "review", "block"}
