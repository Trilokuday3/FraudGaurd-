"""Tests for serving/replay_worker.py."""

import asyncio

import pytest

from serving.replay_worker import replay_worker_loop


class _FakeSession:
    def __init__(self, store: list):
        self.store = store

    def add(self, record):
        self.store.append(record)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.mark.asyncio
async def test_replay_worker_loop_scores_and_persists_each_row():
    store: list = []
    calls = []

    def fake_compute_score(row):
        calls.append(row)
        return ("response", f"record-for-{row['transaction_id']}")

    def fake_session_factory():
        return _FakeSession(store)

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError  # stop the infinite loop for the test

    rows = [{"transaction_id": "a"}, {"transaction_id": "b"}]

    with pytest.raises(asyncio.CancelledError):
        await replay_worker_loop(
            compute_score=fake_compute_score,
            session_factory=fake_session_factory,
            rows=rows,
            interval_seconds=0.01,
            sleep=fake_sleep,
        )

    assert len(calls) == 3  # a, b, a (loops back to the start)
    assert store == ["record-for-a", "record-for-b", "record-for-a"]
