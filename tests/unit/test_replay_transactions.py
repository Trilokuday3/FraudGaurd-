import numpy as np
import pandas as pd

from scripts.replay_transactions import _to_jsonable, replay, select_rows


def test_to_jsonable_converts_numpy_scalars_to_native_python():
    assert isinstance(_to_jsonable(np.int64(5)), int)
    assert not isinstance(_to_jsonable(np.int64(5)), np.integer)
    assert isinstance(_to_jsonable(np.float64(1.5)), float)
    assert isinstance(_to_jsonable(np.bool_(True)), bool)
    assert _to_jsonable(pd.Timestamp("2026-01-01T10:00:00")) == "2026-01-01T10:00:00"
    assert _to_jsonable("plain-string") == "plain-string"


def test_select_rows_respects_count():
    features = pd.DataFrame({"transaction_id": [f"TXN{i}" for i in range(10)]})
    rows = select_rows(features, count=3, loop=False)
    assert len(rows) == 3


def test_select_rows_returns_all_rows_when_count_is_none():
    features = pd.DataFrame({"transaction_id": [f"TXN{i}" for i in range(5)]})
    rows = select_rows(features, count=None, loop=False)
    assert len(rows) == 5


def test_replay_posts_each_row_once_and_sleeps_between_each():
    rows = [{"transaction_id": "TXN1"}, {"transaction_id": "TXN2"}]
    posted = []
    slept = []

    class _FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, json, timeout):
        posted.append((url, json))
        return _FakeResponse()

    def fake_sleep(seconds):
        slept.append(seconds)

    sent = replay(
        rows, "http://localhost:8000", 2.5, loop=False, post=fake_post, sleep=fake_sleep
    )

    assert sent == 2
    assert [p[0] for p in posted] == [
        "http://localhost:8000/score",
        "http://localhost:8000/score",
    ]
    assert slept == [2.5, 2.5]


def test_replay_converts_numpy_and_timestamp_values_in_payload():
    rows = [{"transaction_id": "TXN1", "amount": np.float64(9.5), "ts": pd.Timestamp("2026-01-01")}]
    posted = []

    class _FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, json, timeout):
        posted.append(json)
        return _FakeResponse()

    replay(rows, "http://localhost:8000", 0.0, loop=False, post=fake_post, sleep=lambda s: None)

    assert posted[0]["amount"] == 9.5
    assert isinstance(posted[0]["amount"], float)
    assert posted[0]["ts"] == "2026-01-01T00:00:00"
