from serving.models import make_session_factory
from streaming.cursor_store import get_cursor, advance_cursor


def test_get_cursor_defaults_to_zero_when_no_row_exists():
    session = make_session_factory("sqlite:///:memory:")()
    assert get_cursor(session) == 0


def test_advance_cursor_then_get_cursor_round_trips():
    session = make_session_factory("sqlite:///:memory:")()
    advance_cursor(session, 42)
    assert get_cursor(session) == 42


def test_advance_cursor_twice_overwrites_not_duplicates():
    session = make_session_factory("sqlite:///:memory:")()
    advance_cursor(session, 10)
    advance_cursor(session, 20)
    assert get_cursor(session) == 20
