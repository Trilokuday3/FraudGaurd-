"""Postgres-backed cursor tracking the producer's position in the
replayed sample pool. A single row (id=1) -- there is only ever one
producer for one pool."""

from sqlalchemy.orm import Session

from serving.models import StreamCursor

_CURSOR_ID = 1


def get_cursor(session: Session) -> int:
    row = session.get(StreamCursor, _CURSOR_ID)
    return row.position if row is not None else 0


def advance_cursor(session: Session, new_position: int) -> None:
    row = session.get(StreamCursor, _CURSOR_ID)
    if row is None:
        session.add(StreamCursor(id=_CURSOR_ID, position=new_position))
    else:
        row.position = new_position
    session.commit()
