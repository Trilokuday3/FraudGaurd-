"""Publishes batches of the existing realistic sample transaction pool
(deploy/sample_transactions.json) onto Kafka, cursoring through it like
serving/replay_worker.py's itertools.cycle does today -- but as a
scheduled, stateless run (GitHub Actions) rather than a standing loop,
so the cursor is persisted in Postgres between runs (streaming.cursor_store)
instead of held in process memory."""

import json


def load_sample_pool(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def select_batch(pool: list[dict], cursor: int, batch_size: int) -> tuple[list[dict], int]:
    """Return the next `batch_size` rows starting at `cursor`, wrapping
    around the pool, plus the cursor position to resume from next time."""
    n = len(pool)
    if n == 0:
        return [], cursor
    batch = [pool[(cursor + i) % n] for i in range(batch_size)]
    new_cursor = (cursor + batch_size) % n
    return batch, new_cursor
