"""In-process background task that replays deploy/sample_transactions.json
through the same scoring path as /score, on a fixed interval, forever. Runs
inside the API's own process (via serving/app.py's lifespan) rather than a
separate service -- see docs/superpowers/specs/2026-09-16-deployment-portfolio-design.md,
"Replay worker", for why: Render's free tier has no cost-free way to run a
standalone background worker."""

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from typing import Any


async def replay_worker_loop(
    compute_score: Callable,
    session_factory: Callable,
    rows: list[Any],
    interval_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Cycle through `rows` forever, scoring and persisting each one.

    Schema-agnostic on purpose: `rows` is whatever `compute_score` expects
    (the real caller, `serving.app._compute_score`, expects `FeatureRow`
    instances, not plain dicts -- confirmed the hard way: passing plain
    dicts raised `AttributeError: 'dict' object has no attribute
    'model_dump'` inside this loop). `compute_score(row)` must return
    `(response, record)` where `record` is ready to `session.add()` -- this
    matches `serving.app._compute_score`'s existing signature exactly, so
    no adapter is needed at the call site. Runs until cancelled (the normal
    way an asyncio task wired into FastAPI's lifespan is stopped on
    shutdown).
    """
    for row in itertools.cycle(rows):
        _response, record = compute_score(row)
        session = session_factory()
        try:
            session.add(record)
            session.commit()
        finally:
            session.close()
        await sleep(interval_seconds)
