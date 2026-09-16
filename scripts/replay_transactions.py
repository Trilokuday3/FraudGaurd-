"""Replay data/features.parquet rows into the running API's /score endpoint
at a slow, visible cadence, so Live Transactions has a genuinely growing
feed to poll. Local-dev stand-in for sub-project 5's skipped streaming
pipeline -- see docs/superpowers/specs/2026-09-15-frontend-design.md."""

import argparse
import time

import numpy as np
import pandas as pd
import requests


def _to_jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def select_rows(features: pd.DataFrame, count: int | None, loop: bool) -> list[dict]:
    """Pick the rows to replay, in file order. `loop` doesn't change which
    rows are selected -- looping back to the start happens in `replay`."""
    rows = features.to_dict(orient="records")
    if count is not None:
        rows = rows[:count]
    return rows


def replay(
    rows: list[dict],
    api_base_url: str,
    interval_seconds: float,
    loop: bool,
    post=requests.post,
    sleep=time.sleep,
) -> int:
    """POST each row to `{api_base_url}/score`, `interval_seconds` apart.

    `post`/`sleep` are injectable so this loop is unit-testable without a
    real network call or a real wait.
    """
    sent = 0
    while True:
        for row in rows:
            payload = {k: _to_jsonable(v) for k, v in row.items()}
            response = post(f"{api_base_url}/score", json=payload, timeout=10)
            response.raise_for_status()
            sent += 1
            sleep(interval_seconds)
        if not loop:
            break
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(prog="replay_transactions")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--api-base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--interval", type=float, default=2.5)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    features = pd.read_parquet(f"{args.data_dir}/features.parquet")
    rows = select_rows(features, args.count, args.loop)
    print(
        f"Replaying {len(rows)} transactions to {args.api_base_url}/score "
        f"every {args.interval}s (loop={args.loop})..."
    )
    sent = replay(rows, args.api_base_url, args.interval, args.loop)
    print(f"Done. Sent {sent} transactions.")


if __name__ == "__main__":
    main()
