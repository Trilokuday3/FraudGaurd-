"""Sample ~200 realistic rows from data/features.parquet, shaped exactly
like serving.schemas.FeatureRow, and write them to deploy/sample_transactions.json
for the deployed replay worker to cycle through (data/ itself is gitignored
and never deployed). See scripts/replay_transactions.py for the equivalent
local-dev tool this reuses the row-shaping logic from."""

import argparse
import json

import numpy as np
import pandas as pd

from serving.schemas import FeatureRow

FEATURE_ROW_FIELDS = list(FeatureRow.model_fields.keys())


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


def build_sample(features: pd.DataFrame, count: int, seed: int) -> list[dict]:
    """Deterministically sample `count` rows, projected to exactly
    FeatureRow's fields, JSON-serializable."""
    sampled = features[FEATURE_ROW_FIELDS].sample(n=count, random_state=seed)
    records = sampled.to_dict(orient="records")
    return [{k: _to_jsonable(v) for k, v in row.items()} for row in records]


def main() -> None:
    parser = argparse.ArgumentParser(prog="generate_sample_transactions")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./deploy/sample_transactions.json")
    args = parser.parse_args()

    features = pd.read_parquet(f"{args.data_dir}/features.parquet")
    sample = build_sample(features, args.count, args.seed)
    with open(args.output, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Wrote {len(sample)} sample transactions to {args.output}")


if __name__ == "__main__":
    main()
