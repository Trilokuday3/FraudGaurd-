"""CLI: python -m features build [--data-dir DIR] [--output-dir DIR]"""

import argparse
import os

import pandas as pd

from features.build import build_feature_table, write_features


def _load(data_dir: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="features")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="Build the gold feature table")
    build_cmd.add_argument("--data-dir", type=str, default="./data")
    build_cmd.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.command == "build":
        output_dir = args.output_dir or args.data_dir
        transactions = _load(args.data_dir, "transactions")
        customers = _load(args.data_dir, "customers")
        devices = _load(args.data_dir, "devices")
        ground_truth = _load(args.data_dir, "ground_truth")

        table = build_feature_table(transactions, customers, devices, ground_truth)
        path = write_features(table, output_dir)
        print(f"Wrote {len(table)} rows to {path}")


if __name__ == "__main__":
    main()
