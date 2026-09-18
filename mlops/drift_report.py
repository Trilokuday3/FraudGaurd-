"""Evidently data-drift report: reference = a sample of the training data
(data/features.parquet), current = the most recently scored transactions'
feature rows from the Decision log (same source check_scored_dq.py reads).
Identifier/timestamp columns are excluded from the comparison -- drift on a
unique transaction_id is meaningless, only the 13 behavioral/feature
columns matter here."""

import argparse
import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from ml.data import load_features_and_labels
from mlops.check_scored_dq import _BOOL_COLUMNS, _load_recent_feature_rows
from serving.config import settings

_NON_FEATURE_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "timestamp"]


def _prepare_current(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in _BOOL_COLUMNS:
        df[col] = df[col].astype(int)
    return df


def _summarize(result) -> dict:
    drifted_columns_count = 0
    drifted_share = 0.0
    columns: dict[str, float] = {}
    for metric in result.dict()["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            drifted_columns_count = metric["value"]["count"]
            drifted_share = metric["value"]["share"]
        elif name.startswith("ValueDrift"):
            columns[metric["config"]["column"]] = metric["value"]
    return {
        "drifted_columns_count": drifted_columns_count,
        "drifted_share": drifted_share,
        "columns": columns,
    }


def generate_drift_report(
    data_dir: str = "./data",
    db_url: str | None = None,
    reference_sample_size: int = 2000,
    current_limit: int = 500,
    output_path: str = "./mlops/reports/drift_report.html",
) -> dict:
    """Compare recent scored traffic against the training reference
    distribution and save an Evidently HTML report. Returns a summary dict;
    if there's no scored traffic yet, skips the report entirely (nothing to
    compare) rather than erroring."""
    db_url = db_url or settings.decision_db_url

    current = _load_recent_feature_rows(db_url, current_limit)
    if current.empty:
        return {
            "checked_rows": 0,
            "drifted_columns_count": 0,
            "drifted_share": 0.0,
            "columns": {},
            "report_path": None,
        }
    current = _prepare_current(current)

    features, _ = load_features_and_labels(data_dir)
    reference = features.sample(min(reference_sample_size, len(features)), random_state=42)

    feature_columns = [c for c in reference.columns if c not in _NON_FEATURE_COLUMNS]
    reference = reference[feature_columns]
    current = current[feature_columns]

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.save_html(output_path)

    summary = _summarize(result)
    summary["checked_rows"] = len(current)
    summary["report_path"] = output_path
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="drift_report")
    parser.add_argument("--output", type=str, default="./mlops/reports/drift_report.html")
    args = parser.parse_args()

    result = generate_drift_report(output_path=args.output)
    if result["report_path"] is None:
        print("No scored traffic yet -- nothing to compare, skipped the report.")
        return
    print(
        f"Drift report: {result['drifted_columns_count']:.0f} drifted columns "
        f"({result['drifted_share']:.1%}) across {result['checked_rows']} scored rows. "
        f"Saved to {result['report_path']}"
    )


if __name__ == "__main__":
    main()
