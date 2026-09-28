"""Validate recently-scored transactions' feature rows against the same
pandera schema (fraudguard_core.schemas.features_schema) sub-project 1's
generator DQ suite (tests/dq/) validates training data with -- extends
automated data-quality checks to live/replayed traffic, not just the
offline dataset. Reuses the schema rather than duplicating it: FeatureRow
(serving/schemas.py) and features_schema describe the same 17 columns.

One check is relaxed for this context: features_schema requires a unique
transaction_id per row, which holds for offline generator output but not
for scored traffic -- the replay worker (serving/replay_worker.py) cycles
forever through a fixed sample file, so the same transaction is legitimately
re-scored many times over."""

import argparse

import pandas as pd
import pandera.errors as pa_errors
from fraudguard_core.schemas import features_schema
from pandera.pandas import Check

from serving.config import settings
from serving.models import Decision, make_session_factory

# features_schema checks these as 0/1 ints; FeatureRow/Decision.feature_row
# stores them as JSON booleans (true/false) -- cast before validating.
_BOOL_COLUMNS = [
    "is_night",
    "is_cross_border",
    "is_new_device",
    "is_new_country_for_customer",
    "ip_country_mismatch",
]

# features_schema enforces transaction_id uniqueness for offline generator
# output, where one row == one transaction. Scored/live traffic doesn't hold
# that invariant: the replay worker (serving/replay_worker.py) cycles
# forever through a fixed sample file, so the same transaction_id is
# legitimately re-scored -- and re-inserted as a new Decision row -- many
# times over. Drop just the uniqueness check for this context; keep every
# other column check as-is.
_SCORED_FEATURES_SCHEMA = features_schema.update_column(
    "transaction_id", unique=False, checks=Check.str_startswith("TXN")
)


def _load_recent_feature_rows(db_url: str, limit: int) -> pd.DataFrame:
    session_factory = make_session_factory(db_url)
    session = session_factory()
    try:
        records = session.query(Decision).order_by(Decision.id.desc()).limit(limit).all()
    finally:
        session.close()
    feature_rows = [r.feature_row for r in records if r.feature_row]
    return pd.DataFrame.from_records(feature_rows)


def check_scored_dq(db_url: str | None = None, limit: int = 500) -> dict:
    """Validate the `limit` most recently scored transactions' feature rows
    against `features_schema`. Returns a summary dict (never raises on
    schema violations -- those are the check's result, not an error in the
    check itself)."""
    db_url = db_url or settings.decision_db_url
    df = _load_recent_feature_rows(db_url, limit)

    if df.empty:
        return {"checked_rows": 0, "passed": True, "failures": []}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in _BOOL_COLUMNS:
        df[col] = df[col].astype(int)

    try:
        _SCORED_FEATURES_SCHEMA.validate(df, lazy=True)
        return {"checked_rows": len(df), "passed": True, "failures": []}
    except pa_errors.SchemaErrors as exc:
        failures = exc.failure_cases[["column", "check", "failure_case"]].to_dict(orient="records")
        return {"checked_rows": len(df), "passed": False, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(prog="check_scored_dq")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    result = check_scored_dq(limit=args.limit)
    status = "PASSED" if result["passed"] else "FAILED"
    print(f"Scored-traffic DQ check: {status} ({result['checked_rows']} rows checked)")
    for failure in result["failures"]:
        print(f"  - {failure['column']}: {failure['check']} (got {failure['failure_case']!r})")

    if not result["passed"]:
        # Non-zero exit so this fails the CI job that runs it -- printing
        # "FAILED" alone wouldn't fail the workflow step.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
