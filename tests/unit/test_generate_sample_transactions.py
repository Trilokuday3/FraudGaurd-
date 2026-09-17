"""Tests for scripts/generate_sample_transactions.py."""

import pandas as pd

from scripts.generate_sample_transactions import FEATURE_ROW_FIELDS, build_sample
from serving.schemas import FeatureRow


def test_build_sample_matches_feature_row_shape():
    rows = [{field: _dummy_value(field) for field in FEATURE_ROW_FIELDS} for _ in range(10)]
    features = pd.DataFrame(rows)

    sample = build_sample(features, count=5, seed=0)

    assert len(sample) == 5
    for row in sample:
        FeatureRow(**row)  # raises if any field is missing or mistyped


def _dummy_value(field: str):
    field_type = FeatureRow.model_fields[field].annotation
    if field_type is str:
        return "x"
    if field_type is bool:
        return False
    if field_type is int:
        return 1
    if field_type is float:
        return 1.0
    return pd.Timestamp("2026-01-01T00:00:00Z")
