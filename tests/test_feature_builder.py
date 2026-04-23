from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_builder import F1FeatureBuilder


def test_feature_builder_adds_historical_columns(synthetic_raw_df):
    builder = F1FeatureBuilder()
    enriched = builder.build(synthetic_raw_df)

    assert "driver_recent_form_score" in enriched.columns
    assert "constructor_recent_form_score" in enriched.columns
    assert "track_specific_driver_score" in enriched.columns
    assert "field_strength" in enriched.columns


def test_feature_builder_keeps_first_history_rows_null(synthetic_raw_df):
    builder = F1FeatureBuilder()
    enriched = builder.build(synthetic_raw_df)

    first_ver = enriched[enriched["driver"] == "VER"].sort_values(["year", "round"]).iloc[0]
    later_ver = enriched[enriched["driver"] == "VER"].sort_values(["year", "round"]).iloc[3]

    assert str(first_ver["driver_recent_form_score"]) == "nan"
    assert later_ver["driver_recent_form_score"] == later_ver["driver_recent_form_score"]
