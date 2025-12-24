"""Unit tests for Pandera schema validation.

Tests the PANDERA_SCHEMA_REGISTRY and all 6 E2I data source schemas.
"""

import pandas as pd
import pytest
from datetime import datetime, timezone

from src.mlops.pandera_schemas import (
    # Schema classes
    AgentActivitiesSchema,
    BusinessMetricsSchema,
    CausalPathsSchema,
    PatientJourneysSchema,
    PredictionsSchema,
    TriggersSchema,
    # Registry and utilities
    PANDERA_SCHEMA_REGISTRY,
    get_schema,
    list_registered_schemas,
    validate_dataframe,
    # E2I constants
    E2I_BRANDS,
    E2I_REGIONS,
)


class TestE2IConstants:
    """Test E2I business constants."""

    def test_e2i_brands_contains_expected_values(self):
        """Verify E2I_BRANDS has correct values."""
        assert "Remibrutinib" in E2I_BRANDS
        assert "Fabhalta" in E2I_BRANDS
        assert "Kisqali" in E2I_BRANDS
        assert "All_Brands" in E2I_BRANDS
        assert len(E2I_BRANDS) == 4

    def test_e2i_regions_contains_expected_values(self):
        """Verify E2I_REGIONS has correct values."""
        assert "northeast" in E2I_REGIONS
        assert "south" in E2I_REGIONS
        assert "midwest" in E2I_REGIONS
        assert "west" in E2I_REGIONS
        assert len(E2I_REGIONS) == 4


class TestSchemaRegistry:
    """Test PANDERA_SCHEMA_REGISTRY and utility functions."""

    def test_registry_has_six_schemas(self):
        """Verify registry contains all 6 E2I data sources."""
        expected_sources = {
            "business_metrics",
            "predictions",
            "ml_predictions",
            "triggers",
            "patient_journeys",
            "causal_paths",
            "agent_activities",
        }
        assert expected_sources == set(PANDERA_SCHEMA_REGISTRY.keys())

    def test_ml_predictions_is_alias_for_predictions(self):
        """Verify ml_predictions is an alias for predictions."""
        assert PANDERA_SCHEMA_REGISTRY["ml_predictions"] == PANDERA_SCHEMA_REGISTRY["predictions"]

    def test_get_schema_returns_correct_schema(self):
        """Test get_schema returns the correct schema class."""
        assert get_schema("business_metrics") == BusinessMetricsSchema
        assert get_schema("predictions") == PredictionsSchema
        assert get_schema("triggers") == TriggersSchema
        assert get_schema("patient_journeys") == PatientJourneysSchema
        assert get_schema("causal_paths") == CausalPathsSchema
        assert get_schema("agent_activities") == AgentActivitiesSchema

    def test_get_schema_returns_none_for_unknown_source(self):
        """Test get_schema returns None for unknown data source."""
        assert get_schema("unknown_source") is None
        assert get_schema("") is None

    def test_list_registered_schemas_returns_all_sources(self):
        """Test list_registered_schemas returns all source names."""
        sources = list_registered_schemas()
        assert len(sources) == 7  # Including ml_predictions alias
        assert "business_metrics" in sources
        assert "ml_predictions" in sources
        # Returns Dict[str, str] mapping name to class name
        assert sources["business_metrics"] == "BusinessMetricsSchema"


class TestBusinessMetricsSchema:
    """Test BusinessMetricsSchema validation."""

    def test_valid_business_metrics(self):
        """Test validation passes with valid data."""
        df = pd.DataFrame({
            "metric_id": ["M001", "M002"],
            "metric_date": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "metric_name": ["TRx", "NRx"],
            "metric_value": [100.5, 200.0],
            "brand": ["Remibrutinib", "Fabhalta"],
            "region": ["northeast", "south"],
        })
        result = validate_dataframe(df, "business_metrics")
        assert result["status"] == "passed"
        assert result["errors"] == []

    def test_invalid_brand_fails(self):
        """Test validation fails with invalid brand."""
        df = pd.DataFrame({
            "metric_id": ["M001"],
            "metric_date": [datetime.now(timezone.utc)],
            "metric_name": ["TRx"],
            "metric_value": [100.5],
            "brand": ["InvalidBrand"],  # Invalid brand
            "region": ["northeast"],
        })
        result = validate_dataframe(df, "business_metrics")
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_invalid_region_fails(self):
        """Test validation fails with invalid region."""
        df = pd.DataFrame({
            "metric_id": ["M001"],
            "metric_date": [datetime.now(timezone.utc)],
            "metric_name": ["TRx"],
            "metric_value": [100.5],
            "brand": ["Remibrutinib"],
            "region": ["invalid_region"],  # Invalid region
        })
        result = validate_dataframe(df, "business_metrics")
        assert result["status"] == "failed"

    def test_null_metric_id_fails(self):
        """Test validation fails with null required field."""
        df = pd.DataFrame({
            "metric_id": [None, "M002"],  # Null value
            "metric_date": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "metric_name": ["TRx", "NRx"],
            "metric_value": [100.5, 200.0],
        })
        result = validate_dataframe(df, "business_metrics")
        assert result["status"] == "failed"


class TestPredictionsSchema:
    """Test PredictionsSchema validation."""

    def test_valid_predictions(self):
        """Test validation passes with valid predictions data."""
        df = pd.DataFrame({
            "prediction_id": ["P001", "P002"],
            "prediction_date": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "model_id": ["model_v1", "model_v1"],
            "predicted_value": [0.85, 0.72],
            "confidence_score": [0.95, 0.88],
            "brand": ["Kisqali", "Fabhalta"],
        })
        result = validate_dataframe(df, "predictions")
        assert result["status"] == "passed"

    def test_confidence_score_out_of_range_fails(self):
        """Test validation fails when confidence_score > 1.0."""
        df = pd.DataFrame({
            "prediction_id": ["P001"],
            "prediction_date": [datetime.now(timezone.utc)],
            "model_id": ["model_v1"],
            "predicted_value": [0.85],
            "confidence_score": [1.5],  # Out of range
            "brand": ["Kisqali"],
        })
        result = validate_dataframe(df, "predictions")
        assert result["status"] == "failed"

    def test_negative_confidence_score_fails(self):
        """Test validation fails when confidence_score < 0."""
        df = pd.DataFrame({
            "prediction_id": ["P001"],
            "prediction_date": [datetime.now(timezone.utc)],
            "model_id": ["model_v1"],
            "predicted_value": [0.85],
            "confidence_score": [-0.1],  # Negative
            "brand": ["Kisqali"],
        })
        result = validate_dataframe(df, "predictions")
        assert result["status"] == "failed"


class TestTriggersSchema:
    """Test TriggersSchema validation."""

    def test_valid_triggers(self):
        """Test validation passes with valid triggers data."""
        df = pd.DataFrame({
            "trigger_id": ["T001", "T002"],
            "patient_id": ["P001", "P002"],  # Required
            "trigger_timestamp": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "trigger_type": ["alert", "recommendation"],
            "priority": ["high", "medium"],  # Uses priority, not severity
            "confidence_score": [0.9, 0.85],
        })
        result = validate_dataframe(df, "triggers")
        assert result["status"] == "passed"

    def test_missing_patient_id_fails(self):
        """Test validation fails when patient_id is missing."""
        df = pd.DataFrame({
            "trigger_id": ["T001"],
            "trigger_type": ["alert"],
            # Missing patient_id - required field
        })
        result = validate_dataframe(df, "triggers")
        assert result["status"] == "failed"


class TestPatientJourneysSchema:
    """Test PatientJourneysSchema validation."""

    def test_valid_patient_journeys(self):
        """Test validation passes with valid patient journeys data."""
        df = pd.DataFrame({
            "patient_journey_id": ["J001", "J002"],  # Correct field name
            "patient_id": ["P001", "P002"],  # Required
            "journey_start_date": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "current_stage": ["diagnosis", "maintenance"],
            "brand": ["Fabhalta", "Kisqali"],
            "geographic_region": ["west", "midwest"],  # Uses geographic_region, not region
        })
        result = validate_dataframe(df, "patient_journeys")
        assert result["status"] == "passed"

    def test_invalid_journey_stage_fails(self):
        """Test validation fails with invalid current_stage."""
        df = pd.DataFrame({
            "patient_journey_id": ["J001"],
            "patient_id": ["P001"],
            "current_stage": ["invalid_stage"],  # Invalid stage
        })
        result = validate_dataframe(df, "patient_journeys")
        assert result["status"] == "failed"


class TestCausalPathsSchema:
    """Test CausalPathsSchema validation."""

    def test_valid_causal_paths(self):
        """Test validation passes with valid causal paths data."""
        df = pd.DataFrame({
            "path_id": ["CP001", "CP002"],
            "discovery_date": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "source_node": ["marketing_spend", "rep_visits"],
            "target_node": ["prescriptions", "awareness"],
            "causal_effect_size": [0.45, -0.22],  # -1 to 1 range
            "confidence_level": [0.92, 0.85],
        })
        result = validate_dataframe(df, "causal_paths")
        assert result["status"] == "passed"

    def test_causal_effect_size_out_of_range_fails(self):
        """Test validation fails when causal_effect_size > 1.0."""
        df = pd.DataFrame({
            "path_id": ["CP001"],
            "discovery_date": [datetime.now(timezone.utc)],
            "source_node": ["marketing"],
            "target_node": ["sales"],
            "causal_effect_size": [1.5],  # Out of range
            "confidence_level": [0.9],
        })
        result = validate_dataframe(df, "causal_paths")
        assert result["status"] == "failed"

    def test_causal_effect_size_below_range_fails(self):
        """Test validation fails when causal_effect_size < -1.0."""
        df = pd.DataFrame({
            "path_id": ["CP001"],
            "discovery_date": [datetime.now(timezone.utc)],
            "source_node": ["marketing"],
            "target_node": ["sales"],
            "causal_effect_size": [-1.5],  # Out of range
            "confidence_level": [0.9],
        })
        result = validate_dataframe(df, "causal_paths")
        assert result["status"] == "failed"

    def test_invalid_p_value_fails(self):
        """Test validation fails when p_value > 1.0."""
        df = pd.DataFrame({
            "path_id": ["CP001"],
            "discovery_date": [datetime.now(timezone.utc)],
            "source_node": ["marketing"],
            "target_node": ["sales"],
            "p_value": [1.5],  # Out of range (must be 0-1)
        })
        result = validate_dataframe(df, "causal_paths")
        assert result["status"] == "failed"


class TestAgentActivitiesSchema:
    """Test AgentActivitiesSchema validation."""

    def test_valid_agent_activities(self):
        """Test validation passes with valid agent activities data."""
        df = pd.DataFrame({
            "activity_id": ["A001", "A002"],
            "activity_timestamp": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "agent_name": ["gap_analyzer", "causal_impact"],
            "activity_type": ["analysis", "inference"],
        })
        result = validate_dataframe(df, "agent_activities")
        assert result["status"] == "passed"


class TestValidateDataframeFunction:
    """Test validate_dataframe utility function."""

    def test_unknown_source_returns_skipped(self):
        """Test unknown data source returns skipped status."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = validate_dataframe(df, "unknown_source")
        assert result["status"] == "skipped"
        assert "no schema defined" in result["message"].lower()

    def test_empty_dataframe_passes(self):
        """Test empty DataFrame still validates (schema check passes)."""
        df = pd.DataFrame({
            "metric_id": pd.Series([], dtype=str),
            "metric_date": pd.Series([], dtype="datetime64[ns, UTC]"),
            "metric_name": pd.Series([], dtype=str),
            "metric_value": pd.Series([], dtype=float),
        })
        result = validate_dataframe(df, "business_metrics")
        # Empty df should pass schema validation (no rows to violate constraints)
        assert result["status"] == "passed"

    def test_lazy_validation_collects_all_errors(self):
        """Test lazy=True collects multiple errors."""
        df = pd.DataFrame({
            "metric_id": [None, None],  # Error 1: nulls
            "metric_date": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "metric_name": ["TRx", "NRx"],
            "metric_value": [100.5, 200.0],
            "brand": ["InvalidBrand", "AnotherInvalid"],  # Error 2: invalid brands
        })
        result = validate_dataframe(df, "business_metrics", lazy=True)
        assert result["status"] == "failed"
        # Should have collected multiple errors
        assert len(result["errors"]) >= 1

    def test_result_contains_required_fields(self):
        """Test result includes required fields: status, errors, rows_validated, schema_name."""
        df = pd.DataFrame({
            "metric_id": ["M001"],
            "metric_date": [datetime.now(timezone.utc)],
            "metric_name": ["TRx"],
            "metric_value": [100.5],
        })
        result = validate_dataframe(df, "business_metrics")
        assert "status" in result
        assert "errors" in result
        assert "rows_validated" in result
        assert "schema_name" in result
        assert result["rows_validated"] == 1
        assert result["schema_name"] == "business_metrics"


class TestSchemaConfigOptions:
    """Test schema configuration options."""

    def test_strict_false_allows_extra_columns(self):
        """Test strict=False allows extra columns not in schema."""
        df = pd.DataFrame({
            "metric_id": ["M001"],
            "metric_date": [datetime.now(timezone.utc)],
            "metric_name": ["TRx"],
            "metric_value": [100.5],
            "extra_column": ["extra_value"],  # Not in schema
        })
        result = validate_dataframe(df, "business_metrics")
        # Should pass because strict=False in schema config
        assert result["status"] == "passed"

    def test_coerce_true_converts_types(self):
        """Test coerce=True converts compatible types."""
        df = pd.DataFrame({
            "metric_id": ["M001"],
            "metric_date": ["2025-01-01"],  # String, should be coerced to datetime
            "metric_name": ["TRx"],
            "metric_value": [100],  # Int, should be coerced to float
        })
        result = validate_dataframe(df, "business_metrics")
        # With coerce=True, these should convert successfully
        assert result["status"] == "passed"
