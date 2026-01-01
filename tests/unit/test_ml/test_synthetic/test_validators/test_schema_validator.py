"""
Tests for SchemaValidator.

Tests schema validation against Supabase table definitions.
"""

import pytest
import pandas as pd
import numpy as np

from src.ml.synthetic.validators.schema_validator import (
    SchemaValidator,
    SchemaValidationResult,
    TABLE_SCHEMAS,
)


class TestSchemaValidator:
    """Test suite for SchemaValidator."""

    @pytest.fixture
    def validator(self):
        """Create a SchemaValidator instance."""
        return SchemaValidator()

    @pytest.fixture
    def valid_hcp_df(self):
        """Create a valid HCP profiles DataFrame."""
        return pd.DataFrame({
            "hcp_id": ["hcp_00001", "hcp_00002", "hcp_00003"],
            "npi": ["1234567890", "1234567891", "1234567892"],
            "specialty": ["dermatology", "hematology", "oncology"],
            "practice_type": ["academic", "community", "private"],
            "geographic_region": ["northeast", "south", "midwest"],
            "years_experience": [10, 20, 5],
            "academic_hcp": [1, 0, 0],
            "total_patient_volume": [200, 300, 150],
            "brand": ["Remibrutinib", "Fabhalta", "Kisqali"],
        })

    @pytest.fixture
    def valid_patient_df(self):
        """Create a valid patient journeys DataFrame."""
        return pd.DataFrame({
            "patient_journey_id": ["patient_000001", "patient_000002"],
            "patient_id": ["pt_000001", "pt_000002"],
            "hcp_id": ["hcp_00001", "hcp_00002"],
            "brand": ["Remibrutinib", "Fabhalta"],
            "journey_start_date": ["2023-01-01", "2023-06-15"],
            "data_split": ["train", "validation"],
            "disease_severity": [5.5, 7.2],
            "academic_hcp": [1, 0],
            "engagement_score": [6.5, 8.0],
            "treatment_initiated": [1, 0],
            "days_to_treatment": [30.0, None],
            "geographic_region": ["northeast", "south"],
            "insurance_type": ["commercial", "medicare"],
            "age_at_diagnosis": [45, 62],
        })

    def test_validate_valid_hcp_profiles(self, validator, valid_hcp_df):
        """Test validation of valid HCP profiles."""
        result = validator.validate(valid_hcp_df, "hcp_profiles")

        assert result.is_valid is True
        assert result.total_rows == 3
        assert len(result.errors) == 0

    def test_validate_valid_patient_journeys(self, validator, valid_patient_df):
        """Test validation of valid patient journeys."""
        result = validator.validate(valid_patient_df, "patient_journeys")

        assert result.is_valid is True
        assert result.total_rows == 2
        assert len(result.errors) == 0

    def test_validate_missing_required_column(self, validator, valid_hcp_df):
        """Test detection of missing required column."""
        df = valid_hcp_df.drop(columns=["specialty"])
        result = validator.validate(df, "hcp_profiles")

        assert result.is_valid is False
        assert any(e.error_type == "missing_column" for e in result.errors)

    def test_validate_null_in_non_nullable_column(self, validator, valid_hcp_df):
        """Test detection of NULL in non-nullable column."""
        df = valid_hcp_df.copy()
        df.loc[0, "specialty"] = None

        result = validator.validate(df, "hcp_profiles")

        assert result.is_valid is False
        assert any(e.error_type == "null_value" for e in result.errors)

    def test_validate_invalid_enum_value(self, validator, valid_hcp_df):
        """Test detection of invalid enum value."""
        df = valid_hcp_df.copy()
        df.loc[0, "specialty"] = "invalid_specialty"

        result = validator.validate(df, "hcp_profiles")

        assert result.is_valid is False
        assert any(e.error_type == "invalid_enum" for e in result.errors)

    def test_validate_value_below_minimum(self, validator, valid_hcp_df):
        """Test detection of value below minimum."""
        df = valid_hcp_df.copy()
        df.loc[0, "years_experience"] = -5

        result = validator.validate(df, "hcp_profiles")

        assert result.is_valid is False
        assert any(e.error_type == "below_min" for e in result.errors)

    def test_validate_value_above_maximum(self, validator, valid_patient_df):
        """Test detection of value above maximum."""
        df = valid_patient_df.copy()
        df.loc[0, "disease_severity"] = 15  # Max is 10

        result = validator.validate(df, "patient_journeys")

        assert result.is_valid is False
        assert any(e.error_type == "above_max" for e in result.errors)

    def test_validate_foreign_key_integrity(self, validator, valid_hcp_df, valid_patient_df):
        """Test foreign key validation."""
        # Create patient with non-existent HCP
        df = valid_patient_df.copy()
        df.loc[0, "hcp_id"] = "non_existent_hcp"

        reference_dfs = {"hcp_profiles": valid_hcp_df}
        result = validator.validate(df, "patient_journeys", reference_dfs=reference_dfs)

        assert result.is_valid is False
        assert any(e.error_type == "orphan_fk" for e in result.errors)

    def test_validate_unknown_table(self, validator, valid_hcp_df):
        """Test validation against unknown table."""
        result = validator.validate(valid_hcp_df, "unknown_table")

        assert result.is_valid is False
        assert any(e.error_type == "unknown_table" for e in result.errors)

    def test_validate_all_datasets(self, validator, valid_hcp_df, valid_patient_df):
        """Test validation of all datasets at once."""
        datasets = {
            "hcp_profiles": valid_hcp_df,
            "patient_journeys": valid_patient_df,
        }

        results = validator.validate_all(datasets)

        assert len(results) == 2
        assert results["hcp_profiles"].is_valid is True
        assert results["patient_journeys"].is_valid is True

    def test_get_validation_summary(self, validator, valid_hcp_df, valid_patient_df):
        """Test validation summary generation."""
        datasets = {
            "hcp_profiles": valid_hcp_df,
            "patient_journeys": valid_patient_df,
        }

        results = validator.validate_all(datasets)
        summary = validator.get_validation_summary(results)

        assert summary["all_valid"] is True
        assert summary["total_tables"] == 2
        assert summary["valid_tables"] == 2
        assert summary["total_errors"] == 0


class TestSchemaValidationResult:
    """Test suite for SchemaValidationResult."""

    def test_add_error_sets_invalid(self):
        """Test that adding an error sets is_valid to False."""
        result = SchemaValidationResult(
            is_valid=True, table_name="test", total_rows=100
        )

        result.add_error("test_col", "test_type", "Test message")

        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_add_warning_preserves_validity(self):
        """Test that adding a warning doesn't affect is_valid."""
        result = SchemaValidationResult(
            is_valid=True, table_name="test", total_rows=100
        )

        result.add_warning("Test warning")

        assert result.is_valid is True
        assert len(result.warnings) == 1
