"""Unit tests for run_schema_validation node.

Tests the Pandera schema validation node in the data_preparer LangGraph pipeline.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.schema_validator import (
    run_schema_validation,
)


def create_valid_business_metrics_df():
    """Create a valid business_metrics DataFrame for testing."""
    return pd.DataFrame(
        {
            "metric_id": ["M001", "M002", "M003"],
            "metric_date": [datetime.now(timezone.utc)] * 3,
            "metric_name": ["TRx", "NRx", "Share"],
            "metric_value": [100.5, 200.0, 0.35],
            "brand": ["Remibrutinib", "Fabhalta", "Kisqali"],
            "region": ["northeast", "south", "midwest"],
        }
    )


def create_invalid_business_metrics_df():
    """Create an invalid business_metrics DataFrame for testing."""
    return pd.DataFrame(
        {
            "metric_id": [None, "M002"],  # Null ID
            "metric_date": [datetime.now(timezone.utc)] * 2,
            "metric_name": ["TRx", "NRx"],
            "metric_value": [100.5, 200.0],
            "brand": ["InvalidBrand", "Fabhalta"],  # Invalid brand
            "region": ["northeast", "invalid_region"],  # Invalid region
        }
    )


def create_valid_predictions_df():
    """Create a valid predictions DataFrame for testing."""
    return pd.DataFrame(
        {
            "prediction_id": ["P001", "P002"],
            "prediction_date": [datetime.now(timezone.utc)] * 2,
            "model_id": ["model_v1", "model_v1"],
            "predicted_value": [0.85, 0.72],
            "confidence_score": [0.95, 0.88],
            "brand": ["Kisqali", "Fabhalta"],
        }
    )


class TestSchemaValidationPassed:
    """Test run_schema_validation with valid data."""

    @pytest.mark.asyncio
    async def test_validation_passes_with_valid_train_df(self):
        """Test validation passes when train_df is valid."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_valid_business_metrics_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "passed"
        assert result["schema_validation_errors"] == []
        assert result["schema_splits_validated"] == 1
        assert result["schema_validation_time_ms"] >= 0
        assert "blocking_issues" not in result or result.get("blocking_issues", []) == []

    @pytest.mark.asyncio
    async def test_validation_passes_with_all_splits(self):
        """Test validation passes when all splits are valid.

        Note: The schema validator only validates train, validation, and test
        splits (not holdout). This is by design - holdout data is kept
        completely separate for final model evaluation.
        """
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_valid_business_metrics_df(),
            "validation_df": create_valid_business_metrics_df(),
            "test_df": create_valid_business_metrics_df(),
            "holdout_df": create_valid_business_metrics_df(),  # Not validated
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "passed"
        # Only train, validation, test are validated (not holdout)
        assert result["schema_splits_validated"] == 3

    @pytest.mark.asyncio
    async def test_validation_passes_with_predictions_schema(self):
        """Test validation passes with predictions data source."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "predictions",
            "train_df": create_valid_predictions_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "passed"


class TestSchemaValidationFailed:
    """Test run_schema_validation with invalid data."""

    @pytest.mark.asyncio
    async def test_validation_fails_with_invalid_data(self):
        """Test validation fails when train_df has invalid data."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_invalid_business_metrics_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "failed"
        assert len(result["schema_validation_errors"]) > 0
        # Should add to blocking_issues
        assert "blocking_issues" in result
        assert len(result["blocking_issues"]) > 0

    @pytest.mark.asyncio
    async def test_failed_validation_blocks_downstream(self):
        """Test that schema failure adds blocking issues."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_invalid_business_metrics_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": ["existing_issue"],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "failed"
        # Should preserve existing blocking issues and add new ones
        assert "existing_issue" in result["blocking_issues"]
        # Should have added schema error
        has_schema_blocking = any(
            "schema" in issue.lower()
            for issue in result["blocking_issues"]
            if issue != "existing_issue"
        )
        assert has_schema_blocking


class TestSchemaValidationSkipped:
    """Test run_schema_validation skip scenarios."""

    @pytest.mark.asyncio
    async def test_validation_skipped_for_unknown_source(self):
        """Test validation is skipped for unknown data source."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "unknown_source",
            "train_df": pd.DataFrame({"col1": [1, 2, 3]}),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "skipped"
        assert result["schema_splits_validated"] == 0
        # Should NOT add blocking issues for skipped validation
        assert result.get("blocking_issues", []) == []

    @pytest.mark.asyncio
    async def test_validation_skipped_when_no_dataframes(self):
        """Test validation is skipped when all DataFrames are None."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": None,
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "skipped"
        assert result["schema_splits_validated"] == 0


class TestSchemaValidationError:
    """Test run_schema_validation error handling."""

    @pytest.mark.asyncio
    async def test_validation_handles_exception_gracefully(self):
        """Test validation returns error status on exception."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": "not_a_dataframe",  # Will cause an error
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "error"
        assert "error" in result or len(result["schema_validation_errors"]) > 0


class TestSchemaValidationPartialSuccess:
    """Test run_schema_validation with mixed valid/invalid splits."""

    @pytest.mark.asyncio
    async def test_fails_if_any_split_invalid(self):
        """Test validation fails if any split has invalid data."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_valid_business_metrics_df(),  # Valid
            "validation_df": create_invalid_business_metrics_df(),  # Invalid
            "test_df": create_valid_business_metrics_df(),  # Valid
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "failed"
        # All 3 non-None splits should still be validated
        assert result["schema_splits_validated"] == 3


class TestSchemaValidationTiming:
    """Test run_schema_validation timing metrics."""

    @pytest.mark.asyncio
    async def test_timing_is_recorded(self):
        """Test validation_time_ms is recorded."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_valid_business_metrics_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        assert "schema_validation_time_ms" in result
        assert isinstance(result["schema_validation_time_ms"], int)
        assert result["schema_validation_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_timing_under_sla(self):
        """Test validation completes quickly (under 100ms typical)."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_valid_business_metrics_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": [],
        }

        result = await run_schema_validation(state)

        # Pandera validation should be fast (~10ms)
        # Allow up to 1000ms for slow CI environments
        assert result["schema_validation_time_ms"] < 1000


class TestSchemaValidationStatePreservation:
    """Test that run_schema_validation preserves existing state."""

    @pytest.mark.asyncio
    async def test_existing_blocking_issues_preserved(self):
        """Test existing blocking_issues are preserved on success."""
        state = {
            "experiment_id": "exp_001",
            "data_source": "business_metrics",
            "train_df": create_valid_business_metrics_df(),
            "validation_df": None,
            "test_df": None,
            "holdout_df": None,
            "blocking_issues": ["pre_existing_issue"],
        }

        result = await run_schema_validation(state)

        assert result["schema_validation_status"] == "passed"
        # On success, blocking_issues should not be in result (not modified)
        # or if present, should not have added new issues
        if "blocking_issues" in result:
            # Only pre-existing issues
            assert (
                result["blocking_issues"] == ["pre_existing_issue"]
                or result["blocking_issues"] == []
            )
