"""
Unit tests for src/ml/synthetic/validation/pipeline.py

Tests the validation pipeline integration for synthetic data.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.ml.synthetic.config import DGPType
from src.ml.synthetic.validation.pipeline import (
    PipelineValidationResult,
    get_combined_summary,
    quick_validate,
    validate_and_log,
    validate_dataset,
    validate_pipeline_output,
)


@pytest.mark.unit
class TestPipelineValidationResult:
    """Test PipelineValidationResult dataclass."""

    def test_pipeline_validation_result_creation(self):
        """Test PipelineValidationResult creation."""
        result = PipelineValidationResult(
            table_name="patient_journeys",
            pandera_valid=True,
        )

        assert result.table_name == "patient_journeys"
        assert result.pandera_valid is True
        assert result.pandera_errors is None
        assert result.gx_result is None

    def test_is_valid_both_pass(self):
        """Test is_valid when both validations pass."""
        mock_gx_result = Mock()
        mock_gx_result.success = True

        result = PipelineValidationResult(
            table_name="test_table",
            pandera_valid=True,
            gx_result=mock_gx_result,
        )

        assert result.is_valid is True

    def test_is_valid_pandera_fails(self):
        """Test is_valid when Pandera validation fails."""
        mock_gx_result = Mock()
        mock_gx_result.success = True

        result = PipelineValidationResult(
            table_name="test_table",
            pandera_valid=False,
            gx_result=mock_gx_result,
        )

        assert result.is_valid is False

    def test_is_valid_gx_fails(self):
        """Test is_valid when GX validation fails."""
        mock_gx_result = Mock()
        mock_gx_result.success = False

        result = PipelineValidationResult(
            table_name="test_table",
            pandera_valid=True,
            gx_result=mock_gx_result,
        )

        assert result.is_valid is False

    def test_is_valid_no_gx(self):
        """Test is_valid when GX is not run."""
        result = PipelineValidationResult(
            table_name="test_table",
            pandera_valid=True,
            gx_result=None,
        )

        assert result.is_valid is True

    def test_summary_property(self):
        """Test summary property."""
        mock_gx_result = Mock()
        mock_gx_result.success = True
        mock_gx_result.success_rate = 0.95

        result = PipelineValidationResult(
            table_name="test_table",
            pandera_valid=True,
            gx_result=mock_gx_result,
        )

        summary = result.summary

        assert summary["table_name"] == "test_table"
        assert summary["pandera_valid"] is True
        assert summary["gx_valid"] is True
        assert summary["gx_success_rate"] == 0.95
        assert summary["overall_valid"] is True
        assert "timestamp" in summary


@pytest.mark.unit
class TestValidateDataset:
    """Test validate_dataset function."""

    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe")
    @patch("src.ml.synthetic.validation.pipeline.GX_AVAILABLE", True)
    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe_with_expectations")
    @patch("src.ml.synthetic.validation.pipeline.SCHEMA_REGISTRY", {"test_table": Mock()})
    def test_validate_dataset_both_validations(self, mock_gx_validate, mock_pandera_validate):
        """Test validate_dataset with both Pandera and GX."""
        df = pd.DataFrame({"id": [1, 2, 3]})

        # Mock Pandera validation
        mock_pandera_validate.return_value = (True, None)

        # Mock GX validation
        mock_gx_result = Mock()
        mock_gx_result.success = True
        mock_gx_validate.return_value = mock_gx_result

        result = validate_dataset(df, "test_table", run_pandera=True, run_gx=True)

        assert result.table_name == "test_table"
        assert result.pandera_valid is True
        assert result.gx_result == mock_gx_result

        mock_pandera_validate.assert_called_once()
        mock_gx_validate.assert_called_once()

    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe")
    @patch("src.ml.synthetic.validation.pipeline.SCHEMA_REGISTRY", {"test_table": Mock()})
    def test_validate_dataset_pandera_only(self, mock_pandera_validate):
        """Test validate_dataset with only Pandera."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        mock_pandera_validate.return_value = (True, None)

        result = validate_dataset(df, "test_table", run_pandera=True, run_gx=False)

        assert result.pandera_valid is True
        assert result.gx_result is None

    @patch("src.ml.synthetic.validation.pipeline.GX_AVAILABLE", True)
    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe_with_expectations")
    def test_validate_dataset_gx_only(self, mock_gx_validate):
        """Test validate_dataset with only GX."""
        df = pd.DataFrame({"id": [1, 2, 3]})

        mock_gx_result = Mock()
        mock_gx_result.success = True
        mock_gx_validate.return_value = mock_gx_result

        result = validate_dataset(df, "test_table", run_pandera=False, run_gx=True)

        assert result.pandera_valid is True  # Default
        assert result.gx_result == mock_gx_result

    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe")
    @patch("src.ml.synthetic.validation.pipeline.SCHEMA_REGISTRY", {"test_table": Mock()})
    def test_validate_dataset_pandera_failure(self, mock_pandera_validate):
        """Test validate_dataset when Pandera validation fails."""
        df = pd.DataFrame({"id": [1, 2, 3]})

        mock_errors = Mock()
        mock_pandera_validate.return_value = (False, mock_errors)

        result = validate_dataset(df, "test_table", run_pandera=True, run_gx=False)

        assert result.pandera_valid is False
        assert result.pandera_errors == mock_errors

    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe")
    @patch("src.ml.synthetic.validation.pipeline.SCHEMA_REGISTRY", {})
    def test_validate_dataset_table_not_in_registry(self, mock_pandera_validate):
        """Test validate_dataset when table not in schema registry."""
        df = pd.DataFrame({"id": [1, 2, 3]})

        result = validate_dataset(df, "unknown_table", run_pandera=True, run_gx=False)

        # Should skip Pandera validation
        mock_pandera_validate.assert_not_called()
        assert result.pandera_valid is True  # Default

    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe")
    @patch("src.ml.synthetic.validation.pipeline.GX_AVAILABLE", True)
    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe_with_expectations")
    @patch("src.ml.synthetic.validation.pipeline.SCHEMA_REGISTRY", {"test_table": Mock()})
    def test_validate_dataset_with_dgp_type(self, mock_gx_validate, mock_pandera_validate):
        """Test validate_dataset passes DGP type to GX."""
        df = pd.DataFrame({"id": [1, 2, 3]})

        mock_pandera_validate.return_value = (True, None)
        mock_gx_result = Mock()
        mock_gx_validate.return_value = mock_gx_result

        validate_dataset(df, "test_table", dgp_type=DGPType.SIMPLE_LINEAR, run_gx=True)

        # Check that DGP type was passed to GX validation
        mock_gx_validate.assert_called_once_with(df, "test_table", DGPType.SIMPLE_LINEAR)


@pytest.mark.unit
class TestValidatePipelineOutput:
    """Test validate_pipeline_output function."""

    @patch("src.ml.synthetic.validation.pipeline.validate_dataset")
    @patch("src.ml.synthetic.validation.pipeline.ValidationObserver")
    @patch("src.ml.synthetic.validation.pipeline.create_validation_span")
    def test_validate_pipeline_output_basic(self, mock_span, mock_observer, mock_validate_dataset):
        """Test validate_pipeline_output basic functionality."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
            "table2": pd.DataFrame({"id": [4, 5, 6]}),
        }

        # Mock validation results
        mock_result1 = Mock()
        mock_result1.is_valid = True
        mock_result1.gx_result = None

        mock_result2 = Mock()
        mock_result2.is_valid = True
        mock_result2.gx_result = None

        mock_validate_dataset.side_effect = [mock_result1, mock_result2]

        # Mock span context manager
        mock_span_ctx = Mock()
        mock_span_ctx.__enter__ = Mock(return_value=mock_span_ctx)
        mock_span_ctx.__exit__ = Mock(return_value=False)
        mock_span.return_value = mock_span_ctx

        # Mock observer
        mock_obs_instance = Mock()
        mock_obs_instance.finalize.return_value = {"mlflow_run_id": "test-run-id"}
        mock_observer.return_value = mock_obs_instance

        results, obs_summary = validate_pipeline_output(datasets, enable_observability=True)

        assert len(results) == 2
        assert "table1" in results
        assert "table2" in results
        assert obs_summary["mlflow_run_id"] == "test-run-id"

    @patch("src.ml.synthetic.validation.pipeline.validate_dataset")
    @patch("src.ml.synthetic.validation.pipeline.create_validation_span")
    def test_validate_pipeline_output_no_observability(self, mock_span, mock_validate_dataset):
        """Test validate_pipeline_output without observability."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
        }

        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.gx_result = None
        mock_validate_dataset.return_value = mock_result

        # Mock span context manager
        mock_span_ctx = Mock()
        mock_span_ctx.__enter__ = Mock(return_value=mock_span_ctx)
        mock_span_ctx.__exit__ = Mock(return_value=False)
        mock_span.return_value = mock_span_ctx

        results, obs_summary = validate_pipeline_output(datasets, enable_observability=False)

        assert len(results) == 1
        assert obs_summary == {}

    @patch("src.ml.synthetic.validation.pipeline.validate_dataset")
    @patch("src.ml.synthetic.validation.pipeline.ValidationObserver")
    @patch("src.ml.synthetic.validation.pipeline.create_validation_span")
    def test_validate_pipeline_output_with_gx_results(
        self, mock_span, mock_observer, mock_validate_dataset
    ):
        """Test validate_pipeline_output logs GX results."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
        }

        # Mock GX result
        mock_gx_result = Mock()
        mock_gx_result.success = True

        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.gx_result = mock_gx_result
        mock_validate_dataset.return_value = mock_result

        # Mock span
        mock_span_ctx = Mock()
        mock_span_ctx.__enter__ = Mock(return_value=mock_span_ctx)
        mock_span_ctx.__exit__ = Mock(return_value=False)
        mock_span.return_value = mock_span_ctx

        # Mock observer
        mock_obs_instance = Mock()
        mock_obs_instance.finalize.return_value = {}
        mock_observer.return_value = mock_obs_instance

        results, obs_summary = validate_pipeline_output(datasets, enable_observability=True)

        # Verify GX result was logged
        mock_span_ctx.log_result.assert_called_once_with(mock_gx_result)
        mock_obs_instance.record_result.assert_called_once_with("table1", mock_gx_result)

    @patch("src.ml.synthetic.validation.pipeline.validate_dataset")
    @patch("src.ml.synthetic.validation.pipeline.create_validation_span")
    def test_validate_pipeline_output_empty_datasets(self, mock_span, mock_validate_dataset):
        """Test validate_pipeline_output with empty datasets dict."""
        datasets = {}

        # Mock span context manager
        mock_span_ctx = Mock()
        mock_span_ctx.__enter__ = Mock(return_value=mock_span_ctx)
        mock_span_ctx.__exit__ = Mock(return_value=False)
        mock_span.return_value = mock_span_ctx

        results, obs_summary = validate_pipeline_output(datasets, enable_observability=False)

        assert results == {}
        assert obs_summary == {}


@pytest.mark.unit
class TestGetCombinedSummary:
    """Test get_combined_summary function."""

    def test_get_combined_summary_all_pass(self):
        """Test summary generation when all validations pass."""
        results = {
            "table1": PipelineValidationResult(
                table_name="table1",
                pandera_valid=True,
                gx_result=Mock(success=True, success_rate=1.0),
            ),
            "table2": PipelineValidationResult(
                table_name="table2",
                pandera_valid=True,
                gx_result=Mock(success=True, success_rate=1.0),
            ),
        }

        summary = get_combined_summary(results)

        assert "ALL PASSED" in summary
        assert "table1" in summary
        assert "table2" in summary

    def test_get_combined_summary_pandera_failure(self):
        """Test summary generation with Pandera failures."""
        mock_errors = Mock()
        mock_errors.failure_cases = [{"error": "test"}]

        results = {
            "table1": PipelineValidationResult(
                table_name="table1",
                pandera_valid=False,
                pandera_errors=mock_errors,
            ),
        }

        summary = get_combined_summary(results)

        assert "FAIL" in summary
        assert "VALIDATION FAILED" in summary

    def test_get_combined_summary_gx_failure(self):
        """Test summary generation with GX failures."""
        results = {
            "table1": PipelineValidationResult(
                table_name="table1",
                pandera_valid=True,
                gx_result=Mock(success=False, success_rate=0.5),
            ),
        }

        summary = get_combined_summary(results)

        assert "FAIL" in summary or "50.0%" in summary

    @patch("src.ml.synthetic.validation.pipeline.GX_AVAILABLE", False)
    def test_get_combined_summary_no_gx(self):
        """Test summary generation when GX is not available."""
        results = {
            "table1": PipelineValidationResult(
                table_name="table1",
                pandera_valid=True,
            ),
        }

        summary = get_combined_summary(results)

        assert "Great Expectations not installed" in summary

    def test_get_combined_summary_with_observability(self):
        """Test summary generation includes observability info."""
        results = {
            "table1": PipelineValidationResult(
                table_name="table1",
                pandera_valid=True,
            ),
        }

        obs_summary = {
            "mlflow_run_id": "test-run-123",
        }

        summary = get_combined_summary(results, obs_summary)

        assert "test-run-123" in summary
        assert "OBSERVABILITY" in summary

    def test_get_combined_summary_skipped_tables(self):
        """Test summary generation with skipped GX validations."""
        results = {
            "table1": PipelineValidationResult(
                table_name="table1",
                pandera_valid=True,
                gx_result=None,
            ),
        }

        summary = get_combined_summary(results)

        assert "SKIPPED" in summary or "table1" in summary


@pytest.mark.unit
class TestQuickValidate:
    """Test quick_validate convenience function."""

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_quick_validate_success(self, mock_validate_pipeline, capsys):
        """Test quick_validate with successful validation."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
        }

        # Mock successful validation
        mock_result = Mock()
        mock_result.is_valid = True

        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        result = quick_validate(datasets, verbose=True)

        assert result is True
        mock_validate_pipeline.assert_called_once()

        # Check that it was called with observability disabled
        call_kwargs = mock_validate_pipeline.call_args[1]
        assert call_kwargs["enable_observability"] is False

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_quick_validate_failure(self, mock_validate_pipeline):
        """Test quick_validate with failed validation."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
        }

        # Mock failed validation
        mock_result = Mock()
        mock_result.is_valid = False

        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        result = quick_validate(datasets, verbose=False)

        assert result is False

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_quick_validate_with_dgp_type(self, mock_validate_pipeline):
        """Test quick_validate passes DGP type."""
        datasets = {"table1": pd.DataFrame({"id": [1, 2, 3]})}

        mock_result = Mock()
        mock_result.is_valid = True
        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        quick_validate(datasets, dgp_type=DGPType.CONFOUNDED)

        call_kwargs = mock_validate_pipeline.call_args[1]
        assert call_kwargs["dgp_type"] == DGPType.CONFOUNDED


@pytest.mark.unit
class TestValidateAndLog:
    """Test validate_and_log convenience function."""

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_validate_and_log_success(self, mock_validate_pipeline):
        """Test validate_and_log with successful validation."""
        datasets = {
            "table1": pd.DataFrame({"id": [1, 2, 3]}),
        }

        mock_result = Mock()
        mock_result.is_valid = True

        mock_obs_summary = {"mlflow_run_id": "test-run-123"}

        mock_validate_pipeline.return_value = ({"table1": mock_result}, mock_obs_summary)

        all_valid, run_id = validate_and_log(datasets, verbose=False)

        assert all_valid is True
        assert run_id == "test-run-123"

        # Check that it was called with observability enabled
        call_kwargs = mock_validate_pipeline.call_args[1]
        assert call_kwargs["enable_observability"] is True

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_validate_and_log_failure(self, mock_validate_pipeline):
        """Test validate_and_log with failed validation."""
        datasets = {"table1": pd.DataFrame({"id": [1, 2, 3]})}

        mock_result = Mock()
        mock_result.is_valid = False

        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        all_valid, run_id = validate_and_log(datasets, verbose=False)

        assert all_valid is False

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_validate_and_log_with_tags(self, mock_validate_pipeline):
        """Test validate_and_log passes tags to pipeline."""
        datasets = {"table1": pd.DataFrame({"id": [1, 2, 3]})}

        mock_result = Mock()
        mock_result.is_valid = True
        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        tags = {"env": "test", "version": "1.0"}

        validate_and_log(datasets, tags=tags, verbose=False)

        call_kwargs = mock_validate_pipeline.call_args[1]
        assert call_kwargs["tags"] == tags

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_validate_and_log_custom_experiment(self, mock_validate_pipeline):
        """Test validate_and_log with custom experiment name."""
        datasets = {"table1": pd.DataFrame({"id": [1, 2, 3]})}

        mock_result = Mock()
        mock_result.is_valid = True
        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        validate_and_log(datasets, experiment_name="custom_experiment", verbose=False)

        call_kwargs = mock_validate_pipeline.call_args[1]
        assert call_kwargs["experiment_name"] == "custom_experiment"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_pipeline_validation_result_timestamp(self):
        """Test that validation result includes timestamp."""
        result = PipelineValidationResult(
            table_name="test_table",
            pandera_valid=True,
        )

        assert isinstance(result.validation_timestamp, datetime)

    @patch("src.ml.synthetic.validation.pipeline.validate_dataframe")
    @patch("src.ml.synthetic.validation.pipeline.SCHEMA_REGISTRY", {"test": Mock()})
    def test_validate_dataset_pandera_exception(self, mock_validate):
        """Test validate_dataset handles Pandera exceptions."""
        df = pd.DataFrame({"id": [1, 2, 3]})

        # Mock Pandera raising exception
        mock_validate.side_effect = Exception("Validation error")

        # Should not crash, but propagate exception
        with pytest.raises(Exception, match="Validation error"):
            validate_dataset(df, "test", run_pandera=True, run_gx=False)

    def test_get_combined_summary_empty_results(self):
        """Test get_combined_summary with empty results."""
        summary = get_combined_summary({})

        assert "Tables Validated: 0" in summary

    @patch("src.ml.synthetic.validation.pipeline.validate_pipeline_output")
    def test_quick_validate_no_verbose(self, mock_validate_pipeline, capsys):
        """Test quick_validate with verbose=False doesn't print."""
        datasets = {"table1": pd.DataFrame({"id": [1, 2, 3]})}

        mock_result = Mock()
        mock_result.is_valid = True
        mock_validate_pipeline.return_value = ({"table1": mock_result}, {})

        quick_validate(datasets, verbose=False)

        captured = capsys.readouterr()
        # Should not print anything
        assert captured.out == ""
