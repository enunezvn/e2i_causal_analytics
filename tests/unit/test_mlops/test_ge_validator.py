"""Tests for Great Expectations Data Quality Validator.

Phase 3 of MLOps Integration Audit - testing the DataQualityValidator class.
Tests are organized into 3 batches:
- Batch 1: Core Validation (5 tests)
- Batch 2: Expectation Suites (5 tests)
- Batch 3: Alerting & Integration (5 tests)
"""

from datetime import datetime

import pandas as pd
import pytest

from src.mlops.data_quality import (
    GE_AVAILABLE,
    AlertSeverity,
    DataQualityAlerter,
    DataQualityCheckpointError,
    DataQualityResult,
    DataQualityValidator,
    ExpectationSuiteBuilder,
    get_data_quality_validator,
)

# =============================================================================
# BATCH 1: Core Validation Tests
# =============================================================================


class TestCoreValidation:
    """Batch 1: Core validation functionality tests."""

    @pytest.fixture
    def validator(self):
        """Create a fresh DataQualityValidator instance."""
        return DataQualityValidator()

    @pytest.fixture
    def valid_business_metrics_df(self):
        """Create a valid business metrics DataFrame."""
        return pd.DataFrame(
            {
                "id": ["BM001", "BM002", "BM003"],
                "brand": ["Remibrutinib", "Fabhalta", "Kisqali"],
                "metric_value": [100.5, 250.0, 75.3],
                "created_at": [datetime.now()] * 3,
            }
        )

    @pytest.fixture
    def invalid_business_metrics_df(self):
        """Create an invalid business metrics DataFrame (missing required columns)."""
        return pd.DataFrame(
            {
                "id": ["BM001", "BM002"],
                # missing 'brand' column
                "metric_value": [-50.0, 100.0],  # negative value (invalid)
                "created_at": [datetime.now()] * 2,
            }
        )

    @pytest.mark.asyncio
    async def test_validate_business_metrics_success(self, validator, valid_business_metrics_df):
        """Should validate valid business metrics data successfully."""
        result = await validator.validate(
            df=valid_business_metrics_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        assert result is not None
        assert isinstance(result, DataQualityResult)
        assert result.table_name == "business_metrics"
        assert result.data_split == "train"
        # With valid data, should pass or warn (depending on GE availability)
        if GE_AVAILABLE:
            assert result.overall_status in ["passed", "warning"]
            assert result.expectations_evaluated > 0
        else:
            assert result.overall_status == "skipped"

    @pytest.mark.asyncio
    async def test_validate_business_metrics_failure(self, validator, invalid_business_metrics_df):
        """Should detect validation failures in invalid data."""
        result = await validator.validate(
            df=invalid_business_metrics_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        assert result is not None
        assert isinstance(result, DataQualityResult)
        if GE_AVAILABLE:
            # Should fail due to missing 'brand' column and negative metric_value
            assert result.expectations_failed > 0 or result.overall_status != "passed"
            assert len(result.failed_expectations) >= 0  # May have failed expectations

    @pytest.mark.asyncio
    async def test_validate_predictions_success(self, validator):
        """Should validate predictions data successfully."""
        predictions_df = pd.DataFrame(
            {
                "id": ["P001", "P002", "P003"],
                "prediction_value": [0.85, 0.92, 0.67],
                "confidence": [0.95, 0.88, 0.72],
            }
        )

        result = await validator.validate(
            df=predictions_df,
            suite_name="predictions",
            table_name="predictions",
            data_split="validation",
        )

        assert result is not None
        assert result.table_name == "predictions"
        assert result.data_split == "validation"
        if GE_AVAILABLE:
            assert result.overall_status in ["passed", "warning"]

    @pytest.mark.asyncio
    async def test_validate_triggers_success(self, validator):
        """Should validate triggers data successfully."""
        triggers_df = pd.DataFrame(
            {
                "id": ["T001", "T002", "T003"],
                "trigger_type": ["threshold", "anomaly", "schedule"],
                "priority": ["high", "medium", "low"],
            }
        )

        result = await validator.validate(
            df=triggers_df,
            suite_name="triggers",
            table_name="triggers",
            data_split="full",
        )

        assert result is not None
        assert result.table_name == "triggers"
        if GE_AVAILABLE:
            assert result.overall_status in ["passed", "warning"]

    @pytest.mark.asyncio
    async def test_validation_result_structure(self, validator, valid_business_metrics_df):
        """Should return properly structured DataQualityResult."""
        result = await validator.validate(
            df=valid_business_metrics_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
            brand="Remibrutinib",
            region="US",
            training_run_id="run_123",
        )

        # Check result structure
        assert hasattr(result, "id")
        assert hasattr(result, "report_name")
        assert hasattr(result, "expectation_suite_name")
        assert hasattr(result, "overall_status")
        assert hasattr(result, "success_rate")
        assert hasattr(result, "expectations_evaluated")
        assert hasattr(result, "expectations_passed")
        assert hasattr(result, "expectations_failed")
        assert hasattr(result, "failed_expectations")

        # Check metadata
        assert result.brand == "Remibrutinib"
        assert result.region == "US"
        assert result.training_run_id == "run_123"
        assert result.data_split == "train"

        # Check to_dict works
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "id" in result_dict
        assert "overall_status" in result_dict
        assert "success_rate" in result_dict


# =============================================================================
# BATCH 2: Expectation Suites Tests
# =============================================================================


class TestExpectationSuites:
    """Batch 2: Expectation suite definition tests."""

    @pytest.fixture
    def validator(self):
        """Create a fresh DataQualityValidator instance."""
        return DataQualityValidator()

    def test_business_metrics_suite_expectations(self, validator):
        """Should have correct expectations in business_metrics suite."""
        suite = validator.get_suite("business_metrics")

        assert isinstance(suite, list)
        assert len(suite) > 0

        # Check for required expectations
        exp_types = [e["expectation_type"] for e in suite]
        assert "expect_table_row_count_to_be_between" in exp_types
        assert "expect_column_to_exist" in exp_types
        assert "expect_column_values_to_not_be_null" in exp_types

        # Check required columns are validated
        column_checks = [
            e.get("kwargs", {}).get("column")
            for e in suite
            if e["expectation_type"] == "expect_column_to_exist"
        ]
        assert "id" in column_checks
        assert "brand" in column_checks
        assert "metric_value" in column_checks

    def test_predictions_suite_expectations(self, validator):
        """Should have correct expectations in predictions suite."""
        suite = validator.get_suite("predictions")

        assert isinstance(suite, list)
        assert len(suite) > 0

        # Check for required columns
        column_checks = [
            e.get("kwargs", {}).get("column")
            for e in suite
            if e["expectation_type"] == "expect_column_to_exist"
        ]
        assert "id" in column_checks
        assert "prediction_value" in column_checks
        assert "confidence" in column_checks

        # Check confidence has range validation
        range_checks = [
            e
            for e in suite
            if e["expectation_type"] == "expect_column_values_to_be_between"
            and e.get("kwargs", {}).get("column") == "confidence"
        ]
        assert len(range_checks) > 0
        range_check = range_checks[0]
        assert range_check["kwargs"].get("min_value") == 0
        assert range_check["kwargs"].get("max_value") == 1

    def test_triggers_suite_expectations(self, validator):
        """Should have correct expectations in triggers suite."""
        suite = validator.get_suite("triggers")

        assert isinstance(suite, list)
        assert len(suite) > 0

        # Check for priority value set validation
        set_checks = [
            e
            for e in suite
            if e["expectation_type"] == "expect_column_values_to_be_in_set"
            and e.get("kwargs", {}).get("column") == "priority"
        ]
        assert len(set_checks) > 0
        assert "high" in set_checks[0]["kwargs"]["value_set"]
        assert "medium" in set_checks[0]["kwargs"]["value_set"]
        assert "low" in set_checks[0]["kwargs"]["value_set"]

    def test_patient_journeys_suite_expectations(self, validator):
        """Should have correct expectations in patient_journeys suite."""
        suite = validator.get_suite("patient_journeys")

        assert isinstance(suite, list)
        assert len(suite) > 0

        # Check required columns
        column_checks = [
            e.get("kwargs", {}).get("column")
            for e in suite
            if e["expectation_type"] == "expect_column_to_exist"
        ]
        assert "patient_id" in column_checks
        assert "event_type" in column_checks
        assert "event_date" in column_checks

    def test_causal_paths_suite_expectations(self, validator):
        """Should have correct expectations in causal_paths suite."""
        suite = validator.get_suite("causal_paths")

        assert isinstance(suite, list)
        assert len(suite) > 0

        # Check required columns
        column_checks = [
            e.get("kwargs", {}).get("column")
            for e in suite
            if e["expectation_type"] == "expect_column_to_exist"
        ]
        assert "source_node" in column_checks
        assert "target_node" in column_checks
        assert "effect_strength" in column_checks

        # Check effect_strength range (-1 to 1)
        range_checks = [
            e
            for e in suite
            if e["expectation_type"] == "expect_column_values_to_be_between"
            and e.get("kwargs", {}).get("column") == "effect_strength"
        ]
        assert len(range_checks) > 0
        range_check = range_checks[0]
        assert range_check["kwargs"].get("min_value") == -1
        assert range_check["kwargs"].get("max_value") == 1


# =============================================================================
# BATCH 3: Alerting & Integration Tests
# =============================================================================


class TestAlertingAndIntegration:
    """Batch 3: Alerting and integration tests."""

    @pytest.fixture
    def result_critical(self):
        """Create a critical severity result."""
        result = DataQualityResult(
            report_name="test_critical",
            expectation_suite_name="test_suite",
            table_name="test_table",
        )
        result.overall_status = "failed"
        result.success_rate = 0.3  # Below 50% = critical
        result.expectations_evaluated = 10
        result.expectations_passed = 3
        result.expectations_failed = 7
        result.failed_expectations = [
            {"expectation_type": "test_exp", "column": "col1"},
        ]
        result.leakage_detected = False
        return result

    @pytest.fixture
    def result_warning(self):
        """Create a warning severity result."""
        result = DataQualityResult(
            report_name="test_warning",
            expectation_suite_name="test_suite",
            table_name="test_table",
        )
        result.overall_status = "warning"
        result.success_rate = 0.9
        result.expectations_evaluated = 10
        result.expectations_passed = 9
        result.expectations_failed = 1
        result.failed_expectations = [
            {"expectation_type": "test_exp", "column": "col1"},
        ]
        return result

    def test_quality_alerter_critical(self, result_critical):
        """Should determine critical severity for low success rate."""
        alerter = DataQualityAlerter()

        severity = alerter.determine_severity(result_critical)

        assert severity == AlertSeverity.CRITICAL

    def test_quality_alerter_warning(self, result_warning):
        """Should determine warning severity for warning status."""
        alerter = DataQualityAlerter()

        severity = alerter.determine_severity(result_warning)

        assert severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_qc_gate_blocking(self):
        """Should block pipeline when checkpoint fails."""
        validator = DataQualityValidator()

        # Create invalid data that should fail validation
        bad_df = pd.DataFrame(
            {
                "id": [None, None],  # null IDs
                "brand": ["Invalid", "Brand"],  # invalid brand values
                "metric_value": [-100, -200],  # negative values
                "created_at": [datetime.now()] * 2,
            }
        )

        # With fail_on_error=True and send_alerts=False (to avoid async issues)
        if GE_AVAILABLE:
            with pytest.raises(DataQualityCheckpointError) as exc_info:
                await validator.run_checkpoint(
                    checkpoint_name="qc_gate_test",
                    df=bad_df,
                    suite_name="business_metrics",
                    table_name="test_table",
                    fail_on_error=True,
                    send_alerts=False,  # Disable alerts for test
                )

            assert exc_info.value.result is not None
            assert exc_info.value.result.overall_status == "failed"
        else:
            # Without GE, validation is skipped
            result = await validator.run_checkpoint(
                checkpoint_name="qc_gate_test",
                df=bad_df,
                suite_name="business_metrics",
                table_name="test_table",
                fail_on_error=True,
                send_alerts=False,
            )
            assert result.overall_status == "skipped"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Should gracefully handle when GE is not available."""
        # This test verifies the graceful degradation pattern
        validator = DataQualityValidator()

        test_df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "brand": ["Remibrutinib", "Fabhalta"],
                "metric_value": [100, 200],
                "created_at": [datetime.now()] * 2,
            }
        )

        result = await validator.validate(
            df=test_df,
            suite_name="business_metrics",
            table_name="test_table",
        )

        # Should always return a result
        assert result is not None
        assert isinstance(result, DataQualityResult)

        if not GE_AVAILABLE:
            assert result.overall_status == "skipped"
            assert "not available" in (result.notes or "").lower()
        else:
            assert result.overall_status in ["passed", "warning", "failed"]

    @pytest.mark.asyncio
    async def test_pandera_ge_pipeline_integration(self):
        """Should work alongside Pandera in a validation pipeline."""
        # Import Pandera schemas
        try:
            from src.mlops.pandera_schemas import BusinessMetricSchema

            PANDERA_AVAILABLE = True
        except ImportError:
            PANDERA_AVAILABLE = False

        # Create test data
        test_df = pd.DataFrame(
            {
                "id": ["BM001", "BM002"],
                "brand": ["Remibrutinib", "Kisqali"],
                "metric_value": [150.0, 200.0],
                "created_at": [datetime.now()] * 2,
            }
        )

        validation_results = {"pandera": None, "ge": None}

        # Step 1: Pandera validation (schema-level)
        if PANDERA_AVAILABLE:
            try:
                BusinessMetricSchema.validate(test_df, lazy=True)
                validation_results["pandera"] = "passed"
            except Exception as e:
                validation_results["pandera"] = f"failed: {e}"
        else:
            validation_results["pandera"] = "skipped"

        # Step 2: GE validation (expectation-level)
        validator = DataQualityValidator()
        ge_result = await validator.validate(
            df=test_df,
            suite_name="business_metrics",
            table_name="business_metrics",
        )
        validation_results["ge"] = ge_result.overall_status

        # Both should succeed or gracefully degrade
        assert validation_results["pandera"] in ["passed", "skipped"]
        assert validation_results["ge"] in ["passed", "warning", "skipped"]


# =============================================================================
# Additional Helper Tests
# =============================================================================


class TestExpectationSuiteBuilder:
    """Tests for ExpectationSuiteBuilder helper class."""

    def test_builder_chaining(self):
        """Should support method chaining."""
        builder = ExpectationSuiteBuilder("test_suite")

        result = (
            builder.expect_column_to_exist("col1")
            .expect_column_values_to_not_be_null("col1")
            .expect_column_values_to_be_between("col2", min_value=0, max_value=100)
            .expect_column_values_to_be_in_set("col3", ["a", "b", "c"])
            .expect_column_values_to_be_unique("col4")
            .build()
        )

        assert len(result) == 5
        assert result[0]["expectation_type"] == "expect_column_to_exist"
        assert result[1]["expectation_type"] == "expect_column_values_to_not_be_null"
        assert result[2]["expectation_type"] == "expect_column_values_to_be_between"
        assert result[3]["expectation_type"] == "expect_column_values_to_be_in_set"
        assert result[4]["expectation_type"] == "expect_column_values_to_be_unique"

    def test_builder_kwargs(self):
        """Should correctly set kwargs for expectations."""
        builder = ExpectationSuiteBuilder("test_suite")

        result = builder.expect_column_values_to_be_between(
            "value", min_value=10, max_value=90, mostly=0.95
        ).build()

        assert result[0]["kwargs"]["column"] == "value"
        assert result[0]["kwargs"]["min_value"] == 10
        assert result[0]["kwargs"]["max_value"] == 90
        assert result[0]["kwargs"]["mostly"] == 0.95


class TestDataQualityResult:
    """Tests for DataQualityResult class."""

    def test_result_properties(self):
        """Should have correct property values."""
        result = DataQualityResult(
            report_name="test_report",
            expectation_suite_name="test_suite",
            table_name="test_table",
        )

        # Default status is skipped
        assert result.passed is False
        assert result.blocking is False

        # Set to passed
        result.overall_status = "passed"
        assert result.passed is True
        assert result.blocking is False

        # Set to failed
        result.overall_status = "failed"
        assert result.passed is False
        assert result.blocking is True

    def test_result_to_dict(self):
        """Should serialize to dict correctly."""
        result = DataQualityResult(
            report_name="test_report",
            expectation_suite_name="test_suite",
            table_name="test_table",
            data_split="train",
            brand="Remibrutinib",
        )
        result.overall_status = "passed"
        result.success_rate = 0.95
        result.expectations_evaluated = 10
        result.expectations_passed = 9
        result.expectations_failed = 1

        d = result.to_dict()

        assert d["report_name"] == "test_report"
        assert d["overall_status"] == "passed"
        assert d["success_rate"] == 0.95
        assert d["brand"] == "Remibrutinib"
        assert d["data_split"] == "train"
        assert "run_at" in d


class TestValidatorSingleton:
    """Tests for validator singleton pattern."""

    def test_get_validator_returns_instance(self):
        """Should return a DataQualityValidator instance."""
        validator = get_data_quality_validator()

        assert validator is not None
        assert isinstance(validator, DataQualityValidator)

    def test_get_validator_is_singleton(self):
        """Should return the same instance on multiple calls."""
        validator1 = get_data_quality_validator()
        validator2 = get_data_quality_validator()

        assert validator1 is validator2
