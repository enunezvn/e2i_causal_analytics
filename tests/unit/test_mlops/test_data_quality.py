"""Unit tests for data quality module (Great Expectations integration)."""

import pytest
import pandas as pd
import numpy as np

from src.mlops.data_quality import (
    DataQualityResult,
    DataQualityValidator,
    ExpectationSuiteBuilder,
    get_data_quality_validator,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(100),
        "brand": np.random.choice(["Remibrutinib", "Fabhalta", "Kisqali"], 100),
        "metric_value": np.random.uniform(0, 100, 100),
        "created_at": pd.date_range("2024-01-01", periods=100, freq="D"),
        "region": np.random.choice(["US", "EU", "APAC"], 100),
    })


@pytest.fixture
def validator():
    """Create a DataQualityValidator instance."""
    return DataQualityValidator()


class TestExpectationSuiteBuilder:
    """Tests for ExpectationSuiteBuilder."""

    def test_build_empty_suite(self):
        """Test building an empty suite."""
        builder = ExpectationSuiteBuilder("test_suite")
        expectations = builder.build()
        assert expectations == []

    def test_build_with_expectations(self):
        """Test building a suite with expectations."""
        builder = (
            ExpectationSuiteBuilder("test_suite")
            .expect_column_to_exist("id")
            .expect_column_values_to_not_be_null("id")
            .expect_column_values_to_be_between("value", min_value=0, max_value=100)
        )
        expectations = builder.build()

        assert len(expectations) == 3
        assert expectations[0]["expectation_type"] == "expect_column_to_exist"
        assert expectations[1]["expectation_type"] == "expect_column_values_to_not_be_null"
        assert expectations[2]["expectation_type"] == "expect_column_values_to_be_between"

    def test_builder_chaining(self):
        """Test that builder methods return self for chaining."""
        builder = ExpectationSuiteBuilder("test_suite")
        result = builder.expect_column_to_exist("id")
        assert result is builder

    def test_expect_column_values_to_be_in_set(self):
        """Test building expectation for value set."""
        builder = ExpectationSuiteBuilder("test_suite")
        builder.expect_column_values_to_be_in_set(
            "brand", ["A", "B", "C"], mostly=0.95
        )
        expectations = builder.build()

        assert len(expectations) == 1
        assert expectations[0]["kwargs"]["value_set"] == ["A", "B", "C"]
        assert expectations[0]["kwargs"]["mostly"] == 0.95


class TestDataQualityValidator:
    """Tests for DataQualityValidator."""

    def test_default_suites_registered(self, validator):
        """Test that default suites are registered."""
        assert "business_metrics" in validator.SUITES
        assert "predictions" in validator.SUITES
        assert "triggers" in validator.SUITES
        assert "patient_journeys" in validator.SUITES
        assert "causal_paths" in validator.SUITES
        assert "agent_activities" in validator.SUITES

    def test_get_suite(self, validator):
        """Test getting a registered suite."""
        suite = validator.get_suite("business_metrics")
        assert isinstance(suite, list)
        assert len(suite) > 0

    def test_get_unknown_suite_raises(self, validator):
        """Test that getting unknown suite raises ValueError."""
        with pytest.raises(ValueError, match="Unknown suite"):
            validator.get_suite("nonexistent_suite")

    def test_register_custom_suite(self, validator):
        """Test registering a custom suite."""
        custom_expectations = [
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "custom_col"}}
        ]
        validator.register_suite("custom_suite", custom_expectations)

        assert "custom_suite" in validator.SUITES
        assert validator.get_suite("custom_suite") == custom_expectations

    @pytest.mark.asyncio
    async def test_validate_clean_data(self, validator, sample_df):
        """Test validation passes for clean data."""
        result = await validator.validate(
            df=sample_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        assert isinstance(result, DataQualityResult)
        assert result.expectations_evaluated > 0
        assert result.overall_status in ["passed", "warning"]

    @pytest.mark.asyncio
    async def test_validate_with_nulls(self, validator):
        """Test validation detects null values."""
        df_with_nulls = pd.DataFrame({
            "id": [1, 2, None, 4, 5],
            "brand": ["Remibrutinib", None, "Kisqali", "Fabhalta", None],
            "metric_value": [10.0, 20.0, 30.0, None, 50.0],
            "created_at": pd.date_range("2024-01-01", periods=5, freq="D"),
        })

        result = await validator.validate(
            df=df_with_nulls,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        # Should have some failed expectations due to nulls
        assert result.expectations_failed > 0 or result.completeness_score < 1.0

    @pytest.mark.asyncio
    async def test_validate_empty_df(self, validator):
        """Test validation handles empty DataFrame."""
        empty_df = pd.DataFrame(columns=["id", "brand", "metric_value", "created_at"])

        result = await validator.validate(
            df=empty_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        # Row count expectation should fail
        assert result.expectations_failed > 0

    @pytest.mark.asyncio
    async def test_validate_splits(self, validator, sample_df):
        """Test validating multiple splits."""
        # Split the data
        train_df = sample_df.iloc[:60]
        val_df = sample_df.iloc[60:80]
        test_df = sample_df.iloc[80:]

        results = await validator.validate_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            suite_name="business_metrics",
            table_name="business_metrics",
        )

        assert "train" in results
        assert "validation" in results
        assert "test" in results
        assert all(isinstance(r, DataQualityResult) for r in results.values())


class TestDataQualityResult:
    """Tests for DataQualityResult."""

    def test_result_initialization(self):
        """Test result initialization."""
        result = DataQualityResult(
            report_name="test_report",
            expectation_suite_name="test_suite",
            table_name="test_table",
            data_split="train",
        )

        assert result.report_name == "test_report"
        assert result.expectation_suite_name == "test_suite"
        assert result.table_name == "test_table"
        assert result.data_split == "train"
        assert result.overall_status == "skipped"

    def test_result_to_dict(self):
        """Test result to_dict method."""
        result = DataQualityResult(
            report_name="test_report",
            expectation_suite_name="test_suite",
            table_name="test_table",
        )
        result.overall_status = "passed"
        result.expectations_evaluated = 10
        result.expectations_passed = 10
        result.success_rate = 1.0

        result_dict = result.to_dict()

        assert result_dict["report_name"] == "test_report"
        assert result_dict["overall_status"] == "passed"
        assert result_dict["expectations_evaluated"] == 10
        assert "run_at" in result_dict

    def test_result_passed_property(self):
        """Test passed property."""
        result = DataQualityResult("test", "suite", "table")

        result.overall_status = "passed"
        assert result.passed is True

        result.overall_status = "warning"
        assert result.passed is False

        result.overall_status = "failed"
        assert result.passed is False

    def test_result_blocking_property(self):
        """Test blocking property."""
        result = DataQualityResult("test", "suite", "table")

        result.overall_status = "failed"
        assert result.blocking is True

        result.overall_status = "passed"
        assert result.blocking is False

        result.overall_status = "warning"
        assert result.blocking is False


class TestGetDataQualityValidator:
    """Tests for singleton getter."""

    def test_get_validator_returns_instance(self):
        """Test that getter returns a validator instance."""
        validator = get_data_quality_validator()
        assert isinstance(validator, DataQualityValidator)

    def test_get_validator_returns_same_instance(self):
        """Test that getter returns the same instance (singleton)."""
        validator1 = get_data_quality_validator()
        validator2 = get_data_quality_validator()
        assert validator1 is validator2


# Import alerting classes for tests
from src.mlops.data_quality import (
    AlertConfig,
    AlertSeverity,
    DataQualityAlerter,
    LogAlertHandler,
    WebhookAlertHandler,
    configure_alerter,
    get_default_alerter,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_severity_ordering(self):
        """Test that severities can be compared."""
        # Enum members should be string subclass
        assert isinstance(AlertSeverity.INFO, str)
        assert AlertSeverity.INFO == "info"


class TestAlertConfig:
    """Tests for AlertConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlertConfig()
        assert config.enabled is True
        assert config.min_severity == AlertSeverity.WARNING
        assert config.include_failed_expectations is True
        assert config.max_failed_to_include == 10
        assert config.webhook_url is None
        assert config.webhook_timeout == 5.0
        assert config.custom_handlers == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AlertConfig(
            enabled=False,
            min_severity=AlertSeverity.ERROR,
            webhook_url="https://example.com/webhook",
            max_failed_to_include=5,
        )
        assert config.enabled is False
        assert config.min_severity == AlertSeverity.ERROR
        assert config.webhook_url == "https://example.com/webhook"
        assert config.max_failed_to_include == 5


class TestDataQualityAlerter:
    """Tests for DataQualityAlerter."""

    def test_default_initialization(self):
        """Test default alerter initialization."""
        alerter = DataQualityAlerter()
        assert alerter.config.enabled is True
        assert len(alerter._handlers) == 1  # Only LogAlertHandler
        assert isinstance(alerter._handlers[0], LogAlertHandler)

    def test_initialization_with_webhook(self):
        """Test alerter initialization with webhook."""
        config = AlertConfig(webhook_url="https://example.com/webhook")
        alerter = DataQualityAlerter(config)
        assert len(alerter._handlers) == 2  # LogAlertHandler + WebhookAlertHandler
        assert any(isinstance(h, WebhookAlertHandler) for h in alerter._handlers)

    def test_determine_severity_passed(self):
        """Test severity determination for passed result."""
        alerter = DataQualityAlerter()
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "passed"

        severity = alerter.determine_severity(result)
        assert severity == AlertSeverity.INFO

    def test_determine_severity_warning(self):
        """Test severity determination for warning result."""
        alerter = DataQualityAlerter()
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "warning"

        severity = alerter.determine_severity(result)
        assert severity == AlertSeverity.WARNING

    def test_determine_severity_failed(self):
        """Test severity determination for failed result."""
        alerter = DataQualityAlerter()
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "failed"
        result.success_rate = 0.7

        severity = alerter.determine_severity(result)
        assert severity == AlertSeverity.ERROR

    def test_determine_severity_critical(self):
        """Test severity determination for critical failures."""
        alerter = DataQualityAlerter()

        # Low success rate
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "failed"
        result.success_rate = 0.3

        severity = alerter.determine_severity(result)
        assert severity == AlertSeverity.CRITICAL

        # Leakage detected
        result2 = DataQualityResult("test", "suite", "table")
        result2.overall_status = "failed"
        result2.success_rate = 0.8
        result2.leakage_detected = True

        severity2 = alerter.determine_severity(result2)
        assert severity2 == AlertSeverity.CRITICAL

    def test_should_alert_enabled(self):
        """Test should_alert when alerts are enabled."""
        alerter = DataQualityAlerter()

        # Failed result should alert
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "failed"

        assert alerter.should_alert(result) is True

    def test_should_alert_disabled(self):
        """Test should_alert when alerts are disabled."""
        config = AlertConfig(enabled=False)
        alerter = DataQualityAlerter(config)

        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "failed"

        assert alerter.should_alert(result) is False

    def test_should_alert_below_min_severity(self):
        """Test should_alert when result is below min severity."""
        config = AlertConfig(min_severity=AlertSeverity.ERROR)
        alerter = DataQualityAlerter(config)

        # Warning result should not alert when min is ERROR
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "warning"

        assert alerter.should_alert(result) is False

    def test_add_and_remove_handler(self):
        """Test adding and removing handlers."""
        alerter = DataQualityAlerter()
        initial_count = len(alerter._handlers)

        # Add handler
        alerter.add_handler(WebhookAlertHandler())
        assert len(alerter._handlers) == initial_count + 1

        # Remove handlers of type
        alerter.remove_handler(WebhookAlertHandler)
        assert len(alerter._handlers) == initial_count
        assert not any(isinstance(h, WebhookAlertHandler) for h in alerter._handlers)

    @pytest.mark.asyncio
    async def test_send_alert_success(self):
        """Test sending alert successfully."""
        alerter = DataQualityAlerter()

        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "failed"
        result.success_rate = 0.7
        result.expectations_evaluated = 10
        result.expectations_passed = 7
        result.expectations_failed = 3

        alert_results = await alerter.send_alert(result)

        assert "LogAlertHandler" in alert_results
        assert alert_results["LogAlertHandler"] is True

    @pytest.mark.asyncio
    async def test_send_alert_suppressed(self):
        """Test alert suppression for passing results."""
        config = AlertConfig(min_severity=AlertSeverity.WARNING)
        alerter = DataQualityAlerter(config)

        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "passed"  # Should not trigger alert

        alert_results = await alerter.send_alert(result)

        # Should return empty dict when suppressed
        assert alert_results == {}


class TestLogAlertHandler:
    """Tests for LogAlertHandler."""

    @pytest.mark.asyncio
    async def test_send_alert(self, caplog):
        """Test log alert sends correctly."""
        handler = LogAlertHandler()
        config = AlertConfig()

        result = DataQualityResult("test_report", "test_suite", "test_table")
        result.overall_status = "failed"
        result.success_rate = 0.7
        result.expectations_evaluated = 10
        result.expectations_passed = 7
        result.expectations_failed = 3
        result.failed_expectations = [
            {"expectation_type": "expect_column_to_exist", "column": "missing_col"}
        ]

        with caplog.at_level("WARNING"):
            success = await handler.send_alert(result, AlertSeverity.ERROR, config)

        assert success is True
        assert "DATA QUALITY ALERT" in caplog.text
        assert "FAILED" in caplog.text

    def test_format_message(self):
        """Test message formatting."""
        handler = LogAlertHandler()
        config = AlertConfig()

        result = DataQualityResult("test_report", "test_suite", "test_table")
        result.overall_status = "failed"
        result.success_rate = 0.7
        result.expectations_evaluated = 10
        result.expectations_passed = 7
        result.expectations_failed = 3
        result.data_split = "train"

        message = handler._format_message(result, config)

        assert "[DATA QUALITY ALERT]" in message
        assert "FAILED" in message
        assert "test_report" in message
        assert "test_table" in message
        assert "70.0%" in message


class TestValidatorWithAlerting:
    """Tests for DataQualityValidator with alerting integration."""

    @pytest.fixture
    def validator_with_alerter(self):
        """Create a validator with custom alerter."""
        config = AlertConfig(min_severity=AlertSeverity.WARNING)
        alerter = DataQualityAlerter(config)
        return DataQualityValidator(alerter=alerter)

    def test_validator_has_alerter(self, validator_with_alerter):
        """Test validator has alerter property."""
        assert validator_with_alerter._alerter is not None
        assert isinstance(validator_with_alerter.alerter, DataQualityAlerter)

    def test_set_alerter(self):
        """Test setting a custom alerter."""
        validator = DataQualityValidator()
        custom_alerter = DataQualityAlerter(AlertConfig(enabled=False))

        validator.set_alerter(custom_alerter)

        assert validator._alerter is custom_alerter
        assert validator.alerter.config.enabled is False

    @pytest.mark.asyncio
    async def test_alert_on_result(self, validator_with_alerter):
        """Test alert_on_result method."""
        result = DataQualityResult("test", "suite", "table")
        result.overall_status = "failed"
        result.success_rate = 0.7

        alert_results = await validator_with_alerter.alert_on_result(result)

        assert isinstance(alert_results, dict)
        assert "LogAlertHandler" in alert_results

    @pytest.mark.asyncio
    async def test_validate_with_alerts(self, validator_with_alerter, sample_df):
        """Test validate_with_alerts method."""
        result = await validator_with_alerter.validate_with_alerts(
            df=sample_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        assert isinstance(result, DataQualityResult)
        assert result.expectations_evaluated > 0

    @pytest.mark.asyncio
    async def test_run_checkpoint_with_alerts(self, validator_with_alerter, sample_df):
        """Test run_checkpoint sends alerts."""
        result = await validator_with_alerter.run_checkpoint(
            checkpoint_name="test_checkpoint",
            df=sample_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            fail_on_error=False,
            send_alerts=True,
        )

        assert isinstance(result, DataQualityResult)

    @pytest.mark.asyncio
    async def test_run_checkpoint_without_alerts(self, validator_with_alerter, sample_df):
        """Test run_checkpoint without alerts."""
        result = await validator_with_alerter.run_checkpoint(
            checkpoint_name="test_checkpoint",
            df=sample_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            fail_on_error=False,
            send_alerts=False,
        )

        assert isinstance(result, DataQualityResult)


class TestAlertingHelpers:
    """Tests for alerting helper functions."""

    def test_get_default_alerter(self):
        """Test getting default alerter."""
        alerter = get_default_alerter()
        assert isinstance(alerter, DataQualityAlerter)

    def test_configure_alerter(self):
        """Test configuring alerter."""
        config = AlertConfig(
            enabled=True,
            min_severity=AlertSeverity.ERROR,
            webhook_url="https://example.com/webhook",
        )

        alerter = configure_alerter(config)

        assert alerter.config.min_severity == AlertSeverity.ERROR
        assert alerter.config.webhook_url == "https://example.com/webhook"
        # Should have webhook handler
        assert any(isinstance(h, WebhookAlertHandler) for h in alerter._handlers)
