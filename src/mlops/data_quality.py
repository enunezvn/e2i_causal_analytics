"""
Data Quality Validation with Great Expectations - Phase 3

Provides data quality validation using Great Expectations framework:
- Expectation suite management
- Async validation support
- Result formatting for ml_data_quality_reports
- Integration with E2I quality dimensions

Version: 1.0.0
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import pandas as pd

try:
    import great_expectations as gx
    from great_expectations.core.expectation_validation_result import (
        ExpectationSuiteValidationResult,
    )
    GE_AVAILABLE = True
except ImportError:
    gx = None
    GE_AVAILABLE = False

logger = logging.getLogger(__name__)

# E2I Brand and Region types
BrandType = Literal["Remibrutinib", "Fabhalta", "Kisqali"]
RegionType = Literal["US", "EU", "APAC", "LATAM", "JP"]
DQStatusType = Literal["passed", "failed", "warning", "skipped"]
DataSplitType = Literal["train", "validation", "test", "holdout", "full"]


class AlertSeverity(str, Enum):
    """Severity levels for data quality alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configuration for data quality alerting."""

    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.WARNING
    include_failed_expectations: bool = True
    max_failed_to_include: int = 10
    webhook_url: Optional[str] = None
    webhook_timeout: float = 5.0
    custom_handlers: List[Callable[["DataQualityResult", AlertSeverity], None]] = field(
        default_factory=list
    )


class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @abstractmethod
    async def send_alert(
        self,
        result: "DataQualityResult",
        severity: AlertSeverity,
        config: AlertConfig,
    ) -> bool:
        """Send an alert for a data quality result.

        Args:
            result: The validation result triggering the alert
            severity: Alert severity level
            config: Alert configuration

        Returns:
            True if alert was sent successfully
        """
        pass


class LogAlertHandler(AlertHandler):
    """Alert handler that logs to the standard logger."""

    async def send_alert(
        self,
        result: "DataQualityResult",
        severity: AlertSeverity,
        config: AlertConfig,
    ) -> bool:
        """Send alert via logging."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.WARNING)

        message = self._format_message(result, config)
        logger.log(log_level, message)
        return True

    def _format_message(self, result: "DataQualityResult", config: AlertConfig) -> str:
        """Format the alert message."""
        lines = [
            f"[DATA QUALITY ALERT] {result.overall_status.upper()}",
            f"  Report: {result.report_name}",
            f"  Table: {result.table_name}",
            f"  Suite: {result.expectation_suite_name}",
            f"  Split: {result.data_split}",
            f"  Success Rate: {result.success_rate:.1%}",
            f"  Passed: {result.expectations_passed}/{result.expectations_evaluated}",
        ]

        if config.include_failed_expectations and result.failed_expectations:
            lines.append("  Failed Expectations:")
            for i, exp in enumerate(result.failed_expectations[: config.max_failed_to_include]):
                col = exp.get("column", "N/A")
                exp_type = exp.get("expectation_type", "unknown")
                lines.append(f"    {i + 1}. {exp_type} (column: {col})")
            if len(result.failed_expectations) > config.max_failed_to_include:
                remaining = len(result.failed_expectations) - config.max_failed_to_include
                lines.append(f"    ... and {remaining} more")

        return "\n".join(lines)


class WebhookAlertHandler(AlertHandler):
    """Alert handler that sends alerts via webhook."""

    async def send_alert(
        self,
        result: "DataQualityResult",
        severity: AlertSeverity,
        config: AlertConfig,
    ) -> bool:
        """Send alert via webhook."""
        if not config.webhook_url:
            logger.warning("Webhook URL not configured, skipping webhook alert")
            return False

        payload = self._build_payload(result, severity, config)

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.webhook_timeout),
                ) as response:
                    if response.status < 300:
                        logger.info(f"Webhook alert sent successfully: {response.status}")
                        return True
                    else:
                        logger.warning(f"Webhook alert failed: {response.status}")
                        return False

        except ImportError:
            logger.warning("aiohttp not installed, falling back to urllib")
            return self._send_sync(payload, config)
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def _send_sync(self, payload: Dict[str, Any], config: AlertConfig) -> bool:
        """Synchronous fallback for webhook."""
        import urllib.request
        import urllib.error

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=config.webhook_timeout) as response:
                return response.status < 300
        except Exception as e:
            logger.error(f"Sync webhook failed: {e}")
            return False

    def _build_payload(
        self,
        result: "DataQualityResult",
        severity: AlertSeverity,
        config: AlertConfig,
    ) -> Dict[str, Any]:
        """Build webhook payload."""
        payload = {
            "type": "data_quality_alert",
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "report_name": result.report_name,
            "table_name": result.table_name,
            "suite_name": result.expectation_suite_name,
            "data_split": result.data_split,
            "overall_status": result.overall_status,
            "success_rate": result.success_rate,
            "expectations_evaluated": result.expectations_evaluated,
            "expectations_passed": result.expectations_passed,
            "expectations_failed": result.expectations_failed,
        }

        if result.brand:
            payload["brand"] = result.brand
        if result.region:
            payload["region"] = result.region
        if result.training_run_id:
            payload["training_run_id"] = result.training_run_id

        if config.include_failed_expectations:
            payload["failed_expectations"] = result.failed_expectations[
                : config.max_failed_to_include
            ]

        return payload


class DataQualityAlerter:
    """Manages data quality alerting across multiple channels."""

    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize the alerter.

        Args:
            config: Alert configuration (uses defaults if not provided)
        """
        self.config = config or AlertConfig()
        self._handlers: List[AlertHandler] = [LogAlertHandler()]

        # Add webhook handler if configured
        if self.config.webhook_url:
            self._handlers.append(WebhookAlertHandler())

    def add_handler(self, handler: AlertHandler) -> None:
        """Add a custom alert handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler_type: type) -> None:
        """Remove handlers of a specific type."""
        self._handlers = [h for h in self._handlers if not isinstance(h, handler_type)]

    def determine_severity(self, result: "DataQualityResult") -> AlertSeverity:
        """Determine alert severity based on validation result."""
        if result.overall_status == "passed":
            return AlertSeverity.INFO
        elif result.overall_status == "warning":
            return AlertSeverity.WARNING
        elif result.overall_status == "failed":
            # Critical if success rate is very low or leakage detected
            if result.success_rate < 0.5 or result.leakage_detected:
                return AlertSeverity.CRITICAL
            return AlertSeverity.ERROR
        else:  # skipped
            return AlertSeverity.INFO

    def should_alert(self, result: "DataQualityResult") -> bool:
        """Check if an alert should be sent based on config and result."""
        if not self.config.enabled:
            return False

        severity = self.determine_severity(result)
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3,
        }

        return severity_order[severity] >= severity_order[self.config.min_severity]

    async def send_alert(self, result: "DataQualityResult") -> Dict[str, bool]:
        """Send alerts through all configured handlers.

        Args:
            result: The validation result

        Returns:
            Dict mapping handler names to success status
        """
        if not self.should_alert(result):
            logger.debug(f"Alert suppressed for {result.report_name} (below min severity)")
            return {}

        severity = self.determine_severity(result)
        results = {}

        # Run all handlers
        for handler in self._handlers:
            handler_name = handler.__class__.__name__
            try:
                success = await handler.send_alert(result, severity, self.config)
                results[handler_name] = success
            except Exception as e:
                logger.error(f"Handler {handler_name} failed: {e}")
                results[handler_name] = False

        # Run custom handlers
        for custom_handler in self.config.custom_handlers:
            try:
                custom_handler(result, severity)
                results[f"custom_{custom_handler.__name__}"] = True
            except Exception as e:
                logger.error(f"Custom handler failed: {e}")
                results[f"custom_{custom_handler.__name__}"] = False

        return results


# Default alerter instance
_default_alerter: Optional[DataQualityAlerter] = None


def get_default_alerter() -> DataQualityAlerter:
    """Get the default alerter instance."""
    global _default_alerter
    if _default_alerter is None:
        _default_alerter = DataQualityAlerter()
    return _default_alerter


def configure_alerter(config: AlertConfig) -> DataQualityAlerter:
    """Configure and return the default alerter."""
    global _default_alerter
    _default_alerter = DataQualityAlerter(config)
    return _default_alerter


class DataQualityResult:
    """Result of a data quality validation run."""

    def __init__(
        self,
        report_name: str,
        expectation_suite_name: str,
        table_name: str,
        data_split: DataSplitType = "full",
        brand: Optional[BrandType] = None,
        region: Optional[RegionType] = None,
        training_run_id: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        self.report_name = report_name
        self.expectation_suite_name = expectation_suite_name
        self.table_name = table_name
        self.data_split = data_split
        self.brand = brand
        self.region = region
        self.training_run_id = training_run_id
        self.run_at = datetime.now()

        # Results (populated after validation)
        self.overall_status: DQStatusType = "skipped"
        self.expectations_evaluated: int = 0
        self.expectations_passed: int = 0
        self.expectations_failed: int = 0
        self.success_rate: float = 0.0
        self.failed_expectations: List[Dict[str, Any]] = []

        # Quality dimension scores (0.0 - 1.0)
        self.completeness_score: Optional[float] = None
        self.validity_score: Optional[float] = None
        self.uniqueness_score: Optional[float] = None
        self.consistency_score: Optional[float] = None
        self.timeliness_score: Optional[float] = None
        self.accuracy_score: Optional[float] = None

        # Metadata
        self.validation_time_ms: Optional[int] = None
        self.leakage_detected: bool = False
        self.notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "report_name": self.report_name,
            "expectation_suite_name": self.expectation_suite_name,
            "table_name": self.table_name,
            "brand": self.brand,
            "region": self.region,
            "overall_status": self.overall_status,
            "expectations_evaluated": self.expectations_evaluated,
            "expectations_passed": self.expectations_passed,
            "expectations_failed": self.expectations_failed,
            "success_rate": self.success_rate,
            "failed_expectations": self.failed_expectations,
            "completeness_score": self.completeness_score,
            "validity_score": self.validity_score,
            "uniqueness_score": self.uniqueness_score,
            "consistency_score": self.consistency_score,
            "timeliness_score": self.timeliness_score,
            "accuracy_score": self.accuracy_score,
            "validation_time_ms": self.validation_time_ms,
            "data_split": self.data_split,
            "training_run_id": self.training_run_id,
            "leakage_detected": self.leakage_detected,
            "notes": self.notes,
            "run_at": self.run_at.isoformat(),
        }

    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.overall_status == "passed"

    @property
    def blocking(self) -> bool:
        """Check if validation is blocking (failed)."""
        return self.overall_status == "failed"


class ExpectationSuiteBuilder:
    """Builder for creating expectation suites programmatically."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.expectations: List[Dict[str, Any]] = []

    def expect_column_to_exist(self, column: str) -> "ExpectationSuiteBuilder":
        """Expect column to exist in the dataset."""
        self.expectations.append({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": column},
        })
        return self

    def expect_column_values_to_not_be_null(
        self, column: str, mostly: float = 1.0
    ) -> "ExpectationSuiteBuilder":
        """Expect column values to not be null."""
        self.expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": column, "mostly": mostly},
        })
        return self

    def expect_column_values_to_be_between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        mostly: float = 1.0,
    ) -> "ExpectationSuiteBuilder":
        """Expect column values to be between min and max."""
        kwargs = {"column": column, "mostly": mostly}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        self.expectations.append({
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": kwargs,
        })
        return self

    def expect_column_values_to_be_in_set(
        self, column: str, value_set: List[Any], mostly: float = 1.0
    ) -> "ExpectationSuiteBuilder":
        """Expect column values to be in a set of allowed values."""
        self.expectations.append({
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": column, "value_set": value_set, "mostly": mostly},
        })
        return self

    def expect_column_values_to_be_unique(
        self, column: str, mostly: float = 1.0
    ) -> "ExpectationSuiteBuilder":
        """Expect column values to be unique."""
        self.expectations.append({
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {"column": column, "mostly": mostly},
        })
        return self

    def expect_table_row_count_to_be_between(
        self, min_value: int = 1, max_value: Optional[int] = None
    ) -> "ExpectationSuiteBuilder":
        """Expect table row count to be between min and max."""
        kwargs = {"min_value": min_value}
        if max_value is not None:
            kwargs["max_value"] = max_value
        self.expectations.append({
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": kwargs,
        })
        return self

    def expect_column_values_to_match_regex(
        self, column: str, regex: str, mostly: float = 1.0
    ) -> "ExpectationSuiteBuilder":
        """Expect column values to match a regex pattern."""
        self.expectations.append({
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {"column": column, "regex": regex, "mostly": mostly},
        })
        return self

    def expect_column_pair_values_to_be_equal(
        self, column_A: str, column_B: str, mostly: float = 1.0
    ) -> "ExpectationSuiteBuilder":
        """Expect values in two columns to be equal."""
        self.expectations.append({
            "expectation_type": "expect_column_pair_values_to_be_equal",
            "kwargs": {"column_A": column_A, "column_B": column_B, "mostly": mostly},
        })
        return self

    def build(self) -> List[Dict[str, Any]]:
        """Build and return the expectations list."""
        return self.expectations.copy()


class DataQualityValidator:
    """
    Data quality validator using Great Expectations.

    Provides async validation of DataFrames against expectation suites.
    Results are formatted for storage in ml_data_quality_reports.

    Example:
        validator = DataQualityValidator()

        # Create a suite
        suite = validator.create_business_metrics_suite()

        # Validate data
        result = await validator.validate(
            df=train_df,
            suite_name="business_metrics",
            table_name="business_metrics",
            data_split="train",
        )

        # Check result
        if result.blocking:
            raise ValueError(f"Data quality check failed: {result.failed_expectations}")
    """

    # Pre-defined expectation suites for E2I tables
    SUITES: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, alerter: Optional[DataQualityAlerter] = None):
        """Initialize the validator.

        Args:
            alerter: Optional custom alerter (uses default if not provided)
        """
        if not GE_AVAILABLE:
            logger.warning(
                "Great Expectations not available. "
                "Install with: pip install great-expectations"
            )
        self._checkpoint_history: List[Dict[str, Any]] = []
        self._alerter = alerter
        self._register_default_suites()

    @property
    def alerter(self) -> DataQualityAlerter:
        """Get the alerter instance."""
        if self._alerter is None:
            self._alerter = get_default_alerter()
        return self._alerter

    def set_alerter(self, alerter: DataQualityAlerter) -> None:
        """Set a custom alerter."""
        self._alerter = alerter

    def _register_default_suites(self) -> None:
        """Register default expectation suites for E2I tables."""
        # Business metrics suite
        self.SUITES["business_metrics"] = (
            ExpectationSuiteBuilder("business_metrics")
            .expect_table_row_count_to_be_between(min_value=1)
            .expect_column_to_exist("id")
            .expect_column_to_exist("brand")
            .expect_column_to_exist("metric_value")
            .expect_column_to_exist("created_at")
            .expect_column_values_to_not_be_null("id")
            .expect_column_values_to_not_be_null("brand")
            .expect_column_values_to_not_be_null("metric_value")
            .expect_column_values_to_be_in_set(
                "brand", ["Remibrutinib", "Fabhalta", "Kisqali"], mostly=0.95
            )
            .expect_column_values_to_be_between("metric_value", min_value=0)
            .build()
        )

        # Predictions suite
        self.SUITES["predictions"] = (
            ExpectationSuiteBuilder("predictions")
            .expect_table_row_count_to_be_between(min_value=1)
            .expect_column_to_exist("id")
            .expect_column_to_exist("prediction_value")
            .expect_column_to_exist("confidence")
            .expect_column_values_to_not_be_null("prediction_value")
            .expect_column_values_to_be_between("confidence", min_value=0, max_value=1)
            .build()
        )

        # Triggers suite
        self.SUITES["triggers"] = (
            ExpectationSuiteBuilder("triggers")
            .expect_table_row_count_to_be_between(min_value=1)
            .expect_column_to_exist("id")
            .expect_column_to_exist("trigger_type")
            .expect_column_to_exist("priority")
            .expect_column_values_to_not_be_null("trigger_type")
            .expect_column_values_to_be_in_set(
                "priority", ["high", "medium", "low"], mostly=0.95
            )
            .build()
        )

        # Patient journeys suite
        self.SUITES["patient_journeys"] = (
            ExpectationSuiteBuilder("patient_journeys")
            .expect_table_row_count_to_be_between(min_value=1)
            .expect_column_to_exist("patient_id")
            .expect_column_to_exist("event_type")
            .expect_column_to_exist("event_date")
            .expect_column_values_to_not_be_null("patient_id")
            .expect_column_values_to_not_be_null("event_type")
            .build()
        )

        # Causal paths suite
        self.SUITES["causal_paths"] = (
            ExpectationSuiteBuilder("causal_paths")
            .expect_table_row_count_to_be_between(min_value=1)
            .expect_column_to_exist("source_node")
            .expect_column_to_exist("target_node")
            .expect_column_to_exist("effect_strength")
            .expect_column_values_to_not_be_null("source_node")
            .expect_column_values_to_not_be_null("target_node")
            .expect_column_values_to_be_between(
                "effect_strength", min_value=-1, max_value=1
            )
            .build()
        )

        # Agent activities suite
        self.SUITES["agent_activities"] = (
            ExpectationSuiteBuilder("agent_activities")
            .expect_table_row_count_to_be_between(min_value=1)
            .expect_column_to_exist("agent_name")
            .expect_column_to_exist("activity_type")
            .expect_column_to_exist("created_at")
            .expect_column_values_to_not_be_null("agent_name")
            .expect_column_values_to_not_be_null("activity_type")
            .build()
        )

    def register_suite(
        self, suite_name: str, expectations: List[Dict[str, Any]]
    ) -> None:
        """Register a custom expectation suite."""
        self.SUITES[suite_name] = expectations
        logger.info(f"Registered suite '{suite_name}' with {len(expectations)} expectations")

    def get_suite(self, suite_name: str) -> List[Dict[str, Any]]:
        """Get an expectation suite by name."""
        if suite_name not in self.SUITES:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(self.SUITES.keys())}")
        return self.SUITES[suite_name]

    async def validate(
        self,
        df: pd.DataFrame,
        suite_name: str,
        table_name: str,
        data_split: DataSplitType = "full",
        brand: Optional[BrandType] = None,
        region: Optional[RegionType] = None,
        training_run_id: Optional[str] = None,
        fail_threshold: float = 0.8,
        warn_threshold: float = 0.95,
    ) -> DataQualityResult:
        """
        Validate a DataFrame against an expectation suite.

        Args:
            df: DataFrame to validate
            suite_name: Name of the expectation suite
            table_name: Source table name
            data_split: Data split type (train, validation, test, holdout, full)
            brand: Optional brand filter
            region: Optional region filter
            training_run_id: Optional training run ID
            fail_threshold: Success rate below this = failed (default 0.8)
            warn_threshold: Success rate below this = warning (default 0.95)

        Returns:
            DataQualityResult with validation results
        """
        start_time = datetime.now()

        result = DataQualityResult(
            report_name=f"{table_name}_{data_split}_{start_time.strftime('%Y%m%d_%H%M%S')}",
            expectation_suite_name=suite_name,
            table_name=table_name,
            data_split=data_split,
            brand=brand,
            region=region,
            training_run_id=training_run_id,
        )

        if not GE_AVAILABLE:
            result.overall_status = "skipped"
            result.notes = "Great Expectations not available"
            return result

        try:
            expectations = self.get_suite(suite_name)
            validation_results = self._run_expectations(df, expectations)

            # Aggregate results
            result.expectations_evaluated = len(validation_results)
            result.expectations_passed = sum(1 for r in validation_results if r["success"])
            result.expectations_failed = result.expectations_evaluated - result.expectations_passed

            if result.expectations_evaluated > 0:
                result.success_rate = result.expectations_passed / result.expectations_evaluated
            else:
                result.success_rate = 1.0

            # Collect failed expectations
            result.failed_expectations = [
                {
                    "expectation_type": r["expectation_type"],
                    "column": r.get("kwargs", {}).get("column"),
                    "kwargs": r.get("kwargs", {}),
                    "observed_value": r.get("result", {}).get("observed_value"),
                    "success": r["success"],
                }
                for r in validation_results
                if not r["success"]
            ]

            # Determine overall status
            if result.success_rate >= warn_threshold:
                result.overall_status = "passed"
            elif result.success_rate >= fail_threshold:
                result.overall_status = "warning"
            else:
                result.overall_status = "failed"

            # Calculate quality dimension scores from expectations
            result.completeness_score = self._calc_completeness_score(df, validation_results)
            result.validity_score = self._calc_validity_score(validation_results)
            result.uniqueness_score = self._calc_uniqueness_score(df, validation_results)

            # Validation time
            result.validation_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            logger.info(
                f"Validation completed: suite={suite_name}, "
                f"status={result.overall_status}, "
                f"passed={result.expectations_passed}/{result.expectations_evaluated}"
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            result.overall_status = "failed"
            result.notes = f"Validation error: {str(e)}"

        return result

    def _run_expectations(
        self, df: pd.DataFrame, expectations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run expectations on a DataFrame using GE 1.x API."""
        results = []

        try:
            # Create ephemeral context
            context = gx.get_context()

            # Create a unique name for this validation
            unique_id = str(uuid.uuid4())[:8]

            # Add pandas datasource
            datasource = context.data_sources.add_pandas(f"ds_{unique_id}")

            # Create data asset
            data_asset = datasource.add_dataframe_asset(f"asset_{unique_id}")

            # Create batch definition
            batch_def = data_asset.add_batch_definition_whole_dataframe(f"batch_{unique_id}")

            # Create expectation suite
            suite = context.suites.add(gx.ExpectationSuite(name=f"suite_{unique_id}"))

            # Map expectation types to GE expectation classes
            expectation_map = {
                "expect_column_to_exist": gx.expectations.ExpectColumnToExist,
                "expect_column_values_to_not_be_null": gx.expectations.ExpectColumnValuesToNotBeNull,
                "expect_column_values_to_be_between": gx.expectations.ExpectColumnValuesToBeBetween,
                "expect_column_values_to_be_in_set": gx.expectations.ExpectColumnValuesToBeInSet,
                "expect_column_values_to_be_unique": gx.expectations.ExpectColumnValuesToBeUnique,
                "expect_table_row_count_to_be_between": gx.expectations.ExpectTableRowCountToBeBetween,
                "expect_column_values_to_match_regex": gx.expectations.ExpectColumnValuesToMatchRegex,
            }

            # Add expectations to suite
            for exp in expectations:
                exp_type = exp["expectation_type"]
                kwargs = exp.get("kwargs", {})

                exp_class = expectation_map.get(exp_type)
                if exp_class:
                    try:
                        suite.add_expectation(exp_class(**kwargs))
                    except Exception as e:
                        logger.warning(f"Failed to add expectation {exp_type}: {e}")
                else:
                    logger.warning(f"Unknown expectation type: {exp_type}")

            # Create validation definition
            validation_def = context.validation_definitions.add(
                gx.ValidationDefinition(
                    name=f"validation_{unique_id}",
                    data=batch_def,
                    suite=suite
                )
            )

            # Run validation
            validation_result = validation_def.run(batch_parameters={"dataframe": df})

            # Extract individual expectation results
            for i, exp in enumerate(expectations):
                exp_type = exp["expectation_type"]
                kwargs = exp.get("kwargs", {})

                # Find matching result
                success = True
                result_data = {}

                if hasattr(validation_result, "results") and validation_result.results:
                    for res in validation_result.results:
                        res_type = getattr(res.expectation_config, "type", "")
                        if res_type == exp_type:
                            # Check if kwargs match
                            res_kwargs = res.expectation_config.kwargs or {}
                            if kwargs.get("column") == res_kwargs.get("column"):
                                success = res.success
                                result_data = dict(res.result) if res.result else {}
                                break

                results.append({
                    "expectation_type": exp_type,
                    "kwargs": kwargs,
                    "success": success,
                    "result": result_data,
                })

        except Exception as e:
            logger.error(f"GE validation failed: {e}", exc_info=True)
            # Mark all expectations as failed
            for exp in expectations:
                results.append({
                    "expectation_type": exp["expectation_type"],
                    "kwargs": exp.get("kwargs", {}),
                    "success": False,
                    "result": {"error": str(e)},
                })

        return results

    def _calc_completeness_score(
        self, df: pd.DataFrame, validation_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate completeness score from null checks."""
        # Use DataFrame's actual completeness
        if df.empty:
            return 0.0
        total_cells = df.size
        non_null_cells = df.notna().sum().sum()
        return float(non_null_cells / total_cells)

    def _calc_validity_score(
        self, validation_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate validity score from type/range checks."""
        validity_types = [
            "expect_column_values_to_be_between",
            "expect_column_values_to_be_in_set",
            "expect_column_values_to_match_regex",
        ]
        validity_results = [
            r for r in validation_results
            if r["expectation_type"] in validity_types
        ]
        if not validity_results:
            return 1.0
        return sum(1 for r in validity_results if r["success"]) / len(validity_results)

    def _calc_uniqueness_score(
        self, df: pd.DataFrame, validation_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate uniqueness score from unique checks."""
        uniqueness_types = [
            "expect_column_values_to_be_unique",
        ]
        uniqueness_results = [
            r for r in validation_results
            if r["expectation_type"] in uniqueness_types
        ]
        if not uniqueness_results:
            # Default: check for duplicate rows
            if df.empty:
                return 1.0
            return 1 - (df.duplicated().sum() / len(df))
        return sum(1 for r in uniqueness_results if r["success"]) / len(uniqueness_results)

    async def validate_splits(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: Optional[pd.DataFrame],
        suite_name: str,
        table_name: str,
        training_run_id: Optional[str] = None,
    ) -> Dict[str, DataQualityResult]:
        """
        Validate all data splits and return results.

        Args:
            train_df: Training data
            val_df: Validation data (optional)
            test_df: Test data (optional)
            suite_name: Expectation suite name
            table_name: Source table name
            training_run_id: Optional training run ID

        Returns:
            Dict mapping split name to validation result
        """
        results = {}

        # Validate train split (required)
        results["train"] = await self.validate(
            df=train_df,
            suite_name=suite_name,
            table_name=table_name,
            data_split="train",
            training_run_id=training_run_id,
        )

        # Validate val split (optional)
        if val_df is not None and not val_df.empty:
            results["validation"] = await self.validate(
                df=val_df,
                suite_name=suite_name,
                table_name=table_name,
                data_split="validation",
                training_run_id=training_run_id,
            )

        # Validate test split (optional)
        if test_df is not None and not test_df.empty:
            results["test"] = await self.validate(
                df=test_df,
                suite_name=suite_name,
                table_name=table_name,
                data_split="test",
                training_run_id=training_run_id,
            )

        return results

    async def run_checkpoint(
        self,
        checkpoint_name: str,
        df: pd.DataFrame,
        suite_name: str,
        table_name: str,
        data_split: DataSplitType = "full",
        brand: Optional[BrandType] = None,
        region: Optional[RegionType] = None,
        training_run_id: Optional[str] = None,
        fail_on_error: bool = True,
        store_result: bool = True,
        send_alerts: bool = True,
    ) -> DataQualityResult:
        """
        Run a named checkpoint for automated validation.

        A checkpoint is a configured validation point in the pipeline that:
        - Validates data against expectations
        - Optionally stores results for auditing
        - Sends alerts on failures (configurable)
        - Can fail the pipeline if validation fails

        Args:
            checkpoint_name: Unique name for this checkpoint
            df: DataFrame to validate
            suite_name: Expectation suite to use
            table_name: Source table name
            data_split: Data split type
            brand: Optional brand filter
            region: Optional region filter
            training_run_id: Optional training run ID
            fail_on_error: If True, raise exception on validation failure
            store_result: If True, add result to checkpoint history
            send_alerts: If True, send alerts on failures

        Returns:
            DataQualityResult with checkpoint results

        Raises:
            DataQualityCheckpointError: If fail_on_error=True and validation fails
        """
        logger.info(f"Running checkpoint '{checkpoint_name}' with suite '{suite_name}'")

        result = await self.validate(
            df=df,
            suite_name=suite_name,
            table_name=table_name,
            data_split=data_split,
            brand=brand,
            region=region,
            training_run_id=training_run_id,
        )

        # Store checkpoint result
        if store_result:
            self._checkpoint_history.append({
                "checkpoint_name": checkpoint_name,
                "result": result,
                "timestamp": datetime.now(),
            })

        # Send alerts if configured
        if send_alerts:
            await self.alert_on_result(result)

        # Handle failure
        if fail_on_error and result.blocking:
            error_msg = (
                f"Checkpoint '{checkpoint_name}' failed: "
                f"{result.expectations_failed} expectations failed, "
                f"success_rate={result.success_rate:.2%}"
            )
            logger.error(error_msg)
            raise DataQualityCheckpointError(error_msg, result)

        logger.info(
            f"Checkpoint '{checkpoint_name}' completed: "
            f"status={result.overall_status}, "
            f"passed={result.expectations_passed}/{result.expectations_evaluated}"
        )

        return result

    async def alert_on_result(
        self, result: DataQualityResult
    ) -> Dict[str, bool]:
        """Send alerts based on validation result.

        Uses the configured alerter to send notifications through
        all registered handlers (logging, webhook, custom).

        Args:
            result: The validation result

        Returns:
            Dict mapping handler names to success status
        """
        return await self.alerter.send_alert(result)

    async def validate_with_alerts(
        self,
        df: pd.DataFrame,
        suite_name: str,
        table_name: str,
        data_split: DataSplitType = "full",
        brand: Optional[BrandType] = None,
        region: Optional[RegionType] = None,
        training_run_id: Optional[str] = None,
    ) -> DataQualityResult:
        """
        Validate data and send alerts on failure.

        Convenience method that combines validation with alerting.

        Args:
            df: DataFrame to validate
            suite_name: Expectation suite name
            table_name: Source table name
            data_split: Data split type
            brand: Optional brand filter
            region: Optional region filter
            training_run_id: Optional training run ID

        Returns:
            DataQualityResult with validation results
        """
        result = await self.validate(
            df=df,
            suite_name=suite_name,
            table_name=table_name,
            data_split=data_split,
            brand=brand,
            region=region,
            training_run_id=training_run_id,
        )

        await self.alert_on_result(result)
        return result

    def get_checkpoint_history(
        self, checkpoint_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get checkpoint run history.

        Args:
            checkpoint_name: Filter by checkpoint name (optional)

        Returns:
            List of checkpoint run records
        """
        if checkpoint_name:
            return [
                h for h in self._checkpoint_history
                if h["checkpoint_name"] == checkpoint_name
            ]
        return list(self._checkpoint_history)

    def clear_checkpoint_history(self) -> None:
        """Clear checkpoint history."""
        self._checkpoint_history.clear()

    async def store_result(
        self,
        result: "DataQualityResult",
        supabase_client=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Store a validation result in the database.

        Args:
            result: DataQualityResult to store
            supabase_client: Optional Supabase client

        Returns:
            Stored record or None if storage failed
        """
        try:
            from src.repositories.data_quality_report import (
                get_data_quality_report_repository,
            )

            repo = get_data_quality_report_repository(supabase_client)
            stored = await repo.store_result(result.to_dict())

            if stored:
                logger.info(f"Stored DQ result: {result.report_name}")
            return stored

        except Exception as e:
            logger.error(f"Failed to store DQ result: {e}", exc_info=True)
            return None

    async def validate_and_store(
        self,
        df: pd.DataFrame,
        suite_name: str,
        table_name: str,
        data_split: DataSplitType = "full",
        brand: Optional[BrandType] = None,
        region: Optional[RegionType] = None,
        training_run_id: Optional[str] = None,
        supabase_client=None,
    ) -> DataQualityResult:
        """
        Validate data and automatically store results in the database.

        This is the recommended method for production pipelines as it
        ensures validation results are persisted for auditing.

        Args:
            df: DataFrame to validate
            suite_name: Expectation suite name
            table_name: Source table name
            data_split: Data split type
            brand: Optional brand filter
            region: Optional region filter
            training_run_id: Optional training run ID
            supabase_client: Optional Supabase client

        Returns:
            DataQualityResult with validation results
        """
        result = await self.validate(
            df=df,
            suite_name=suite_name,
            table_name=table_name,
            data_split=data_split,
            brand=brand,
            region=region,
            training_run_id=training_run_id,
        )

        # Store in database
        await self.store_result(result, supabase_client)

        return result


class DataQualityCheckpointError(Exception):
    """Raised when a data quality checkpoint fails."""

    def __init__(self, message: str, result: DataQualityResult):
        super().__init__(message)
        self.result = result
        self.failed_expectations = result.failed_expectations
        self.success_rate = result.success_rate


# Singleton instance
_validator: Optional[DataQualityValidator] = None


def get_data_quality_validator() -> DataQualityValidator:
    """Get the singleton DataQualityValidator instance."""
    global _validator
    if _validator is None:
        _validator = DataQualityValidator()
    return _validator
