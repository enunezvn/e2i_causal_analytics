"""
Observability Integration for Synthetic Data Validation.

Integrates validation results with MLflow and Opik for tracking and monitoring.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

# Optional imports for observability tools
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]

try:
    import opik

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    opik = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .expectations import ValidationResult


# =============================================================================
# MLFLOW INTEGRATION
# =============================================================================


def log_validation_to_mlflow(
    results: Dict[str, "ValidationResult"],
    run_name: Optional[str] = None,
    experiment_name: str = "synthetic_data_validation",
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Log validation results to MLflow.

    Args:
        results: Dictionary of table_name -> ValidationResult
        run_name: Optional name for the MLflow run
        experiment_name: MLflow experiment name
        tags: Optional tags to add to the run

    Returns:
        MLflow run ID if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Generate run name if not provided
        if run_name is None:
            run_name = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            # Add tags
            mlflow.set_tag("validation_type", "synthetic_data")
            mlflow.set_tag("validation_timestamp", datetime.now().isoformat())
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            # Log overall metrics
            total_tables = len(results)
            passed_tables = sum(1 for r in results.values() if r.success)
            total_expectations = sum(
                r.statistics.get("evaluated_expectations", 0) for r in results.values()
            )
            passed_expectations = sum(
                r.statistics.get("successful_expectations", 0) for r in results.values()
            )

            mlflow.log_metrics(
                {
                    "total_tables_validated": total_tables,
                    "tables_passed": passed_tables,
                    "tables_failed": total_tables - passed_tables,
                    "total_expectations": total_expectations,
                    "passed_expectations": passed_expectations,
                    "failed_expectations": total_expectations - passed_expectations,
                    "overall_success_rate": (
                        passed_expectations / total_expectations if total_expectations > 0 else 0.0
                    ),
                }
            )

            # Log per-table metrics
            for table_name, result in results.items():
                prefix = f"{table_name}_"
                mlflow.log_metrics(
                    {
                        f"{prefix}success": 1 if result.success else 0,
                        f"{prefix}success_rate": result.success_rate,
                        f"{prefix}evaluated_expectations": result.statistics.get(
                            "evaluated_expectations", 0
                        ),
                        f"{prefix}successful_expectations": result.statistics.get(
                            "successful_expectations", 0
                        ),
                        f"{prefix}run_time_seconds": result.run_time_seconds,
                    }
                )

            # Log failed expectations as artifact
            failed_expectations = {}
            for table_name, result in results.items():
                if result.failed_expectations:
                    failed_expectations[table_name] = [
                        {
                            "expectation_type": exp["expectation_type"],
                            "kwargs": _serialize_kwargs(exp.get("kwargs", {})),
                        }
                        for exp in result.failed_expectations
                    ]

            if failed_expectations:
                mlflow.log_dict(failed_expectations, "failed_expectations.json")

            # Log full results as artifact
            full_results = {table_name: result.to_dict() for table_name, result in results.items()}
            mlflow.log_dict(full_results, "validation_results.json")

            return cast(str, run.info.run_id)

    except Exception as e:
        # Log error but don't fail the validation
        print(f"Warning: Failed to log to MLflow: {e}")
        return None


def _serialize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize kwargs for JSON logging."""
    result = {}
    for key, value in kwargs.items():
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = str(value)
    return result


# =============================================================================
# OPIK INTEGRATION
# =============================================================================


class ValidationSpan:
    """Context manager for Opik validation spans."""

    def __init__(
        self,
        operation_name: str,
        table_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.operation_name = operation_name
        self.table_name = table_name
        self.metadata = metadata or {}
        self.span: Optional[Any] = None
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> "ValidationSpan":
        self.start_time = datetime.now()
        if OPIK_AVAILABLE and opik:
            try:
                self.span = opik.trace(  # type: ignore[attr-defined]
                    name=self.operation_name,
                    metadata={
                        "table_name": self.table_name,
                        "validation_type": "synthetic_data",
                        **self.metadata,
                    },
                )
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            try:
                duration = (datetime.now() - self.start_time).total_seconds()
                self.span.end(
                    metadata={
                        "duration_seconds": duration,
                        "success": exc_type is None,
                        "error": str(exc_val) if exc_val else None,
                    }
                )
            except Exception:
                pass
        return False

    def log_result(self, result: "ValidationResult") -> None:
        """Log validation result to the span."""
        if self.span:
            try:
                self.span.log(
                    metadata={
                        "success": result.success,
                        "success_rate": result.success_rate,
                        "statistics": result.statistics,
                        "failed_count": len(result.failed_expectations),
                    }
                )
            except Exception:
                pass


def create_validation_span(
    operation_name: str = "synthetic_data_validation",
    table_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ValidationSpan:
    """
    Create a validation span for Opik tracing.

    Args:
        operation_name: Name of the validation operation
        table_name: Optional table being validated
        metadata: Additional metadata to include

    Returns:
        ValidationSpan context manager
    """
    return ValidationSpan(
        operation_name=operation_name,
        table_name=table_name,
        metadata=metadata,
    )


# =============================================================================
# COMBINED OBSERVABILITY
# =============================================================================


class ValidationObserver:
    """
    Combined observability for validation pipelines.

    Coordinates logging to MLflow and tracing with Opik.
    """

    def __init__(
        self,
        experiment_name: str = "synthetic_data_validation",
        enable_mlflow: bool = True,
        enable_opik: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.enable_opik = enable_opik and OPIK_AVAILABLE
        self.tags = tags or {}
        self._results: Dict[str, "ValidationResult"] = {}
        self._run_id: Optional[str] = None

    def record_result(
        self,
        table_name: str,
        result: "ValidationResult",
    ) -> None:
        """Record a validation result."""
        self._results[table_name] = result

    def finalize(
        self,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Finalize observability and log all results.

        Args:
            run_name: Optional name for the MLflow run

        Returns:
            Summary dictionary with run IDs and status
        """
        summary: Dict[str, Any] = {
            "mlflow_run_id": None,
            "tables_validated": len(self._results),
            "all_passed": all(r.success for r in self._results.values()),
        }

        # Log to MLflow
        if self.enable_mlflow and self._results:
            self._run_id = log_validation_to_mlflow(
                results=self._results,
                run_name=run_name,
                experiment_name=self.experiment_name,
                tags=self.tags,
            )
            summary["mlflow_run_id"] = self._run_id

        return summary


def get_observability_status() -> Dict[str, bool]:
    """Get status of observability integrations."""
    return {
        "mlflow_available": MLFLOW_AVAILABLE,
        "opik_available": OPIK_AVAILABLE,
    }
