"""CohortConstructor Observability Integration.

Provides MLflow experiment tracking and Opik distributed tracing for
cohort construction workflows.

Integrates with existing E2I observability connectors while providing
cohort-specific logging capabilities.

Usage:
    from src.agents.cohort_constructor.observability import (
        CohortMLflowLogger,
        CohortOpikTracer,
        get_cohort_mlflow_logger,
        get_cohort_opik_tracer,
    )

    # MLflow logging
    logger = get_cohort_mlflow_logger()
    logger.log_cohort_execution(result)
    logger.log_eligibility_funnel(stats)

    # Opik tracing
    tracer = get_cohort_opik_tracer()
    with tracer.trace_cohort_construction(config) as span:
        # ... construction logic ...
        span.log_criterion_evaluation(criterion, removed, remaining)
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .constants import AGENT_METADATA, SLAThreshold
from .types import CohortConfig, CohortExecutionResult, Criterion

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# MLflow Integration
# =============================================================================


class CohortMLflowLogger:
    """MLflow experiment tracking for CohortConstructor.

    Provides cohort-specific logging capabilities that integrate with
    the E2I MLflow connector.

    Logs:
    - Cohort execution metrics (size, rates, timing)
    - Eligibility funnel breakdown
    - Cohort characteristics (demographics, temporal)
    - Configuration artifacts
    """

    def __init__(self, connector: Optional[Any] = None):
        """Initialize CohortMLflowLogger.

        Args:
            connector: Optional MLflowConnector instance. If None, will
                       attempt to get the default connector.
        """
        self._connector = connector
        self._initialized = False

    def _get_connector(self) -> Optional[Any]:
        """Lazily get MLflow connector."""
        if self._connector is None:
            try:
                from src.mlops.mlflow_connector import get_mlflow_connector

                self._connector = get_mlflow_connector()
            except Exception as e:
                logger.warning(f"Could not get MLflow connector: {e}")
                return None
        return self._connector

    def log_cohort_execution(
        self,
        result: CohortExecutionResult,
        config: Optional[CohortConfig] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Log cohort execution metrics to MLflow.

        Args:
            result: Execution result with metrics
            config: Optional cohort configuration
            run_id: Optional existing run ID to log to

        Returns:
            MLflow run ID if successful, None otherwise
        """
        connector = self._get_connector()
        if connector is None:
            return None

        try:
            # Prepare metrics
            metrics = {
                "cohort_size": len(result.eligible_patient_ids),
                "execution_time_ms": result.execution_metadata.get("execution_time_ms", 0),
                "status_success": 1 if result.status == "success" else 0,
            }

            # Add eligibility stats if available
            if result.eligibility_stats:
                stats = result.eligibility_stats
                metrics.update(
                    {
                        "initial_population": stats.get("initial_population", 0),
                        "after_inclusion": stats.get("after_inclusion_criteria", 0),
                        "after_exclusion": stats.get("after_exclusion_criteria", 0),
                        "final_eligible": stats.get("final_eligible", 0),
                        "inclusion_removal_rate": stats.get("inclusion_removal_rate", 0),
                        "exclusion_removal_rate": stats.get("exclusion_removal_rate", 0),
                        "temporal_removal_rate": stats.get("temporal_removal_rate", 0),
                        "overall_eligibility_rate": stats.get("overall_eligibility_rate", 0),
                    }
                )

            # Prepare parameters
            params = {
                "cohort_id": result.cohort_id,
                "execution_id": result.execution_id,
                "agent_name": AGENT_METADATA["name"],
                "agent_tier": str(AGENT_METADATA["tier"]),
            }

            if config:
                params.update(
                    {
                        "brand": config.brand,
                        "indication": config.indication,
                        "cohort_name": config.cohort_name,
                        "inclusion_count": str(len(config.inclusion_criteria)),
                        "exclusion_count": str(len(config.exclusion_criteria)),
                        "lookback_days": str(config.temporal_requirements.lookback_days),
                        "followup_days": str(config.temporal_requirements.followup_days),
                    }
                )

            # Log using connector
            if hasattr(connector, "log_metrics") and callable(connector.log_metrics):
                connector.log_metrics(metrics, run_id=run_id)

            if hasattr(connector, "log_params") and callable(connector.log_params):
                connector.log_params(params, run_id=run_id)

            logger.debug(f"Logged cohort execution to MLflow: {result.cohort_id}")
            return run_id

        except Exception as e:
            logger.warning(f"Could not log cohort execution to MLflow: {e}")
            return None

    def log_eligibility_funnel(
        self,
        funnel_data: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> None:
        """Log eligibility funnel breakdown to MLflow.

        Args:
            funnel_data: List of criterion application results
            run_id: Optional run ID
        """
        connector = self._get_connector()
        if connector is None:
            return

        try:
            # Log each step as a metric with step number
            for i, step in enumerate(funnel_data):
                step_metrics = {
                    f"funnel_step_{i}_removed": step.get("removed", 0),
                    f"funnel_step_{i}_remaining": step.get("remaining", 0),
                }

                if hasattr(connector, "log_metrics"):
                    connector.log_metrics(step_metrics, run_id=run_id)

            logger.debug(f"Logged eligibility funnel with {len(funnel_data)} steps")

        except Exception as e:
            logger.warning(f"Could not log eligibility funnel: {e}")

    def log_cohort_characteristics(
        self,
        characteristics: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> None:
        """Log cohort demographic and clinical characteristics.

        Args:
            characteristics: Dict with demographics, clinical features
            run_id: Optional run ID
        """
        connector = self._get_connector()
        if connector is None:
            return

        try:
            # Convert characteristics to metrics
            metrics = {}

            # Demographics
            if "demographics" in characteristics:
                demo = characteristics["demographics"]
                metrics.update(
                    {
                        "mean_age": demo.get("mean_age", 0),
                        "female_pct": demo.get("female_pct", 0),
                        "male_pct": demo.get("male_pct", 0),
                    }
                )

            # Temporal characteristics
            if "temporal" in characteristics:
                temporal = characteristics["temporal"]
                metrics.update(
                    {
                        "mean_lookback_days": temporal.get("mean_lookback_days", 0),
                        "mean_followup_days": temporal.get("mean_followup_days", 0),
                        "temporal_coverage_pct": temporal.get("coverage_pct", 0),
                    }
                )

            # Clinical characteristics (brand-specific)
            if "clinical" in characteristics:
                for key, value in characteristics["clinical"].items():
                    if isinstance(value, (int, float)):
                        metrics[f"clinical_{key}"] = value

            if hasattr(connector, "log_metrics"):
                connector.log_metrics(metrics, run_id=run_id)

            logger.debug("Logged cohort characteristics to MLflow")

        except Exception as e:
            logger.warning(f"Could not log cohort characteristics: {e}")

    def log_config_artifact(
        self,
        config: CohortConfig,
        run_id: Optional[str] = None,
    ) -> None:
        """Log cohort configuration as MLflow artifact.

        Args:
            config: Cohort configuration
            run_id: Optional run ID
        """
        connector = self._get_connector()
        if connector is None:
            return

        try:
            import json
            import tempfile
            from pathlib import Path

            # Serialize config to JSON
            config_dict = config.to_dict()
            config_json = json.dumps(config_dict, indent=2, default=str)

            # Write to temp file and log
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(config_json)
                temp_path = f.name

            if hasattr(connector, "log_artifact"):
                connector.log_artifact(temp_path, "cohort_config.json", run_id=run_id)

            # Cleanup
            Path(temp_path).unlink(missing_ok=True)

            logger.debug("Logged cohort config artifact to MLflow")

        except Exception as e:
            logger.warning(f"Could not log config artifact: {e}")

    def log_sla_compliance(
        self,
        execution_time_ms: float,
        patient_count: int,
        run_id: Optional[str] = None,
    ) -> None:
        """Log SLA compliance metrics.

        Args:
            execution_time_ms: Execution time in milliseconds
            patient_count: Number of patients processed
            run_id: Optional run ID
        """
        connector = self._get_connector()
        if connector is None:
            return

        try:
            # Calculate SLA threshold based on patient count
            if patient_count <= 1000:
                sla_threshold = SLAThreshold.SMALL_COHORT_MS
            elif patient_count <= 10000:
                sla_threshold = SLAThreshold.MEDIUM_COHORT_MS
            elif patient_count <= 100000:
                sla_threshold = SLAThreshold.LARGE_COHORT_MS
            else:
                sla_threshold = SLAThreshold.VERY_LARGE_COHORT_MS

            sla_compliant = execution_time_ms <= sla_threshold
            sla_margin_pct = ((sla_threshold - execution_time_ms) / sla_threshold) * 100

            metrics = {
                "sla_threshold_ms": sla_threshold,
                "sla_compliant": 1 if sla_compliant else 0,
                "sla_margin_pct": sla_margin_pct,
                "patients_per_second": (patient_count / execution_time_ms) * 1000
                if execution_time_ms > 0
                else 0,
            }

            if hasattr(connector, "log_metrics"):
                connector.log_metrics(metrics, run_id=run_id)

            logger.debug(
                f"Logged SLA compliance: compliant={sla_compliant}, margin={sla_margin_pct:.1f}%"
            )

        except Exception as e:
            logger.warning(f"Could not log SLA compliance: {e}")


# =============================================================================
# Opik Integration
# =============================================================================


class CohortOpikTracer:
    """Opik distributed tracing for CohortConstructor.

    Provides hierarchical tracing at multiple granularity levels:
    - Top-level cohort construction
    - Individual criterion evaluation
    - Batch criteria processing
    - Temporal validation

    Each span captures structured telemetry with inputs, outputs, and metadata.
    """

    def __init__(self, connector: Optional[Any] = None):
        """Initialize CohortOpikTracer.

        Args:
            connector: Optional OpikConnector instance
        """
        self._connector = connector
        self._active_traces: Dict[str, Any] = {}

    def _get_connector(self) -> Optional[Any]:
        """Lazily get Opik connector."""
        if self._connector is None:
            try:
                from src.mlops.opik_connector import get_opik_connector

                self._connector = get_opik_connector()
            except Exception as e:
                logger.warning(f"Could not get Opik connector: {e}")
                return None
        return self._connector

    @contextmanager
    def trace_cohort_construction(
        self,
        config: CohortConfig,
        patient_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing cohort construction.

        Args:
            config: Cohort configuration
            patient_count: Number of patients in source population
            metadata: Optional additional metadata

        Yields:
            CohortTraceContext for logging within the trace
        """
        connector = self._get_connector()
        trace_context = CohortTraceContext(connector)

        try:
            # Start trace
            trace_context.start_trace(
                name="cohort_construction",
                inputs={
                    "brand": config.brand,
                    "indication": config.indication,
                    "cohort_name": config.cohort_name,
                    "patient_count": patient_count,
                    "inclusion_criteria_count": len(config.inclusion_criteria),
                    "exclusion_criteria_count": len(config.exclusion_criteria),
                },
                metadata={
                    "agent": AGENT_METADATA["name"],
                    "tier": AGENT_METADATA["tier"],
                    **(metadata or {}),
                },
            )

            yield trace_context

        except Exception as e:
            trace_context.log_error(str(e))
            raise

        finally:
            trace_context.end_trace()

    @asynccontextmanager
    async def trace_cohort_construction_async(
        self,
        config: CohortConfig,
        patient_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Async context manager for tracing cohort construction.

        Args:
            config: Cohort configuration
            patient_count: Number of patients
            metadata: Optional additional metadata

        Yields:
            CohortTraceContext for logging within the trace
        """
        with self.trace_cohort_construction(config, patient_count, metadata) as ctx:
            yield ctx


class CohortTraceContext:
    """Context for logging within a cohort construction trace."""

    def __init__(self, connector: Optional[Any] = None):
        """Initialize trace context.

        Args:
            connector: Opik connector
        """
        self._connector = connector
        self._trace_id: Optional[str] = None
        self._spans: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None
        self._criterion_spans: Dict[str, Any] = {}

    def start_trace(
        self,
        name: str,
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Start a new trace.

        Args:
            name: Trace name
            inputs: Input parameters
            metadata: Optional metadata

        Returns:
            Trace ID if successful
        """
        self._start_time = datetime.utcnow()

        if self._connector is None:
            return None

        try:
            if hasattr(self._connector, "start_trace"):
                self._trace_id = self._connector.start_trace(
                    name=name,
                    inputs=inputs,
                    metadata=metadata,
                )
                return self._trace_id
        except Exception as e:
            logger.warning(f"Could not start Opik trace: {e}")

        return None

    def end_trace(self, outputs: Optional[Dict[str, Any]] = None) -> None:
        """End the current trace.

        Args:
            outputs: Optional output data
        """
        if self._connector is None or self._trace_id is None:
            return

        try:
            duration_ms = 0.0
            if self._start_time:
                duration_ms = (datetime.utcnow() - self._start_time).total_seconds() * 1000

            if hasattr(self._connector, "end_trace"):
                self._connector.end_trace(
                    trace_id=self._trace_id,
                    outputs=outputs,
                    metadata={"duration_ms": duration_ms},
                )
        except Exception as e:
            logger.warning(f"Could not end Opik trace: {e}")

    def log_criterion_evaluation(
        self,
        criterion: Union[Criterion, Dict[str, Any]],
        criterion_type: str,
        initial_count: int,
        removed_count: int,
        remaining_count: int,
    ) -> None:
        """Log evaluation of a single criterion.

        Args:
            criterion: The criterion being evaluated
            criterion_type: "inclusion" or "exclusion"
            initial_count: Count before applying criterion
            removed_count: Number of patients removed
            remaining_count: Number of patients remaining
        """
        if self._connector is None:
            return

        try:
            # Convert criterion to dict if needed
            crit_dict = criterion.to_dict() if hasattr(criterion, "to_dict") else dict(criterion)

            span_data = {
                "name": f"evaluate_{criterion_type}_criterion",
                "inputs": {
                    "field": crit_dict.get("field", "unknown"),
                    "operator": crit_dict.get("operator", "unknown"),
                    "value": str(crit_dict.get("value", "")),
                    "initial_count": initial_count,
                },
                "outputs": {
                    "removed_count": removed_count,
                    "remaining_count": remaining_count,
                    "removal_rate": (removed_count / initial_count * 100)
                    if initial_count > 0
                    else 0,
                },
                "metadata": {
                    "criterion_type": criterion_type,
                    "description": crit_dict.get("description", ""),
                },
            }

            self._spans.append(span_data)

            if hasattr(self._connector, "log_span"):
                self._connector.log_span(
                    trace_id=self._trace_id,
                    **span_data,
                )

        except Exception as e:
            logger.warning(f"Could not log criterion evaluation: {e}")

    def log_temporal_validation(
        self,
        initial_count: int,
        passed_count: int,
        failed_count: int,
        lookback_days: int,
        followup_days: int,
    ) -> None:
        """Log temporal validation results.

        Args:
            initial_count: Count before temporal validation
            passed_count: Patients passing temporal validation
            failed_count: Patients failing temporal validation
            lookback_days: Required lookback period
            followup_days: Required followup period
        """
        if self._connector is None:
            return

        try:
            span_data = {
                "name": "validate_temporal_eligibility",
                "inputs": {
                    "initial_count": initial_count,
                    "lookback_days": lookback_days,
                    "followup_days": followup_days,
                },
                "outputs": {
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "pass_rate": (passed_count / initial_count * 100) if initial_count > 0 else 0,
                },
            }

            self._spans.append(span_data)

            if hasattr(self._connector, "log_span"):
                self._connector.log_span(
                    trace_id=self._trace_id,
                    **span_data,
                )

        except Exception as e:
            logger.warning(f"Could not log temporal validation: {e}")

    def log_execution_complete(
        self,
        eligible_count: int,
        total_count: int,
        execution_time_ms: float,
        status: str,
    ) -> None:
        """Log execution completion.

        Args:
            eligible_count: Final eligible patient count
            total_count: Total patients processed
            execution_time_ms: Total execution time
            status: Execution status
        """
        if self._connector is None:
            return

        try:
            outputs = {
                "eligible_count": eligible_count,
                "total_count": total_count,
                "eligibility_rate": (eligible_count / total_count * 100) if total_count > 0 else 0,
                "execution_time_ms": execution_time_ms,
                "status": status,
            }

            self.end_trace(outputs)

            # Log feedback score for dashboards
            if hasattr(self._connector, "log_feedback") and self._trace_id:
                try:
                    eligibility_rate = outputs.get("eligibility_rate", 0.0)
                    self._connector.log_feedback(
                        trace_id=self._trace_id,
                        score=float(eligible_count),
                        feedback_type="cohort_size",
                        reason=f"Eligible: {eligible_count}/{total_count} ({eligibility_rate:.1f}%)",
                    )
                except Exception as fb_e:
                    logger.warning(f"Could not log feedback: {fb_e}")

        except Exception as e:
            logger.warning(f"Could not log execution complete: {e}")

    def log_error(self, error_message: str, error_code: Optional[str] = None) -> None:
        """Log an error during execution.

        Args:
            error_message: Error description
            error_code: Optional error code
        """
        if self._connector is None:
            return

        try:
            if hasattr(self._connector, "log_span"):
                self._connector.log_span(
                    trace_id=self._trace_id,
                    name="error",
                    inputs={"error_code": error_code},
                    outputs={"error_message": error_message},
                    metadata={"level": "error"},
                )
        except Exception as e:
            logger.warning(f"Could not log error to Opik: {e}")


# =============================================================================
# Decorators
# =============================================================================


def track_cohort_step(step_name: str) -> Callable[[F], F]:
    """Decorator for tracking individual cohort construction steps.

    Args:
        step_name: Name of the step being tracked

    Returns:
        Decorated function with Opik tracing

    Usage:
        @track_cohort_step("custom_filter")
        def apply_custom_filter(df, config):
            # ... filtering logic ...
            return filtered_df
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_cohort_opik_tracer()
            connector = tracer._get_connector()

            start_time = datetime.utcnow()
            error = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                error = str(e)
                raise

            finally:
                if connector is not None:
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                    try:
                        if hasattr(connector, "log_span"):
                            connector.log_span(
                                name=step_name,
                                inputs={"args_count": len(args)},
                                outputs={
                                    "success": error is None,
                                    "error": error,
                                },
                                metadata={"duration_ms": duration_ms},
                            )
                    except Exception as e:
                        logger.warning(f"Could not log step {step_name}: {e}")

        return wrapper  # type: ignore

    return decorator


def track_cohort_construction(
    config_arg: str = "config",
    patient_count_arg: str = "patient_df",
) -> Callable[[F], F]:
    """Decorator for tracing entire cohort construction functions.

    Args:
        config_arg: Name of config argument
        patient_count_arg: Name of patient DataFrame argument

    Returns:
        Decorated function with full Opik tracing

    Usage:
        @track_cohort_construction()
        def construct_cohort(patient_df, config):
            # ... construction logic ...
            return eligible_df, result
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config and patient count
            config = kwargs.get(config_arg)
            patient_df = kwargs.get(patient_count_arg)
            patient_count = len(patient_df) if patient_df is not None else 0

            if config is None:
                # Try to find in positional args
                return func(*args, **kwargs)

            tracer = get_cohort_opik_tracer()

            with tracer.trace_cohort_construction(config, patient_count) as ctx:
                result = func(*args, **kwargs)

                # Log completion if result is tuple (eligible_df, execution_result)
                if isinstance(result, tuple) and len(result) == 2:
                    eligible_df, exec_result = result
                    if hasattr(exec_result, "status"):
                        ctx.log_execution_complete(
                            eligible_count=len(eligible_df),
                            total_count=patient_count,
                            execution_time_ms=exec_result.execution_metadata.get(
                                "execution_time_ms", 0
                            ),
                            status=exec_result.status,
                        )

                return result

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Factory Functions
# =============================================================================

# Singleton instances
_mlflow_logger: Optional[CohortMLflowLogger] = None
_opik_tracer: Optional[CohortOpikTracer] = None


def get_cohort_mlflow_logger() -> CohortMLflowLogger:
    """Get singleton CohortMLflowLogger instance.

    Returns:
        CohortMLflowLogger instance
    """
    global _mlflow_logger
    if _mlflow_logger is None:
        _mlflow_logger = CohortMLflowLogger()
    return _mlflow_logger


def get_cohort_opik_tracer() -> CohortOpikTracer:
    """Get singleton CohortOpikTracer instance.

    Returns:
        CohortOpikTracer instance
    """
    global _opik_tracer
    if _opik_tracer is None:
        _opik_tracer = CohortOpikTracer()
    return _opik_tracer


def reset_observability_singletons() -> None:
    """Reset singleton instances (useful for testing).

    Clears the cached MLflow logger and Opik tracer instances.
    """
    global _mlflow_logger, _opik_tracer
    _mlflow_logger = None
    _opik_tracer = None
