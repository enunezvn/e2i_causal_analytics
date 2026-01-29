"""MLflow Tracker for Drift Monitor Agent.

Provides experiment tracking, metric logging, and artifact storage for
drift detection analyses. Tracks PSI scores, drift severity, and alerts.

Tier: 3 (Monitoring)
Pattern: Based on EnergyScoreMLflowTracker
"""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MLflow experiment prefix for this agent
EXPERIMENT_PREFIX = "e2i_causal/drift_monitor"


@dataclass
class DriftMonitorContext:
    """Context for drift monitoring MLflow tracking.

    Captures metadata about the monitoring context for logging.
    """

    experiment_name: str = "default"
    brand: Optional[str] = None
    model_id: Optional[str] = None
    time_window: Optional[str] = None
    query_id: Optional[str] = None
    run_id: Optional[str] = None
    start_time: Optional[datetime] = None


@dataclass
class DriftMonitorMetrics:
    """Metrics collected during drift monitoring.

    These metrics are logged to MLflow for tracking and comparison.
    """

    # Drift detection counts
    features_checked: int = 0
    features_with_drift: int = 0
    data_drift_count: int = 0
    model_drift_count: int = 0
    concept_drift_count: int = 0
    structural_drift_detected: bool = False

    # Drift scores
    overall_drift_score: float = 0.0
    avg_psi_score: float = 0.0
    max_psi_score: float = 0.0
    avg_ks_statistic: float = 0.0
    max_ks_statistic: float = 0.0

    # Severity breakdown
    critical_severity_count: int = 0
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    none_severity_count: int = 0

    # Alerts
    alerts_total: int = 0
    alerts_critical: int = 0
    alerts_warning: int = 0

    # Latency
    total_latency_ms: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)


class DriftMonitorMLflowTracker:
    """MLflow tracker for Drift Monitor Agent.

    Provides:
    - Experiment run management with async context manager
    - Metric logging for drift detection results
    - Artifact logging for detailed drift analysis
    - Historical query methods for dashboard integration

    Usage:
        tracker = DriftMonitorMLflowTracker()

        async with tracker.start_monitoring_run(
            experiment_name="pharma_drift_monitoring",
            brand="Kisqali",
            model_id="engagement_model_v1",
            time_window="7d",
        ) as ctx:
            # Run drift detection
            result = await agent.run(input_data)

            # Log results
            await tracker.log_monitoring_result(result, state)

        # Query historical monitoring
        history = tracker.get_monitoring_history(brand="Kisqali", limit=10)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI. If None, uses default.
        """
        self._mlflow = None
        self._tracking_uri = tracking_uri
        self._current_context: Optional[DriftMonitorContext] = None

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import errors if not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                self._mlflow = mlflow
                if self._tracking_uri:
                    mlflow.set_tracking_uri(self._tracking_uri)
            except (ImportError, OSError, PermissionError) as e:
                logger.warning(f"MLflow tracking unavailable ({type(e).__name__}): {e}")
                return None
        return self._mlflow

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create MLflow experiment.

        Args:
            experiment_name: Name suffix for experiment

        Returns:
            Experiment ID
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return ""

        full_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"

        try:
            experiment = mlflow.get_experiment_by_name(full_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_name,
                    artifact_location="mlflow-artifacts:/",
                    tags={
                        "agent": "drift_monitor",
                        "tier": "3",
                        "domain": "e2i_causal",
                    },
                )
            else:
                experiment_id = experiment.experiment_id

            return experiment_id
        except Exception as e:
            logger.warning(f"MLflow connection failed, continuing without tracking: {e}")
            self._mlflow = None  # Disable MLflow for subsequent calls
            return ""

    @asynccontextmanager
    async def start_monitoring_run(
        self,
        experiment_name: str = "default",
        brand: Optional[str] = None,
        model_id: Optional[str] = None,
        time_window: Optional[str] = None,
        query_id: Optional[str] = None,
    ):
        """Start an MLflow run for drift monitoring.

        Args:
            experiment_name: Name for the MLflow experiment
            brand: Brand being monitored
            model_id: Model ID being monitored
            time_window: Time window for drift comparison
            query_id: Unique identifier for this monitoring query

        Yields:
            DriftMonitorContext with run information

        Example:
            async with tracker.start_monitoring_run(
                experiment_name="pharma_monitoring",
                brand="Kisqali",
                model_id="engagement_model_v1",
                time_window="7d",
            ) as ctx:
                result = await agent.run(input_data)
                await tracker.log_monitoring_result(result, state)
        """
        mlflow = self._get_mlflow()
        context = DriftMonitorContext(
            experiment_name=experiment_name,
            brand=brand,
            model_id=model_id,
            time_window=time_window,
            query_id=query_id,
            start_time=datetime.utcnow(),
        )
        self._current_context = context

        if mlflow is None:
            yield context
            return

        experiment_id = self._get_or_create_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            context.run_id = run.info.run_id

            # Log initial parameters
            mlflow.log_params(
                {
                    "brand": brand or "all",
                    "model_id": model_id or "none",
                    "time_window": time_window or "7d",
                    "query_id": query_id or "",
                }
            )

            try:
                yield context
            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:250])
                raise
            finally:
                self._current_context = None

    async def log_monitoring_result(
        self,
        output: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log drift monitoring results to MLflow.

        Args:
            output: DriftMonitorOutput from agent
            state: Optional final state with additional metrics
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        # Extract metrics from output
        metrics = self._extract_metrics(output, state)

        # Log numeric metrics
        metric_dict = {
            # Drift counts
            "features_checked": metrics.features_checked,
            "features_with_drift": metrics.features_with_drift,
            "data_drift_count": metrics.data_drift_count,
            "model_drift_count": metrics.model_drift_count,
            "concept_drift_count": metrics.concept_drift_count,
            "structural_drift_detected": 1 if metrics.structural_drift_detected else 0,
            # Drift scores
            "overall_drift_score": metrics.overall_drift_score,
            "avg_psi_score": metrics.avg_psi_score,
            "max_psi_score": metrics.max_psi_score,
            "avg_ks_statistic": metrics.avg_ks_statistic,
            "max_ks_statistic": metrics.max_ks_statistic,
            # Severity counts
            "critical_severity_count": metrics.critical_severity_count,
            "high_severity_count": metrics.high_severity_count,
            "medium_severity_count": metrics.medium_severity_count,
            "low_severity_count": metrics.low_severity_count,
            "none_severity_count": metrics.none_severity_count,
            # Alerts
            "alerts_total": metrics.alerts_total,
            "alerts_critical": metrics.alerts_critical,
            "alerts_warning": metrics.alerts_warning,
            # Latency
            "total_latency_ms": metrics.total_latency_ms,
        }

        mlflow.log_metrics(metric_dict)

        # Log status tags
        drift_level = "none"
        if metrics.overall_drift_score >= 0.7:
            drift_level = "critical"
        elif metrics.overall_drift_score >= 0.5:
            drift_level = "high"
        elif metrics.overall_drift_score >= 0.3:
            drift_level = "medium"
        elif metrics.overall_drift_score > 0.0:
            drift_level = "low"

        mlflow.set_tag("drift_level", drift_level)
        mlflow.set_tag("status", "completed")

        # Log warnings count
        if metrics.warnings:
            mlflow.set_tag("warnings_count", str(len(metrics.warnings)))

        # Log artifacts
        await self._log_artifacts(output, state, metrics)

    def _extract_metrics(
        self,
        output: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> DriftMonitorMetrics:
        """Extract metrics from output and state.

        Args:
            output: DriftMonitorOutput (Pydantic model or dict)
            state: Optional final state

        Returns:
            DriftMonitorMetrics dataclass
        """
        metrics = DriftMonitorMetrics()

        # Handle both Pydantic models and dicts
        if hasattr(output, "model_dump"):
            output_dict = output.model_dump()
        elif hasattr(output, "dict"):
            output_dict = output.dict()
        elif isinstance(output, dict):
            output_dict = output
        else:
            return metrics

        # Basic counts
        metrics.features_checked = output_dict.get("features_checked", 0)
        features_with_drift = output_dict.get("features_with_drift", [])
        metrics.features_with_drift = len(features_with_drift)
        metrics.overall_drift_score = output_dict.get("overall_drift_score", 0.0)
        metrics.total_latency_ms = output_dict.get("total_latency_ms", 0)

        # Extract PSI and KS statistics from drift results
        psi_scores = []
        ks_statistics = []

        # Data drift results
        data_results = output_dict.get("data_drift_results", [])
        for result in data_results:
            if isinstance(result, dict):
                if result.get("drift_detected"):
                    metrics.data_drift_count += 1
                    self._count_severity(result.get("severity", "none"), metrics)

                # Extract test statistics
                stat = result.get("test_statistic", 0.0)
                if "psi" in result.get("drift_type", "").lower():
                    psi_scores.append(stat)
                else:
                    ks_statistics.append(stat)

        # Model drift results
        model_results = output_dict.get("model_drift_results", [])
        for result in model_results:
            if isinstance(result, dict):
                if result.get("drift_detected"):
                    metrics.model_drift_count += 1
                    self._count_severity(result.get("severity", "none"), metrics)

                stat = result.get("test_statistic", 0.0)
                ks_statistics.append(stat)

        # Concept drift results
        concept_results = output_dict.get("concept_drift_results", [])
        for result in concept_results:
            if isinstance(result, dict):
                if result.get("drift_detected"):
                    metrics.concept_drift_count += 1
                    self._count_severity(result.get("severity", "none"), metrics)

        # Calculate PSI statistics
        if psi_scores:
            metrics.avg_psi_score = sum(psi_scores) / len(psi_scores)
            metrics.max_psi_score = max(psi_scores)

        # Calculate KS statistics
        if ks_statistics:
            metrics.avg_ks_statistic = sum(ks_statistics) / len(ks_statistics)
            metrics.max_ks_statistic = max(ks_statistics)

        # Alerts
        alerts = output_dict.get("alerts", [])
        metrics.alerts_total = len(alerts)
        for alert in alerts:
            if isinstance(alert, dict):
                severity = alert.get("severity", "warning")
                if severity == "critical":
                    metrics.alerts_critical += 1
                else:
                    metrics.alerts_warning += 1

        # Structural drift (V4.4)
        if state:
            structural_details = state.get("structural_drift_details")
            if structural_details and isinstance(structural_details, dict):
                metrics.structural_drift_detected = structural_details.get("detected", False)

        # Warnings
        metrics.warnings = output_dict.get("warnings", [])

        return metrics

    def _count_severity(self, severity: str, metrics: DriftMonitorMetrics) -> None:
        """Count severity level.

        Args:
            severity: Severity string
            metrics: Metrics object to update
        """
        if severity == "critical":
            metrics.critical_severity_count += 1
        elif severity == "high":
            metrics.high_severity_count += 1
        elif severity == "medium":
            metrics.medium_severity_count += 1
        elif severity == "low":
            metrics.low_severity_count += 1
        else:
            metrics.none_severity_count += 1

    async def _log_artifacts(
        self,
        output: Any,
        state: Optional[Dict[str, Any]],
        metrics: DriftMonitorMetrics,
    ) -> None:
        """Log artifacts to MLflow.

        Args:
            output: DriftMonitorOutput
            state: Optional final state
            metrics: Extracted metrics
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        import tempfile
        import os

        # Handle both Pydantic models and dicts
        if hasattr(output, "model_dump"):
            output_dict = output.model_dump()
        elif hasattr(output, "dict"):
            output_dict = output.dict()
        elif isinstance(output, dict):
            output_dict = output
        else:
            output_dict = {}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log monitoring summary
                summary = {
                    "overall_drift_score": output_dict.get("overall_drift_score", 0.0),
                    "features_checked": output_dict.get("features_checked", 0),
                    "features_with_drift": output_dict.get("features_with_drift", []),
                    "drift_summary": output_dict.get("drift_summary", ""),
                    "recommended_actions": output_dict.get("recommended_actions", []),
                    "baseline_timestamp": output_dict.get("baseline_timestamp", ""),
                    "timestamp": output_dict.get("timestamp", ""),
                    "context": {
                        "brand": self._current_context.brand if self._current_context else None,
                        "model_id": self._current_context.model_id if self._current_context else None,
                        "time_window": self._current_context.time_window if self._current_context else None,
                    },
                }
                summary_path = os.path.join(tmpdir, "monitoring_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                mlflow.log_artifact(summary_path, "monitoring")

                # Log data drift results
                data_results = output_dict.get("data_drift_results", [])
                if data_results:
                    results_path = os.path.join(tmpdir, "data_drift_results.json")
                    with open(results_path, "w") as f:
                        json.dump(data_results, f, indent=2, default=str)
                    mlflow.log_artifact(results_path, "drift")

                # Log model drift results
                model_results = output_dict.get("model_drift_results", [])
                if model_results:
                    results_path = os.path.join(tmpdir, "model_drift_results.json")
                    with open(results_path, "w") as f:
                        json.dump(model_results, f, indent=2, default=str)
                    mlflow.log_artifact(results_path, "drift")

                # Log concept drift results
                concept_results = output_dict.get("concept_drift_results", [])
                if concept_results:
                    results_path = os.path.join(tmpdir, "concept_drift_results.json")
                    with open(results_path, "w") as f:
                        json.dump(concept_results, f, indent=2, default=str)
                    mlflow.log_artifact(results_path, "drift")

                # Log alerts
                alerts = output_dict.get("alerts", [])
                if alerts:
                    alerts_path = os.path.join(tmpdir, "alerts.json")
                    with open(alerts_path, "w") as f:
                        json.dump(alerts, f, indent=2, default=str)
                    mlflow.log_artifact(alerts_path, "alerts")

                # Log structural drift details (V4.4)
                if state:
                    structural_details = state.get("structural_drift_details")
                    if structural_details:
                        structural_path = os.path.join(tmpdir, "structural_drift.json")
                        with open(structural_path, "w") as f:
                            json.dump(structural_details, f, indent=2, default=str)
                        mlflow.log_artifact(structural_path, "drift")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to log MLflow artifacts: {e}")

    def get_monitoring_history(
        self,
        experiment_name: str = "default",
        brand: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query historical drift monitoring runs.

        Args:
            experiment_name: Name of the MLflow experiment
            brand: Filter by brand
            model_id: Filter by model ID
            limit: Maximum number of runs to return

        Returns:
            List of run data dictionaries
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return []

        full_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment is None:
            return []

        # Build filter string
        filter_parts = []
        if brand:
            filter_parts.append(f"params.brand = '{brand}'")
        if model_id:
            filter_parts.append(f"params.model_id = '{model_id}'")

        filter_string = " AND ".join(filter_parts) if filter_parts else ""

        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"],
        )

        if runs.empty:
            return []

        # Convert to list of dicts
        result = []
        for _, row in runs.iterrows():
            run_data = {
                "run_id": row.get("run_id"),
                "start_time": row.get("start_time"),
                "brand": row.get("params.brand"),
                "model_id": row.get("params.model_id"),
                "time_window": row.get("params.time_window"),
                "drift_level": row.get("tags.drift_level"),
                "status": row.get("tags.status"),
                # Key metrics
                "overall_drift_score": row.get("metrics.overall_drift_score"),
                "features_checked": row.get("metrics.features_checked"),
                "features_with_drift": row.get("metrics.features_with_drift"),
                "alerts_total": row.get("metrics.alerts_total"),
                "alerts_critical": row.get("metrics.alerts_critical"),
                "total_latency_ms": row.get("metrics.total_latency_ms"),
            }
            result.append(run_data)

        return result

    def get_drift_trend(
        self,
        experiment_name: str = "default",
        brand: Optional[str] = None,
        model_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get drift trend statistics over time.

        Args:
            experiment_name: Name of the MLflow experiment
            brand: Filter by brand
            model_id: Filter by model ID
            days: Number of days to look back

        Returns:
            Trend statistics dictionary
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return {}

        full_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment is None:
            return {}

        # Build filter
        from datetime import timedelta

        start_time = datetime.utcnow() - timedelta(days=days)
        filter_parts = [f"attributes.start_time >= {int(start_time.timestamp() * 1000)}"]
        if brand:
            filter_parts.append(f"params.brand = '{brand}'")
        if model_id:
            filter_parts.append(f"params.model_id = '{model_id}'")

        filter_string = " AND ".join(filter_parts)

        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=1000,
        )

        if runs.empty:
            return {
                "total_checks": 0,
                "period_days": days,
                "brand": brand,
                "model_id": model_id,
            }

        # Calculate trend stats
        return {
            "total_checks": len(runs),
            "period_days": days,
            "brand": brand,
            "model_id": model_id,
            # Drift score stats
            "avg_drift_score": runs["metrics.overall_drift_score"].mean(),
            "max_drift_score": runs["metrics.overall_drift_score"].max(),
            "min_drift_score": runs["metrics.overall_drift_score"].min(),
            "drift_score_trend": (
                "increasing"
                if runs["metrics.overall_drift_score"].diff().mean() > 0
                else "decreasing"
            ),
            # Alert stats
            "total_alerts": runs["metrics.alerts_total"].sum(),
            "total_critical_alerts": runs["metrics.alerts_critical"].sum(),
            "avg_alerts_per_check": runs["metrics.alerts_total"].mean(),
            # Drift level distribution
            "drift_levels": runs["tags.drift_level"].value_counts().to_dict()
            if "tags.drift_level" in runs.columns
            else {},
            # Performance stats
            "avg_latency_ms": runs["metrics.total_latency_ms"].mean(),
            "p95_latency_ms": runs["metrics.total_latency_ms"].quantile(0.95),
        }


def create_tracker(tracking_uri: Optional[str] = None) -> DriftMonitorMLflowTracker:
    """Factory function to create MLflow tracker.

    Args:
        tracking_uri: Optional MLflow tracking URI

    Returns:
        DriftMonitorMLflowTracker instance
    """
    return DriftMonitorMLflowTracker(tracking_uri=tracking_uri)
