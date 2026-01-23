"""MLflow Experiment Tracking for Data Preparer Agent.

This module provides MLflow integration for tracking data preparation runs,
including data quality scores, QC gate decisions, leakage detection, and
validation metrics.

The tracker follows the established E2I pattern with:
- Lazy MLflow loading to avoid import overhead
- Async context managers for clean resource management
- Comprehensive metric logging with dimension breakdowns
- Artifact logging for detailed QC reports
- Historical query methods for dashboard integration

Version: 1.0.0
"""

import json
import logging
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataQualityContext:
    """Context for a data preparation run.

    Attributes:
        experiment_id: Unique experiment identifier
        data_source: Name of the data source (table/view)
        split_id: ML split ID if using existing split
        validation_suite: Great Expectations suite name
        tags: Additional tags for the run
    """

    experiment_id: str
    data_source: str
    split_id: Optional[str] = None
    validation_suite: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataPreparerMetrics:
    """Metrics captured from a data preparation run.

    Captures data quality scores, QC gate status, sample counts,
    and validation performance metrics.
    """

    # Quality dimension scores (0.0 - 1.0)
    completeness_score: float = 0.0
    validity_score: float = 0.0
    consistency_score: float = 0.0
    uniqueness_score: float = 0.0
    timeliness_score: float = 0.0
    overall_score: float = 0.0

    # QC gate status
    qc_status: str = "unknown"  # passed, failed, warning, skipped
    qc_passed: bool = False
    qc_score: float = 0.0
    gate_passed: bool = False

    # Schema validation
    schema_validation_status: str = "unknown"
    schema_splits_validated: int = 0
    schema_validation_time_ms: int = 0

    # Leakage detection
    leakage_detected: bool = False
    leakage_issues_count: int = 0

    # Sample counts
    total_samples: int = 0
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    holdout_samples: int = 0

    # Feature counts
    column_count: int = 0
    available_features_count: int = 0
    missing_required_features_count: int = 0

    # Expectations
    total_expectations: int = 0
    failed_expectations_count: int = 0
    warnings_count: int = 0
    blocking_issues_count: int = 0

    # Feast registration
    feast_registration_status: str = "unknown"
    feast_features_registered: int = 0

    # Timing
    validation_duration_seconds: float = 0.0

    # Ready status
    is_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        return {
            # Quality scores
            "completeness_score": self.completeness_score,
            "validity_score": self.validity_score,
            "consistency_score": self.consistency_score,
            "uniqueness_score": self.uniqueness_score,
            "timeliness_score": self.timeliness_score,
            "overall_score": self.overall_score,
            # QC gate
            "qc_passed": float(self.qc_passed),
            "qc_score": self.qc_score,
            "gate_passed": float(self.gate_passed),
            # Schema
            "schema_splits_validated": float(self.schema_splits_validated),
            "schema_validation_time_ms": float(self.schema_validation_time_ms),
            # Leakage
            "leakage_detected": float(self.leakage_detected),
            "leakage_issues_count": float(self.leakage_issues_count),
            # Samples
            "total_samples": float(self.total_samples),
            "train_samples": float(self.train_samples),
            "validation_samples": float(self.validation_samples),
            "test_samples": float(self.test_samples),
            "holdout_samples": float(self.holdout_samples),
            # Features
            "column_count": float(self.column_count),
            "available_features_count": float(self.available_features_count),
            "missing_required_features_count": float(
                self.missing_required_features_count
            ),
            # Expectations
            "total_expectations": float(self.total_expectations),
            "failed_expectations_count": float(self.failed_expectations_count),
            "warnings_count": float(self.warnings_count),
            "blocking_issues_count": float(self.blocking_issues_count),
            # Feast
            "feast_features_registered": float(self.feast_features_registered),
            # Timing
            "validation_duration_seconds": self.validation_duration_seconds,
            # Ready
            "is_ready": float(self.is_ready),
        }


class DataPreparerMLflowTracker:
    """MLflow tracker for Data Preparer agent.

    Tracks data quality validation runs with:
    - Quality dimension scores (completeness, validity, etc.)
    - QC gate decisions
    - Leakage detection results
    - Schema validation status
    - Sample distribution across splits

    Usage:
        tracker = DataPreparerMLflowTracker(project_name="data_quality")

        context = DataQualityContext(
            experiment_id="exp_123",
            data_source="ml_features_v3",
        )

        async with tracker.track_preparation_run(context) as run:
            # Run preparation pipeline
            state = await run_data_preparer(initial_state)

            # Log metrics from state
            metrics = tracker.extract_metrics(state)
            await run.log_metrics(metrics)
    """

    def __init__(
        self,
        project_name: str = "data_preparer",
        tracking_uri: Optional[str] = None,
    ):
        """Initialize the MLflow tracker.

        Args:
            project_name: MLflow experiment name prefix
            tracking_uri: Optional MLflow tracking URI override
        """
        self.project_name = project_name
        self.tracking_uri = tracking_uri
        self._mlflow = None
        self._connector = None

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import overhead."""
        if self._mlflow is None:
            try:
                import mlflow

                self._mlflow = mlflow
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
            except ImportError:
                logger.warning("MLflow not available - tracking disabled")
                self._mlflow = None
        return self._mlflow

    def _get_connector(self):
        """Lazy load MLflow connector."""
        if self._connector is None:
            try:
                from src.mlops.mlflow_connector import get_mlflow_connector

                self._connector = get_mlflow_connector()
            except ImportError:
                logger.warning("MLflow connector not available")
                self._connector = None
        return self._connector

    @asynccontextmanager
    async def track_preparation_run(
        self,
        context: DataQualityContext,
    ) -> AsyncGenerator[Any, None]:
        """Track a data preparation run with MLflow.

        Creates an MLflow run, yields it for metric logging,
        and ensures proper cleanup on exit.

        Args:
            context: DataQualityContext with run metadata

        Yields:
            MLflow run object for logging metrics and artifacts
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            # Yield a no-op run object
            yield _NoOpRun()
            return

        connector = self._get_connector()
        if connector is None:
            yield _NoOpRun()
            return

        # Create experiment name
        experiment_name = f"{self.project_name}_data_quality"

        try:
            # Get or create experiment
            experiment_id = await connector.get_or_create_experiment(
                name=experiment_name,
                tags={
                    "agent": "data_preparer",
                    "tier": "0",
                    "source": "ml_foundation",
                },
            )

            # Generate run name
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"data_prep_{context.experiment_id}_{timestamp}"

            # Prepare tags
            tags = {
                "experiment_id": context.experiment_id,
                "data_source": context.data_source,
                "agent": "data_preparer",
                "tier": "0",
                **context.tags,
            }

            if context.split_id:
                tags["split_id"] = context.split_id
            if context.validation_suite:
                tags["validation_suite"] = context.validation_suite

            # Start run
            async with connector.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags,
                description=f"Data preparation for {context.data_source}",
            ) as run:
                # Log parameters
                await run.log_params(
                    {
                        "experiment_id": context.experiment_id,
                        "data_source": context.data_source,
                        "split_id": context.split_id or "auto",
                        "validation_suite": context.validation_suite or "default",
                    }
                )

                yield run

        except Exception as e:
            logger.error(f"MLflow tracking error: {e}")
            yield _NoOpRun()

    def extract_metrics(self, state: Dict[str, Any]) -> DataPreparerMetrics:
        """Extract metrics from DataPreparerState.

        Args:
            state: DataPreparerState dictionary

        Returns:
            DataPreparerMetrics with all tracked values
        """
        return DataPreparerMetrics(
            # Quality scores
            completeness_score=state.get("completeness_score", 0.0),
            validity_score=state.get("validity_score", 0.0),
            consistency_score=state.get("consistency_score", 0.0),
            uniqueness_score=state.get("uniqueness_score", 0.0),
            timeliness_score=state.get("timeliness_score", 0.0),
            overall_score=state.get("overall_score", 0.0),
            # QC gate
            qc_status=state.get("qc_status", "unknown"),
            qc_passed=state.get("qc_passed", False),
            qc_score=state.get("qc_score", 0.0),
            gate_passed=state.get("gate_passed", False),
            # Schema
            schema_validation_status=state.get(
                "schema_validation_status", "unknown"
            ),
            schema_splits_validated=state.get("schema_splits_validated", 0),
            schema_validation_time_ms=state.get("schema_validation_time_ms", 0),
            # Leakage
            leakage_detected=state.get("leakage_detected", False),
            leakage_issues_count=len(state.get("leakage_issues", [])),
            # Samples
            total_samples=state.get("total_samples", 0),
            train_samples=state.get("train_samples", 0),
            validation_samples=state.get("validation_samples", 0),
            test_samples=state.get("test_samples", 0),
            holdout_samples=state.get("holdout_samples", 0),
            # Features
            column_count=state.get("column_count", 0),
            available_features_count=len(state.get("available_features", [])),
            missing_required_features_count=len(
                state.get("missing_required_features", [])
            ),
            # Expectations
            total_expectations=len(state.get("expectation_results", [])),
            failed_expectations_count=len(state.get("failed_expectations", [])),
            warnings_count=len(state.get("warnings", [])),
            blocking_issues_count=len(state.get("blocking_issues", [])),
            # Feast
            feast_registration_status=state.get(
                "feast_registration_status", "unknown"
            ),
            feast_features_registered=state.get("feast_features_registered", 0),
            # Timing
            validation_duration_seconds=state.get(
                "validation_duration_seconds", 0.0
            ),
            # Ready
            is_ready=state.get("is_ready", False),
        )

    async def log_qc_report(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log detailed QC report as artifact.

        Args:
            run: MLflow run object
            state: DataPreparerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        report = {
            "report_id": state.get("report_id"),
            "experiment_id": state.get("experiment_id"),
            "data_source": state.get("data_source"),
            "validated_at": state.get("validated_at"),
            "qc_status": state.get("qc_status"),
            "gate_passed": state.get("gate_passed"),
            "quality_scores": {
                "completeness": state.get("completeness_score"),
                "validity": state.get("validity_score"),
                "consistency": state.get("consistency_score"),
                "uniqueness": state.get("uniqueness_score"),
                "timeliness": state.get("timeliness_score"),
                "overall": state.get("overall_score"),
            },
            "schema_validation": {
                "status": state.get("schema_validation_status"),
                "splits_validated": state.get("schema_splits_validated"),
                "errors": state.get("schema_validation_errors", []),
            },
            "leakage_detection": {
                "detected": state.get("leakage_detected"),
                "issues": state.get("leakage_issues", []),
            },
            "sample_distribution": {
                "total": state.get("total_samples"),
                "train": state.get("train_samples"),
                "validation": state.get("validation_samples"),
                "test": state.get("test_samples"),
                "holdout": state.get("holdout_samples"),
            },
            "failed_expectations": state.get("failed_expectations", []),
            "warnings": state.get("warnings", []),
            "blocking_issues": state.get("blocking_issues", []),
            "remediation_steps": state.get("remediation_steps", []),
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(report, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "qc_report.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log QC report artifact: {e}")

    async def log_feature_stats(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log feature statistics as artifact.

        Args:
            run: MLflow run object
            state: DataPreparerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        feature_stats = state.get("feature_stats", {})
        if not feature_stats:
            return

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(feature_stats, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "feature_stats.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log feature stats artifact: {e}")

    async def log_expectation_results(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log Great Expectations results as artifact.

        Args:
            run: MLflow run object
            state: DataPreparerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        results = state.get("expectation_results", [])
        if not results:
            return

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(results, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "expectation_results.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log expectation results artifact: {e}")

    async def get_quality_history(
        self,
        data_source: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get historical quality scores for a data source.

        Args:
            data_source: Data source name to filter by
            limit: Maximum number of runs to return

        Returns:
            List of historical quality metrics
        """
        connector = self._get_connector()
        if connector is None:
            return []

        try:
            experiment_name = f"{self.project_name}_data_quality"
            runs = await connector.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.data_source = '{data_source}'",
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )

            history = []
            for run in runs:
                history.append(
                    {
                        "run_id": run.info.run_id,
                        "timestamp": run.info.start_time,
                        "completeness_score": run.data.metrics.get(
                            "completeness_score"
                        ),
                        "validity_score": run.data.metrics.get("validity_score"),
                        "consistency_score": run.data.metrics.get(
                            "consistency_score"
                        ),
                        "uniqueness_score": run.data.metrics.get("uniqueness_score"),
                        "timeliness_score": run.data.metrics.get("timeliness_score"),
                        "overall_score": run.data.metrics.get("overall_score"),
                        "qc_passed": run.data.metrics.get("qc_passed"),
                        "gate_passed": run.data.metrics.get("gate_passed"),
                        "total_samples": run.data.metrics.get("total_samples"),
                        "leakage_detected": run.data.metrics.get("leakage_detected"),
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Failed to get quality history: {e}")
            return []

    async def get_leakage_incidents(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get recent data leakage incidents.

        Args:
            days: Number of days to look back

        Returns:
            List of runs where leakage was detected
        """
        connector = self._get_connector()
        if connector is None:
            return []

        try:
            experiment_name = f"{self.project_name}_data_quality"
            runs = await connector.search_runs(
                experiment_names=[experiment_name],
                filter_string="metrics.leakage_detected = 1",
                max_results=100,
                order_by=["attributes.start_time DESC"],
            )

            incidents = []
            for run in runs:
                incidents.append(
                    {
                        "run_id": run.info.run_id,
                        "timestamp": run.info.start_time,
                        "data_source": run.data.tags.get("data_source"),
                        "experiment_id": run.data.tags.get("experiment_id"),
                        "leakage_issues_count": run.data.metrics.get(
                            "leakage_issues_count"
                        ),
                    }
                )

            return incidents

        except Exception as e:
            logger.error(f"Failed to get leakage incidents: {e}")
            return []


class _NoOpRun:
    """No-op run object when MLflow is unavailable."""

    run_id = None

    async def log_params(self, params: Dict[str, Any]) -> None:
        """No-op parameter logging."""
        pass

    async def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """No-op metric logging."""
        pass

    async def log_artifact(self, local_path: str, artifact_path: str) -> None:
        """No-op artifact logging."""
        pass


def create_tracker(
    project_name: str = "data_preparer",
    tracking_uri: Optional[str] = None,
) -> DataPreparerMLflowTracker:
    """Factory function to create a DataPreparerMLflowTracker.

    Args:
        project_name: MLflow experiment name prefix
        tracking_uri: Optional MLflow tracking URI override

    Returns:
        Configured DataPreparerMLflowTracker instance
    """
    return DataPreparerMLflowTracker(
        project_name=project_name,
        tracking_uri=tracking_uri,
    )
