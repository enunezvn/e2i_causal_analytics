"""
MLflow Integration for Prediction Synthesizer Agent.

Provides comprehensive MLflow tracking for ensemble prediction aggregation,
logging model agreement metrics, confidence intervals, and individual
model contributions.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - Dashboard metrics queries

Usage:
    tracker = PredictionSynthesizerMLflowTracker()
    async with tracker.start_prediction_run(
        experiment_name="churn_prediction",
        entity_type="hcp"
    ):
        output = await agent.run(input_data)
        await tracker.log_prediction_result(output, state)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

if TYPE_CHECKING:
    from .state import PredictionSynthesizerState

logger = logging.getLogger(__name__)

# Experiment prefix for Prediction Synthesizer Agent
EXPERIMENT_PREFIX = "e2i_causal/prediction_synthesizer"


@dataclass
class PredictionContext:
    """Context for an MLflow prediction run."""

    run_id: str
    experiment_name: str
    entity_type: str
    prediction_target: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional context
    brand: Optional[str] = None
    region: Optional[str] = None
    time_horizon: Optional[str] = None


@dataclass
class PredictionSynthesizerMetrics:
    """Structured metrics for prediction synthesizer tracking."""

    # Ensemble metrics
    point_estimate: Optional[float] = None
    prediction_interval_lower: Optional[float] = None
    prediction_interval_upper: Optional[float] = None
    ensemble_confidence: float = 0.0
    model_agreement: float = 0.0

    # Model execution metrics
    models_succeeded: int = 0
    models_failed: int = 0
    models_total: int = 0

    # Context metrics
    historical_accuracy: Optional[float] = None
    similar_cases_count: int = 0

    # Latency metrics
    orchestration_latency_ms: int = 0
    ensemble_latency_ms: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        metrics = {
            "ensemble_confidence": self.ensemble_confidence,
            "model_agreement": self.model_agreement,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "models_total": self.models_total,
            "similar_cases_count": self.similar_cases_count,
            "orchestration_latency_ms": self.orchestration_latency_ms,
            "ensemble_latency_ms": self.ensemble_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }

        # Add optional metrics
        if self.point_estimate is not None:
            metrics["point_estimate"] = self.point_estimate
        if self.prediction_interval_lower is not None:
            metrics["prediction_interval_lower"] = self.prediction_interval_lower
        if self.prediction_interval_upper is not None:
            metrics["prediction_interval_upper"] = self.prediction_interval_upper
        if self.historical_accuracy is not None:
            metrics["historical_accuracy"] = self.historical_accuracy

        return metrics


class PredictionSynthesizerMLflowTracker:
    """
    Tracks Prediction Synthesizer Agent metrics in MLflow.

    Integrates with MLflow to log:
    - Ensemble prediction metrics
    - Model agreement statistics
    - Individual model contributions
    - Confidence intervals
    - Performance metrics

    Example:
        tracker = PredictionSynthesizerMLflowTracker()

        async with tracker.start_prediction_run("churn", entity_type="hcp"):
            output = await agent.run(input_data)
            await tracker.log_prediction_result(output, final_state)

        # Query historical results
        history = await tracker.get_prediction_history(days=30)
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        enable_artifact_logging: bool = True,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: from env)
            enable_artifact_logging: Whether to log artifacts
        """
        self._mlflow = None
        self._tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.enable_artifact_logging = enable_artifact_logging
        self._current_run_id: Optional[str] = None

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import errors when not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not installed, tracking disabled")
                return None
        return self._mlflow

    @asynccontextmanager
    async def start_prediction_run(
        self,
        experiment_name: str = "default",
        entity_type: str = "unknown",
        prediction_target: str = "unknown",
        brand: Optional[str] = None,
        region: Optional[str] = None,
        time_horizon: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[PredictionContext]:
        """
        Start an MLflow run for prediction tracking.

        Args:
            experiment_name: Name of the experiment
            entity_type: Type of entity being predicted (hcp, territory, etc.)
            prediction_target: What is being predicted (churn, conversion, etc.)
            brand: E2I brand context
            region: E2I region context
            time_horizon: Prediction time horizon
            tags: Additional MLflow tags

        Yields:
            PredictionContext with run information
        """
        mlflow = self._get_mlflow()

        if mlflow is None:
            yield PredictionContext(
                run_id="no-mlflow",
                experiment_name=experiment_name,
                entity_type=entity_type,
                prediction_target=prediction_target,
                brand=brand,
                region=region,
                time_horizon=time_horizon,
            )
            return

        full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"

        try:
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_experiment_name,
                    artifact_location="mlflow-artifacts:/",
                    tags={
                        "framework": "e2i_causal",
                        "agent": "prediction_synthesizer",
                        "tier": "4",
                    },
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create/get experiment: {e}")
            yield PredictionContext(
                run_id="experiment-error",
                experiment_name=experiment_name,
                entity_type=entity_type,
                prediction_target=prediction_target,
            )
            return

        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                self._current_run_id = run.info.run_id

                # Log run parameters
                mlflow.log_params(
                    {
                        "agent": "prediction_synthesizer",
                        "tier": 4,
                        "entity_type": entity_type,
                        "prediction_target": prediction_target,
                    }
                )

                # Log context tags
                mlflow.set_tags(
                    {
                        "agent_type": "ml_predictions",
                        "framework_version": "4.3",
                    }
                )
                if brand:
                    mlflow.set_tag("brand", brand)
                if region:
                    mlflow.set_tag("region", region)
                if time_horizon:
                    mlflow.set_tag("time_horizon", time_horizon)

                # Custom tags
                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                ctx = PredictionContext(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    entity_type=entity_type,
                    prediction_target=prediction_target,
                    brand=brand,
                    region=region,
                    time_horizon=time_horizon,
                )

                yield ctx

                self._current_run_id = None

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            self._current_run_id = None
            raise

    async def log_prediction_result(
        self,
        state: "PredictionSynthesizerState",
    ) -> None:
        """
        Log prediction results to MLflow.

        Args:
            state: Final PredictionSynthesizerState from agent execution
        """
        mlflow = self._get_mlflow()
        if mlflow is None or self._current_run_id is None:
            return

        try:
            # Extract metrics from state
            metrics = self._extract_metrics(state)
            mlflow.log_metrics(metrics.to_dict())

            # Log additional parameters
            ensemble_pred = state.get("ensemble_prediction")
            if ensemble_pred:
                method = ensemble_pred.get("ensemble_method", "unknown") if isinstance(ensemble_pred, dict) else "unknown"
                mlflow.log_param("ensemble_method", method)

            # Log confidence and agreement tags
            mlflow.set_tags(
                {
                    "high_confidence": str(metrics.ensemble_confidence >= 0.7).lower(),
                    "strong_agreement": str(metrics.model_agreement >= 0.8).lower(),
                    "has_failures": str(metrics.models_failed > 0).lower(),
                }
            )

            # Log artifacts
            if self.enable_artifact_logging:
                await self._log_artifacts(state)

            logger.debug(
                f"Logged prediction metrics to MLflow run {self._current_run_id}: "
                f"estimate={metrics.point_estimate}, agreement={metrics.model_agreement}"
            )

        except Exception as e:
            logger.warning(f"Failed to log prediction metrics to MLflow: {e}")

    def _extract_metrics(
        self,
        state: "PredictionSynthesizerState",
    ) -> PredictionSynthesizerMetrics:
        """Extract metrics from state."""
        metrics = PredictionSynthesizerMetrics()

        # Ensemble prediction metrics
        ensemble = state.get("ensemble_prediction")
        if ensemble and isinstance(ensemble, dict):
            metrics.point_estimate = ensemble.get("point_estimate")
            metrics.prediction_interval_lower = ensemble.get("prediction_interval_lower")
            metrics.prediction_interval_upper = ensemble.get("prediction_interval_upper")
            metrics.ensemble_confidence = ensemble.get("confidence", 0.0)
            metrics.model_agreement = ensemble.get("model_agreement", 0.0)

        # Model execution metrics
        metrics.models_succeeded = state.get("models_succeeded", 0)
        metrics.models_failed = state.get("models_failed", 0)
        metrics.models_total = metrics.models_succeeded + metrics.models_failed

        # Context metrics
        pred_context = state.get("prediction_context")
        if pred_context and isinstance(pred_context, dict):
            metrics.historical_accuracy = pred_context.get("historical_accuracy")
            similar_cases = pred_context.get("similar_cases", [])
            metrics.similar_cases_count = len(similar_cases) if similar_cases else 0

        # Latency metrics
        metrics.orchestration_latency_ms = state.get("orchestration_latency_ms", 0)
        metrics.ensemble_latency_ms = state.get("ensemble_latency_ms", 0)
        metrics.total_latency_ms = state.get("total_latency_ms", 0)

        return metrics

    async def _log_artifacts(
        self,
        state: "PredictionSynthesizerState",
    ) -> None:
        """Log artifacts to MLflow."""
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log individual predictions
                individual_preds = state.get("individual_predictions", [])
                if individual_preds:
                    preds_path = os.path.join(tmpdir, "individual_predictions.json")
                    with open(preds_path, "w") as f:
                        json.dump(individual_preds, f, indent=2, default=str)
                    mlflow.log_artifact(preds_path, "predictions")

                # Log ensemble prediction
                ensemble_data = state.get("ensemble_prediction")
                if ensemble_data:
                    ensemble_path = os.path.join(tmpdir, "ensemble_prediction.json")
                    with open(ensemble_path, "w") as f:
                        json.dump(ensemble_data, f, indent=2, default=str)
                    mlflow.log_artifact(ensemble_path, "predictions")

                # Log prediction context
                context_data = state.get("prediction_context")
                if context_data:
                    context_path = os.path.join(tmpdir, "prediction_context.json")
                    with open(context_path, "w") as f:
                        json.dump(context_data, f, indent=2, default=str)
                    mlflow.log_artifact(context_path, "context")

        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    async def get_prediction_history(
        self,
        experiment_name: str = "default",
        entity_type: Optional[str] = None,
        prediction_target: Optional[str] = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query historical prediction runs.

        Args:
            experiment_name: Name of the experiment to query
            entity_type: Filter by entity type
            prediction_target: Filter by prediction target
            max_results: Maximum number of results to return

        Returns:
            List of historical prediction results
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return []

        try:
            full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                return []

            # Build filter string
            filters = []
            if entity_type:
                filters.append(f"params.entity_type = '{entity_type}'")
            if prediction_target:
                filters.append(f"params.prediction_target = '{prediction_target}'")

            filter_string = " AND ".join(filters) if filters else None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            history = []
            for _, row in runs.iterrows():
                history.append(
                    {
                        "run_id": row["run_id"],
                        "timestamp": row["start_time"],
                        "point_estimate": row.get("metrics.point_estimate"),
                        "ensemble_confidence": row.get("metrics.ensemble_confidence"),
                        "model_agreement": row.get("metrics.model_agreement"),
                        "models_succeeded": row.get("metrics.models_succeeded"),
                        "models_failed": row.get("metrics.models_failed"),
                        "total_latency_ms": row.get("metrics.total_latency_ms"),
                        "entity_type": row.get("params.entity_type"),
                        "prediction_target": row.get("params.prediction_target"),
                        "ensemble_method": row.get("params.ensemble_method"),
                    }
                )

            return history

        except Exception as e:
            logger.warning(f"Failed to query prediction history: {e}")
            return []

    async def get_model_performance_summary(
        self,
        experiment_name: str = "default",
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get summary of model performance over time.

        Args:
            experiment_name: Experiment to analyze
            days: Number of days to look back

        Returns:
            Dictionary with performance summary
        """
        history = await self.get_prediction_history(experiment_name, max_results=1000)

        if not history:
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "avg_model_agreement": 0.0,
                "model_success_rate": 0.0,
            }

        confidences = [h["ensemble_confidence"] for h in history if h.get("ensemble_confidence")]
        agreements = [h["model_agreement"] for h in history if h.get("model_agreement")]
        total_succeeded = sum(h.get("models_succeeded", 0) for h in history)
        total_models = sum(
            (h.get("models_succeeded", 0) + h.get("models_failed", 0)) for h in history
        )

        return {
            "total_predictions": len(history),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "avg_model_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
            "model_success_rate": total_succeeded / total_models if total_models > 0 else 0.0,
            "predictions_by_entity_type": self._count_by_field(history, "entity_type"),
            "predictions_by_target": self._count_by_field(history, "prediction_target"),
        }

    def _count_by_field(self, history: list[dict[str, Any]], field: str) -> dict[str, int]:
        """Count records by a specific field."""
        counts: dict[str, int] = {}
        for h in history:
            value = h.get(field) or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts


def create_tracker(tracking_uri: Optional[str] = None) -> PredictionSynthesizerMLflowTracker:
    """Factory function to create a Prediction Synthesizer MLflow tracker."""
    return PredictionSynthesizerMLflowTracker(tracking_uri=tracking_uri)
