"""MLflow Experiment Tracking for Feature Analyzer Agent.

This module provides MLflow integration for tracking feature analysis runs,
including SHAP values, feature importance, feature selection, and
interaction detection results.

The tracker follows the established E2I pattern with:
- Lazy MLflow loading to avoid import overhead
- Async context managers for clean resource management
- Comprehensive metric logging for SHAP analysis
- Artifact logging for importance rankings and interactions
- Historical query methods for trend analysis

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
class FeatureAnalysisContext:
    """Context for a feature analysis run.

    Attributes:
        experiment_id: Unique experiment identifier
        model_uri: MLflow model URI being analyzed
        training_run_id: Training run that produced the model
        problem_type: Classification or regression
        tags: Additional tags for the run
    """

    experiment_id: str
    model_uri: Optional[str] = None
    training_run_id: Optional[str] = None
    problem_type: str = "classification"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureAnalyzerMetrics:
    """Metrics captured from a feature analysis run.

    Captures SHAP analysis, feature selection, and interaction
    detection metrics.
    """

    # SHAP Analysis Metrics
    shap_analysis_id: Optional[str] = None
    samples_analyzed: int = 0
    explainer_type: str = "unknown"
    base_value: float = 0.0
    shap_computation_time_seconds: float = 0.0

    # Feature Importance Metrics
    top_feature_importance: float = 0.0
    avg_feature_importance: float = 0.0
    feature_count: int = 0
    top_features_count: int = 0

    # Feature Direction Distribution
    positive_direction_count: int = 0
    negative_direction_count: int = 0
    mixed_direction_count: int = 0
    neutral_direction_count: int = 0

    # Feature Generation Metrics
    original_feature_count: int = 0
    new_feature_count: int = 0
    feature_generation_time_seconds: float = 0.0

    # Feature Selection Metrics
    selected_feature_count: int = 0
    removed_variance_count: int = 0
    removed_correlation_count: int = 0
    removed_vif_count: int = 0
    selection_time_seconds: float = 0.0

    # Interaction Detection Metrics
    top_interactions_count: int = 0
    max_interaction_strength: float = 0.0
    avg_interaction_strength: float = 0.0
    interaction_computation_time_seconds: float = 0.0

    # Causal Discovery (V4.4)
    discovery_enabled: bool = False
    discovery_edge_count: int = 0
    discovery_gate_decision: str = "unknown"
    discovery_gate_confidence: float = 0.0
    rank_correlation: float = 0.0
    divergent_features_count: int = 0

    # Overall Timing
    total_computation_time_seconds: float = 0.0

    # Status
    status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        return {
            # SHAP metrics
            "samples_analyzed": float(self.samples_analyzed),
            "base_value": self.base_value,
            "shap_computation_time_seconds": self.shap_computation_time_seconds,
            # Importance metrics
            "top_feature_importance": self.top_feature_importance,
            "avg_feature_importance": self.avg_feature_importance,
            "feature_count": float(self.feature_count),
            "top_features_count": float(self.top_features_count),
            # Direction distribution
            "positive_direction_count": float(self.positive_direction_count),
            "negative_direction_count": float(self.negative_direction_count),
            "mixed_direction_count": float(self.mixed_direction_count),
            "neutral_direction_count": float(self.neutral_direction_count),
            # Feature generation
            "original_feature_count": float(self.original_feature_count),
            "new_feature_count": float(self.new_feature_count),
            "feature_generation_time_seconds": self.feature_generation_time_seconds,
            # Feature selection
            "selected_feature_count": float(self.selected_feature_count),
            "removed_variance_count": float(self.removed_variance_count),
            "removed_correlation_count": float(self.removed_correlation_count),
            "removed_vif_count": float(self.removed_vif_count),
            "selection_time_seconds": self.selection_time_seconds,
            # Interaction metrics
            "top_interactions_count": float(self.top_interactions_count),
            "max_interaction_strength": self.max_interaction_strength,
            "avg_interaction_strength": self.avg_interaction_strength,
            "interaction_computation_time_seconds": self.interaction_computation_time_seconds,
            # Causal discovery
            "discovery_enabled": float(self.discovery_enabled),
            "discovery_edge_count": float(self.discovery_edge_count),
            "discovery_gate_confidence": self.discovery_gate_confidence,
            "rank_correlation": self.rank_correlation,
            "divergent_features_count": float(self.divergent_features_count),
            # Overall
            "total_computation_time_seconds": self.total_computation_time_seconds,
        }


class FeatureAnalyzerMLflowTracker:
    """MLflow tracker for Feature Analyzer agent.

    Tracks feature analysis runs with:
    - SHAP computation metrics and values
    - Feature importance rankings
    - Feature selection decisions
    - Interaction detection results
    - Causal discovery integration metrics

    Usage:
        tracker = FeatureAnalyzerMLflowTracker(project_name="feature_analysis")

        context = FeatureAnalysisContext(
            experiment_id="exp_123",
            model_uri="runs:/abc123/model",
        )

        async with tracker.track_analysis_run(context) as run:
            # Run analysis pipeline
            state = await run_feature_analyzer(initial_state)

            # Log metrics from state
            metrics = tracker.extract_metrics(state)
            await run.log_metrics(metrics.to_dict())
    """

    def __init__(
        self,
        project_name: str = "feature_analyzer",
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
    async def track_analysis_run(
        self,
        context: FeatureAnalysisContext,
    ) -> AsyncGenerator[Any, None]:
        """Track a feature analysis run with MLflow.

        Creates an MLflow run, yields it for metric logging,
        and ensures proper cleanup on exit.

        Args:
            context: FeatureAnalysisContext with run metadata

        Yields:
            MLflow run object for logging metrics and artifacts
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            yield _NoOpRun()
            return

        connector = self._get_connector()
        if connector is None:
            yield _NoOpRun()
            return

        # Create experiment name
        experiment_name = f"{self.project_name}_shap_analysis"

        try:
            # Get or create experiment
            experiment_id = await connector.get_or_create_experiment(
                name=experiment_name,
                tags={
                    "agent": "feature_analyzer",
                    "tier": "0",
                    "source": "ml_foundation",
                    "analysis_type": "shap",
                },
            )

            # Generate run name
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"shap_{context.experiment_id}_{timestamp}"

            # Prepare tags
            tags = {
                "experiment_id": context.experiment_id,
                "problem_type": context.problem_type,
                "agent": "feature_analyzer",
                "tier": "0",
                **context.tags,
            }

            if context.model_uri:
                tags["model_uri"] = context.model_uri
            if context.training_run_id:
                tags["training_run_id"] = context.training_run_id

            # Start run
            async with connector.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags,
                description=f"SHAP analysis for {context.experiment_id}",
            ) as run:
                # Log parameters
                await run.log_params(
                    {
                        "experiment_id": context.experiment_id,
                        "model_uri": context.model_uri or "none",
                        "problem_type": context.problem_type,
                    }
                )

                yield run

        except Exception as e:
            logger.error(f"MLflow tracking error: {e}")
            yield _NoOpRun()

    def extract_metrics(self, state: Dict[str, Any]) -> FeatureAnalyzerMetrics:
        """Extract metrics from FeatureAnalyzerState.

        Args:
            state: FeatureAnalyzerState dictionary

        Returns:
            FeatureAnalyzerMetrics with all tracked values
        """
        # Extract feature importance stats
        global_importance = state.get("global_importance", {})
        importance_values = list(global_importance.values()) if global_importance else []

        # Extract feature direction counts
        feature_directions = state.get("feature_directions", {})
        direction_counts = {
            "positive": 0,
            "negative": 0,
            "mixed": 0,
            "neutral": 0,
        }
        for direction in feature_directions.values():
            if direction in direction_counts:
                direction_counts[direction] += 1

        # Extract removed feature counts
        removed_features = state.get("removed_features", {})

        # Extract interaction stats
        top_interactions = state.get("top_interactions_raw", [])
        interaction_strengths = [t[2] for t in top_interactions] if top_interactions else []

        return FeatureAnalyzerMetrics(
            # SHAP analysis
            shap_analysis_id=state.get("shap_analysis_id"),
            samples_analyzed=state.get("samples_analyzed", 0),
            explainer_type=state.get("explainer_type", "unknown"),
            base_value=float(state.get("base_value", 0.0)),
            shap_computation_time_seconds=state.get("shap_computation_time_seconds", 0.0),
            # Importance metrics
            top_feature_importance=(max(importance_values) if importance_values else 0.0),
            avg_feature_importance=(
                sum(importance_values) / len(importance_values) if importance_values else 0.0
            ),
            feature_count=len(state.get("feature_names", [])),
            top_features_count=len(state.get("top_features", [])),
            # Direction distribution
            positive_direction_count=direction_counts["positive"],
            negative_direction_count=direction_counts["negative"],
            mixed_direction_count=direction_counts["mixed"],
            neutral_direction_count=direction_counts["neutral"],
            # Feature generation
            original_feature_count=state.get("original_feature_count", 0),
            new_feature_count=state.get("new_feature_count", 0),
            feature_generation_time_seconds=state.get("feature_generation_time_seconds", 0.0),
            # Feature selection
            selected_feature_count=state.get("selected_feature_count", 0),
            removed_variance_count=len(removed_features.get("variance", [])),
            removed_correlation_count=len(removed_features.get("correlation", [])),
            removed_vif_count=len(removed_features.get("vif", [])),
            selection_time_seconds=state.get("selection_time_seconds", 0.0),
            # Interaction metrics
            top_interactions_count=len(top_interactions),
            max_interaction_strength=(max(interaction_strengths) if interaction_strengths else 0.0),
            avg_interaction_strength=(
                sum(interaction_strengths) / len(interaction_strengths)
                if interaction_strengths
                else 0.0
            ),
            interaction_computation_time_seconds=state.get(
                "interaction_computation_time_seconds", 0.0
            ),
            # Causal discovery
            discovery_enabled=state.get("discovery_enabled", False),
            discovery_edge_count=state.get("discovery_result", {}).get("n_edges", 0),
            discovery_gate_decision=state.get("discovery_gate_decision", "unknown"),
            discovery_gate_confidence=state.get("discovery_gate_confidence", 0.0),
            rank_correlation=state.get("rank_correlation", 0.0),
            divergent_features_count=len(state.get("divergent_features", [])),
            # Overall
            total_computation_time_seconds=state.get("total_computation_time_seconds", 0.0),
            status=state.get("status", "unknown"),
        )

    async def log_feature_importance(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log feature importance as artifact.

        Args:
            run: MLflow run object
            state: FeatureAnalyzerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        global_importance_ranked = state.get("global_importance_ranked", [])
        if not global_importance_ranked:
            return

        importance_data = {
            "shap_analysis_id": state.get("shap_analysis_id"),
            "experiment_id": state.get("experiment_id"),
            "model_uri": state.get("model_uri"),
            "explainer_type": state.get("explainer_type"),
            "samples_analyzed": state.get("samples_analyzed"),
            "top_features": state.get("top_features", []),
            "feature_importance_ranked": [
                {"feature": f, "importance": float(i)} for f, i in global_importance_ranked
            ],
            "feature_directions": state.get("feature_directions", {}),
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(importance_data, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "feature_importance.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log feature importance artifact: {e}")

    async def log_interactions(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log feature interactions as artifact.

        Args:
            run: MLflow run object
            state: FeatureAnalyzerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        top_interactions = state.get("top_interactions_raw", [])
        if not top_interactions:
            return

        interaction_data = {
            "shap_analysis_id": state.get("shap_analysis_id"),
            "interaction_method": state.get("interaction_method"),
            "computation_time_seconds": state.get("interaction_computation_time_seconds"),
            "top_interactions": [
                {
                    "feature_1": feat1,
                    "feature_2": feat2,
                    "strength": float(strength),
                }
                for feat1, feat2, strength in top_interactions
            ],
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(interaction_data, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "feature_interactions.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log interactions artifact: {e}")

    async def log_selection_summary(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log feature selection summary as artifact.

        Args:
            run: MLflow run object
            state: FeatureAnalyzerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        selection_history = state.get("selection_history", [])
        if not selection_history:
            return

        selection_data = {
            "experiment_id": state.get("experiment_id"),
            "original_feature_count": state.get("original_feature_count"),
            "selected_feature_count": state.get("selected_feature_count"),
            "selected_features": state.get("selected_features", []),
            "removed_features": state.get("removed_features", {}),
            "selection_history": selection_history,
            "selection_time_seconds": state.get("selection_time_seconds"),
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(selection_data, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "feature_selection.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log selection summary artifact: {e}")

    async def log_causal_comparison(
        self,
        run: Any,
        state: Dict[str, Any],
    ) -> None:
        """Log causal vs predictive comparison as artifact.

        Args:
            run: MLflow run object
            state: FeatureAnalyzerState dictionary
        """
        if not hasattr(run, "log_artifact"):
            return

        if not state.get("discovery_enabled", False):
            return

        comparison_data = {
            "experiment_id": state.get("experiment_id"),
            "discovery_gate_decision": state.get("discovery_gate_decision"),
            "discovery_gate_confidence": state.get("discovery_gate_confidence"),
            "rank_correlation": state.get("rank_correlation"),
            "causal_rankings": state.get("causal_rankings", []),
            "divergent_features": state.get("divergent_features", []),
            "concordant_features": state.get("concordant_features", []),
            "direct_cause_features": state.get("direct_cause_features", []),
            "causal_interpretation": state.get("causal_interpretation"),
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(comparison_data, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "causal_comparison.json")
                Path(f.name).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to log causal comparison artifact: {e}")

    async def get_analysis_history(
        self,
        model_uri: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get historical analysis results.

        Args:
            model_uri: Filter by model URI (optional)
            limit: Maximum number of runs to return

        Returns:
            List of historical analysis metrics
        """
        connector = self._get_connector()
        if connector is None:
            return []

        try:
            experiment_name = f"{self.project_name}_shap_analysis"
            filter_string = f"tags.model_uri = '{model_uri}'" if model_uri else ""

            runs = await connector.search_runs(
                experiment_names=[experiment_name],
                filter_string=filter_string,
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )

            history = []
            for run in runs:
                history.append(
                    {
                        "run_id": run.info.run_id,
                        "timestamp": run.info.start_time,
                        "experiment_id": run.data.tags.get("experiment_id"),
                        "model_uri": run.data.tags.get("model_uri"),
                        "samples_analyzed": run.data.metrics.get("samples_analyzed"),
                        "feature_count": run.data.metrics.get("feature_count"),
                        "top_feature_importance": run.data.metrics.get("top_feature_importance"),
                        "shap_computation_time": run.data.metrics.get(
                            "shap_computation_time_seconds"
                        ),
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Failed to get analysis history: {e}")
            return []

    async def get_importance_trends(
        self,
        feature_name: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get importance trends for a specific feature.

        Args:
            feature_name: Feature to track
            limit: Maximum number of data points

        Returns:
            List of importance values over time
        """
        # Note: This requires loading artifacts which is expensive
        # In production, consider storing this in a time-series database
        connector = self._get_connector()
        if connector is None:
            return []

        try:
            experiment_name = f"{self.project_name}_shap_analysis"
            runs = await connector.search_runs(
                experiment_names=[experiment_name],
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )

            # For now, return run metadata - full implementation would
            # download and parse the feature_importance.json artifacts
            return [
                {
                    "run_id": run.info.run_id,
                    "timestamp": run.info.start_time,
                    "top_feature_importance": run.data.metrics.get("top_feature_importance"),
                }
                for run in runs
            ]

        except Exception as e:
            logger.error(f"Failed to get importance trends: {e}")
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
    project_name: str = "feature_analyzer",
    tracking_uri: Optional[str] = None,
) -> FeatureAnalyzerMLflowTracker:
    """Factory function to create a FeatureAnalyzerMLflowTracker.

    Args:
        project_name: MLflow experiment name prefix
        tracking_uri: Optional MLflow tracking URI override

    Returns:
        Configured FeatureAnalyzerMLflowTracker instance
    """
    return FeatureAnalyzerMLflowTracker(
        project_name=project_name,
        tracking_uri=tracking_uri,
    )
