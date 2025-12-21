"""
E2I Prediction Synthesizer Agent - Main Agent Class
Version: 4.2
Purpose: Multi-model prediction aggregation and ensemble synthesis
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .graph import build_prediction_synthesizer_graph, build_simple_prediction_graph
from .state import (
    EnsemblePrediction,
    ModelPrediction,
    PredictionContext,
    PredictionSynthesizerState,
)

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONTRACTS
# ============================================================================


class PredictionSynthesizerInput(BaseModel):
    """Input contract for Prediction Synthesizer agent"""

    query: str = ""
    entity_id: str
    entity_type: str = "hcp"
    prediction_target: str
    features: Dict[str, Any] = Field(default_factory=dict)
    time_horizon: str = "30d"
    models_to_use: Optional[List[str]] = None
    ensemble_method: Literal["average", "weighted", "stacking", "voting"] = "weighted"
    confidence_level: float = 0.95
    include_context: bool = True


class PredictionSynthesizerOutput(BaseModel):
    """Output contract for Prediction Synthesizer agent"""

    ensemble_prediction: Optional[EnsemblePrediction] = None
    individual_predictions: List[ModelPrediction] = Field(default_factory=list)
    prediction_context: Optional[PredictionContext] = None
    prediction_summary: str = ""
    models_succeeded: int = 0
    models_failed: int = 0
    total_latency_ms: int = 0
    timestamp: str = ""
    status: str = "pending"
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# AGENT CLASS
# ============================================================================


class PredictionSynthesizerAgent:
    """
    Tier 4 Prediction Synthesizer Agent.

    Responsibilities:
    - Orchestrate predictions from multiple models
    - Combine predictions using ensemble methods
    - Quantify prediction uncertainty
    - Enrich predictions with context
    """

    def __init__(
        self,
        model_registry: Optional[Any] = None,
        model_clients: Optional[Dict[str, Any]] = None,
        context_store: Optional[Any] = None,
        feature_store: Optional[Any] = None,
    ):
        """
        Initialize Prediction Synthesizer agent.

        Args:
            model_registry: Registry of available models
            model_clients: Dict mapping model_id to prediction client
            context_store: Store for historical context
            feature_store: Store for feature metadata
        """
        self.model_registry = model_registry
        self.model_clients = model_clients or {}
        self.context_store = context_store
        self.feature_store = feature_store

        self._full_graph = None
        self._simple_graph = None

    @property
    def full_graph(self):
        """Lazy-load full prediction graph."""
        if self._full_graph is None:
            self._full_graph = build_prediction_synthesizer_graph(
                model_registry=self.model_registry,
                model_clients=self.model_clients,
                context_store=self.context_store,
                feature_store=self.feature_store,
            )
        return self._full_graph

    @property
    def simple_graph(self):
        """Lazy-load simple prediction graph."""
        if self._simple_graph is None:
            self._simple_graph = build_simple_prediction_graph(
                model_clients=self.model_clients,
            )
        return self._simple_graph

    async def synthesize(
        self,
        entity_id: str,
        prediction_target: str,
        features: Optional[Dict[str, Any]] = None,
        entity_type: str = "hcp",
        time_horizon: str = "30d",
        models_to_use: Optional[List[str]] = None,
        ensemble_method: str = "weighted",
        include_context: bool = True,
        query: str = "",
    ) -> PredictionSynthesizerOutput:
        """
        Synthesize predictions from multiple models.

        Args:
            entity_id: ID of entity to predict for
            prediction_target: What to predict (e.g., "churn", "conversion")
            features: Feature values for prediction
            entity_type: Type of entity (hcp, territory, patient)
            time_horizon: Prediction time horizon
            models_to_use: Specific models to use (None for all)
            ensemble_method: How to combine predictions
            include_context: Whether to add context enrichment
            query: Original query text

        Returns:
            PredictionSynthesizerOutput with ensemble prediction
        """
        initial_state: PredictionSynthesizerState = {
            "query": query,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "prediction_target": prediction_target,
            "features": features or {},
            "time_horizon": time_horizon,
            "models_to_use": models_to_use,
            "ensemble_method": ensemble_method,
            "confidence_level": 0.95,
            "include_context": include_context,
            "individual_predictions": None,
            "models_succeeded": 0,
            "models_failed": 0,
            "ensemble_prediction": None,
            "prediction_summary": None,
            "prediction_context": None,
            "orchestration_latency_ms": 0,
            "ensemble_latency_ms": 0,
            "total_latency_ms": 0,
            "timestamp": "",
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        # Choose graph based on context requirement
        graph = self.full_graph if include_context else self.simple_graph

        logger.info(
            f"Starting prediction synthesis: entity={entity_id}, "
            f"target={prediction_target}, method={ensemble_method}"
        )

        result = await graph.ainvoke(initial_state)

        return PredictionSynthesizerOutput(
            ensemble_prediction=result.get("ensemble_prediction"),
            individual_predictions=result.get("individual_predictions") or [],
            prediction_context=result.get("prediction_context"),
            prediction_summary=result.get("prediction_summary", ""),
            models_succeeded=result.get("models_succeeded", 0),
            models_failed=result.get("models_failed", 0),
            total_latency_ms=result.get("total_latency_ms", 0),
            timestamp=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            status=result.get("status", "failed"),
            errors=result.get("errors") or [],
            warnings=result.get("warnings") or [],
        )

    async def quick_predict(
        self,
        entity_id: str,
        prediction_target: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> PredictionSynthesizerOutput:
        """
        Quick prediction without context enrichment.

        Args:
            entity_id: ID of entity to predict for
            prediction_target: What to predict
            features: Feature values for prediction

        Returns:
            PredictionSynthesizerOutput with ensemble prediction
        """
        return await self.synthesize(
            entity_id=entity_id,
            prediction_target=prediction_target,
            features=features,
            include_context=False,
        )

    def get_handoff(self, output: PredictionSynthesizerOutput) -> Dict[str, Any]:
        """
        Generate handoff for orchestrator.

        Args:
            output: Prediction synthesis output

        Returns:
            Handoff dictionary for other agents
        """
        ensemble = output.ensemble_prediction or {}

        recommendations = []
        if output.status == "failed":
            recommendations.append("Review model availability and health")
            recommendations.append("Check input features for completeness")
        elif ensemble.get("model_agreement", 0) < 0.5:
            recommendations.append(
                "Low model agreement - consider investigating model discrepancies"
            )
        elif ensemble.get("confidence", 0) < 0.5:
            recommendations.append("Low confidence - may need more data or model retraining")

        context = output.prediction_context or {}

        return {
            "agent": "prediction_synthesizer",
            "analysis_type": "prediction",
            "key_findings": {
                "prediction": ensemble.get("point_estimate"),
                "confidence_interval": [
                    ensemble.get("prediction_interval_lower"),
                    ensemble.get("prediction_interval_upper"),
                ],
                "confidence": ensemble.get("confidence"),
                "model_agreement": ensemble.get("model_agreement"),
            },
            "models": {
                "succeeded": output.models_succeeded,
                "failed": output.models_failed,
            },
            "context": {
                "trend": context.get("trend_direction", "stable"),
                "historical_accuracy": context.get("historical_accuracy", 0.0),
            },
            "recommendations": recommendations,
            "requires_further_analysis": ensemble.get("confidence", 0) < 0.5,
            "suggested_next_agent": "explainer" if output.status == "completed" else None,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def synthesize_predictions(
    entity_id: str,
    prediction_target: str,
    features: Optional[Dict[str, Any]] = None,
    model_clients: Optional[Dict[str, Any]] = None,
    ensemble_method: str = "weighted",
    include_context: bool = False,
) -> PredictionSynthesizerOutput:
    """
    Convenience function for prediction synthesis.

    Args:
        entity_id: ID of entity to predict for
        prediction_target: What to predict
        features: Feature values
        model_clients: Model prediction clients
        ensemble_method: How to combine predictions
        include_context: Whether to add context

    Returns:
        PredictionSynthesizerOutput
    """
    agent = PredictionSynthesizerAgent(model_clients=model_clients)
    return await agent.synthesize(
        entity_id=entity_id,
        prediction_target=prediction_target,
        features=features,
        ensemble_method=ensemble_method,
        include_context=include_context,
    )


def synthesize_predictions_sync(
    entity_id: str,
    prediction_target: str,
    features: Optional[Dict[str, Any]] = None,
    model_clients: Optional[Dict[str, Any]] = None,
) -> PredictionSynthesizerOutput:
    """
    Synchronous wrapper for prediction synthesis.
    """
    return asyncio.run(
        synthesize_predictions(
            entity_id=entity_id,
            prediction_target=prediction_target,
            features=features,
            model_clients=model_clients,
        )
    )
