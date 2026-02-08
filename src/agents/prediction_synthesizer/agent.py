"""
E2I Prediction Synthesizer Agent - Main Agent Class
Version: 4.3
Purpose: Multi-model prediction aggregation and ensemble synthesis
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .memory_hooks import PredictionSynthesizerMemoryHooks
    from .opik_tracer import PredictionSynthesizerOpikTracer

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
        enable_memory: bool = True,
        enable_dspy: bool = True,
        enable_opik: bool = True,
    ):
        """
        Initialize Prediction Synthesizer agent.

        Args:
            model_registry: Registry of available models
            model_clients: Dict mapping model_id to prediction client
            context_store: Store for historical context
            feature_store: Store for feature metadata
            enable_memory: Whether to enable memory integration (default: True)
            enable_dspy: Whether to enable DSPy signal emission (default: True)
            enable_opik: Whether to enable Opik distributed tracing (default: True)
        """
        self.model_registry = model_registry
        self.model_clients = model_clients or {}
        self.context_store = context_store
        self.feature_store = feature_store
        self.enable_memory = enable_memory
        self.enable_dspy = enable_dspy
        self.enable_opik = enable_opik

        self._full_graph = None
        self._simple_graph = None
        self._memory_hooks: Optional["PredictionSynthesizerMemoryHooks"] = None
        self._tracer: Optional["PredictionSynthesizerOpikTracer"] = None

    @property
    def memory_hooks(self) -> Optional["PredictionSynthesizerMemoryHooks"]:
        """Lazy-load memory hooks."""
        if self._memory_hooks is None and self.enable_memory:
            try:
                from .memory_hooks import get_prediction_synthesizer_memory_hooks

                self._memory_hooks = get_prediction_synthesizer_memory_hooks()
            except ImportError:
                logger.warning("Memory hooks not available")
                return None
        return self._memory_hooks

    @property
    def tracer(self) -> Optional["PredictionSynthesizerOpikTracer"]:
        """Lazy-load Opik tracer."""
        if self._tracer is None and self.enable_opik:
            try:
                from .opik_tracer import get_prediction_synthesizer_tracer

                self._tracer = get_prediction_synthesizer_tracer()
            except ImportError:
                logger.warning("Opik tracer not available")
                return None
        return self._tracer

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
        ensemble_method: Literal["average", "weighted", "stacking", "voting"] = "weighted",
        include_context: bool = True,
        query: str = "",
        session_id: Optional[str] = None,
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
            session_id: Optional session identifier for memory context

        Returns:
            PredictionSynthesizerOutput with ensemble prediction
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Retrieve memory context if enabled
        memory_context = None
        if self.enable_memory and self.memory_hooks:
            try:
                memory_context = await self.memory_hooks.get_context(
                    session_id=session_id,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    prediction_target=prediction_target,
                    time_horizon=time_horizon,
                )
                logger.debug(
                    f"Retrieved memory context: "
                    f"cached={len(memory_context.cached_predictions)}, "
                    f"episodic={len(memory_context.episodic_context)}, "
                    f"models={len(memory_context.model_performance)}"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve memory context: {e}")

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

        # Execute with optional Opik tracing
        result = None
        output = None

        if self.enable_opik and self.tracer:
            async with self.tracer.trace_synthesis(
                entity_type=entity_type,
                prediction_target=prediction_target,
                ensemble_method=ensemble_method,
                synthesis_id=session_id,
                query=query,
            ) as trace_ctx:
                # Log synthesis started
                models_requested = len(models_to_use) if models_to_use else 3
                trace_ctx.log_synthesis_started(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    prediction_target=prediction_target,
                    time_horizon=time_horizon,
                    models_requested=models_requested,
                    ensemble_method=ensemble_method,
                    include_context=include_context,
                )

                # Execute graph
                result = await graph.ainvoke(initial_state)

                # Log model orchestration
                trace_ctx.log_model_orchestration(
                    models_requested=models_requested,
                    models_succeeded=result.get("models_succeeded", 0),
                    models_failed=result.get("models_failed", 0),
                    orchestration_latency_ms=result.get("orchestration_latency_ms", 0),
                )

                # Log ensemble combination
                ensemble = result.get("ensemble_prediction") or {}
                trace_ctx.log_ensemble_combination(
                    ensemble_method=ensemble.get("ensemble_method", ensemble_method),
                    point_estimate=ensemble.get("point_estimate", 0.0),
                    prediction_interval_lower=ensemble.get("prediction_interval_lower", 0.0),
                    prediction_interval_upper=ensemble.get("prediction_interval_upper", 0.0),
                    confidence=ensemble.get("confidence", 0.0),
                    model_agreement=ensemble.get("model_agreement", 0.0),
                    ensemble_latency_ms=result.get("ensemble_latency_ms", 0),
                )

                # Log context enrichment if applicable
                context = result.get("prediction_context") or {}
                if include_context and context:
                    enrichment_latency = (
                        result.get("total_latency_ms", 0)
                        - result.get("orchestration_latency_ms", 0)
                        - result.get("ensemble_latency_ms", 0)
                    )
                    trace_ctx.log_context_enrichment(
                        similar_cases_found=len(context.get("similar_cases", [])),
                        feature_importance_calculated=bool(context.get("feature_importance")),
                        historical_accuracy=context.get("historical_accuracy", 0.0),
                        trend_direction=context.get("trend_direction", ""),
                        enrichment_latency_ms=max(0, enrichment_latency),
                    )

                # Build output
                output = PredictionSynthesizerOutput(
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

                # Log synthesis complete
                trace_ctx.log_synthesis_complete(
                    status=output.status,
                    success=output.status != "failed",
                    total_duration_ms=output.total_latency_ms,
                    point_estimate=ensemble.get("point_estimate"),
                    confidence=ensemble.get("confidence"),
                    model_agreement=ensemble.get("model_agreement"),
                    models_succeeded=output.models_succeeded,
                    models_failed=output.models_failed,
                    prediction_summary=output.prediction_summary,
                    errors=output.errors,
                    warnings=output.warnings,
                )
        else:
            # Execute without tracing
            result = await graph.ainvoke(initial_state)

            output = PredictionSynthesizerOutput(
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

        # Contribute to memory after successful prediction
        if self.enable_memory and self.memory_hooks and output.status != "failed":
            try:
                from .memory_hooks import contribute_to_memory

                await contribute_to_memory(
                    result=output.model_dump(),
                    state=result,
                    memory_hooks=self.memory_hooks,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to contribute to memory: {e}")

        # Emit DSPy training signal after successful prediction
        if self.enable_dspy and output.status != "failed":
            try:
                from .dspy_integration import collect_and_emit_signal

                await collect_and_emit_signal(
                    session_id=session_id,
                    state=result,
                    output=output.model_dump(),
                    min_reward_threshold=0.5,
                )
            except Exception as e:
                logger.warning(f"Failed to emit DSPy training signal: {e}")

        return output

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
        ensemble: Dict[str, Any] = dict(output.ensemble_prediction) if output.ensemble_prediction else {}

        recommendations: List[str] = []
        if output.status == "failed":
            recommendations.append("Review model availability and health")
            recommendations.append("Check input features for completeness")
        elif ensemble.get("model_agreement", 0) < 0.5:
            recommendations.append(
                "Low model agreement - consider investigating model discrepancies"
            )
        elif ensemble.get("confidence", 0) < 0.5:
            recommendations.append("Low confidence - may need more data or model retraining")

        context: Dict[str, Any] = dict(output.prediction_context) if output.prediction_context else {}

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
    ensemble_method: Literal["average", "weighted", "stacking", "voting"] = "weighted",
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
