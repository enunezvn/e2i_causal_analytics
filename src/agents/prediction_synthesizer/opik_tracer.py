"""
E2I Prediction Synthesizer Agent - Opik Distributed Tracing
Version: 4.3
Purpose: Distributed tracing for Prediction Synthesizer agent operations

This module provides Opik integration for the Prediction Synthesizer agent,
enabling distributed tracing across prediction synthesis operations.

Architecture:
    - PredictionSynthesizerOpikTracer: Main tracer class with singleton pattern
    - SynthesisTraceContext: Async context manager for traces
    - NodeSpanContext: Context for individual node spans

Pipeline Nodes Traced:
    - orchestrate: Model orchestration (parallel model predictions)
    - combine: Ensemble combination (weighted/average/voting/stacking)
    - enrich: Context enrichment (similar cases, trends, accuracy)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from opik import Opik, Trace
    from opik.api_objects.span import Span

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

ENSEMBLE_METHODS = ["average", "weighted", "stacking", "voting"]
ENTITY_TYPES = ["hcp", "territory", "patient"]
PIPELINE_NODES = ["orchestrate", "combine", "enrich"]


# ============================================================================
# SPAN CONTEXT
# ============================================================================


@dataclass
class NodeSpanContext:
    """Context for individual prediction pipeline node spans."""

    span: Optional["Span"]
    node_name: str
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the span."""
        self.metadata[key] = value

    def end(self, status: str = "completed") -> None:
        """End the span with final metadata."""
        if self.span:
            try:
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                self.span.end(
                    metadata={
                        **self.metadata,
                        "status": status,
                        "duration_ms": elapsed_ms,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to end node span: {e}")


# ============================================================================
# TRACE CONTEXT
# ============================================================================


@dataclass
class SynthesisTraceContext:
    """Async context manager for prediction synthesis traces."""

    trace: Optional["Trace"]
    tracer: "PredictionSynthesizerOpikTracer"
    entity_type: str
    prediction_target: str
    start_time: float = field(default_factory=time.time)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    active_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)

    def log_synthesis_started(
        self,
        entity_id: str,
        entity_type: str,
        prediction_target: str,
        time_horizon: str,
        models_requested: int,
        ensemble_method: str,
        include_context: bool,
    ) -> None:
        """Log that a synthesis has started."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                        "prediction_target": prediction_target,
                        "time_horizon": time_horizon,
                        "models_requested": models_requested,
                        "ensemble_method": ensemble_method,
                        "include_context": include_context,
                        "synthesis_started_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log synthesis started: {e}")

    def start_node_span(
        self,
        node_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> NodeSpanContext:
        """Start a span for a prediction pipeline node."""
        span = None
        if self.trace and self.tracer._client:
            try:
                span = self.trace.span(
                    name=f"prediction_synthesizer.{node_name}",
                    input=input_data or {},
                    metadata={
                        "node_name": node_name,
                        "entity_type": self.entity_type,
                        "prediction_target": self.prediction_target,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to create node span: {e}")

        ctx = NodeSpanContext(span=span, node_name=node_name)
        self.active_spans[node_name] = ctx
        return ctx

    def end_node_span(
        self,
        node_name: str,
        output_data: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ) -> None:
        """End a node span."""
        if node_name in self.active_spans:
            ctx = self.active_spans[node_name]
            if ctx.span:
                try:
                    elapsed_ms = int((time.time() - ctx.start_time) * 1000)
                    ctx.span.end(
                        output=output_data or {},
                        metadata={
                            **ctx.metadata,
                            "status": status,
                            "duration_ms": elapsed_ms,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to end node span {node_name}: {e}")
            del self.active_spans[node_name]

    def log_model_orchestration(
        self,
        models_requested: int,
        models_succeeded: int,
        models_failed: int,
        orchestration_latency_ms: int,
    ) -> None:
        """Log model orchestration results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "models_requested": models_requested,
                        "models_succeeded": models_succeeded,
                        "models_failed": models_failed,
                        "orchestration_latency_ms": orchestration_latency_ms,
                        "model_success_rate": (
                            models_succeeded / models_requested if models_requested > 0 else 0.0
                        ),
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log model orchestration: {e}")

    def log_ensemble_combination(
        self,
        ensemble_method: str,
        point_estimate: float,
        prediction_interval_lower: float,
        prediction_interval_upper: float,
        confidence: float,
        model_agreement: float,
        ensemble_latency_ms: int,
    ) -> None:
        """Log ensemble combination results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "ensemble_method": ensemble_method,
                        "point_estimate": point_estimate,
                        "prediction_interval_lower": prediction_interval_lower,
                        "prediction_interval_upper": prediction_interval_upper,
                        "prediction_interval_width": (
                            prediction_interval_upper - prediction_interval_lower
                        ),
                        "confidence": confidence,
                        "model_agreement": model_agreement,
                        "ensemble_latency_ms": ensemble_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log ensemble combination: {e}")

    def log_context_enrichment(
        self,
        similar_cases_found: int,
        feature_importance_calculated: bool,
        historical_accuracy: float,
        trend_direction: str,
        enrichment_latency_ms: int,
    ) -> None:
        """Log context enrichment results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "similar_cases_found": similar_cases_found,
                        "feature_importance_calculated": feature_importance_calculated,
                        "historical_accuracy": historical_accuracy,
                        "trend_direction": trend_direction,
                        "enrichment_latency_ms": enrichment_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log context enrichment: {e}")

    def log_synthesis_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        point_estimate: Optional[float],
        confidence: Optional[float],
        model_agreement: Optional[float],
        models_succeeded: int,
        models_failed: int,
        prediction_summary: str,
        errors: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Log synthesis completion."""
        if self.trace:
            try:
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                self.trace_metadata.update(
                    {
                        "status": status,
                        "success": success,
                        "total_duration_ms": total_duration_ms,
                        "trace_duration_ms": elapsed_ms,
                        "point_estimate": point_estimate,
                        "confidence": confidence,
                        "model_agreement": model_agreement,
                        "models_succeeded": models_succeeded,
                        "models_failed": models_failed,
                        "prediction_summary_length": len(prediction_summary),
                        "errors_count": len(errors),
                        "warnings_count": len(warnings),
                        "warnings": warnings[:5],  # Limit for trace size
                        "synthesis_completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.trace.update(
                    metadata=self.trace_metadata,
                    output={
                        "point_estimate": point_estimate,
                        "confidence": confidence,
                        "status": status,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to log synthesis complete: {e}")


# ============================================================================
# MAIN TRACER CLASS
# ============================================================================


class PredictionSynthesizerOpikTracer:
    """
    Opik distributed tracer for Prediction Synthesizer agent.

    Provides tracing for:
    - Prediction synthesis operations
    - Model orchestration (parallel predictions)
    - Ensemble combination (weighted, average, voting, stacking)
    - Context enrichment (similar cases, trends, accuracy)

    Usage:
        tracer = PredictionSynthesizerOpikTracer()
        async with tracer.trace_synthesis(
            entity_type="hcp",
            prediction_target="churn"
        ) as ctx:
            ctx.log_synthesis_started(...)
            # ... perform synthesis ...
            ctx.log_synthesis_complete(...)

    Thread Safety:
        Uses singleton pattern - safe for concurrent access.
    """

    _instance: Optional["PredictionSynthesizerOpikTracer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "PredictionSynthesizerOpikTracer":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_name: str = "e2i-prediction-synthesizer",
        sampling_rate: float = 1.0,
        enabled: bool = True,
    ):
        """
        Initialize the Opik tracer.

        Args:
            project_name: Opik project name for traces
            sampling_rate: Fraction of traces to capture (0.0-1.0)
            enabled: Whether tracing is enabled
        """
        if self._initialized:
            return

        self.project_name = project_name
        self.sampling_rate = sampling_rate
        self.enabled = enabled
        self._client: Optional["Opik"] = None
        self._initialized = True

        logger.info(
            f"PredictionSynthesizerOpikTracer initialized: project={project_name}, "
            f"sampling_rate={sampling_rate}, enabled={enabled}"
        )

    def _get_client(self) -> Optional["Opik"]:
        """Get or create Opik client (lazy initialization)."""
        if not self.enabled:
            return None

        if self._client is None:
            try:
                from opik import Opik

                self._client = Opik(project_name=self.project_name)
                logger.debug("Opik client initialized for Prediction Synthesizer")
            except ImportError:
                logger.warning("Opik not available - tracing disabled")
                self.enabled = False
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Opik client: {e}")
                self.enabled = False
                return None

        return self._client

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random

        return random.random() < self.sampling_rate

    def _generate_trace_id(self) -> str:
        """Generate a UUID v7 compatible trace ID for Opik."""
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        random_uuid = uuid.uuid4()
        uuid_bytes = random_uuid.bytes
        timestamp_bytes = timestamp_ms.to_bytes(6, byteorder="big")
        new_bytes = (
            timestamp_bytes
            + bytes([0x70 | (uuid_bytes[6] & 0x0F)])  # Version 7
            + uuid_bytes[7:8]
            + bytes([0x80 | (uuid_bytes[8] & 0x3F)])  # Variant 10
            + uuid_bytes[9:16]
        )
        return str(uuid.UUID(bytes=new_bytes))

    @asynccontextmanager
    async def trace_synthesis(
        self,
        entity_type: str = "hcp",
        prediction_target: str = "churn",
        ensemble_method: str = "weighted",
        synthesis_id: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """
        Async context manager for tracing a prediction synthesis operation.

        Args:
            entity_type: Type of entity being predicted for
            prediction_target: What is being predicted
            ensemble_method: Method used for ensemble combination
            synthesis_id: Optional unique identifier for this synthesis
            query: Optional original query text

        Yields:
            SynthesisTraceContext for logging trace data
        """
        if not self.enabled or not self._should_sample():
            yield SynthesisTraceContext(
                trace=None,
                tracer=self,
                entity_type=entity_type,
                prediction_target=prediction_target,
            )
            return

        client = self._get_client()
        if client is None:
            yield SynthesisTraceContext(
                trace=None,
                tracer=self,
                entity_type=entity_type,
                prediction_target=prediction_target,
            )
            return

        trace = None
        try:
            trace_id = self._generate_trace_id()
            trace = client.trace(
                name=f"prediction_synthesizer.{entity_type}.{prediction_target}",
                id=trace_id,
                input={
                    "entity_type": entity_type,
                    "prediction_target": prediction_target,
                    "ensemble_method": ensemble_method,
                    "query": query or "",
                },
                metadata={
                    "agent": "prediction_synthesizer",
                    "tier": 4,
                    "agent_type": "ml_predictions",
                    "entity_type": entity_type,
                    "prediction_target": prediction_target,
                    "ensemble_method": ensemble_method,
                    "synthesis_id": synthesis_id or trace_id,
                    "trace_started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug(f"Started synthesis trace: {trace_id}")

            ctx = SynthesisTraceContext(
                trace=trace,
                tracer=self,
                entity_type=entity_type,
                prediction_target=prediction_target,
                trace_metadata={
                    "entity_type": entity_type,
                    "prediction_target": prediction_target,
                    "ensemble_method": ensemble_method,
                },
            )
            yield ctx

        except Exception as e:
            logger.warning(f"Failed to create synthesis trace: {e}")
            yield SynthesisTraceContext(
                trace=None,
                tracer=self,
                entity_type=entity_type,
                prediction_target=prediction_target,
            )
        finally:
            if trace:
                try:
                    trace.end()
                except Exception as e:
                    logger.debug(f"Failed to end trace: {e}")

    def flush(self) -> None:
        """Flush any pending traces to Opik backend."""
        if self._client:
            try:
                self._client.flush()
                logger.debug("Flushed Opik traces for Prediction Synthesizer")
            except Exception as e:
                logger.debug(f"Failed to flush Opik traces: {e}")


# ============================================================================
# SINGLETON FACTORY
# ============================================================================

_tracer_instance: Optional[PredictionSynthesizerOpikTracer] = None


def get_prediction_synthesizer_tracer(
    project_name: str = "e2i-prediction-synthesizer",
    sampling_rate: float = 1.0,
    enabled: bool = True,
) -> PredictionSynthesizerOpikTracer:
    """
    Get or create the singleton Prediction Synthesizer Opik tracer.

    Args:
        project_name: Opik project name
        sampling_rate: Fraction of traces to capture
        enabled: Whether tracing is enabled

    Returns:
        PredictionSynthesizerOpikTracer instance
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = PredictionSynthesizerOpikTracer(
            project_name=project_name,
            sampling_rate=sampling_rate,
            enabled=enabled,
        )
    return _tracer_instance


def reset_tracer() -> None:
    """Reset the tracer singleton (for testing)."""
    global _tracer_instance
    _tracer_instance = None
    PredictionSynthesizerOpikTracer._instance = None
    PredictionSynthesizerOpikTracer._initialized = False
