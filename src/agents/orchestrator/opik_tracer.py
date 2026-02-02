"""
E2I Orchestrator Agent - Opik Distributed Tracing
Version: 4.2
Purpose: Distributed tracing for Orchestrator agent operations

This module provides Opik integration for the Orchestrator agent,
enabling distributed tracing across query orchestration operations.

Architecture:
    - OrchestratorOpikTracer: Main tracer class with singleton pattern
    - OrchestrationTraceContext: Async context manager for traces
    - NodeSpanContext: Context for individual node spans

Pipeline Nodes Traced:
    - classify: Intent classification (<500ms target)
    - rag_context: RAG context retrieval
    - route: Agent routing (<50ms target)
    - dispatch: Parallel agent dispatch
    - synthesize: Response synthesis
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

ORCHESTRATION_PHASES = [
    "classifying",
    "retrieving_context",
    "routing",
    "dispatching",
    "synthesizing",
]
PIPELINE_NODES = ["classify", "rag_context", "route", "dispatch", "synthesize"]


# ============================================================================
# SPAN CONTEXT
# ============================================================================


@dataclass
class NodeSpanContext:
    """Context for individual orchestration pipeline node spans."""

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
class OrchestrationTraceContext:
    """Async context manager for orchestration traces."""

    trace: Optional["Trace"]
    tracer: "OrchestratorOpikTracer"
    query_id: str
    start_time: float = field(default_factory=time.time)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    active_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)

    def log_orchestration_started(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log that an orchestration has started."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "query": query[:500],  # Truncate for trace size
                        "user_id": user_id,
                        "session_id": session_id,
                        "orchestration_started_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log orchestration started: {e}")

    def start_node_span(
        self,
        node_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> NodeSpanContext:
        """Start a span for an orchestration pipeline node."""
        span = None
        if self.trace and self.tracer._client:
            try:
                span = self.trace.span(
                    name=f"orchestrator.{node_name}",
                    input=input_data or {},
                    metadata={
                        "node_name": node_name,
                        "query_id": self.query_id,
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

    def log_intent_classification(
        self,
        primary_intent: str,
        confidence: float,
        secondary_intents: List[str],
        classification_latency_ms: int,
    ) -> None:
        """Log intent classification results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "primary_intent": primary_intent,
                        "intent_confidence": confidence,
                        "secondary_intents": secondary_intents[:3],
                        "classification_latency_ms": classification_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log intent classification: {e}")

    def log_rag_retrieval(
        self,
        context_retrieved: bool,
        chunks_count: int,
        rag_latency_ms: int,
    ) -> None:
        """Log RAG context retrieval results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "context_retrieved": context_retrieved,
                        "rag_chunks_count": chunks_count,
                        "rag_latency_ms": rag_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log RAG retrieval: {e}")

    def log_agent_routing(
        self,
        agents_selected: List[str],
        routing_rationale: Optional[str],
        routing_latency_ms: int,
    ) -> None:
        """Log agent routing results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "agents_selected": agents_selected,
                        "agents_count": len(agents_selected),
                        "routing_rationale": routing_rationale[:200] if routing_rationale else None,
                        "routing_latency_ms": routing_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log agent routing: {e}")

    def log_agent_dispatch(
        self,
        agents_dispatched: List[str],
        successful_agents: List[str],
        failed_agents: List[str],
        dispatch_latency_ms: int,
    ) -> None:
        """Log agent dispatch results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "agents_dispatched": agents_dispatched,
                        "successful_agents": successful_agents,
                        "failed_agents": failed_agents,
                        "dispatch_success_rate": len(successful_agents)
                        / max(len(agents_dispatched), 1),
                        "dispatch_latency_ms": dispatch_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log agent dispatch: {e}")

    def log_response_synthesis(
        self,
        response_length: int,
        citations_count: int,
        synthesis_latency_ms: int,
    ) -> None:
        """Log response synthesis results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "response_length": response_length,
                        "citations_count": citations_count,
                        "synthesis_latency_ms": synthesis_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log response synthesis: {e}")

    def log_orchestration_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        response_confidence: float,
        agents_dispatched: List[str],
        successful_agents: List[str],
        failed_agents: List[str],
        has_partial_failure: bool,
        primary_intent: Optional[str],
        classification_latency_ms: int,
        rag_latency_ms: int,
        routing_latency_ms: int,
        dispatch_latency_ms: int,
        synthesis_latency_ms: int,
        errors: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Log orchestration completion."""
        if self.trace:
            try:
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                self.trace_metadata.update(
                    {
                        "status": status,
                        "success": success,
                        "total_duration_ms": total_duration_ms,
                        "trace_duration_ms": elapsed_ms,
                        "response_confidence": response_confidence,
                        "agents_dispatched": agents_dispatched,
                        "successful_agents": successful_agents,
                        "failed_agents": failed_agents,
                        "has_partial_failure": has_partial_failure,
                        "primary_intent": primary_intent,
                        "classification_latency_ms": classification_latency_ms,
                        "rag_latency_ms": rag_latency_ms,
                        "routing_latency_ms": routing_latency_ms,
                        "dispatch_latency_ms": dispatch_latency_ms,
                        "synthesis_latency_ms": synthesis_latency_ms,
                        "errors_count": len(errors),
                        "warnings_count": len(warnings),
                        "warnings": warnings[:5],  # Limit for trace size
                        "orchestration_completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.trace.update(
                    metadata=self.trace_metadata,
                    output={
                        "status": status,
                        "response_confidence": response_confidence,
                        "agents_count": len(agents_dispatched),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to log orchestration complete: {e}")


# ============================================================================
# MAIN TRACER CLASS
# ============================================================================


class OrchestratorOpikTracer:
    """
    Opik distributed tracer for Orchestrator agent.

    Provides tracing for:
    - Query orchestration operations
    - Intent classification
    - RAG context retrieval
    - Agent routing
    - Parallel agent dispatch
    - Response synthesis

    Usage:
        tracer = OrchestratorOpikTracer()
        async with tracer.trace_orchestration(query_id="q-123") as ctx:
            ctx.log_orchestration_started(query, user_id)
            # ... perform orchestration ...
            ctx.log_orchestration_complete(...)

    Thread Safety:
        Uses singleton pattern - safe for concurrent access.
    """

    _instance: Optional["OrchestratorOpikTracer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "OrchestratorOpikTracer":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_name: str = "e2i-orchestrator",
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
            f"OrchestratorOpikTracer initialized: project={project_name}, "
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
                logger.debug("Opik client initialized for Orchestrator")
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
    async def trace_orchestration(
        self,
        query_id: str,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Async context manager for tracing an orchestration operation.

        Args:
            query_id: Unique identifier for this query
            query: Optional query text
            user_id: Optional user identifier
            session_id: Optional session identifier

        Yields:
            OrchestrationTraceContext for logging trace data
        """
        if not self.enabled or not self._should_sample():
            yield OrchestrationTraceContext(
                trace=None,
                tracer=self,
                query_id=query_id,
            )
            return

        client = self._get_client()
        if client is None:
            yield OrchestrationTraceContext(
                trace=None,
                tracer=self,
                query_id=query_id,
            )
            return

        trace = None
        try:
            trace_id = self._generate_trace_id()
            trace = client.trace(
                name="orchestrator.query",
                id=trace_id,
                input={
                    "query_id": query_id,
                    "query": (query[:500] if query else ""),
                    "user_id": user_id,
                    "session_id": session_id,
                },
                metadata={
                    "agent": "orchestrator",
                    "tier": 1,
                    "agent_type": "coordination",
                    "query_id": query_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "trace_started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug(f"Started orchestration trace: {trace_id}")

            ctx = OrchestrationTraceContext(
                trace=trace,
                tracer=self,
                query_id=query_id,
                trace_metadata={
                    "query_id": query_id,
                },
            )
            yield ctx

        except Exception as e:
            logger.warning(f"Failed to create orchestration trace: {e}")
            yield OrchestrationTraceContext(
                trace=None,
                tracer=self,
                query_id=query_id,
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
                logger.debug("Flushed Opik traces for Orchestrator")
            except Exception as e:
                logger.debug(f"Failed to flush Opik traces: {e}")


# ============================================================================
# SINGLETON FACTORY
# ============================================================================

_tracer_instance: Optional[OrchestratorOpikTracer] = None


def get_orchestrator_tracer(
    project_name: str = "e2i-orchestrator",
    sampling_rate: float = 1.0,
    enabled: bool = True,
) -> OrchestratorOpikTracer:
    """
    Get or create the singleton Orchestrator Opik tracer.

    Args:
        project_name: Opik project name
        sampling_rate: Fraction of traces to capture
        enabled: Whether tracing is enabled

    Returns:
        OrchestratorOpikTracer instance
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = OrchestratorOpikTracer(
            project_name=project_name,
            sampling_rate=sampling_rate,
            enabled=enabled,
        )
    return _tracer_instance
