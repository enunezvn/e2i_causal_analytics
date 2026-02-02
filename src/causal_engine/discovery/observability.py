"""
E2I Causal Analytics - Discovery Observability
===============================================

Opik integration for causal discovery observability.

Provides:
- DiscoveryTracer: Trace discovery runs with algorithm details
- Log gate decisions (ACCEPT/REVIEW/REJECT/AUGMENT)
- Track cache performance (hit rate, latency savings)
- Monitor ensemble agreement across algorithms

Traces:
- discovery_run: Overall discovery operation
  - algorithm_execution: Per-algorithm timing
  - ensemble_voting: Edge voting process
  - gate_evaluation: Decision making
  - cache_lookup: Cache performance

Metrics:
- discovery_latency_ms: Total discovery time
- algorithm_agreement_ratio: Edge consensus
- gate_decision_distribution: ACCEPT/REVIEW/REJECT/AUGMENT counts
- cache_hit_rate: Cache effectiveness
- edges_discovered: Edge count per run

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from uuid_utils import uuid7 as uuid7_func

from .base import (
    AlgorithmResult,
    DiscoveryConfig,
    GateDecision,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoverySpanMetadata:
    """Metadata for a discovery span.

    Attributes:
        session_id: Session ID for tracking
        algorithms: Algorithms being run
        n_variables: Number of variables in data
        n_samples: Number of samples in data
        config: Discovery configuration
    """

    session_id: Optional[UUID] = None
    algorithms: List[str] = field(default_factory=list)
    n_variables: int = 0
    n_samples: int = 0
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Opik metadata."""
        return {
            "session_id": str(self.session_id) if self.session_id else None,
            "algorithms": self.algorithms,
            "n_variables": self.n_variables,
            "n_samples": self.n_samples,
            "config": self.config,
        }


@dataclass
class DiscoverySpan:
    """Context object for a discovery trace span.

    Provides methods for enriching the span with discovery-specific data.
    """

    span_id: str
    trace_id: str
    name: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: DiscoverySpanMetadata = field(default_factory=DiscoverySpanMetadata)
    algorithm_results: List[Dict[str, Any]] = field(default_factory=list)
    gate_decision: Optional[str] = None
    gate_confidence: float = 0.0
    gate_reasons: List[str] = field(default_factory=list)
    cache_hit: Optional[bool] = None
    cache_latency_ms: Optional[float] = None
    n_edges_discovered: int = 0
    algorithm_agreement: float = 0.0
    error: Optional[str] = None
    _opik_span: Any = None
    _tracer: Optional["DiscoveryTracer"] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute on the span."""
        if self._opik_span:
            try:
                current_metadata = getattr(self._opik_span, "_metadata", {}) or {}
                current_metadata[key] = value
                self._opik_span.update(metadata=current_metadata)
            except Exception as e:
                logger.debug(f"Failed to update Opik span attribute: {e}")

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        if self._opik_span:
            try:
                # Events are added to metadata
                current_metadata = getattr(self._opik_span, "_metadata", {}) or {}
                events = current_metadata.get("events", [])
                events.append(
                    {
                        "name": name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "attributes": attributes or {},
                    }
                )
                current_metadata["events"] = events
                self._opik_span.update(metadata=current_metadata)
            except Exception as e:
                logger.debug(f"Failed to add Opik span event: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata.to_dict(),
            "algorithm_results": self.algorithm_results,
            "gate_decision": self.gate_decision,
            "gate_confidence": self.gate_confidence,
            "gate_reasons": self.gate_reasons,
            "cache_hit": self.cache_hit,
            "cache_latency_ms": self.cache_latency_ms,
            "n_edges_discovered": self.n_edges_discovered,
            "algorithm_agreement": self.algorithm_agreement,
            "error": self.error,
        }


class DiscoveryTracer:
    """Traces causal discovery operations to Opik.

    Provides tracing for discovery runs, algorithm executions, gate decisions,
    and cache operations.

    Example:
        tracer = DiscoveryTracer()

        async with tracer.trace_discovery(
            session_id=session_id,
            algorithms=["ges", "pc"]
        ) as span:
            result = await runner.discover_dag(data, config)
            span.n_edges_discovered = result.n_edges
            span.algorithm_agreement = result.algorithm_agreement
    """

    def __init__(
        self,
        opik: Optional[Any] = None,
        enabled: bool = True,
        project_name: str = "e2i-causal-analytics",
    ):
        """Initialize DiscoveryTracer.

        Args:
            opik: OpikConnector instance. If None, attempts to import.
            enabled: Whether tracing is enabled
            project_name: Project name for Opik
        """
        self._opik = opik
        self._enabled = enabled
        self._project_name = project_name
        self._initialized = False

        if enabled and opik is None:
            self._init_opik()

    def _init_opik(self) -> None:
        """Initialize Opik connector."""
        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik = get_opik_connector()
            self._initialized = True
            logger.debug("DiscoveryTracer initialized with OpikConnector")
        except ImportError:
            logger.warning("OpikConnector not available, discovery tracing disabled")
            self._enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpikConnector: {e}")
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        if not self._enabled:
            return False
        if self._opik is None:
            return False
        return getattr(self._opik, "is_enabled", False)

    @asynccontextmanager
    async def trace_discovery(
        self,
        session_id: Optional[UUID] = None,
        algorithms: Optional[List[str]] = None,
        n_variables: int = 0,
        n_samples: int = 0,
        config: Optional[DiscoveryConfig] = None,
        tags: Optional[List[str]] = None,
    ) -> AsyncGenerator[DiscoverySpan, None]:
        """Trace a complete discovery run.

        Args:
            session_id: Session ID for tracking
            algorithms: List of algorithms being run
            n_variables: Number of variables in data
            n_samples: Number of samples in data
            config: Discovery configuration
            tags: Tags for categorizing the trace

        Yields:
            DiscoverySpan for enriching the trace
        """
        span_id = str(uuid7_func())
        trace_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        metadata = DiscoverySpanMetadata(
            session_id=session_id,
            algorithms=algorithms or [],
            n_variables=n_variables,
            n_samples=n_samples,
            config=config.to_dict() if config else None,
        )

        span = DiscoverySpan(
            span_id=span_id,
            trace_id=trace_id,
            name="causal_discovery.discover_dag",
            start_time=start_time,
            metadata=metadata,
            _tracer=self,
        )

        error_occurred = False
        error_message = None

        try:
            # Create Opik trace/span if enabled
            if self.is_enabled:
                try:
                    async with self._opik.trace_agent(
                        agent_name="causal_discovery",
                        operation="discover_dag",
                        metadata={
                            "session_id": str(session_id) if session_id else None,
                            "algorithms": algorithms or [],
                            "n_variables": n_variables,
                            "n_samples": n_samples,
                            "discovery_type": "ensemble",
                        },
                        tags=tags or ["causal_discovery", "ensemble"],
                        input_data={
                            "algorithms": algorithms,
                            "n_variables": n_variables,
                            "n_samples": n_samples,
                            "config": config.to_dict() if config else None,
                        },
                    ) as opik_ctx:
                        span._opik_span = opik_ctx
                        yield span
                except Exception as e:
                    logger.debug(f"Opik tracing failed, continuing without: {e}")
                    yield span
            else:
                yield span

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            span.error = error_message
            raise

        finally:
            end_time = datetime.now(timezone.utc)
            span.end_time = end_time
            span.duration_ms = (end_time - start_time).total_seconds() * 1000

            # Log span completion
            if error_occurred:
                logger.warning(
                    f"Discovery trace completed with error: {span.name} "
                    f"[{span.duration_ms:.2f}ms] - {error_message}"
                )
            else:
                logger.debug(
                    f"Discovery trace completed: {span.name} "
                    f"[{span.duration_ms:.2f}ms] "
                    f"edges={span.n_edges_discovered} "
                    f"agreement={span.algorithm_agreement:.2%}"
                )

    async def log_algorithm_result(
        self,
        parent_span: DiscoverySpan,
        result: AlgorithmResult,
    ) -> None:
        """Log individual algorithm execution.

        Args:
            parent_span: Parent discovery span
            result: Algorithm result to log
        """
        algorithm_data = {
            "algorithm": result.algorithm.value,
            "runtime_seconds": result.runtime_seconds,
            "n_edges": len(result.edge_list),
            "converged": result.converged,
            "score": result.score,
        }
        parent_span.algorithm_results.append(algorithm_data)

        # Add event to parent span
        parent_span.add_event(
            f"algorithm_complete_{result.algorithm.value}",
            algorithm_data,
        )

        # Log metric if Opik enabled
        if self.is_enabled and self._opik:
            try:
                self._opik.log_metric(
                    name=f"discovery_algorithm_{result.algorithm.value}_runtime_seconds",
                    value=result.runtime_seconds,
                    trace_id=parent_span.trace_id,
                    metadata={"algorithm": result.algorithm.value},
                )
            except Exception as e:
                logger.debug(f"Failed to log algorithm metric: {e}")

        logger.debug(
            f"Algorithm {result.algorithm.value} completed: "
            f"{len(result.edge_list)} edges in {result.runtime_seconds:.2f}s"
        )

    async def log_gate_decision(
        self,
        parent_span: DiscoverySpan,
        decision: GateDecision,
        confidence: float,
        reasons: List[str],
    ) -> None:
        """Log gate evaluation result.

        Args:
            parent_span: Parent discovery span
            decision: Gate decision
            confidence: Confidence score
            reasons: List of reasons for decision
        """
        parent_span.gate_decision = decision.value
        parent_span.gate_confidence = confidence
        parent_span.gate_reasons = reasons

        parent_span.add_event(
            "gate_decision",
            {
                "decision": decision.value,
                "confidence": confidence,
                "reasons": reasons,
            },
        )

        # Log metrics if Opik enabled
        if self.is_enabled and self._opik:
            try:
                self._opik.log_metric(
                    name="discovery_gate_confidence",
                    value=confidence,
                    trace_id=parent_span.trace_id,
                    metadata={"decision": decision.value},
                )

                # Log feedback for the trace
                self._opik.log_feedback(
                    trace_id=parent_span.trace_id,
                    score=confidence,
                    feedback_type="gate_confidence",
                    reason=f"Gate decision: {decision.value}",
                )
            except Exception as e:
                logger.debug(f"Failed to log gate decision metric: {e}")

        logger.info(f"Gate decision: {decision.value} (confidence: {confidence:.2%})")

    async def log_cache_event(
        self,
        parent_span: DiscoverySpan,
        hit: bool,
        latency_ms: float,
    ) -> None:
        """Log cache lookup result.

        Args:
            parent_span: Parent discovery span
            hit: Whether cache was hit
            latency_ms: Cache lookup latency
        """
        parent_span.cache_hit = hit
        parent_span.cache_latency_ms = latency_ms

        parent_span.add_event(
            "cache_lookup",
            {
                "hit": hit,
                "latency_ms": latency_ms,
            },
        )

        # Log metrics if Opik enabled
        if self.is_enabled and self._opik:
            try:
                self._opik.log_metric(
                    name="discovery_cache_hit",
                    value=1.0 if hit else 0.0,
                    trace_id=parent_span.trace_id,
                    metadata={"latency_ms": latency_ms},
                )
                self._opik.log_metric(
                    name="discovery_cache_latency_ms",
                    value=latency_ms,
                    trace_id=parent_span.trace_id,
                )
            except Exception as e:
                logger.debug(f"Failed to log cache event metric: {e}")

        logger.debug(f"Cache {'hit' if hit else 'miss'} in {latency_ms:.2f}ms")

    async def log_ensemble_result(
        self,
        parent_span: DiscoverySpan,
        n_edges: int,
        agreement: float,
        runtime_seconds: float,
    ) -> None:
        """Log ensemble voting result.

        Args:
            parent_span: Parent discovery span
            n_edges: Number of edges discovered
            agreement: Algorithm agreement ratio
            runtime_seconds: Total runtime
        """
        parent_span.n_edges_discovered = n_edges
        parent_span.algorithm_agreement = agreement

        parent_span.add_event(
            "ensemble_complete",
            {
                "n_edges": n_edges,
                "agreement": agreement,
                "runtime_seconds": runtime_seconds,
            },
        )

        # Log metrics if Opik enabled
        if self.is_enabled and self._opik:
            try:
                self._opik.log_metric(
                    name="discovery_edges_discovered",
                    value=float(n_edges),
                    trace_id=parent_span.trace_id,
                )
                self._opik.log_metric(
                    name="discovery_algorithm_agreement",
                    value=agreement,
                    trace_id=parent_span.trace_id,
                )
                self._opik.log_metric(
                    name="discovery_total_runtime_seconds",
                    value=runtime_seconds,
                    trace_id=parent_span.trace_id,
                )
            except Exception as e:
                logger.debug(f"Failed to log ensemble result metric: {e}")

        logger.info(
            f"Ensemble complete: {n_edges} edges, "
            f"{agreement:.2%} agreement in {runtime_seconds:.2f}s"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get tracer status."""
        return {
            "enabled": self._enabled,
            "is_enabled": self.is_enabled,
            "initialized": self._initialized,
            "project_name": self._project_name,
            "opik_available": self._opik is not None,
        }


# Module-level singleton
_discovery_tracer: Optional[DiscoveryTracer] = None


def get_discovery_tracer(
    force_new: bool = False,
    **kwargs: Any,
) -> DiscoveryTracer:
    """Get the DiscoveryTracer singleton instance.

    Args:
        force_new: Force creation of a new instance
        **kwargs: Arguments to pass to DiscoveryTracer

    Returns:
        DiscoveryTracer instance
    """
    global _discovery_tracer

    if _discovery_tracer is None or force_new:
        _discovery_tracer = DiscoveryTracer(**kwargs)

    return _discovery_tracer


def reset_discovery_tracer() -> None:
    """Reset the DiscoveryTracer singleton.

    Useful for testing to ensure clean state between tests.
    """
    global _discovery_tracer
    _discovery_tracer = None
