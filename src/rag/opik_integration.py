"""
Opik Integration for RAG Evaluation.

This module provides Opik tracing utilities specifically for RAGAS evaluation,
enabling observability of RAG pipeline evaluations with:
- Trace creation for evaluation runs
- Score logging as feedback
- Rubric score integration
- Circuit breaker protection

Usage:
    from src.rag.opik_integration import (
        OpikEvaluationTracer,
        log_ragas_scores,
        log_rubric_scores,
    )

    tracer = OpikEvaluationTracer()

    # Trace an evaluation run
    async with tracer.trace_evaluation("eval_001") as trace:
        results = await evaluator.evaluate_batch(samples)
        trace.log_scores(results)

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTraceContext:
    """Context for an evaluation trace with score logging utilities."""

    trace_id: str
    run_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    sample_count: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    rubric_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    _opik_trace: Any = None  # Actual Opik trace object

    def log_ragas_scores(
        self,
        faithfulness: Optional[float] = None,
        answer_relevancy: Optional[float] = None,
        context_precision: Optional[float] = None,
        context_recall: Optional[float] = None,
        overall_score: Optional[float] = None,
    ) -> None:
        """Log RAGAS metric scores to the trace.

        Args:
            faithfulness: Faithfulness score (0-1)
            answer_relevancy: Answer relevancy score (0-1)
            context_precision: Context precision score (0-1)
            context_recall: Context recall score (0-1)
            overall_score: Overall aggregated score (0-1)
        """
        if faithfulness is not None:
            self.scores["faithfulness"] = faithfulness
        if answer_relevancy is not None:
            self.scores["answer_relevancy"] = answer_relevancy
        if context_precision is not None:
            self.scores["context_precision"] = context_precision
        if context_recall is not None:
            self.scores["context_recall"] = context_recall
        if overall_score is not None:
            self.scores["overall_score"] = overall_score

        # Log to Opik if available
        self._log_scores_to_opik(self.scores, prefix="ragas")

    def log_rubric_scores(
        self,
        causal_validity: Optional[float] = None,
        actionability: Optional[float] = None,
        evidence_chain: Optional[float] = None,
        regulatory_awareness: Optional[float] = None,
        uncertainty_communication: Optional[float] = None,
        weighted_score: Optional[float] = None,
        decision: Optional[str] = None,
    ) -> None:
        """Log rubric evaluation scores to the trace.

        Args:
            causal_validity: Causal validity score (1-5)
            actionability: Actionability score (1-5)
            evidence_chain: Evidence chain score (1-5)
            regulatory_awareness: Regulatory awareness score (1-5)
            uncertainty_communication: Uncertainty communication score (1-5)
            weighted_score: Weighted overall score (1-5)
            decision: Improvement decision (acceptable, suggestion, auto_update, escalate)
        """
        if causal_validity is not None:
            self.rubric_scores["causal_validity"] = causal_validity
        if actionability is not None:
            self.rubric_scores["actionability"] = actionability
        if evidence_chain is not None:
            self.rubric_scores["evidence_chain"] = evidence_chain
        if regulatory_awareness is not None:
            self.rubric_scores["regulatory_awareness"] = regulatory_awareness
        if uncertainty_communication is not None:
            self.rubric_scores["uncertainty_communication"] = uncertainty_communication
        if weighted_score is not None:
            self.rubric_scores["weighted_score"] = weighted_score
        if decision is not None:
            self.metadata["rubric_decision"] = decision

        # Log to Opik if available
        self._log_scores_to_opik(self.rubric_scores, prefix="rubric")

    def _log_scores_to_opik(self, scores: Dict[str, float], prefix: str) -> None:
        """Log scores to Opik trace as feedback scores."""
        if not self._opik_trace:
            return

        try:
            for name, value in scores.items():
                score_name = f"{prefix}_{name}" if prefix else name
                self._opik_trace.log_feedback_score(
                    name=score_name,
                    value=value,
                    reason=f"Automated {prefix} evaluation score",
                )
        except Exception as e:
            logger.debug(f"Failed to log scores to Opik: {e}")

    def set_sample_count(self, count: int) -> None:
        """Set the number of samples evaluated."""
        self.sample_count = count
        self.metadata["sample_count"] = count

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary."""
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "started_at": self.start_time.isoformat(),
            "ended_at": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": (
                (self.end_time - self.start_time).total_seconds() * 1000
                if self.end_time
                else None
            ),
            "sample_count": self.sample_count,
            "ragas_scores": self.scores,
            "rubric_scores": self.rubric_scores,
            "metadata": self.metadata,
            "error": self.error,
        }


class OpikEvaluationTracer:
    """Opik tracer specialized for RAG evaluation operations.

    Provides tracing for:
    - Individual sample evaluations
    - Batch evaluation runs
    - Full pipeline evaluation with RAGAS + rubric scores

    Uses the shared OpikConnector for circuit breaker protection.

    Example:
        tracer = OpikEvaluationTracer()

        async with tracer.trace_evaluation("eval_run_001") as trace:
            results = await evaluator.evaluate_batch(samples)
            trace.set_sample_count(len(results))
            trace.log_ragas_scores(
                faithfulness=0.85,
                answer_relevancy=0.90,
                overall_score=0.87,
            )
    """

    def __init__(
        self,
        project_name: str = "e2i-rag-evaluation",
        enabled: bool = True,
    ):
        """Initialize the evaluation tracer.

        Args:
            project_name: Opik project name for evaluation traces
            enabled: Whether tracing is enabled
        """
        self.project_name = project_name
        self.enabled = enabled
        self._opik_connector = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of OpikConnector."""
        if self._initialized:
            return

        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik_connector = get_opik_connector()
            self._initialized = True
            logger.debug("OpikEvaluationTracer initialized")
        except ImportError:
            logger.warning("OpikConnector not available, tracing disabled")
            self._opik_connector = None
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize OpikConnector: {e}")
            self._opik_connector = None
            self._initialized = True

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and available."""
        self._ensure_initialized()
        return (
            self.enabled
            and self._opik_connector is not None
            and self._opik_connector.is_enabled
        )

    @property
    def circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        if not self._opik_connector:
            return {"state": "unknown", "reason": "connector_not_initialized"}
        return self._opik_connector.circuit_breaker.get_status()

    @asynccontextmanager
    async def trace_evaluation(
        self,
        run_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing an evaluation run.

        Creates an Opik trace for the evaluation and provides a context
        object for logging scores and metadata.

        Args:
            run_id: Unique identifier for this evaluation run
            metadata: Additional metadata for the trace

        Yields:
            EvaluationTraceContext: Context for logging scores

        Example:
            async with tracer.trace_evaluation("eval_001") as trace:
                results = await run_evaluation()
                trace.log_ragas_scores(faithfulness=0.85)
        """
        self._ensure_initialized()

        trace_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Create trace context
        trace_ctx = EvaluationTraceContext(
            trace_id=trace_id,
            run_id=run_id,
            start_time=start_time,
            metadata=metadata or {},
        )

        opik_trace = None

        try:
            # Create Opik trace if enabled and circuit breaker allows
            if self.is_enabled and self._opik_connector.circuit_breaker.allow_request():
                try:
                    async with self._opik_connector.trace_agent(
                        agent_name="rag_evaluator",
                        operation="evaluate",
                        trace_id=trace_id,
                        metadata={
                            "run_id": run_id,
                            "evaluation_type": "ragas",
                            **(metadata or {}),
                        },
                        tags=["evaluation", "ragas", run_id],
                    ) as span:
                        trace_ctx._opik_trace = span
                        yield trace_ctx

                        # Log final scores on success
                        if trace_ctx.scores:
                            span.set_output(
                                {
                                    "ragas_scores": trace_ctx.scores,
                                    "rubric_scores": trace_ctx.rubric_scores,
                                    "sample_count": trace_ctx.sample_count,
                                }
                            )

                    self._opik_connector.circuit_breaker.record_success()

                except Exception as e:
                    self._opik_connector.circuit_breaker.record_failure()
                    logger.warning(f"Opik tracing failed: {e}")
                    trace_ctx.error = str(e)
                    yield trace_ctx
            else:
                # No Opik tracing, just yield context for local use
                yield trace_ctx

        except Exception as e:
            trace_ctx.error = str(e)
            raise

        finally:
            trace_ctx.end_time = datetime.now(timezone.utc)

    @asynccontextmanager
    async def trace_sample_evaluation(
        self,
        sample_id: str,
        parent_trace_id: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """Trace evaluation of a single sample.

        Creates a child span under the parent trace for individual sample evaluation.

        Args:
            sample_id: Unique identifier for this sample
            parent_trace_id: Parent trace ID to attach this span to
            query: The query being evaluated

        Yields:
            EvaluationTraceContext: Context for this sample
        """
        self._ensure_initialized()

        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        trace_ctx = EvaluationTraceContext(
            trace_id=span_id,
            run_id=sample_id,
            start_time=start_time,
            metadata={"query": query} if query else {},
        )

        try:
            if self.is_enabled and self._opik_connector.circuit_breaker.allow_request():
                try:
                    async with self._opik_connector.trace_agent(
                        agent_name="rag_evaluator",
                        operation="evaluate_sample",
                        trace_id=parent_trace_id,
                        parent_span_id=None,
                        metadata={"sample_id": sample_id, "query": query},
                        input_data={"query": query} if query else None,
                    ) as span:
                        trace_ctx._opik_trace = span
                        yield trace_ctx

                except Exception as e:
                    logger.debug(f"Sample tracing failed: {e}")
                    yield trace_ctx
            else:
                yield trace_ctx

        finally:
            trace_ctx.end_time = datetime.now(timezone.utc)


def log_ragas_scores_to_opik(
    run_id: str,
    faithfulness: Optional[float] = None,
    answer_relevancy: Optional[float] = None,
    context_precision: Optional[float] = None,
    context_recall: Optional[float] = None,
    overall_score: Optional[float] = None,
    sample_count: int = 0,
) -> None:
    """Convenience function to log RAGAS scores to Opik.

    Used for non-async contexts or when not using the tracer context manager.

    Args:
        run_id: Evaluation run identifier
        faithfulness: Faithfulness score
        answer_relevancy: Answer relevancy score
        context_precision: Context precision score
        context_recall: Context recall score
        overall_score: Overall score
        sample_count: Number of samples evaluated
    """
    try:
        from src.mlops.opik_connector import get_opik_connector

        connector = get_opik_connector()
        if not connector.is_enabled:
            return

        if not connector.circuit_breaker.allow_request():
            logger.debug("Circuit open, skipping RAGAS score logging")
            return

        # Log each score as a metric
        if faithfulness is not None:
            connector.log_metric("ragas_faithfulness", faithfulness)
        if answer_relevancy is not None:
            connector.log_metric("ragas_answer_relevancy", answer_relevancy)
        if context_precision is not None:
            connector.log_metric("ragas_context_precision", context_precision)
        if context_recall is not None:
            connector.log_metric("ragas_context_recall", context_recall)
        if overall_score is not None:
            connector.log_metric("ragas_overall_score", overall_score)

        connector.circuit_breaker.record_success()
        logger.debug(f"Logged RAGAS scores for run {run_id}")

    except Exception as e:
        logger.warning(f"Failed to log RAGAS scores: {e}")


def log_rubric_scores_to_opik(
    run_id: str,
    weighted_score: Optional[float] = None,
    decision: Optional[str] = None,
    criterion_scores: Optional[Dict[str, float]] = None,
) -> None:
    """Convenience function to log rubric evaluation scores to Opik.

    Args:
        run_id: Evaluation run identifier
        weighted_score: Weighted rubric score (1-5)
        decision: Improvement decision
        criterion_scores: Individual criterion scores
    """
    try:
        from src.mlops.opik_connector import get_opik_connector

        connector = get_opik_connector()
        if not connector.is_enabled:
            return

        if not connector.circuit_breaker.allow_request():
            logger.debug("Circuit open, skipping rubric score logging")
            return

        # Log weighted score
        if weighted_score is not None:
            # Normalize to 0-1 for Opik (rubric is 1-5)
            normalized_score = (weighted_score - 1.0) / 4.0
            connector.log_metric("rubric_weighted_score", normalized_score)

        # Log individual criterion scores
        if criterion_scores:
            for criterion, score in criterion_scores.items():
                # Normalize to 0-1
                normalized = (score - 1.0) / 4.0
                connector.log_metric(f"rubric_{criterion}", normalized)

        connector.circuit_breaker.record_success()
        logger.debug(f"Logged rubric scores for run {run_id}")

    except Exception as e:
        logger.warning(f"Failed to log rubric scores: {e}")


@dataclass
class CombinedEvaluationResult:
    """Combined result from RAGAS + rubric evaluation."""

    run_id: str
    timestamp: str

    # RAGAS scores
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    ragas_context_recall: Optional[float] = None
    ragas_overall: Optional[float] = None

    # Rubric scores
    rubric_weighted_score: Optional[float] = None
    rubric_decision: Optional[str] = None
    rubric_criterion_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    sample_count: int = 0
    evaluation_time_seconds: float = 0.0
    passed_thresholds: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "ragas": {
                "faithfulness": self.ragas_faithfulness,
                "answer_relevancy": self.ragas_answer_relevancy,
                "context_precision": self.ragas_context_precision,
                "context_recall": self.ragas_context_recall,
                "overall": self.ragas_overall,
            },
            "rubric": {
                "weighted_score": self.rubric_weighted_score,
                "decision": self.rubric_decision,
                "criterion_scores": self.rubric_criterion_scores,
            },
            "metadata": {
                "sample_count": self.sample_count,
                "evaluation_time_seconds": self.evaluation_time_seconds,
                "passed_thresholds": self.passed_thresholds,
            },
        }

    def log_to_opik(self) -> None:
        """Log all scores to Opik."""
        log_ragas_scores_to_opik(
            run_id=self.run_id,
            faithfulness=self.ragas_faithfulness,
            answer_relevancy=self.ragas_answer_relevancy,
            context_precision=self.ragas_context_precision,
            context_recall=self.ragas_context_recall,
            overall_score=self.ragas_overall,
            sample_count=self.sample_count,
        )

        log_rubric_scores_to_opik(
            run_id=self.run_id,
            weighted_score=self.rubric_weighted_score,
            decision=self.rubric_decision,
            criterion_scores=self.rubric_criterion_scores,
        )
