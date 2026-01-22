"""
Opik Feedback Loop Integration for E2I Causal Analytics.

This module provides feedback integration between:
- User feedback collected via the Feedback API
- Opik traces for agent observability
- GEPA optimization for prompt improvement

Features:
- Log user feedback to specific Opik traces
- Aggregate feedback scores per agent
- Feed feedback signals to GEPA optimization
- Track feedback quality over time

Phase 4 - G23: Opik Feedback Loop Integration
Version: 4.3.0

Usage:
    from src.mlops.opik_feedback import (
        OpikFeedbackCollector,
        get_feedback_collector,
        log_user_feedback,
    )

    # Log feedback to a trace
    await log_user_feedback(
        trace_id="trace_abc123",
        score=0.8,
        feedback_type="rating",
        metadata={"user_rating": 4, "helpful": True},
    )

    # Get aggregated feedback for an agent
    collector = get_feedback_collector()
    stats = collector.get_agent_stats("causal_impact")

Author: E2I Causal Analytics Team
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class FeedbackRecord:
    """A single feedback record tied to an Opik trace."""

    feedback_id: str
    trace_id: str
    span_id: Optional[str] = None
    agent_name: str = "unknown"
    score: float = 0.0
    feedback_type: str = "rating"  # rating, correction, outcome, explicit
    category: Optional[str] = None  # accuracy, relevance, latency, format
    user_feedback: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    response: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_opik_format(self) -> Dict[str, Any]:
        """Convert to Opik feedback format."""
        return {
            "name": f"user_feedback_{self.feedback_type}",
            "value": self.score,
            "reason": self.category or self.feedback_type,
            "metadata": {
                "feedback_id": self.feedback_id,
                "feedback_type": self.feedback_type,
                "agent_name": self.agent_name,
                "category": self.category,
                "timestamp": self.timestamp.isoformat(),
                **(self.metadata or {}),
            },
        }


@dataclass
class AgentFeedbackStats:
    """Aggregated feedback statistics for an agent."""

    agent_name: str
    total_feedback: int = 0
    average_score: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    score_trend: List[float] = field(default_factory=list)  # Recent scores
    last_feedback_time: Optional[datetime] = None

    @property
    def positive_ratio(self) -> float:
        """Ratio of positive feedback."""
        if self.total_feedback == 0:
            return 0.0
        return self.positive_count / self.total_feedback


@dataclass
class FeedbackSignal:
    """Feedback signal for GEPA optimization."""

    agent_name: str
    signal_type: str  # "positive", "negative", "correction"
    weight: float  # 0.0 to 1.0
    description: str
    source_feedback_ids: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None
    confidence: float = 0.7

    def to_gepa_format(self) -> Dict[str, Any]:
        """Convert to GEPA-compatible format."""
        return {
            "signal_type": self.signal_type,
            "weight": self.weight,
            "feedback": self.description,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
        }


# =============================================================================
# Opik Feedback Collector
# =============================================================================


class OpikFeedbackCollector:
    """
    Collects and manages user feedback for Opik traces.

    Provides bidirectional integration between:
    - User feedback API → Opik traces
    - Opik traces → GEPA optimization signals

    Example:
        collector = OpikFeedbackCollector()

        # Record feedback
        collector.record_feedback(
            trace_id="trace_123",
            agent_name="causal_impact",
            score=0.8,
            feedback_type="rating",
        )

        # Get stats for optimization
        stats = collector.get_agent_stats("causal_impact")
        signals = collector.generate_optimization_signals("causal_impact")
    """

    def __init__(
        self,
        max_records: int = 10000,
        positive_threshold: float = 0.7,
        negative_threshold: float = 0.4,
    ):
        """
        Initialize feedback collector.

        Args:
            max_records: Maximum feedback records to keep in memory
            positive_threshold: Score threshold for positive feedback
            negative_threshold: Score threshold for negative feedback
        """
        self.max_records = max_records
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        # In-memory storage
        self._records: List[FeedbackRecord] = []
        self._by_trace: Dict[str, List[FeedbackRecord]] = {}
        self._by_agent: Dict[str, List[FeedbackRecord]] = {}

        # Opik connector
        self._opik_connector = None
        self._init_opik()

    def _init_opik(self) -> None:
        """Initialize Opik connector."""
        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik_connector = get_opik_connector()
            logger.debug("OpikFeedbackCollector initialized with Opik connector")
        except ImportError:
            logger.warning("Opik connector not available for feedback collection")
            self._opik_connector = None

    @property
    def opik_enabled(self) -> bool:
        """Check if Opik integration is available."""
        return self._opik_connector is not None and self._opik_connector.is_enabled

    def record_feedback(
        self,
        trace_id: str,
        agent_name: str,
        score: float,
        feedback_type: str = "rating",
        span_id: Optional[str] = None,
        category: Optional[str] = None,
        user_feedback: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """
        Record user feedback and optionally log to Opik.

        Args:
            trace_id: Opik trace ID (or internal ID if not traced)
            agent_name: Name of the agent that generated the response
            score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback (rating, correction, outcome, explicit)
            span_id: Optional specific span ID
            category: Optional feedback category (accuracy, relevance, etc.)
            user_feedback: Raw user feedback data
            query: Original user query
            response: Agent response
            metadata: Additional metadata

        Returns:
            Created FeedbackRecord
        """
        # Create feedback record
        record = FeedbackRecord(
            feedback_id=f"fb_{uuid4().hex[:12]}",
            trace_id=trace_id,
            span_id=span_id,
            agent_name=agent_name,
            score=score,
            feedback_type=feedback_type,
            category=category,
            user_feedback=user_feedback,
            query=query,
            response=response,
            metadata=metadata or {},
        )

        # Store record
        self._add_record(record)

        # Log to Opik if available
        if self.opik_enabled and trace_id not in ("unknown", "disabled", "error"):
            self._log_to_opik(record)

        logger.debug(
            f"Recorded feedback {record.feedback_id} for agent {agent_name}: "
            f"score={score:.2f}, type={feedback_type}"
        )

        return record

    def _add_record(self, record: FeedbackRecord) -> None:
        """Add record to storage, respecting max_records limit."""
        self._records.append(record)

        # Index by trace
        if record.trace_id not in self._by_trace:
            self._by_trace[record.trace_id] = []
        self._by_trace[record.trace_id].append(record)

        # Index by agent
        if record.agent_name not in self._by_agent:
            self._by_agent[record.agent_name] = []
        self._by_agent[record.agent_name].append(record)

        # Enforce max records
        if len(self._records) > self.max_records:
            oldest = self._records.pop(0)
            # Clean up indexes
            if oldest.trace_id in self._by_trace:
                self._by_trace[oldest.trace_id] = [
                    r for r in self._by_trace[oldest.trace_id] if r.feedback_id != oldest.feedback_id
                ]
            if oldest.agent_name in self._by_agent:
                self._by_agent[oldest.agent_name] = [
                    r for r in self._by_agent[oldest.agent_name] if r.feedback_id != oldest.feedback_id
                ]

    def _log_to_opik(self, record: FeedbackRecord) -> None:
        """Log feedback to Opik trace."""
        try:
            if self._opik_connector:
                self._opik_connector.log_feedback(
                    trace_id=record.trace_id,
                    name=f"user_feedback_{record.feedback_type}",
                    value=record.score,
                    reason=record.category or record.feedback_type,
                )
                logger.debug(f"Logged feedback to Opik trace {record.trace_id}")
        except Exception as e:
            logger.warning(f"Failed to log feedback to Opik: {e}")

    def get_trace_feedback(self, trace_id: str) -> List[FeedbackRecord]:
        """Get all feedback for a specific trace."""
        return self._by_trace.get(trace_id, [])

    def get_agent_feedback(
        self,
        agent_name: str,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[FeedbackRecord]:
        """
        Get feedback records for an agent.

        Args:
            agent_name: Agent name
            limit: Maximum records to return
            since: Only return records after this time

        Returns:
            List of FeedbackRecords, most recent first
        """
        records = self._by_agent.get(agent_name, [])

        if since:
            records = [r for r in records if r.timestamp > since]

        # Sort by timestamp descending
        records = sorted(records, key=lambda r: r.timestamp, reverse=True)

        return records[:limit]

    def get_agent_stats(self, agent_name: str) -> AgentFeedbackStats:
        """
        Get aggregated feedback statistics for an agent.

        Args:
            agent_name: Agent name

        Returns:
            AgentFeedbackStats with aggregated metrics
        """
        records = self._by_agent.get(agent_name, [])

        if not records:
            return AgentFeedbackStats(agent_name=agent_name)

        # Calculate stats
        scores = [r.score for r in records]
        positive_count = sum(1 for s in scores if s >= self.positive_threshold)
        negative_count = sum(1 for s in scores if s <= self.negative_threshold)

        # Count by type and category
        by_type: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for r in records:
            by_type[r.feedback_type] = by_type.get(r.feedback_type, 0) + 1
            if r.category:
                by_category[r.category] = by_category.get(r.category, 0) + 1

        # Recent score trend (last 20)
        recent_records = sorted(records, key=lambda r: r.timestamp, reverse=True)[:20]
        score_trend = [r.score for r in reversed(recent_records)]

        return AgentFeedbackStats(
            agent_name=agent_name,
            total_feedback=len(records),
            average_score=sum(scores) / len(scores),
            positive_count=positive_count,
            negative_count=negative_count,
            by_type=by_type,
            by_category=by_category,
            score_trend=score_trend,
            last_feedback_time=max(r.timestamp for r in records),
        )

    def generate_optimization_signals(
        self,
        agent_name: str,
        min_feedback_count: int = 5,
    ) -> List[FeedbackSignal]:
        """
        Generate optimization signals for GEPA from accumulated feedback.

        Analyzes feedback patterns to generate actionable signals that
        GEPA can use to improve agent prompts.

        Args:
            agent_name: Agent name
            min_feedback_count: Minimum feedback required

        Returns:
            List of FeedbackSignals for optimization
        """
        records = self._by_agent.get(agent_name, [])

        if len(records) < min_feedback_count:
            logger.debug(
                f"Insufficient feedback for {agent_name}: {len(records)} < {min_feedback_count}"
            )
            return []

        signals: List[FeedbackSignal] = []
        stats = self.get_agent_stats(agent_name)

        # Signal: Overall quality trend
        if stats.average_score < 0.5:
            signals.append(
                FeedbackSignal(
                    agent_name=agent_name,
                    signal_type="negative",
                    weight=0.8,
                    description=f"Low overall satisfaction (avg: {stats.average_score:.2f})",
                    suggested_action="Review and improve response quality",
                    confidence=0.8,
                )
            )
        elif stats.average_score > 0.8:
            signals.append(
                FeedbackSignal(
                    agent_name=agent_name,
                    signal_type="positive",
                    weight=0.6,
                    description=f"High user satisfaction (avg: {stats.average_score:.2f})",
                    suggested_action="Maintain current approach",
                    confidence=0.8,
                )
            )

        # Signal: Specific category issues
        for category, count in stats.by_category.items():
            if count >= 3:
                # Get average score for this category
                cat_records = [r for r in records if r.category == category]
                cat_avg = sum(r.score for r in cat_records) / len(cat_records)

                if cat_avg < 0.5:
                    signals.append(
                        FeedbackSignal(
                            agent_name=agent_name,
                            signal_type="negative",
                            weight=0.7,
                            description=f"{category} issues detected (avg: {cat_avg:.2f})",
                            source_feedback_ids=[r.feedback_id for r in cat_records[:5]],
                            suggested_action=f"Focus on improving {category}",
                            confidence=0.7,
                        )
                    )

        # Signal: Correction feedback patterns
        correction_records = [r for r in records if r.feedback_type == "correction"]
        if len(correction_records) >= 3:
            signals.append(
                FeedbackSignal(
                    agent_name=agent_name,
                    signal_type="correction",
                    weight=0.9,
                    description=f"{len(correction_records)} user corrections received",
                    source_feedback_ids=[r.feedback_id for r in correction_records[:5]],
                    suggested_action="Analyze corrections for prompt refinement",
                    confidence=0.85,
                )
            )

        return signals

    def get_gepa_feedback_batch(
        self,
        agent_name: str,
        batch_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get a batch of feedback examples for GEPA training.

        Formats feedback records as GEPA-compatible training examples.

        Args:
            agent_name: Agent name
            batch_size: Number of examples to return

        Returns:
            List of GEPA-formatted feedback examples
        """
        records = self.get_agent_feedback(agent_name, limit=batch_size)

        examples = []
        for record in records:
            if record.query and record.response:
                examples.append({
                    "question": record.query,
                    "answer": record.response,
                    "feedback_score": record.score,
                    "feedback_type": record.feedback_type,
                    "feedback_text": (
                        str(record.user_feedback) if record.user_feedback else ""
                    ),
                    "metadata": {
                        "feedback_id": record.feedback_id,
                        "trace_id": record.trace_id,
                        "timestamp": record.timestamp.isoformat(),
                    },
                })

        return examples

    def clear(self) -> None:
        """Clear all stored feedback."""
        self._records.clear()
        self._by_trace.clear()
        self._by_agent.clear()


# =============================================================================
# Global Instance and Convenience Functions
# =============================================================================

_feedback_collector: Optional[OpikFeedbackCollector] = None


def get_feedback_collector() -> OpikFeedbackCollector:
    """Get or create the global feedback collector."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = OpikFeedbackCollector()
    return _feedback_collector


async def log_user_feedback(
    trace_id: str,
    score: float,
    feedback_type: str = "rating",
    agent_name: str = "unknown",
    span_id: Optional[str] = None,
    category: Optional[str] = None,
    user_feedback: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
    response: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeedbackRecord:
    """
    Convenience function to log user feedback.

    Args:
        trace_id: Opik trace ID
        score: Feedback score (0.0 to 1.0)
        feedback_type: Type of feedback
        agent_name: Agent that generated the response
        span_id: Optional specific span ID
        category: Optional feedback category
        user_feedback: Raw user feedback data
        query: Original user query
        response: Agent response
        metadata: Additional metadata

    Returns:
        Created FeedbackRecord
    """
    collector = get_feedback_collector()
    return collector.record_feedback(
        trace_id=trace_id,
        agent_name=agent_name,
        score=score,
        feedback_type=feedback_type,
        span_id=span_id,
        category=category,
        user_feedback=user_feedback,
        query=query,
        response=response,
        metadata=metadata,
    )


def get_feedback_signals_for_gepa(
    agent_name: str,
    min_feedback_count: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get feedback signals for GEPA optimization.

    Args:
        agent_name: Agent name
        min_feedback_count: Minimum feedback required

    Returns:
        List of GEPA-formatted feedback signals
    """
    collector = get_feedback_collector()
    signals = collector.generate_optimization_signals(agent_name, min_feedback_count)
    return [s.to_gepa_format() for s in signals]


__all__ = [
    "FeedbackRecord",
    "AgentFeedbackStats",
    "FeedbackSignal",
    "OpikFeedbackCollector",
    "get_feedback_collector",
    "log_user_feedback",
    "get_feedback_signals_for_gepa",
]
