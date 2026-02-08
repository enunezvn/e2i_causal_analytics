"""
E2I Causal Analytics - Per-Agent Cost Tracking
===============================================

Tracks LLM API costs per agent for budget monitoring and cost optimization.

Features:
- Per-agent cost aggregation by day/hour
- Token-to-cost conversion for Claude/OpenAI models
- Prometheus metrics export
- Cost alerts and budgets
- Historical cost queries

Model Pricing (as of 2024):
- Claude 3.5 Sonnet: $3/1M input, $15/1M output
- Claude 3 Opus: $15/1M input, $75/1M output
- Claude 3 Haiku: $0.25/1M input, $1.25/1M output
- GPT-4: $30/1M input, $60/1M output
- GPT-4 Turbo: $10/1M input, $30/1M output

Usage:
    from src.mlops.agent_cost_tracker import (
        AgentCostTracker,
        get_cost_tracker,
        record_agent_cost,
    )

    # Record cost for an agent operation
    record_agent_cost(
        agent_name="gap_analyzer",
        model="claude-sonnet-4-20250514",
        input_tokens=1500,
        output_tokens=500,
    )

    # Get cost summary
    tracker = get_cost_tracker()
    summary = tracker.get_agent_summary("gap_analyzer")

Author: E2I Causal Analytics Team
Version: 1.0.0 (Phase 4 - G25)
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Model Pricing (per 1M tokens)
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class ModelPricing:
    """Pricing per 1M tokens for a model."""

    input_cost_per_million: float
    output_cost_per_million: float
    provider: LLMProvider = LLMProvider.ANTHROPIC


# Model pricing lookup (as of Jan 2025)
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Claude 3.5 Sonnet (current)
    "claude-sonnet-4-20250514": ModelPricing(3.0, 15.0, LLMProvider.ANTHROPIC),
    "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0, LLMProvider.ANTHROPIC),
    "claude-3-5-sonnet-20240620": ModelPricing(3.0, 15.0, LLMProvider.ANTHROPIC),
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.0, LLMProvider.ANTHROPIC),
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(15.0, 75.0, LLMProvider.ANTHROPIC),
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, LLMProvider.ANTHROPIC),
    # OpenAI GPT-4
    "gpt-4": ModelPricing(30.0, 60.0, LLMProvider.OPENAI),
    "gpt-4-turbo": ModelPricing(10.0, 30.0, LLMProvider.OPENAI),
    "gpt-4-turbo-preview": ModelPricing(10.0, 30.0, LLMProvider.OPENAI),
    "gpt-4o": ModelPricing(2.5, 10.0, LLMProvider.OPENAI),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, LLMProvider.OPENAI),
    # GPT-3.5
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50, LLMProvider.OPENAI),
}

# Default pricing for unknown models
DEFAULT_PRICING = ModelPricing(3.0, 15.0, LLMProvider.ANTHROPIC)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost in USD for token usage.

    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_million

    return input_cost + output_cost


# =============================================================================
# Cost Record
# =============================================================================


@dataclass
class CostRecord:
    """A single cost record for an agent operation."""

    record_id: str
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation": self.operation,
        }


@dataclass
class AgentCostSummary:
    """Cost summary for an agent."""

    agent_name: str
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    average_cost_per_call: float
    period_start: datetime
    period_end: datetime
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_operation: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "average_cost_per_call": round(self.average_cost_per_call, 6),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "by_model": self.by_model,
            "by_operation": self.by_operation,
        }


# =============================================================================
# Agent Cost Tracker
# =============================================================================


class AgentCostTracker:
    """Tracks and aggregates LLM costs per agent.

    Thread-safe cost tracking with periodic cleanup of old records.

    Attributes:
        max_records: Maximum records to keep in memory
        retention_hours: Hours to retain records (default 24)
    """

    def __init__(
        self,
        max_records: int = 50000,
        retention_hours: int = 24,
    ):
        """Initialize cost tracker.

        Args:
            max_records: Maximum records to keep
            retention_hours: Hours to retain records
        """
        self._records: List[CostRecord] = []
        self._lock = threading.RLock()
        self._max_records = max_records
        self._retention_hours = retention_hours
        self._record_counter = 0

        # Aggregated stats (updated on each record)
        self._total_by_agent: Dict[str, float] = defaultdict(float)
        self._calls_by_agent: Dict[str, int] = defaultdict(int)
        self._tokens_by_agent: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))

        # Budget tracking
        self._budgets: Dict[str, float] = {}  # agent_name -> daily budget USD
        self._daily_spend: Dict[str, float] = defaultdict(float)
        self._daily_spend_date: Optional[date] = None

        logger.info(
            f"AgentCostTracker initialized (max_records={max_records}, "
            f"retention_hours={retention_hours})"
        )

    def record(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """Record a cost entry for an agent operation.

        Args:
            agent_name: Name of the agent
            model: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count
            trace_id: Associated Opik trace ID
            span_id: Associated Opik span ID
            operation: Operation name
            metadata: Additional metadata

        Returns:
            Created CostRecord
        """
        cost_usd = calculate_cost(model, input_tokens, output_tokens)

        with self._lock:
            self._record_counter += 1
            record_id = f"cost_{self._record_counter:08d}"

            record = CostRecord(
                record_id=record_id,
                agent_name=agent_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                trace_id=trace_id,
                span_id=span_id,
                operation=operation,
                metadata=metadata or {},
            )

            self._records.append(record)

            # Update aggregates
            self._total_by_agent[agent_name] += cost_usd
            self._calls_by_agent[agent_name] += 1
            prev_in, prev_out = self._tokens_by_agent[agent_name]
            self._tokens_by_agent[agent_name] = (
                prev_in + input_tokens,
                prev_out + output_tokens,
            )

            # Update daily spend
            self._update_daily_spend(agent_name, cost_usd)

            # Check budget
            self._check_budget(agent_name)

            # Cleanup if needed
            if len(self._records) > self._max_records:
                self._cleanup()

        logger.debug(
            f"Recorded cost: agent={agent_name}, model={model}, "
            f"tokens={input_tokens}+{output_tokens}, cost=${cost_usd:.6f}"
        )

        return record

    def _update_daily_spend(self, agent_name: str, cost_usd: float) -> None:
        """Update daily spend tracking."""
        today = datetime.now(timezone.utc).date()

        if self._daily_spend_date != today:
            # New day, reset spend
            self._daily_spend.clear()
            self._daily_spend_date = today

        self._daily_spend[agent_name] += cost_usd

    def _check_budget(self, agent_name: str) -> None:
        """Check if agent has exceeded daily budget."""
        if agent_name not in self._budgets:
            return

        budget = self._budgets[agent_name]
        spend = self._daily_spend[agent_name]

        if spend > budget:
            logger.warning(
                f"Agent {agent_name} exceeded daily budget: ${spend:.4f} > ${budget:.4f}"
            )

    def _cleanup(self) -> None:
        """Remove old records."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._retention_hours)

        # Remove records older than cutoff
        old_count = len(self._records)
        self._records = [r for r in self._records if r.timestamp > cutoff]

        # If still too many, remove oldest
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

        removed = old_count - len(self._records)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old cost records")

    def set_budget(self, agent_name: str, daily_budget_usd: float) -> None:
        """Set daily budget for an agent.

        Args:
            agent_name: Agent name
            daily_budget_usd: Daily budget in USD
        """
        with self._lock:
            self._budgets[agent_name] = daily_budget_usd
            logger.info(f"Set budget for {agent_name}: ${daily_budget_usd:.2f}/day")

    def get_agent_summary(
        self,
        agent_name: str,
        hours: int = 24,
    ) -> AgentCostSummary:
        """Get cost summary for an agent.

        Args:
            agent_name: Agent name
            hours: Hours to look back

        Returns:
            AgentCostSummary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._lock:
            agent_records = [
                r for r in self._records if r.agent_name == agent_name and r.timestamp > cutoff
            ]

        if not agent_records:
            return AgentCostSummary(
                agent_name=agent_name,
                total_cost_usd=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_calls=0,
                average_cost_per_call=0.0,
                period_start=cutoff,
                period_end=datetime.now(timezone.utc),
            )

        total_cost = sum(r.cost_usd for r in agent_records)
        total_input = sum(r.input_tokens for r in agent_records)
        total_output = sum(r.output_tokens for r in agent_records)
        total_calls = len(agent_records)

        # Group by model
        by_model: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "calls": 0})
        for r in agent_records:
            by_model[r.model]["cost"] += r.cost_usd
            by_model[r.model]["calls"] += 1

        # Group by operation
        by_operation: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "calls": 0})
        for r in agent_records:
            op = r.operation or "unknown"
            by_operation[op]["cost"] += r.cost_usd
            by_operation[op]["calls"] += 1

        return AgentCostSummary(
            agent_name=agent_name,
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_calls=total_calls,
            average_cost_per_call=total_cost / total_calls if total_calls > 0 else 0.0,
            period_start=cutoff,
            period_end=datetime.now(timezone.utc),
            by_model=dict(by_model),
            by_operation=dict(by_operation),
        )

    def get_all_agents_summary(self, hours: int = 24) -> Dict[str, AgentCostSummary]:
        """Get cost summary for all agents.

        Args:
            hours: Hours to look back

        Returns:
            Dict mapping agent names to summaries
        """
        with self._lock:
            agent_names = {r.agent_name for r in self._records}

        return {name: self.get_agent_summary(name, hours) for name in agent_names}

    def get_total_cost(self, hours: int = 24) -> Dict[str, Any]:
        """Get total cost across all agents.

        Args:
            hours: Hours to look back

        Returns:
            Total cost summary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._lock:
            recent_records = [r for r in self._records if r.timestamp > cutoff]

        total_cost = sum(r.cost_usd for r in recent_records)
        total_input = sum(r.input_tokens for r in recent_records)
        total_output = sum(r.output_tokens for r in recent_records)
        total_calls = len(recent_records)

        # By agent
        by_agent: Dict[str, float] = defaultdict(float)
        for r in recent_records:
            by_agent[r.agent_name] += r.cost_usd

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_calls": total_calls,
            "period_hours": hours,
            "by_agent": dict(by_agent),
        }

    def get_metrics_for_prometheus(self) -> List[Dict[str, Any]]:
        """Get metrics formatted for Prometheus.

        Returns list of metric dictionaries with labels.
        """
        metrics = []

        with self._lock:
            for agent_name, total_cost in self._total_by_agent.items():
                calls = self._calls_by_agent[agent_name]
                input_tokens, output_tokens = self._tokens_by_agent[agent_name]

                metrics.append(
                    {
                        "name": "e2i_agent_cost_total_usd",
                        "value": total_cost,
                        "labels": {"agent": agent_name},
                        "type": "counter",
                    }
                )
                metrics.append(
                    {
                        "name": "e2i_agent_calls_total",
                        "value": calls,
                        "labels": {"agent": agent_name},
                        "type": "counter",
                    }
                )
                metrics.append(
                    {
                        "name": "e2i_agent_input_tokens_total",
                        "value": input_tokens,
                        "labels": {"agent": agent_name},
                        "type": "counter",
                    }
                )
                metrics.append(
                    {
                        "name": "e2i_agent_output_tokens_total",
                        "value": output_tokens,
                        "labels": {"agent": agent_name},
                        "type": "counter",
                    }
                )

        return metrics


# =============================================================================
# Singleton Instance
# =============================================================================

_cost_tracker: Optional[AgentCostTracker] = None
_cost_tracker_lock = threading.Lock()


def get_cost_tracker() -> AgentCostTracker:
    """Get singleton cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        with _cost_tracker_lock:
            if _cost_tracker is None:
                _cost_tracker = AgentCostTracker()
    return _cost_tracker


def reset_cost_tracker() -> None:
    """Reset the singleton tracker (mainly for testing)."""
    global _cost_tracker
    with _cost_tracker_lock:
        _cost_tracker = None


# =============================================================================
# Convenience Functions
# =============================================================================


def record_agent_cost(
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    operation: Optional[str] = None,
) -> CostRecord:
    """Record cost for an agent operation.

    Args:
        agent_name: Agent name
        model: Model identifier
        input_tokens: Input token count
        output_tokens: Output token count
        trace_id: Optional Opik trace ID
        span_id: Optional Opik span ID
        operation: Optional operation name

    Returns:
        Created CostRecord
    """
    return get_cost_tracker().record(
        agent_name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trace_id=trace_id,
        span_id=span_id,
        operation=operation,
    )


def get_agent_cost_summary(agent_name: str, hours: int = 24) -> Dict[str, Any]:
    """Get cost summary for an agent.

    Args:
        agent_name: Agent name
        hours: Hours to look back

    Returns:
        Summary dictionary
    """
    summary = get_cost_tracker().get_agent_summary(agent_name, hours)
    return summary.to_dict()


def get_total_cost_summary(hours: int = 24) -> Dict[str, Any]:
    """Get total cost across all agents.

    Args:
        hours: Hours to look back

    Returns:
        Total cost summary
    """
    return get_cost_tracker().get_total_cost(hours)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Classes
    "AgentCostTracker",
    "CostRecord",
    "AgentCostSummary",
    "ModelPricing",
    "LLMProvider",
    # Pricing
    "MODEL_PRICING",
    "calculate_cost",
    # Singleton
    "get_cost_tracker",
    "reset_cost_tracker",
    # Convenience
    "record_agent_cost",
    "get_agent_cost_summary",
    "get_total_cost_summary",
]
