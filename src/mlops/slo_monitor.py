"""
E2I Causal Analytics - SLO Monitoring
=====================================

Service Level Objective (SLO) monitoring for the E2I multi-agent platform.
Tracks availability, latency, and error rate targets per agent tier.

Features:
- Per-tier SLO definitions with configurable targets
- Rolling window compliance calculation
- Prometheus metrics for SLO tracking
- Error budget tracking and alerts
- SLO burn rate monitoring

SLO Targets by Tier:
- Tier 0 (Foundation): 99.9% availability, p99 < 5s
- Tier 1 (Orchestrator): 99.9% availability, p99 < 10s
- Tier 2 (Causal): 99.5% availability, p99 < 30s
- Tier 3 (Monitoring): 99.5% availability, p99 < 15s
- Tier 4 (ML): 99.0% availability, p99 < 60s
- Tier 5 (Learning): 99.0% availability, p99 < 30s

Usage:
    from src.mlops.slo_monitor import (
        SLOMonitor,
        get_slo_monitor,
        record_request,
        get_slo_compliance,
    )

    # Record a request
    record_request(
        agent_name="gap_analyzer",
        latency_ms=1500,
        success=True,
    )

    # Get SLO compliance
    compliance = get_slo_compliance("gap_analyzer")

Author: E2I Causal Analytics Team
Version: 1.0.0 (Phase 4 - G26)
"""

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from src.agents.factory import AGENT_TIER_NUMBERS

logger = logging.getLogger(__name__)


# =============================================================================
# SLO Configuration
# =============================================================================


class AgentTier(str, Enum):
    """Agent tier classification."""

    TIER_0_FOUNDATION = "tier_0"
    TIER_1_ORCHESTRATOR = "tier_1"
    TIER_2_CAUSAL = "tier_2"
    TIER_3_MONITORING = "tier_3"
    TIER_4_ML = "tier_4"
    TIER_5_LEARNING = "tier_5"


@dataclass
class SLOTarget:
    """SLO targets for a tier or agent."""

    availability_target: float  # e.g., 0.999 for 99.9%
    latency_p50_ms: float  # Target p50 latency
    latency_p99_ms: float  # Target p99 latency
    error_rate_target: float  # Max error rate (e.g., 0.001 for 0.1%)
    description: str = ""


# Default SLO targets per tier
DEFAULT_SLO_TARGETS: Dict[AgentTier, SLOTarget] = {
    AgentTier.TIER_0_FOUNDATION: SLOTarget(
        availability_target=0.999,
        latency_p50_ms=1000,
        latency_p99_ms=5000,
        error_rate_target=0.001,
        description="ML Foundation agents - fast, reliable data/feature operations",
    ),
    AgentTier.TIER_1_ORCHESTRATOR: SLOTarget(
        availability_target=0.999,
        latency_p50_ms=2000,
        latency_p99_ms=10000,
        error_rate_target=0.001,
        description="Orchestrator - query routing and coordination",
    ),
    AgentTier.TIER_2_CAUSAL: SLOTarget(
        availability_target=0.995,
        latency_p50_ms=5000,
        latency_p99_ms=30000,
        error_rate_target=0.005,
        description="Causal agents - complex causal inference",
    ),
    AgentTier.TIER_3_MONITORING: SLOTarget(
        availability_target=0.995,
        latency_p50_ms=3000,
        latency_p99_ms=15000,
        error_rate_target=0.005,
        description="Monitoring agents - drift, health, experiments",
    ),
    AgentTier.TIER_4_ML: SLOTarget(
        availability_target=0.99,
        latency_p50_ms=10000,
        latency_p99_ms=60000,
        error_rate_target=0.01,
        description="ML prediction agents - model inference",
    ),
    AgentTier.TIER_5_LEARNING: SLOTarget(
        availability_target=0.99,
        latency_p50_ms=5000,
        latency_p99_ms=30000,
        error_rate_target=0.01,
        description="Learning agents - explanations and feedback",
    ),
}


class UnknownAgentError(KeyError):
    """The agent name is not in :data:`~src.agents.factory.AGENT_REGISTRY_CONFIG`.

    Subclasses ``KeyError`` because that is what the lookup it replaces would
    have raised without its default, so ``except KeyError`` still catches it.
    """


#: ``{tier NUMBER: AgentTier}``, derived from the enum rather than typed out.
#:
#: A hand-written version of exactly this table is the second probe bug behind
#: #1791: it omitted ``TIER_4_ML`` and ``TIER_5_LEARNING`` and confidently
#: reported the registry as "7 of 22 wrong". Deriving it from ``AgentTier``
#: means a new tier member cannot be forgotten here.
_TIER_BY_NUMBER: Dict[int, AgentTier] = {
    # ``AgentTier`` members are STRINGS ("tier_0" .. "tier_5") while the
    # registry stores ints, so the coercion is explicit in both directions.
    # Comparing the two without it makes all 22 agents look wrong -- the third
    # probe bug, which reported "20 of 22".
    int(member.value.split("_", 1)[1]): member
    for member in AgentTier
}


def _build_agent_tier_map() -> Dict[str, AgentTier]:
    """Project ``AGENT_REGISTRY_CONFIG``'s tier column onto :class:`AgentTier`.

    #1791: this was a hand-written literal that never agreed with the registry.
    It was short ``cohort_profiler`` and ``experiment_monitor``, and it placed
    ``cohort_constructor`` in ``TIER_2_CAUSAL`` while the registry said tier 0 --
    wrong in the very commit that introduced the map (f272766ae), so this is a
    roster that never matched its source rather than one that fell behind.

    Deriving it means the map is complete by construction: registering an agent
    updates the SLO roster, and no separate edit can be forgotten.

    Raises:
        RuntimeError: if the registry uses a tier number with no ``AgentTier``
            member. Fails closed at import rather than silently dropping the
            agent from SLO monitoring, which is the failure mode this whole
            issue is about. ``tests/unit/test_agents/test_config_roster_ssot_1779.py``
            catches it in CI first.
    """
    tier_map: Dict[str, AgentTier] = {}
    unmapped: Dict[str, int] = {}
    for agent_name, tier_number in AGENT_TIER_NUMBERS.items():
        member = _TIER_BY_NUMBER.get(tier_number)
        if member is None:
            unmapped[agent_name] = tier_number
        else:
            tier_map[agent_name] = member

    if unmapped:
        raise RuntimeError(
            f"AGENT_REGISTRY_CONFIG places {unmapped} in tier number(s) that "
            f"AgentTier does not define (it defines {sorted(_TIER_BY_NUMBER)}). "
            f"Add the missing AgentTier member and a DEFAULT_SLO_TARGETS entry for it."
        )
    return tier_map


#: Agent to tier mapping -- a projection of the registry, NOT a second roster.
#: Still a plain ``Dict[str, AgentTier]``: ``SLOMonitor.get_tier_compliance``
#: iterates it with ``.items()`` and it is re-exported from ``src/mlops/__init__``.
AGENT_TIER_MAP: Dict[str, AgentTier] = _build_agent_tier_map()


def get_agent_tier(agent_name: str) -> AgentTier:
    """Get the tier for an agent.

    Raises:
        UnknownAgentError: if ``agent_name`` is not a registered agent.

    #1791: this used to return ``AgentTier.TIER_2_CAUSAL`` for anything it did
    not recognise, which meant the map could not distinguish "I have never
    heard of this agent" from "this agent is deliberately in tier 2" -- and
    ``get_slo_target`` then handed back real TIER_2 SLO targets either way. That
    default was doing real damage while the map was incomplete: two registered
    agents were silently measured against the wrong SLO.

    Now that :data:`AGENT_TIER_MAP` is complete by construction, an unrecognised
    name can only be a typo, a component that is not an agent, or an agent
    removed from the registry. In all three cases the honest answer is "no SLO
    is defined for this", and a monitor that instead reports 99.5%-availability
    compliance against a fabricated target is worse than one that refuses --
    a confidently wrong number is the worst possible output for a monitoring
    system. So this raises rather than guessing.
    """
    try:
        return AGENT_TIER_MAP[agent_name]
    except KeyError:
        raise UnknownAgentError(
            f"{agent_name!r} is not a registered agent, so it has no SLO tier. "
            f"Known agents: {sorted(AGENT_TIER_MAP)}"
        ) from None


def get_slo_target(agent_name: str) -> SLOTarget:
    """Get SLO target for an agent."""
    tier = get_agent_tier(agent_name)
    return DEFAULT_SLO_TARGETS[tier]


# =============================================================================
# Request Record
# =============================================================================


@dataclass
class RequestRecord:
    """A single request record for SLO tracking."""

    agent_name: str
    latency_ms: float
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_type: Optional[str] = None
    trace_id: Optional[str] = None


# =============================================================================
# SLO Compliance
# =============================================================================


@dataclass
class SLOCompliance:
    """SLO compliance status for an agent."""

    agent_name: str
    tier: AgentTier
    target: SLOTarget

    # Actual metrics
    actual_availability: float
    actual_error_rate: float
    actual_p50_ms: float
    actual_p99_ms: float

    # Compliance flags
    availability_met: bool
    error_rate_met: bool
    latency_p50_met: bool
    latency_p99_met: bool
    overall_compliant: bool

    # Sample info
    sample_count: int
    window_hours: int
    period_start: datetime
    period_end: datetime

    # Error budget
    error_budget_remaining: float  # 0.0 to 1.0 (percentage remaining)
    burn_rate: float  # How fast we're consuming budget (1.0 = normal)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "tier": self.tier.value,
            "target": {
                "availability": self.target.availability_target,
                "error_rate": self.target.error_rate_target,
                "latency_p50_ms": self.target.latency_p50_ms,
                "latency_p99_ms": self.target.latency_p99_ms,
            },
            "actual": {
                "availability": round(self.actual_availability, 4),
                "error_rate": round(self.actual_error_rate, 4),
                "latency_p50_ms": round(self.actual_p50_ms, 2),
                "latency_p99_ms": round(self.actual_p99_ms, 2),
            },
            "compliance": {
                "availability_met": self.availability_met,
                "error_rate_met": self.error_rate_met,
                "latency_p50_met": self.latency_p50_met,
                "latency_p99_met": self.latency_p99_met,
                "overall_compliant": self.overall_compliant,
            },
            "error_budget": {
                "remaining": round(self.error_budget_remaining, 4),
                "burn_rate": round(self.burn_rate, 2),
            },
            "sample_count": self.sample_count,
            "window_hours": self.window_hours,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


# =============================================================================
# SLO Monitor
# =============================================================================


class SLOMonitor:
    """
    SLO monitoring for E2I agents.

    Tracks request latency and success rate per agent with rolling windows.

    Attributes:
        window_hours: Rolling window size for compliance calculation
        max_records: Maximum records per agent
    """

    def __init__(
        self,
        window_hours: int = 24,
        max_records: int = 10000,
    ):
        """
        Initialize SLO monitor.

        Args:
            window_hours: Rolling window for compliance calculation
            max_records: Maximum records to keep per agent
        """
        self._records: Dict[str, Deque[RequestRecord]] = defaultdict(
            lambda: deque(maxlen=max_records)
        )
        self._lock = threading.RLock()
        self._window_hours = window_hours
        self._max_records = max_records

        # Aggregate counters (lifetime)
        self._total_requests: Dict[str, int] = defaultdict(int)
        self._total_successes: Dict[str, int] = defaultdict(int)
        self._total_errors: Dict[str, int] = defaultdict(int)

        # Custom SLO targets (override defaults)
        self._custom_targets: Dict[str, SLOTarget] = {}

        logger.info(f"SLOMonitor initialized (window={window_hours}h, max_records={max_records})")

    def record(
        self,
        agent_name: str,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> RequestRecord:
        """
        Record a request for SLO tracking.

        Args:
            agent_name: Agent name
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            error_type: Optional error type if failed
            trace_id: Optional trace ID

        Returns:
            Created RequestRecord
        """
        record = RequestRecord(
            agent_name=agent_name,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            trace_id=trace_id,
        )

        with self._lock:
            self._records[agent_name].append(record)
            self._total_requests[agent_name] += 1

            if success:
                self._total_successes[agent_name] += 1
            else:
                self._total_errors[agent_name] += 1

        logger.debug(
            f"SLO record: agent={agent_name}, latency={latency_ms:.1f}ms, success={success}"
        )

        return record

    def set_custom_target(self, agent_name: str, target: SLOTarget) -> None:
        """Set custom SLO target for an agent."""
        with self._lock:
            self._custom_targets[agent_name] = target
            logger.info(f"Set custom SLO for {agent_name}: {target}")

    def get_target(self, agent_name: str) -> SLOTarget:
        """Get SLO target for an agent (custom or default)."""
        if agent_name in self._custom_targets:
            return self._custom_targets[agent_name]
        return get_slo_target(agent_name)

    def get_compliance(
        self,
        agent_name: str,
        window_hours: Optional[int] = None,
    ) -> SLOCompliance:
        """
        Get SLO compliance for an agent.

        Args:
            agent_name: Agent name
            window_hours: Optional override for window size

        Returns:
            SLOCompliance with metrics and compliance status
        """
        window = window_hours or self._window_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window)
        target = self.get_target(agent_name)
        tier = get_agent_tier(agent_name)

        with self._lock:
            # Get records in window
            records = [r for r in self._records[agent_name] if r.timestamp > cutoff]

        if not records:
            # No data - return compliant by default
            now = datetime.now(timezone.utc)
            return SLOCompliance(
                agent_name=agent_name,
                tier=tier,
                target=target,
                actual_availability=1.0,
                actual_error_rate=0.0,
                actual_p50_ms=0.0,
                actual_p99_ms=0.0,
                availability_met=True,
                error_rate_met=True,
                latency_p50_met=True,
                latency_p99_met=True,
                overall_compliant=True,
                sample_count=0,
                window_hours=window,
                period_start=cutoff,
                period_end=now,
                error_budget_remaining=1.0,
                burn_rate=0.0,
            )

        # Calculate metrics
        successes = sum(1 for r in records if r.success)
        failures = len(records) - successes
        availability = successes / len(records)
        error_rate = failures / len(records)

        # Calculate latency percentiles
        latencies = sorted([r.latency_ms for r in records])
        n = len(latencies)
        p50_idx = int(n * 0.50)
        p99_idx = min(int(n * 0.99), n - 1)
        p50_ms = latencies[p50_idx] if n > 0 else 0.0
        p99_ms = latencies[p99_idx] if n > 0 else 0.0

        # Check compliance
        availability_met = availability >= target.availability_target
        error_rate_met = error_rate <= target.error_rate_target
        latency_p50_met = p50_ms <= target.latency_p50_ms
        latency_p99_met = p99_ms <= target.latency_p99_ms
        overall_compliant = (
            availability_met and error_rate_met and latency_p50_met and latency_p99_met
        )

        # Calculate error budget
        # Error budget = allowed errors based on target
        allowed_errors = (1 - target.availability_target) * len(records)
        actual_errors = failures

        if allowed_errors > 0:
            error_budget_remaining = max(0.0, 1 - (actual_errors / allowed_errors))
            # Burn rate: how fast are we consuming budget vs expected rate
            expected_errors = (1 - target.availability_target) * len(records)
            burn_rate = actual_errors / expected_errors if expected_errors > 0 else 0.0
        else:
            error_budget_remaining = 1.0 if failures == 0 else 0.0
            burn_rate = float(failures) if failures > 0 else 0.0

        return SLOCompliance(
            agent_name=agent_name,
            tier=tier,
            target=target,
            actual_availability=availability,
            actual_error_rate=error_rate,
            actual_p50_ms=p50_ms,
            actual_p99_ms=p99_ms,
            availability_met=availability_met,
            error_rate_met=error_rate_met,
            latency_p50_met=latency_p50_met,
            latency_p99_met=latency_p99_met,
            overall_compliant=overall_compliant,
            sample_count=len(records),
            window_hours=window,
            period_start=cutoff,
            period_end=datetime.now(timezone.utc),
            error_budget_remaining=error_budget_remaining,
            burn_rate=burn_rate,
        )

    def get_all_compliance(
        self,
        window_hours: Optional[int] = None,
    ) -> Dict[str, SLOCompliance]:
        """Get SLO compliance for all agents."""
        with self._lock:
            agent_names = list(self._records.keys())

        return {name: self.get_compliance(name, window_hours) for name in agent_names}

    def get_tier_compliance(
        self,
        tier: AgentTier,
        window_hours: Optional[int] = None,
    ) -> Dict[str, SLOCompliance]:
        """Get SLO compliance for all agents in a tier."""
        agents = [a for a, t in AGENT_TIER_MAP.items() if t == tier]

        result = {}
        for agent in agents:
            if agent in self._records:
                result[agent] = self.get_compliance(agent, window_hours)

        return result

    def get_violated_slos(
        self,
        window_hours: Optional[int] = None,
    ) -> List[SLOCompliance]:
        """Get all agents with SLO violations."""
        all_compliance = self.get_all_compliance(window_hours)
        return [c for c in all_compliance.values() if not c.overall_compliant]

    def get_metrics_for_prometheus(self) -> List[Dict[str, Any]]:
        """Get metrics formatted for Prometheus."""
        metrics = []
        all_compliance = self.get_all_compliance()

        for agent_name, compliance in all_compliance.items():
            # Availability gauge
            metrics.append(
                {
                    "name": "e2i_slo_availability",
                    "value": compliance.actual_availability,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )

            # Error rate gauge
            metrics.append(
                {
                    "name": "e2i_slo_error_rate",
                    "value": compliance.actual_error_rate,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )

            # Latency gauges
            metrics.append(
                {
                    "name": "e2i_slo_latency_p50_ms",
                    "value": compliance.actual_p50_ms,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )
            metrics.append(
                {
                    "name": "e2i_slo_latency_p99_ms",
                    "value": compliance.actual_p99_ms,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )

            # Compliance gauges (1 = met, 0 = violated)
            metrics.append(
                {
                    "name": "e2i_slo_compliant",
                    "value": 1 if compliance.overall_compliant else 0,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )

            # Error budget remaining
            metrics.append(
                {
                    "name": "e2i_slo_error_budget_remaining",
                    "value": compliance.error_budget_remaining,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )

            # Burn rate
            metrics.append(
                {
                    "name": "e2i_slo_burn_rate",
                    "value": compliance.burn_rate,
                    "labels": {"agent": agent_name, "tier": compliance.tier.value},
                    "type": "gauge",
                }
            )

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get overall SLO summary."""
        all_compliance = self.get_all_compliance()

        if not all_compliance:
            return {
                "total_agents": 0,
                "compliant_agents": 0,
                "compliance_rate": 1.0,
                "by_tier": {},
            }

        compliant = sum(1 for c in all_compliance.values() if c.overall_compliant)
        total = len(all_compliance)

        # Group by tier
        by_tier: Dict[str, Dict[str, Any]] = {}
        for tier in AgentTier:
            tier_compliance = [c for c in all_compliance.values() if c.tier == tier]
            if tier_compliance:
                tier_compliant = sum(1 for c in tier_compliance if c.overall_compliant)
                by_tier[tier.value] = {
                    "total": len(tier_compliance),
                    "compliant": tier_compliant,
                    "compliance_rate": tier_compliant / len(tier_compliance),
                    "agents": [c.agent_name for c in tier_compliance],
                }

        return {
            "total_agents": total,
            "compliant_agents": compliant,
            "compliance_rate": compliant / total if total > 0 else 1.0,
            "violated_agents": [
                c.agent_name for c in all_compliance.values() if not c.overall_compliant
            ],
            "by_tier": by_tier,
            "window_hours": self._window_hours,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_slo_monitor: Optional[SLOMonitor] = None
_slo_monitor_lock = threading.Lock()


def get_slo_monitor() -> SLOMonitor:
    """Get singleton SLO monitor instance."""
    global _slo_monitor
    if _slo_monitor is None:
        with _slo_monitor_lock:
            if _slo_monitor is None:
                _slo_monitor = SLOMonitor()
    return _slo_monitor


def reset_slo_monitor() -> None:
    """Reset the singleton monitor (mainly for testing)."""
    global _slo_monitor
    with _slo_monitor_lock:
        _slo_monitor = None


# =============================================================================
# Convenience Functions
# =============================================================================


def record_request(
    agent_name: str,
    latency_ms: float,
    success: bool,
    error_type: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> RequestRecord:
    """
    Record a request for SLO tracking.

    Args:
        agent_name: Agent name
        latency_ms: Request latency in milliseconds
        success: Whether request succeeded
        error_type: Optional error type if failed
        trace_id: Optional trace ID

    Returns:
        Created RequestRecord
    """
    return get_slo_monitor().record(
        agent_name=agent_name,
        latency_ms=latency_ms,
        success=success,
        error_type=error_type,
        trace_id=trace_id,
    )


def get_slo_compliance(
    agent_name: str,
    window_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get SLO compliance for an agent.

    Args:
        agent_name: Agent name
        window_hours: Optional window override

    Returns:
        Compliance dictionary
    """
    compliance = get_slo_monitor().get_compliance(agent_name, window_hours)
    return compliance.to_dict()


def get_all_slo_compliance(window_hours: Optional[int] = None) -> Dict[str, Any]:
    """Get SLO compliance for all agents."""
    all_compliance = get_slo_monitor().get_all_compliance(window_hours)
    return {name: c.to_dict() for name, c in all_compliance.items()}


def get_slo_summary() -> Dict[str, Any]:
    """Get overall SLO summary."""
    return get_slo_monitor().get_summary()


def get_violated_slos(window_hours: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all agents with SLO violations."""
    violations = get_slo_monitor().get_violated_slos(window_hours)
    return [v.to_dict() for v in violations]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "AgentTier",
    # Classes
    "SLOTarget",
    "RequestRecord",
    "SLOCompliance",
    "SLOMonitor",
    # Configuration
    "DEFAULT_SLO_TARGETS",
    "AGENT_TIER_MAP",
    "get_agent_tier",
    "get_slo_target",
    # Errors
    "UnknownAgentError",
    # Singleton
    "get_slo_monitor",
    "reset_slo_monitor",
    # Convenience
    "record_request",
    "get_slo_compliance",
    "get_all_slo_compliance",
    "get_slo_summary",
    "get_violated_slos",
]
