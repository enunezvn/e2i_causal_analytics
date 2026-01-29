"""
E2I Health Score Agent - Agent Health Node
Version: 4.2
Purpose: Check health of other agents in the system
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Protocol

from ..metrics import DEFAULT_THRESHOLDS
from ..state import AgentStatus, HealthScoreState

logger = logging.getLogger(__name__)


class AgentRegistry(Protocol):
    """Protocol for agent registry"""

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get list of all registered agents"""
        ...

    async def get_agent_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Get metrics for a specific agent"""
        ...


class AgentHealthNode:
    """
    Check health of other agents in the system.
    Monitors availability, success rates, and latency.
    """

    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        min_success_rate: float = DEFAULT_THRESHOLDS.min_success_rate,
        max_avg_latency_ms: int = DEFAULT_THRESHOLDS.max_avg_latency_ms,
    ):
        """
        Initialize agent health node.

        Args:
            agent_registry: Registry of system agents
            min_success_rate: Minimum acceptable success rate
            max_avg_latency_ms: Maximum acceptable average latency
        """
        self.agent_registry = agent_registry
        self.min_success_rate = min_success_rate
        self.max_avg_latency_ms = max_avg_latency_ms

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Execute agent health checks."""
        start_time = time.time()

        # Skip if scope doesn't include agents
        if state.get("check_scope") not in ["full", "agents"]:
            logger.debug("Skipping agent health for non-agent scope")
            return {
                **state,
                "agent_statuses": [],
                "agent_health_score": 1.0,
            }

        try:
            if self.agent_registry:
                # Fetch all agents
                agents = await self.agent_registry.get_all_agents()

                # Fetch metrics for each agent in parallel
                if agents:
                    tasks = [self._get_agent_status(agent) for agent in agents]
                    statuses = await asyncio.gather(*tasks)
                else:
                    statuses = []
            else:
                # No registry - return empty for testing
                statuses = []

            # Calculate overall agent health
            if statuses:
                available_count = sum(1 for s in statuses if s["available"])
                high_success_count = sum(
                    1
                    for s in statuses
                    if s["available"] and s["success_rate"] >= self.min_success_rate
                )
                # Available with good success rate = 1.0
                # Available with low success rate = 0.5
                # Unavailable = 0.0
                score_sum = high_success_count + ((available_count - high_success_count) * 0.5)
                health_score = score_sum / len(statuses)
            else:
                health_score = 1.0  # No agents tracked = healthy by default

            check_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Agent health check complete: {len(statuses)} agents, "
                f"score={health_score:.2f}, duration={check_time}ms"
            )

            return {
                **state,
                "agent_statuses": statuses,
                "agent_health_score": health_score,
                "total_latency_ms": state.get("total_latency_ms", 0) + check_time,
            }

        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            return {
                **state,
                "errors": [{"node": "agent_health", "error": str(e)}],
                "agent_health_score": 0.5,  # Unknown = degraded
                "agent_statuses": [],
            }

    async def _get_agent_status(self, agent: Dict[str, Any]) -> AgentStatus:
        """Get status for a single agent."""
        agent_name = agent.get("name", "unknown")

        try:
            metrics = await self.agent_registry.get_agent_metrics(agent_name)

            return AgentStatus(
                agent_name=agent_name,
                tier=agent.get("tier", 0),
                available=metrics.get("available", False),
                avg_latency_ms=metrics.get("avg_latency_ms", 0),
                success_rate=metrics.get("success_rate", 0.0),
                last_invocation=metrics.get("last_invocation", ""),
            )

        except Exception as e:
            logger.warning(f"Failed to get status for agent {agent_name}: {e}")
            return AgentStatus(
                agent_name=agent_name,
                tier=agent.get("tier", 0),
                available=False,
                avg_latency_ms=0,
                success_rate=0.0,
                last_invocation="",
            )
