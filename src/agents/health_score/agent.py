"""
E2I Health Score Agent - Main Agent Class
Version: 4.2
Purpose: Tier 3 Fast Path Agent for system health monitoring

This agent provides:
- Quick health checks (<1s)
- Full system health assessment (<5s)
- No LLM usage - pure computation
- Dashboard-ready metrics
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .graph import build_health_score_graph, build_quick_check_graph
from .state import HealthScoreState

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT/OUTPUT CONTRACTS
# ============================================================================


class HealthScoreInput(BaseModel):
    """Input contract for Health Score agent"""

    query: str = Field(default="", description="Optional query text")
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"] = Field(
        default="full",
        description="Scope of health check to perform",
    )


class HealthScoreOutput(BaseModel):
    """Output contract for Health Score agent"""

    overall_health_score: float = Field(
        description="Overall health score (0-100)"
    )
    health_grade: str = Field(description="Letter grade (A-F)")
    component_health_score: float = Field(
        description="Component health score (0-1)"
    )
    model_health_score: float = Field(description="Model health score (0-1)")
    pipeline_health_score: float = Field(
        description="Pipeline health score (0-1)"
    )
    agent_health_score: float = Field(description="Agent health score (0-1)")
    critical_issues: List[str] = Field(
        default_factory=list, description="List of critical issues"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of warnings"
    )
    health_summary: str = Field(description="Human-readable health summary")
    check_latency_ms: int = Field(description="Total check latency in ms")
    timestamp: str = Field(description="Timestamp of health check")


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================


class HealthScoreAgent:
    """
    Tier 3 Health Score Agent

    A Fast Path agent that monitors system health with no LLM usage.
    Designed for quick dashboard updates and system monitoring.

    Usage:
        agent = HealthScoreAgent()
        result = await agent.check_health(scope="full")
        print(f"Health: {result.health_grade} ({result.overall_health_score}/100)")

    Performance targets:
        - Full check: <5s
        - Quick check: <1s
    """

    def __init__(
        self,
        health_client: Optional[Any] = None,
        metrics_store: Optional[Any] = None,
        pipeline_store: Optional[Any] = None,
        agent_registry: Optional[Any] = None,
    ):
        """
        Initialize Health Score agent.

        Args:
            health_client: Client for component health checks
            metrics_store: Store for model metrics
            pipeline_store: Store for pipeline status
            agent_registry: Registry of system agents
        """
        self.health_client = health_client
        self.metrics_store = metrics_store
        self.pipeline_store = pipeline_store
        self.agent_registry = agent_registry

        # Build graphs
        self._full_graph = build_health_score_graph(
            health_client=health_client,
            metrics_store=metrics_store,
            pipeline_store=pipeline_store,
            agent_registry=agent_registry,
        )
        self._quick_graph = build_quick_check_graph(
            health_client=health_client,
        )

        logger.info("HealthScoreAgent initialized")

    async def check_health(
        self,
        scope: Literal["full", "quick", "models", "pipelines", "agents"] = "full",
        query: str = "",
    ) -> HealthScoreOutput:
        """
        Run a health check.

        Args:
            scope: Scope of health check
            query: Optional query text

        Returns:
            HealthScoreOutput with health metrics
        """
        start_time = time.time()
        logger.info(f"Starting health check with scope: {scope}")

        # Create initial state
        initial_state: HealthScoreState = {
            "query": query,
            "check_scope": scope,
            "component_statuses": None,
            "component_health_score": None,
            "model_metrics": None,
            "model_health_score": None,
            "pipeline_statuses": None,
            "pipeline_health_score": None,
            "agent_statuses": None,
            "agent_health_score": None,
            "overall_health_score": None,
            "health_grade": None,
            "critical_issues": None,
            "warnings": None,
            "health_summary": None,
            "check_latency_ms": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "errors": [],
            "status": "pending",
        }

        try:
            # Select appropriate graph
            if scope == "quick":
                graph = self._quick_graph
            else:
                graph = self._full_graph

            # Run graph
            result = await graph.ainvoke(initial_state)

            # Build output
            output = HealthScoreOutput(
                overall_health_score=result.get("overall_health_score", 0.0),
                health_grade=result.get("health_grade", "F"),
                component_health_score=result.get("component_health_score", 0.0),
                model_health_score=result.get("model_health_score", 1.0),
                pipeline_health_score=result.get("pipeline_health_score", 1.0),
                agent_health_score=result.get("agent_health_score", 1.0),
                critical_issues=result.get("critical_issues", []),
                warnings=result.get("warnings", []),
                health_summary=result.get(
                    "health_summary", "Health check completed"
                ),
                check_latency_ms=result.get("check_latency_ms", 0),
                timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
            )

            elapsed = int((time.time() - start_time) * 1000)
            logger.info(
                f"Health check complete: grade={output.health_grade}, "
                f"score={output.overall_health_score:.1f}, latency={elapsed}ms"
            )

            return output

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            elapsed = int((time.time() - start_time) * 1000)
            return HealthScoreOutput(
                overall_health_score=0.0,
                health_grade="F",
                component_health_score=0.0,
                model_health_score=0.0,
                pipeline_health_score=0.0,
                agent_health_score=0.0,
                critical_issues=[f"Health check failed: {e}"],
                warnings=[],
                health_summary="Health check failed due to an error.",
                check_latency_ms=elapsed,
                timestamp=datetime.utcnow().isoformat(),
            )

    async def quick_check(self) -> HealthScoreOutput:
        """
        Run a quick health check (<1s target).

        Returns:
            HealthScoreOutput with component health only
        """
        return await self.check_health(scope="quick")

    async def full_check(self) -> HealthScoreOutput:
        """
        Run a full health check (<5s target).

        Returns:
            HealthScoreOutput with complete health metrics
        """
        return await self.check_health(scope="full")

    def get_handoff(self, output: HealthScoreOutput) -> Dict[str, Any]:
        """
        Generate handoff format for orchestrator.

        Args:
            output: Health score output

        Returns:
            Handoff dictionary for orchestrator
        """
        return {
            "agent": "health_score",
            "analysis_type": "system_health",
            "key_findings": {
                "overall_score": output.overall_health_score,
                "grade": output.health_grade,
                "critical_issues": len(output.critical_issues),
            },
            "component_scores": {
                "component": output.component_health_score,
                "model": output.model_health_score,
                "pipeline": output.pipeline_health_score,
                "agent": output.agent_health_score,
            },
            "issues": output.critical_issues,
            "warnings": output.warnings,
            "recommendations": self._generate_recommendations(output),
            "requires_further_analysis": output.health_grade in ["D", "F"],
            "suggested_next_agent": (
                "drift_monitor" if output.model_health_score < 0.8 else None
            ),
        }

    def _generate_recommendations(
        self, output: HealthScoreOutput
    ) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []

        if output.component_health_score < 0.8:
            recommendations.append(
                "Investigate unhealthy components and restore services"
            )

        if output.model_health_score < 0.8:
            recommendations.append(
                "Review model performance metrics and consider retraining"
            )

        if output.pipeline_health_score < 0.8:
            recommendations.append(
                "Check data pipeline freshness and resolve any failures"
            )

        if output.agent_health_score < 0.8:
            recommendations.append(
                "Verify agent availability and address any connectivity issues"
            )

        if not recommendations:
            recommendations.append("Continue monitoring - system is healthy")

        return recommendations


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def check_system_health(
    scope: Literal["full", "quick"] = "full",
    health_client: Optional[Any] = None,
    metrics_store: Optional[Any] = None,
    pipeline_store: Optional[Any] = None,
    agent_registry: Optional[Any] = None,
) -> HealthScoreOutput:
    """
    Convenience function to check system health.

    Args:
        scope: "full" or "quick" check
        health_client: Optional health check client
        metrics_store: Optional metrics store
        pipeline_store: Optional pipeline store
        agent_registry: Optional agent registry

    Returns:
        HealthScoreOutput with health metrics
    """
    agent = HealthScoreAgent(
        health_client=health_client,
        metrics_store=metrics_store,
        pipeline_store=pipeline_store,
        agent_registry=agent_registry,
    )
    return await agent.check_health(scope=scope)


def check_system_health_sync(
    scope: Literal["full", "quick"] = "full",
    **kwargs,
) -> HealthScoreOutput:
    """
    Synchronous wrapper for health check.
    """
    import asyncio

    return asyncio.run(check_system_health(scope=scope, **kwargs))
