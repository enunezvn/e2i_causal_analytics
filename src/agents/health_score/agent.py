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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .graph import build_health_score_graph, build_quick_check_graph
from .state import HealthScoreState

if TYPE_CHECKING:
    from .mlflow_tracker import HealthScoreMLflowTracker
    from .opik_tracer import HealthScoreOpikTracer

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

    overall_health_score: float = Field(description="Overall health score (0-100)")
    health_grade: str = Field(description="Letter grade (A-F)")
    # F1 (Codex #1): per-dimension scores are Optional — None means the dimension
    # was NOT measured (no real backend wired). They must NEVER fabricate a
    # healthy 1.0 for an unmeasured dimension.
    component_health_score: Optional[float] = Field(
        default=None, description="Component health score (0-1), None if unmeasured"
    )
    model_health_score: Optional[float] = Field(
        default=None, description="Model health score (0-1), None if unmeasured"
    )
    pipeline_health_score: Optional[float] = Field(
        default=None, description="Pipeline health score (0-1), None if unmeasured"
    )
    agent_health_score: Optional[float] = Field(
        default=None, description="Agent health score (0-1), None if unmeasured"
    )
    critical_issues: List[str] = Field(default_factory=list, description="List of critical issues")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations based on health scores"
    )
    health_summary: str = Field(description="Human-readable health summary")
    total_latency_ms: int = Field(description="Total check latency in ms")
    timestamp: str = Field(description="Timestamp of health check")

    # F1 fail-closed: provenance of the composite score so the route/dashboard
    # never presents an unmeasured score as a real measurement. "measured" (all
    # 4 dims measured), "partial" (1-3), or "unknown" (0). Defaults to "unknown"
    # so any path that forgets to set it fails closed, not open.
    data_provenance: str = Field(
        default="unknown",
        description="Provenance of the score: measured | partial | unknown",
    )

    # Contract-required fields (v4.3 fix: must be in output model for contract validation)
    errors: List[dict] = Field(default_factory=list, description="Error details from workflow")
    status: str = Field(default="completed", description="Agent execution status")


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
        enable_mlflow: bool = True,
        enable_opik: bool = True,
    ):
        """
        Initialize Health Score agent.

        Args:
            health_client: Client for component health checks
            metrics_store: Store for model metrics
            pipeline_store: Store for pipeline status
            agent_registry: Registry of system agents
            enable_mlflow: Whether to enable MLflow tracking (default: True)
            enable_opik: Whether to enable Opik distributed tracing (default: True)
        """
        self.health_client = health_client
        self.metrics_store = metrics_store
        self.pipeline_store = pipeline_store
        self.agent_registry = agent_registry
        self.enable_mlflow = enable_mlflow
        self.enable_opik = enable_opik
        self._mlflow_tracker: Optional["HealthScoreMLflowTracker"] = None
        self._opik_tracer: Optional["HealthScoreOpikTracer"] = None

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

    def _get_mlflow_tracker(self) -> Optional["HealthScoreMLflowTracker"]:
        """Get or create MLflow tracker instance (lazy initialization)."""
        if not self.enable_mlflow:
            return None

        if self._mlflow_tracker is None:
            try:
                from .mlflow_tracker import HealthScoreMLflowTracker

                self._mlflow_tracker = HealthScoreMLflowTracker()
            except ImportError:
                logger.warning("MLflow tracker not available")
                return None

        return self._mlflow_tracker

    def _get_opik_tracer(self) -> Optional["HealthScoreOpikTracer"]:
        """Get or create Opik tracer instance (lazy initialization)."""
        if not self.enable_opik:
            return None

        if self._opik_tracer is None:
            try:
                from .opik_tracer import get_health_score_tracer

                self._opik_tracer = get_health_score_tracer()
            except ImportError:
                logger.warning("Opik tracer not available")
                return None

        return self._opik_tracer

    async def check_health(
        self,
        scope: Literal["full", "quick", "models", "pipelines", "agents"] = "full",
        query: str = "",
        experiment_name: str = "default",
    ) -> HealthScoreOutput:
        """
        Run a health check.

        Args:
            scope: Scope of health check
            query: Optional query text
            experiment_name: Name of MLflow experiment (default: "default")

        Returns:
            HealthScoreOutput with health metrics
        """
        start_time = time.time()
        logger.info(f"Starting health check with scope: {scope}")

        # Create initial state
        # NotRequired fields (component/model/pipeline/agent statuses and scores)
        # are omitted - they will be populated during graph execution
        initial_state: HealthScoreState = {
            "query": query,
            "check_scope": scope,
            "overall_health_score": 0.0,
            "health_grade": "F",
            "health_summary": "",
            "total_latency_ms": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": [],
            "status": "pending",
        }

        # Get trackers
        mlflow_tracker = self._get_mlflow_tracker()
        opik_tracer = self._get_opik_tracer()

        async def execute_workflow() -> Dict[str, Any]:
            """Execute the health check workflow."""
            # Select appropriate graph
            if scope == "quick":
                graph = self._quick_graph
            else:
                graph = self._full_graph

            # Run graph
            result: Dict[str, Any] = await graph.ainvoke(initial_state)
            return result

        async def run_with_mlflow(trace_ctx=None) -> HealthScoreOutput:
            """Execute workflow with optional MLflow tracking."""
            if mlflow_tracker:
                async with mlflow_tracker.start_health_run(
                    experiment_name=experiment_name,
                    check_scope=scope,
                ):
                    result = await execute_workflow()

                    # Build output
                    # Extract errors as list of dicts (convert ErrorDetails TypedDicts)
                    raw_errors = result.get("errors", [])
                    errors = [dict(e) if hasattr(e, "keys") else e for e in raw_errors]

                    # F1: resolve per-dim scores honoring the measured flags
                    # (None == unmeasured, never a fabricated 1.0).
                    comp_s = HealthScoreAgent._resolve_dim_score(
                        result, "component_health_score", "component_health_measured"
                    )
                    model_s = HealthScoreAgent._resolve_dim_score(
                        result, "model_health_score", "model_health_measured"
                    )
                    pipe_s = HealthScoreAgent._resolve_dim_score(
                        result, "pipeline_health_score", "pipeline_health_measured"
                    )
                    agent_s = HealthScoreAgent._resolve_dim_score(
                        result, "agent_health_score", "agent_health_measured"
                    )
                    output = HealthScoreOutput(
                        overall_health_score=result.get("overall_health_score", 0.0),
                        health_grade=result.get("health_grade", "F"),
                        component_health_score=comp_s,
                        model_health_score=model_s,
                        pipeline_health_score=pipe_s,
                        agent_health_score=agent_s,
                        critical_issues=result.get("critical_issues", []),
                        warnings=result.get("warnings", []),
                        recommendations=HealthScoreAgent._recommendations_from_scores(
                            component=comp_s,
                            model=model_s,
                            pipeline=pipe_s,
                            agent=agent_s,
                        ),
                        health_summary=result.get("health_summary", "Health check completed"),
                        total_latency_ms=result.get("total_latency_ms", 0),
                        timestamp=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                        data_provenance=result.get("data_provenance", "unknown"),
                        # Contract-required fields (v4.3 fix)
                        errors=errors,
                        status=result.get("status", "completed"),
                    )

                    # Log to MLflow
                    await mlflow_tracker.log_health_result(output, result)  # type: ignore[arg-type]

                    return output
            else:
                # Execute without MLflow tracking
                result = await execute_workflow()

                # Build output
                # Extract errors as list of dicts (convert ErrorDetails TypedDicts)
                raw_errors = result.get("errors", [])
                errors = [dict(e) if hasattr(e, "keys") else e for e in raw_errors]

                # F1: resolve per-dim scores honoring the measured flags
                # (None == unmeasured, never a fabricated 1.0).
                comp_s = HealthScoreAgent._resolve_dim_score(
                    result, "component_health_score", "component_health_measured"
                )
                model_s = HealthScoreAgent._resolve_dim_score(
                    result, "model_health_score", "model_health_measured"
                )
                pipe_s = HealthScoreAgent._resolve_dim_score(
                    result, "pipeline_health_score", "pipeline_health_measured"
                )
                agent_s = HealthScoreAgent._resolve_dim_score(
                    result, "agent_health_score", "agent_health_measured"
                )
                return HealthScoreOutput(
                    overall_health_score=result.get("overall_health_score", 0.0),
                    health_grade=result.get("health_grade", "F"),
                    component_health_score=comp_s,
                    model_health_score=model_s,
                    pipeline_health_score=pipe_s,
                    agent_health_score=agent_s,
                    critical_issues=result.get("critical_issues", []),
                    warnings=result.get("warnings", []),
                    recommendations=HealthScoreAgent._recommendations_from_scores(
                        component=comp_s,
                        model=model_s,
                        pipeline=pipe_s,
                        agent=agent_s,
                    ),
                    health_summary=result.get("health_summary", "Health check completed"),
                    total_latency_ms=result.get("total_latency_ms", 0),
                    timestamp=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    data_provenance=result.get("data_provenance", "unknown"),
                    # Contract-required fields (v4.3 fix)
                    errors=errors,
                    status=result.get("status", "completed"),
                )

        try:
            # Execute with Opik tracing if available
            if opik_tracer:
                async with opik_tracer.trace_health_check(
                    check_scope=scope,
                    experiment_name=experiment_name,
                ) as trace_ctx:
                    trace_ctx.log_check_started(check_scope=scope, query=query)

                    # Run workflow (nested MLflow tracking if available)
                    output = await run_with_mlflow(trace_ctx)

                    # Log to Opik
                    elapsed = int((time.time() - start_time) * 1000)
                    trace_ctx.log_check_complete(
                        status="success",
                        success=True,
                        total_duration_ms=output.total_latency_ms,
                        overall_score=output.overall_health_score,
                        health_grade=output.health_grade,
                        component_score=output.component_health_score,
                        model_score=output.model_health_score,
                        pipeline_score=output.pipeline_health_score,
                        agent_score=output.agent_health_score,
                        critical_issues=output.critical_issues,
                        warnings=output.warnings,
                    )

                    logger.info(
                        f"Health check complete: grade={output.health_grade}, "
                        f"score={output.overall_health_score:.1f}, latency={elapsed}ms"
                    )

                    return output
            else:
                # Execute without Opik tracing
                output = await run_with_mlflow()

                elapsed = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Health check complete: grade={output.health_grade}, "
                    f"score={output.overall_health_score:.1f}, latency={elapsed}ms"
                )

                return output

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            elapsed = int((time.time() - start_time) * 1000)
            # F1: a hard failure means NOTHING was measured -> all dims None
            # (unmeasured), never a fabricated 0.0/1.0.
            return HealthScoreOutput(
                overall_health_score=0.0,
                health_grade="F",
                component_health_score=None,
                model_health_score=None,
                pipeline_health_score=None,
                agent_health_score=None,
                critical_issues=[f"Health check failed: {e}"],
                warnings=[],
                recommendations=HealthScoreAgent._recommendations_from_scores(
                    component=None,
                    model=None,
                    pipeline=None,
                    agent=None,
                ),
                health_summary="Health check failed due to an error.",
                total_latency_ms=elapsed,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data_provenance="unknown",
                # Contract-required fields (v4.3 fix)
                errors=[
                    {
                        "node": "health_check",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
                status="failed",
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
            # F1: guard against None (unmeasured) — an unmeasured model is NOT a
            # measured-degraded model, so it must not auto-route to drift_monitor.
            "suggested_next_agent": (
                "drift_monitor"
                if output.model_health_score is not None and output.model_health_score < 0.8
                else None
            ),
        }

    @staticmethod
    def _resolve_dim_score(
        result: Dict[str, Any], score_key: str, measured_key: str
    ) -> Optional[float]:
        """Resolve a per-dimension score from the workflow result.

        F1 (Codex #1): return the float ONLY when the dimension was actually
        measured (its ``<dim>_health_measured`` flag is True). Otherwise return
        ``None`` (unmeasured) — NEVER a fabricated healthy 1.0/0.0. This keeps
        the output honest end-to-end (the composer already excludes unmeasured
        dims; the OUTPUT must not re-expose them as healthy).
        """
        if not result.get(measured_key, False):
            return None
        raw = result.get(score_key)
        return float(raw) if raw is not None else None

    @staticmethod
    def _recommendations_from_scores(
        component: Optional[float] = None,
        model: Optional[float] = None,
        pipeline: Optional[float] = None,
        agent: Optional[float] = None,
    ) -> List[str]:
        """Generate recommendations from per-dimension health scores.

        F1 (Codex #1): a ``None`` dimension is UNMEASURED (no real backend
        wired). It must produce a "wire a real backend" recommendation and must
        NOT count toward the "system is healthy" message — only genuinely
        measured-and-healthy dimensions may yield that.

        Args:
            component: Component health score (0-1), or None if unmeasured
            model: Model health score (0-1), or None if unmeasured
            pipeline: Pipeline health score (0-1), or None if unmeasured
            agent: Agent health score (0-1), or None if unmeasured

        Returns:
            List of actionable recommendation strings
        """
        recommendations = []
        # Track whether any dimension is unmeasured so we never claim "healthy".
        any_unmeasured = False

        for name, score, low_msg in (
            ("component", component, "Investigate unhealthy components and restore services"),
            ("model", model, "Review model performance metrics and consider retraining"),
            ("pipeline", pipeline, "Check data pipeline freshness and resolve any failures"),
            ("agent", agent, "Verify agent availability and address any connectivity issues"),
        ):
            if score is None:
                any_unmeasured = True
                recommendations.append(
                    f"Wire a real {name} health backend — {name} health is UNMEASURED"
                )
            elif score < 0.8:
                recommendations.append(low_msg)

        # Only claim "healthy" when EVERY dimension was measured AND none flagged.
        if not recommendations and not any_unmeasured:
            recommendations.append("Continue monitoring - system is healthy")

        return recommendations

    def _generate_recommendations(self, output: HealthScoreOutput) -> List[str]:
        """Generate recommendations based on health status."""
        return self._recommendations_from_scores(
            component=output.component_health_score,
            model=output.model_health_score,
            pipeline=output.pipeline_health_score,
            agent=output.agent_health_score,
        )


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
