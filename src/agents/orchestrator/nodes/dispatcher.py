"""Dispatcher node for orchestrator agent.

Parallel agent dispatch with timeout handling.
"""

import asyncio
import functools
import importlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from .._agent_method_map import AgentMethodSpec, get_method_spec
from ..state import AgentDispatch, AgentResult, OrchestratorState

logger = logging.getLogger(__name__)


def _generate_dispatch_id() -> str:
    """Generate unique dispatch identifier."""
    return f"disp_{uuid.uuid4().hex[:16]}"


def _generate_span_id() -> str:
    """Generate unique span identifier for observability."""
    return f"span_{uuid.uuid4().hex[:16]}"


class DispatcherNode:
    """Parallel agent dispatch with timeout handling."""

    def __init__(self, agent_registry: Optional[Dict[str, Any]] = None):
        """Initialize dispatcher with agent registry.

        Args:
            agent_registry: Dict mapping agent_name to agent instance
        """
        self.agents = agent_registry or {}

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute agent dispatch.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with agent results
        """
        start_time = time.time()

        dispatch_plan = state.get("dispatch_plan") or []
        parallel_groups = state.get("parallel_groups") or []
        all_results: List[AgentResult] = []

        # Execute each parallel group sequentially
        for group in parallel_groups:
            group_dispatches = [d for d in dispatch_plan if d["agent_name"] in group]

            # Run agents in parallel within group
            tasks = [self._dispatch_agent(d, state) for d in group_dispatches]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for dispatch, result in zip(group_dispatches, group_results, strict=False):
                if isinstance(result, Exception):
                    # Handle unexpected exceptions from asyncio.gather
                    failed_result = AgentResult(
                        agent_name=dispatch["agent_name"],
                        success=False,
                        result=None,
                        error=str(result),
                        latency_ms=0,
                    )
                    all_results.append(failed_result)

                    # Try fallback if available
                    fallback_agent = dispatch.get("fallback_agent")
                    if fallback_agent:
                        fallback_result = await self._dispatch_fallback(str(fallback_agent), state)
                        all_results.append(fallback_result)
                elif isinstance(result, dict) and not result.get("success", True):
                    # AgentResult returned with success=False
                    all_results.append(result)  # type: ignore[arg-type]

                    # Try fallback if available
                    fallback_agent2 = dispatch.get("fallback_agent")
                    if fallback_agent2:
                        fallback_result = await self._dispatch_fallback(str(fallback_agent2), state)
                        all_results.append(fallback_result)
                else:
                    # Result is AgentResult (TypedDict cannot use isinstance, check dict)
                    if isinstance(result, dict) and "agent_name" in result:
                        all_results.append(result)

        dispatch_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "agent_results": all_results,
            "dispatch_latency_ms": dispatch_time,
            "current_phase": "synthesizing",
        }

    async def _dispatch_agent(
        self, dispatch: AgentDispatch, state: OrchestratorState
    ) -> AgentResult:
        """Dispatch to a single agent with timeout.

        Real agents are reached via the per-agent dispatch spec in
        ``AGENT_METHOD_MAP`` (method name, async vs sync, kwargs splat, optional
        Pydantic input model). Mock execution is only used when the agent name
        is absent from the registry — never as a silent fallback when the
        registered agent exists but is missing the configured method.
        """
        agent_name = dispatch["agent_name"]
        start_time = time.time()

        # Mock implementation when no registry entry exists (used by unit
        # tests that exercise routing without instantiating real agents).
        if agent_name not in self.agents:
            return await self._mock_agent_execution(dispatch, state)

        agent = self.agents[agent_name]
        timeout_ms = dispatch["timeout_ms"]
        spec = get_method_spec(agent_name)

        try:
            agent_input = self._prepare_agent_input(state, dispatch)

            # Wrap input in a Pydantic / dataclass model when the agent expects
            # one (e.g. DriftMonitorInput, ExperimentMonitorInput).
            if spec.input_model and spec.input_module:
                try:
                    input_module = importlib.import_module(spec.input_module)
                    input_cls = getattr(input_module, spec.input_model)
                    agent_input = input_cls(**agent_input)
                except (ImportError, AttributeError, TypeError) as e:
                    latency = int((time.time() - start_time) * 1000)
                    return AgentResult(
                        agent_name=agent_name,
                        success=False,
                        result=None,
                        error=f"Failed to build {spec.input_model}: {e}",
                        latency_ms=latency,
                    )

            method = getattr(agent, spec.method, None)
            if method is None:
                latency = int((time.time() - start_time) * 1000)
                return AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result=None,
                    error=(
                        f"Agent '{agent_name}' is registered but has no "
                        f"method '{spec.method}'. Check AGENT_METHOD_MAP."
                    ),
                    latency_ms=latency,
                )

            timeout_seconds = timeout_ms / 1000

            if spec.is_async:
                if spec.uses_kwargs:
                    coro = method(**agent_input)
                else:
                    coro = method(agent_input)
                raw_result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            else:
                loop = asyncio.get_event_loop()
                if spec.uses_kwargs:
                    call = functools.partial(method, **agent_input)
                else:
                    call = functools.partial(method, agent_input)
                raw_result = await asyncio.wait_for(
                    loop.run_in_executor(None, call), timeout=timeout_seconds
                )

            latency = int((time.time() - start_time) * 1000)
            return AgentResult(
                agent_name=agent_name,
                success=True,
                result=_normalize_agent_result(raw_result),
                error=None,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=f"Agent timed out after {timeout_ms}ms",
                latency_ms=timeout_ms,
            )
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency,
            )

    async def _mock_agent_execution(
        self, dispatch: AgentDispatch, state: OrchestratorState
    ) -> AgentResult:
        """Mock agent execution for testing.

        Args:
            dispatch: Dispatch configuration
            state: Current state

        Returns:
            Mock agent result
        """
        agent_name = dispatch["agent_name"]

        # Simulate processing time
        await asyncio.sleep(0.05)  # 50ms

        # Mock responses by agent type
        mock_responses = {
            "causal_impact": {
                "narrative": "Analysis shows that HCP engagement has a significant positive effect on patient conversion (ATE=0.12, p<0.01).",
                "recommendations": [
                    "Increase HCP engagement in oncology segment",
                    "Focus on high-potential HCPs",
                ],
                "confidence": 0.87,
            },
            "gap_analyzer": {
                "narrative": "Identified 3 key gaps with combined ROI potential of $2.5M: underperforming regions, undertreated patients, and suboptimal messaging.",
                "recommendations": [
                    "Expand coverage in Northeast region",
                    "Increase patient identification initiatives",
                ],
                "confidence": 0.82,
            },
            "heterogeneous_optimizer": {
                "narrative": "Segment-level analysis reveals heterogeneous treatment effects. Oncology specialists show 2x higher response rate compared to general practitioners.",
                "recommendations": [
                    "Differentiate strategies by HCP specialty",
                    "Allocate more resources to oncology segment",
                ],
                "confidence": 0.79,
            },
            "prediction_synthesizer": {
                "narrative": "Forecast indicates 15% increase in conversions over next quarter, driven by recent HCP engagement initiatives.",
                "recommendations": [
                    "Maintain current engagement levels",
                    "Monitor conversion trends weekly",
                ],
                "confidence": 0.75,
            },
            "explainer": {
                "narrative": f"Based on the query '{state.get('query', '')}', here's a detailed explanation of the analysis approach and findings.",
                "recommendations": ["Review additional metrics", "Compare with benchmarks"],
                "confidence": 0.70,
            },
            "resource_optimizer": {
                "narrative": "Optimal resource allocation suggests reallocating 20% of budget from low-ROI channels to high-performing HCP engagement.",
                "recommendations": [
                    "Reallocate budget to top-performing channels",
                    "Monitor ROI weekly",
                ],
                "confidence": 0.81,
            },
            "health_score": {
                "narrative": "System health is nominal. All models performing within expected ranges. No critical issues detected.",
                "recommendations": ["Continue monitoring", "Schedule quarterly review"],
                "confidence": 0.95,
            },
            "drift_monitor": {
                "narrative": "Slight data drift detected in HCP engagement patterns (0.05 Jensen-Shannon divergence). Within acceptable thresholds.",
                "recommendations": [
                    "Monitor drift trends",
                    "Consider retraining in 2 months",
                ],
                "confidence": 0.88,
            },
            "experiment_designer": {
                "narrative": "Designed A/B test for HCP engagement strategy. Required sample size: 500 HCPs per arm. Expected runtime: 8 weeks.",
                "recommendations": [
                    "Preregister experiment",
                    "Set up monitoring dashboard",
                ],
                "confidence": 0.83,
            },
            "feedback_learner": {
                "narrative": "Analyzed feedback from previous campaigns. Key learning: personalized messaging increases engagement by 25%.",
                "recommendations": [
                    "Implement personalization in next campaign",
                    "Track engagement metrics",
                ],
                "confidence": 0.76,
            },
            "cohort_constructor": {
                "narrative": "Cohort construction complete. Applied inclusion/exclusion criteria to patient population.",
                "recommendations": [
                    "Review eligibility log for detailed filtering breakdown",
                    "Monitor cohort size against SLA thresholds",
                ],
                "confidence": 0.92,
                "eligible_count": 150,
                "total_input": 500,
                "eligibility_rate": 0.30,
            },
        }

        # Get mock response or default
        mock_result = mock_responses.get(
            agent_name,
            {
                "narrative": f"Mock response from {agent_name} agent.",
                "recommendations": ["Follow up with additional analysis"],
                "confidence": 0.70,
            },
        )

        return AgentResult(
            agent_name=agent_name,
            success=True,
            result=mock_result,
            error=None,
            latency_ms=50,
        )

    def _prepare_agent_input(
        self, state: OrchestratorState, dispatch: AgentDispatch
    ) -> Dict[str, Any]:
        """Prepare input for specific agent.

        Args:
            state: Current state
            dispatch: Dispatch configuration

        Returns:
            Agent input data with contract-required pass-through fields
        """
        # Generate dispatch_id if not already set
        dispatch_id = dispatch.get("dispatch_id") or _generate_dispatch_id()

        # Generate span_id for observability
        span_id = _generate_span_id()

        return {
            "query": state.get("query"),
            "user_context": state.get("user_context", {}),
            "parameters": dispatch.get("parameters", {}),
            # Contract: BaseAgentState pass-through fields
            "session_id": state.get("session_id"),
            "parsed_query": state.get("parsed_query"),
            # Contract: Orchestrator dispatch fields
            "dispatch_id": dispatch_id,
            "span_id": span_id,
            "execution_mode": dispatch.get("execution_mode", "sequential"),
        }

    async def _dispatch_fallback(self, agent_name: str, state: OrchestratorState) -> AgentResult:
        """Dispatch to fallback agent.

        Args:
            agent_name: Fallback agent name
            state: Current state

        Returns:
            Fallback agent result
        """
        fallback_dispatch = AgentDispatch(
            agent_name=agent_name,
            priority="low",  # Contract: Literal priority type
            parameters={},
            timeout_ms=30000,
            fallback_agent=None,
        )
        return await self._dispatch_agent(fallback_dispatch, state)


def _normalize_agent_result(raw: Any) -> Dict[str, Any]:
    """Coerce an agent's return value to the dict shape AgentResult expects.

    Agents return one of: a TypedDict (already a dict), a dataclass output
    object (e.g. ExperimentMonitorOutput, DriftMonitorOutput), or a plain
    string. ``isinstance(raw, dict)`` short-circuits the TypedDict case;
    dataclasses are flattened via ``__dict__``; anything else is wrapped.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return cast(Dict[str, Any], raw)
    if hasattr(raw, "to_dict") and callable(getattr(raw, "to_dict")):
        try:
            result = raw.to_dict()
            if isinstance(result, dict):
                return cast(Dict[str, Any], result)
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(raw, "__dict__"):
        return {k: v for k, v in vars(raw).items() if not k.startswith("_")}
    return {"raw_output": str(raw)}


# Export for use in graph
async def dispatch_to_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for agent dispatch.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    dispatcher = DispatcherNode()
    result = await dispatcher.execute(cast(OrchestratorState, state))
    return cast(Dict[str, Any], result)
