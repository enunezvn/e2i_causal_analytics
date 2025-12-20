"""Dispatcher node for orchestrator agent.

Parallel agent dispatch with timeout handling.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

from ..state import OrchestratorState, AgentResult, AgentDispatch


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

        dispatch_plan = state.get("dispatch_plan", [])
        parallel_groups = state.get("parallel_groups", [])
        all_results = []

        # Execute each parallel group sequentially
        for group in parallel_groups:
            group_dispatches = [d for d in dispatch_plan if d["agent_name"] in group]

            # Run agents in parallel within group
            tasks = [self._dispatch_agent(d, state) for d in group_dispatches]

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for dispatch, result in zip(group_dispatches, group_results):
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
                    if dispatch.get("fallback_agent"):
                        fallback_result = await self._dispatch_fallback(
                            dispatch["fallback_agent"], state
                        )
                        all_results.append(fallback_result)
                elif isinstance(result, dict) and not result.get("success", True):
                    # AgentResult returned with success=False
                    all_results.append(result)

                    # Try fallback if available
                    if dispatch.get("fallback_agent"):
                        fallback_result = await self._dispatch_fallback(
                            dispatch["fallback_agent"], state
                        )
                        all_results.append(fallback_result)
                else:
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

        Args:
            dispatch: Dispatch configuration
            state: Current state

        Returns:
            Agent result
        """
        agent_name = dispatch["agent_name"]
        start_time = time.time()

        # Mock implementation: simulate agent execution
        # In production, this would call actual agent instances
        if agent_name not in self.agents:
            # Simulate agent execution with mock data
            return await self._mock_agent_execution(dispatch, state)

        agent = self.agents[agent_name]
        timeout_ms = dispatch["timeout_ms"]

        try:
            # Prepare agent input
            agent_input = self._prepare_agent_input(state, dispatch)

            # Execute with timeout
            result = await asyncio.wait_for(
                agent.analyze(agent_input), timeout=timeout_ms / 1000
            )

            latency = int((time.time() - start_time) * 1000)

            return AgentResult(
                agent_name=agent_name,
                success=True,
                result=result,
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
            Agent input data
        """
        return {
            "query": state.get("query"),
            "user_context": state.get("user_context", {}),
            "parameters": dispatch.get("parameters", {}),
        }

    async def _dispatch_fallback(
        self, agent_name: str, state: OrchestratorState
    ) -> AgentResult:
        """Dispatch to fallback agent.

        Args:
            agent_name: Fallback agent name
            state: Current state

        Returns:
            Fallback agent result
        """
        fallback_dispatch = AgentDispatch(
            agent_name=agent_name,
            priority=99,
            parameters={},
            timeout_ms=30000,
            fallback_agent=None,
        )
        return await self._dispatch_agent(fallback_dispatch, state)


# Export for use in graph
async def dispatch_to_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for agent dispatch.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    dispatcher = DispatcherNode()
    return await dispatcher.execute(state)
