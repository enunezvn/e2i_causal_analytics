"""Router node for orchestrator agent.

Fast routing decisions based on intent classification.
No LLM calls - pure logic.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List

from ..state import AgentDispatch, OrchestratorState


class RouterNode:
    """Fast routing decisions based on intent classification.

    No LLM calls - pure logic.
    """

    # Priority mapping: critical > high > medium > low
    PRIORITY_ORDER = {"critical": 1, "high": 2, "medium": 3, "low": 4}

    # Agent capabilities mapping
    INTENT_TO_AGENTS = {
        "causal_effect": [
            AgentDispatch(
                agent_name="causal_impact",
                priority="critical",
                parameters={"interpretation_depth": "standard"},
                timeout_ms=30000,
                fallback_agent="explainer",
            )
        ],
        "performance_gap": [
            AgentDispatch(
                agent_name="gap_analyzer",
                priority="critical",
                parameters={},
                timeout_ms=20000,
                fallback_agent=None,
            )
        ],
        "segment_analysis": [
            AgentDispatch(
                agent_name="heterogeneous_optimizer",
                priority="critical",
                parameters={},
                timeout_ms=25000,
                fallback_agent="gap_analyzer",
            )
        ],
        "experiment_design": [
            AgentDispatch(
                agent_name="experiment_designer",
                priority="critical",
                parameters={"preregistration_formality": "medium"},
                timeout_ms=60000,
                fallback_agent=None,
            )
        ],
        "prediction": [
            AgentDispatch(
                agent_name="prediction_synthesizer",
                priority="critical",
                parameters={},
                timeout_ms=15000,
                fallback_agent=None,
            )
        ],
        "resource_allocation": [
            AgentDispatch(
                agent_name="resource_optimizer",
                priority="critical",
                parameters={},
                timeout_ms=20000,
                fallback_agent=None,
            )
        ],
        "explanation": [
            AgentDispatch(
                agent_name="explainer",
                priority="critical",
                parameters={"depth": "standard"},
                timeout_ms=45000,
                fallback_agent=None,
            )
        ],
        "system_health": [
            AgentDispatch(
                agent_name="health_score",
                priority="critical",
                parameters={},
                timeout_ms=5000,
                fallback_agent=None,
            )
        ],
        "drift_check": [
            AgentDispatch(
                agent_name="drift_monitor",
                priority="critical",
                parameters={},
                timeout_ms=10000,
                fallback_agent=None,
            )
        ],
        "feedback": [
            AgentDispatch(
                agent_name="feedback_learner",
                priority="critical",
                parameters={},
                timeout_ms=30000,
                fallback_agent=None,
            )
        ],
    }

    # Multi-agent patterns for complex queries (priority: critical > high > medium > low)
    MULTI_AGENT_PATTERNS = {
        ("causal_effect", "segment_analysis"): [
            ("causal_impact", "critical"),
            ("heterogeneous_optimizer", "high"),
        ],
        ("performance_gap", "resource_allocation"): [
            ("gap_analyzer", "critical"),
            ("resource_optimizer", "high"),
        ],
        ("prediction", "explanation"): [
            ("prediction_synthesizer", "critical"),
            ("explainer", "high"),
        ],
    }

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute routing logic.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with dispatch plan
        """
        start_time = time.time()

        intent = state.get("intent")
        if not intent:
            # No intent classified, default to explainer
            return self._default_routing(state, start_time)

        dispatch_plan = []
        parallel_groups = []

        # Check for multi-agent patterns
        if intent.get("requires_multi_agent") and intent.get("secondary_intents"):
            key = (intent["primary_intent"], intent["secondary_intents"][0])
            if key in self.MULTI_AGENT_PATTERNS:
                pattern = self.MULTI_AGENT_PATTERNS[key]
                for agent_name, priority in pattern:
                    dispatch_plan.append(self._get_dispatch_for_agent(agent_name, priority))
                # Group by priority for parallel execution
                parallel_groups = self._group_by_priority(dispatch_plan)

        # Single agent dispatch
        if not dispatch_plan:
            primary = intent["primary_intent"]
            if primary in self.INTENT_TO_AGENTS:
                dispatch_plan = self.INTENT_TO_AGENTS[primary]
            else:
                # Default to explainer for general queries
                dispatch_plan = [
                    AgentDispatch(
                        agent_name="explainer",
                        priority="medium",
                        parameters={"depth": "minimal"},
                        timeout_ms=30000,
                        fallback_agent=None,
                    )
                ]

        routing_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": parallel_groups or [[d["agent_name"] for d in dispatch_plan]],
            "routing_latency_ms": routing_time,
            "current_phase": "dispatching",
        }

    def _default_routing(self, state: OrchestratorState, start_time: float) -> OrchestratorState:
        """Default routing when intent classification fails.

        Args:
            state: Current state
            start_time: Routing start time

        Returns:
            Updated state with default dispatch plan
        """
        dispatch_plan = [
            AgentDispatch(
                agent_name="explainer",
                priority="medium",
                parameters={"depth": "minimal"},
                timeout_ms=30000,
                fallback_agent=None,
            )
        ]

        routing_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": [["explainer"]],
            "routing_latency_ms": routing_time,
            "current_phase": "dispatching",
        }

    def _get_dispatch_for_agent(self, agent_name: str, priority: str) -> AgentDispatch:
        """Get dispatch config for a specific agent.

        Args:
            agent_name: Name of agent
            priority: Priority level ("critical", "high", "medium", "low")

        Returns:
            Agent dispatch configuration
        """
        for intent_agents in self.INTENT_TO_AGENTS.values():
            for dispatch in intent_agents:
                if dispatch["agent_name"] == agent_name:
                    return AgentDispatch(**{**dispatch, "priority": priority})

        # Default dispatch
        return AgentDispatch(
            agent_name=agent_name,
            priority=priority,
            parameters={},
            timeout_ms=30000,
            fallback_agent=None,
        )

    def _group_by_priority(self, dispatches: List[AgentDispatch]) -> List[List[str]]:
        """Group agents by priority for parallel execution.

        Args:
            dispatches: List of dispatch configurations

        Returns:
            List of agent groups by priority (critical first, then high, medium, low)
        """
        groups = defaultdict(list)
        for d in dispatches:
            groups[d["priority"]].append(d["agent_name"])
        # Sort by priority order: critical=1, high=2, medium=3, low=4
        return [groups[p] for p in sorted(groups.keys(), key=lambda x: self.PRIORITY_ORDER.get(x, 99))]


# Export for use in graph
async def route_to_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for routing.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    router = RouterNode()
    return await router.execute(state)
