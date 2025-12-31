"""Router node for orchestrator agent.

Fast routing decisions based on intent classification.
No LLM calls - pure logic.

V4.4: Added discovery routing to pass DAG data to discovery-aware agents.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..state import AgentDispatch, OrchestratorState


class RouterNode:
    """Fast routing decisions based on intent classification.

    No LLM calls - pure logic.

    V4.4: Added discovery routing to pass DAG data to discovery-aware agents.
    """

    # Priority mapping: critical > high > medium > low
    PRIORITY_ORDER = {"critical": 1, "high": 2, "medium": 3, "low": 4}

    # V4.4: Agents that can use discovered DAG for validation
    DISCOVERY_AWARE_AGENTS = [
        "causal_impact",
        "gap_analyzer",
        "heterogeneous_optimizer",
        "experiment_designer",
    ]

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

        # V4.4: Apply discovery routing to enhance dispatch parameters
        discovery_routing_applied = False
        discovery_aware_agents: List[str] = []

        if self._should_apply_discovery_routing(state):
            dispatch_plan, discovery_aware_agents = self._enhance_with_discovery_data(
                dispatch_plan, state
            )
            discovery_routing_applied = len(discovery_aware_agents) > 0

        routing_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": parallel_groups or [[d["agent_name"] for d in dispatch_plan]],
            "routing_latency_ms": routing_time,
            "current_phase": "dispatching",
            # V4.4: Discovery routing metadata
            "discovery_routing_applied": discovery_routing_applied,
            "discovery_aware_agents": discovery_aware_agents if discovery_aware_agents else None,
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

    # ========================================================================
    # V4.4: Discovery Routing Methods
    # ========================================================================

    def _should_apply_discovery_routing(self, state: OrchestratorState) -> bool:
        """Check if discovery routing should be applied.

        Discovery routing is applied when:
        1. enable_discovery is True OR propagate_discovered_dag is True
        2. Gate decision is NOT 'reject'

        Args:
            state: Current orchestrator state

        Returns:
            True if discovery routing should be applied
        """
        # Check if discovery is enabled or DAG propagation is requested
        enable_discovery = state.get("enable_discovery", False)
        propagate_dag = state.get("propagate_discovered_dag", False)

        if not (enable_discovery or propagate_dag):
            return False

        # Check gate decision - reject means don't use DAG
        gate_decision = state.get("discovery_gate_decision")
        if gate_decision == "reject":
            return False

        return True

    def _enhance_with_discovery_data(
        self,
        dispatch_plan: List[AgentDispatch],
        state: OrchestratorState,
    ) -> tuple[List[AgentDispatch], List[str]]:
        """Enhance dispatch parameters with discovery data for discovery-aware agents.

        Args:
            dispatch_plan: Current dispatch plan
            state: Current orchestrator state with discovery data

        Returns:
            Tuple of (enhanced dispatch plan, list of agents that received DAG data)
        """
        enhanced_plan: List[AgentDispatch] = []
        discovery_aware_agents: List[str] = []

        # Extract discovery data from state
        discovery_config = state.get("discovery_config")
        dag_adjacency = state.get("discovered_dag_adjacency")
        dag_nodes = state.get("discovered_dag_nodes")
        dag_edge_types = state.get("discovered_dag_edge_types")
        gate_decision = state.get("discovery_gate_decision")
        gate_confidence = state.get("discovery_gate_confidence")

        # Check if we have DAG data to propagate
        has_dag_data = dag_adjacency is not None and dag_nodes is not None

        for dispatch in dispatch_plan:
            agent_name = dispatch.get("agent_name", "")

            # Check if this agent is discovery-aware
            if agent_name in self.DISCOVERY_AWARE_AGENTS:
                # Create enhanced parameters
                enhanced_params = dict(dispatch.get("parameters", {}))

                # Add discovery config if available
                if discovery_config:
                    enhanced_params["discovery_config"] = discovery_config

                # Add DAG data if available and propagation is enabled
                if has_dag_data and state.get("propagate_discovered_dag", True):
                    enhanced_params["discovered_dag_adjacency"] = dag_adjacency
                    enhanced_params["discovered_dag_nodes"] = dag_nodes
                    if dag_edge_types:
                        enhanced_params["discovered_dag_edge_types"] = dag_edge_types

                    # Add gate decision for validation
                    if gate_decision:
                        enhanced_params["discovery_gate_decision"] = gate_decision
                    if gate_confidence is not None:
                        enhanced_params["discovery_gate_confidence"] = gate_confidence

                    discovery_aware_agents.append(agent_name)

                # Create enhanced dispatch
                enhanced_dispatch = AgentDispatch(
                    agent_name=agent_name,
                    priority=dispatch.get("priority", "medium"),
                    parameters=enhanced_params,
                    timeout_ms=dispatch.get("timeout_ms", 30000),
                    fallback_agent=dispatch.get("fallback_agent"),
                )
                enhanced_plan.append(enhanced_dispatch)
            else:
                # Non-discovery-aware agent, keep original dispatch
                enhanced_plan.append(dispatch)

        return enhanced_plan, discovery_aware_agents


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
