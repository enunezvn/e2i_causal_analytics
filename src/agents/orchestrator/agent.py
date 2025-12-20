"""Orchestrator Agent - Tier 1 Coordination.

The orchestrator is the entry point for all queries. It performs:
- Fast intent classification (<500ms)
- Agent routing (<50ms)
- Parallel agent dispatch
- Response synthesis

Total orchestration overhead target: <2 seconds
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid

from .graph import create_orchestrator_graph
from .state import OrchestratorState


class OrchestratorAgent:
    """Orchestrator Agent - Central coordination hub.

    Tier: 1 (Coordination)
    Type: Standard (Fast Path)
    Latency: <2s orchestration overhead
    Critical Path: Yes - all queries pass through
    """

    # Agent metadata
    tier = 1
    tier_name = "coordination"
    agent_type = "standard"
    sla_seconds = 2  # Orchestration overhead only

    def __init__(
        self,
        agent_registry: Optional[Dict[str, Any]] = None,
        enable_checkpointing: bool = False,
    ):
        """Initialize orchestrator agent.

        Args:
            agent_registry: Optional dict mapping agent_name to agent instance
            enable_checkpointing: Whether to enable graph checkpointing
        """
        self.agent_registry = agent_registry or {}
        self.graph = create_orchestrator_graph(
            agent_registry=agent_registry, enable_checkpointing=enable_checkpointing
        )

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute orchestrator workflow.

        Args:
            input_data: Input data with query and context

        Returns:
            Orchestrator output with synthesized response

        Raises:
            ValueError: If required input fields are missing
            RuntimeError: If orchestration fails
        """
        # Validate required fields
        if "query" not in input_data:
            raise ValueError("Missing required field: query")

        # Prepare initial state
        initial_state: OrchestratorState = {
            "query": input_data["query"],
            "query_id": input_data.get("query_id", self._generate_query_id()),
            "user_id": input_data.get("user_id"),
            "session_id": input_data.get("session_id"),
            "user_context": input_data.get("user_context", {}),
            "conversation_history": input_data.get("conversation_history"),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "current_phase": "classifying",
            "status": "pending",
            "agent_results": [],
            "errors": [],
            "warnings": [],
            "fallback_used": False,
            "total_latency_ms": 0,
            "classification_latency_ms": 0,
            "routing_latency_ms": 0,
            "dispatch_latency_ms": 0,
            "synthesis_latency_ms": 0,
            "response_confidence": 0.0,
            "agents_dispatched": [],
        }

        # Execute LangGraph workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            error_msg = final_state["error"]
            error_type = final_state.get("error_type", "unknown")
            raise RuntimeError(f"{error_type}: {error_msg}")

        # Build output conforming to contract
        output = self._build_output(final_state)

        return output

    def _build_output(self, state: OrchestratorState) -> Dict[str, Any]:
        """Build output conforming to OrchestratorOutput contract.

        Args:
            state: Final orchestrator state

        Returns:
            Output data
        """
        # Collect agents used
        agents_dispatched = [r["agent_name"] for r in state.get("agent_results", [])]

        return {
            # Query identification
            "query_id": state.get("query_id"),
            # Status
            "status": state.get("status", "failed"),
            # Synthesized response
            "response_text": state.get("synthesized_response", ""),
            "response_confidence": state.get("response_confidence", 0.0),
            # Agent execution details
            "agents_dispatched": agents_dispatched,
            "agent_results": state.get("agent_results", []),
            # Metadata
            "citations": state.get("citations", []),
            "visualizations": state.get("visualizations", []),
            "follow_up_suggestions": state.get("follow_up_suggestions", []),
            "recommendations": state.get("recommendations", []),
            # Performance
            "total_latency_ms": state.get("total_latency_ms", 0),
            "timestamp": datetime.now(timezone.utc),
            # Additional metadata (not in contract but useful)
            "classification_latency_ms": state.get("classification_latency_ms", 0),
            "routing_latency_ms": state.get("routing_latency_ms", 0),
            "dispatch_latency_ms": state.get("dispatch_latency_ms", 0),
            "synthesis_latency_ms": state.get("synthesis_latency_ms", 0),
            "intent_classified": state.get("intent", {}).get("primary_intent"),
            "intent_confidence": state.get("intent", {}).get("confidence", 0.0),
        }

    def _generate_query_id(self) -> str:
        """Generate unique query ID.

        Returns:
            UUID string
        """
        return f"q-{uuid.uuid4().hex[:12]}"

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify intent of a query (for standalone use).

        Args:
            query: User query

        Returns:
            Intent classification
        """
        from .nodes import IntentClassifierNode

        classifier = IntentClassifierNode()
        result = await classifier.execute({"query": query})
        return result.get("intent", {})

    async def route_query(self, query: str) -> List[str]:
        """Route a query to agents (for standalone use).

        Args:
            query: User query

        Returns:
            List of agent names to dispatch to
        """
        from .nodes import IntentClassifierNode, RouterNode

        # Classify intent
        classifier = IntentClassifierNode()
        state_with_intent = await classifier.execute({"query": query})

        # Route to agents
        router = RouterNode()
        routed_state = await router.execute(state_with_intent)

        # Extract agent names
        dispatch_plan = routed_state.get("dispatch_plan", [])
        return [d["agent_name"] for d in dispatch_plan]

    def get_agent_registry(self) -> Dict[str, Any]:
        """Get current agent registry.

        Returns:
            Agent registry dict
        """
        return self.agent_registry

    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent for dispatch.

        Args:
            agent_name: Name of agent
            agent_instance: Agent instance (must implement analyze method)
        """
        self.agent_registry[agent_name] = agent_instance

        # Rebuild graph with updated registry
        self.graph = create_orchestrator_graph(
            agent_registry=self.agent_registry, enable_checkpointing=False
        )

    def unregister_agent(self, agent_name: str):
        """Unregister an agent.

        Args:
            agent_name: Name of agent to remove
        """
        if agent_name in self.agent_registry:
            del self.agent_registry[agent_name]

            # Rebuild graph with updated registry
            self.graph = create_orchestrator_graph(
                agent_registry=self.agent_registry, enable_checkpointing=False
            )
