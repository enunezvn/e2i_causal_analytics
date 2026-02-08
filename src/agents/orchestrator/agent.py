"""Orchestrator Agent - Tier 1 Coordination.

The orchestrator is the entry point for all queries. It performs:
- Fast intent classification (<500ms)
- Agent routing (<50ms)
- Parallel agent dispatch
- Response synthesis

Total orchestration overhead target: <2 seconds
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from .graph import create_orchestrator_graph
from .state import OrchestratorState

if TYPE_CHECKING:
    from .opik_tracer import OrchestratorOpikTracer

logger = logging.getLogger(__name__)


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
        enable_opik: bool = True,
    ):
        """Initialize orchestrator agent.

        Args:
            agent_registry: Optional dict mapping agent_name to agent instance
            enable_checkpointing: Whether to enable graph checkpointing
            enable_opik: Whether to enable Opik distributed tracing (default: True)
        """
        self.agent_registry = agent_registry or {}
        self.graph = create_orchestrator_graph(
            agent_registry=agent_registry, enable_checkpointing=enable_checkpointing
        )
        self.enable_opik = enable_opik
        self._opik_tracer: Optional["OrchestratorOpikTracer"] = None

    def _get_opik_tracer(self) -> Optional["OrchestratorOpikTracer"]:
        """Get or create Opik tracer instance (lazy initialization).

        Returns:
            OrchestratorOpikTracer instance if enabled, None otherwise
        """
        if not self.enable_opik:
            return None

        if self._opik_tracer is None:
            try:
                from .opik_tracer import get_orchestrator_tracer

                self._opik_tracer = get_orchestrator_tracer()
            except ImportError:
                logger.warning("Opik tracer not available for Orchestrator")
                return None

        return self._opik_tracer

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
        start_time = time.time()

        # Validate required fields
        if "query" not in input_data:
            raise ValueError("Missing required field: query")

        # Generate query_id early for tracing
        query_id = input_data.get("query_id", self._generate_query_id())
        query = input_data["query"]
        user_id = input_data.get("user_id")
        session_id = input_data.get("session_id")

        # Prepare initial state
        initial_state: OrchestratorState = {
            "query": query,
            "query_id": query_id,
            "user_id": user_id,
            "session_id": session_id,
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
            "rag_latency_ms": 0,
            "routing_latency_ms": 0,
            "dispatch_latency_ms": 0,
            "synthesis_latency_ms": 0,
            "response_confidence": 0.0,
            "agents_dispatched": [],
        }

        # Get Opik tracer
        opik_tracer = self._get_opik_tracer()

        async def execute_and_build_output() -> Dict[str, Any]:
            """Execute workflow and build output."""
            final_state = cast(OrchestratorState, await self.graph.ainvoke(initial_state))
            return self._build_output(final_state)

        if opik_tracer:
            async with opik_tracer.trace_orchestration(
                query_id=query_id,
                query=query,
                user_id=user_id,
                session_id=session_id,
            ) as trace_ctx:
                trace_ctx.log_orchestration_started(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                )

                output = await execute_and_build_output()

                # Log orchestration completion with full details
                elapsed_ms = int((time.time() - start_time) * 1000)
                trace_ctx.log_orchestration_complete(
                    status=output.get("status", "unknown"),
                    success=output.get("status") == "completed",
                    total_duration_ms=output.get("total_latency_ms", elapsed_ms),
                    response_confidence=output.get("response_confidence", 0.0),
                    agents_dispatched=output.get("agents_dispatched", []),
                    successful_agents=output.get("successful_agents", []),
                    failed_agents=output.get("failed_agents", []),
                    has_partial_failure=output.get("has_partial_failure", False),
                    primary_intent=output.get("intent_classified"),
                    classification_latency_ms=output.get("classification_latency_ms", 0),
                    rag_latency_ms=output.get("rag_latency_ms", 0),
                    routing_latency_ms=output.get("routing_latency_ms", 0),
                    dispatch_latency_ms=output.get("dispatch_latency_ms", 0),
                    synthesis_latency_ms=output.get("synthesis_latency_ms", 0),
                    errors=output.get("failure_details", []),
                    warnings=[],
                )

                logger.info(
                    f"Orchestration complete: query_id={query_id}, "
                    f"status={output.get('status')}, latency={elapsed_ms}ms"
                )

                return output
        else:
            # Execute without Opik tracing
            output = await execute_and_build_output()

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Orchestration complete: query_id={query_id}, "
                f"status={output.get('status')}, latency={elapsed_ms}ms"
            )

            return output

    def _build_output(self, state: OrchestratorState) -> Dict[str, Any]:
        """Build output conforming to OrchestratorOutput contract.

        Includes partial failure information when some agents fail but
        others succeed. This allows callers to display partial results
        with appropriate warnings.

        Args:
            state: Final orchestrator state

        Returns:
            Output data with partial failure info if applicable
        """
        agent_results = state.get("agent_results", [])

        # Separate successful and failed agents
        successful_results = [r for r in agent_results if r.get("success")]
        failed_results = [r for r in agent_results if not r.get("success")]

        # Collect all agents that were dispatched (deduplicated, preserving order)
        # LangGraph's Annotated[List, operator.add] accumulates results across steps,
        # which can produce duplicate entries when agents are retried or re-dispatched.
        agents_dispatched = list(dict.fromkeys(r["agent_name"] for r in agent_results))
        successful_agents = list(dict.fromkeys(r["agent_name"] for r in successful_results))
        failed_agents = list(dict.fromkeys(r["agent_name"] for r in failed_results))

        # Determine status based on partial vs complete failure
        status = state.get("status", "failed")
        has_partial_failure = len(successful_results) > 0 and len(failed_results) > 0

        if has_partial_failure:
            status = "partial_success"

        # Build failure details for failed agents
        failure_details = []
        for r in failed_results:
            failure_details.append(
                {
                    "agent_name": r["agent_name"],
                    "error": r.get("error", "Unknown error"),
                    "latency_ms": r.get("latency_ms", 0),
                }
            )

        # Include orchestrator-level error if present
        orchestrator_error = state.get("error")
        orchestrator_error_type = state.get("error_type")

        return {
            # Query identification
            "query_id": state.get("query_id"),
            # Status - now includes "partial_success"
            "status": status,
            # Synthesized response (from successful agents only)
            "response_text": state.get("synthesized_response", ""),
            "response_confidence": state.get("response_confidence", 0.0),
            # Agent execution details - now with success/failure breakdown
            "agents_dispatched": agents_dispatched,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "agent_results": agent_results,
            # Partial failure info - new fields for Phase 3 enhancement
            "has_partial_failure": has_partial_failure,
            "failure_details": failure_details if failure_details else None,
            "orchestrator_error": orchestrator_error,
            "orchestrator_error_type": orchestrator_error_type,
            # RAG context
            "rag_context": state.get("rag_context"),
            # Metadata
            "citations": state.get("citations", []),
            "visualizations": state.get("visualizations", []),
            "follow_up_suggestions": state.get("follow_up_suggestions", []),
            "recommendations": state.get("recommendations", []),
            # Performance
            "total_latency_ms": state.get("total_latency_ms", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Additional metadata
            "classification_latency_ms": state.get("classification_latency_ms", 0),
            "rag_latency_ms": state.get("rag_latency_ms", 0),
            "routing_latency_ms": state.get("routing_latency_ms", 0),
            "dispatch_latency_ms": state.get("dispatch_latency_ms", 0),
            "synthesis_latency_ms": state.get("synthesis_latency_ms", 0),
            "intent_classified": (intent := state.get("intent")) and intent["primary_intent"],
            "intent_confidence": (intent := state.get("intent")) and intent["confidence"] or 0.0,
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
        intent = result.get("intent")
        return dict(intent) if intent else {}

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
        dispatch_plan = routed_state.get("dispatch_plan") or []
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
