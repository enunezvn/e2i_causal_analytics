"""Orchestrator Agent - Tier 1 Coordination.

The orchestrator is the entry point for all queries. It performs:
- Fast intent classification (<500ms)
- Agent routing (<50ms)
- Parallel agent dispatch
- Response synthesis

Total orchestration overhead target: <2 seconds
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from .graph import create_orchestrator_graph
from .memory_hooks import (
    OrchestratorMemoryHooks,
    contribute_to_memory,
    get_orchestrator_memory_hooks,
)
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

    # Hard latency budget for the pre-graph working-memory read (#883
    # read-side deferral, wired after PR #886 landed the write side). The
    # read is a single Redis LRANGE (sub-10ms healthy); the budget exists so
    # a hung/unreachable Redis can never spend the <2s orchestration-overhead
    # target waiting on context that is best-effort by contract. On
    # timeout/error the turn proceeds with NO context — fail-open, never
    # fabricated.
    MEMORY_READ_BUDGET_SECONDS = 0.5

    def __init__(
        self,
        agent_registry: Optional[Dict[str, Any]] = None,
        enable_checkpointing: bool = False,
        enable_opik: bool = True,
        enable_memory: bool = True,
        allow_mock: bool = False,
    ):
        """Initialize orchestrator agent.

        Args:
            agent_registry: Optional dict mapping agent_name to agent instance
            enable_checkpointing: Whether to enable graph checkpointing
            enable_opik: Whether to enable Opik distributed tracing (default: True)
            enable_memory: Whether to contribute completed turns to the
                tri-memory architecture (default: True). #883 PR B — the hooks
                existed since the 4-memory rollout but had no caller, leaving
                CONTRACT_VALIDATION.md §10 (memory integration, BLOCKING)
                unsatisfied and the readers get_conversation_history /
                get_routing_decisions permanently empty.
            allow_mock: TEST-ONLY. When True, a dispatch to an agent absent from
                the registry returns the canned dispatcher mock scaffold (used by
                orchestrator integration tests that exercise the graph without real
                agents). Default False makes a missing/partial registry FAIL CLOSED
                (#814); production never sets this.
        """
        self.agent_registry = agent_registry or {}
        self._allow_mock = allow_mock
        self.graph = create_orchestrator_graph(
            agent_registry=agent_registry,
            enable_checkpointing=enable_checkpointing,
            allow_mock=allow_mock,
        )
        self.enable_opik = enable_opik
        self.enable_memory = enable_memory
        self._opik_tracer: Optional["OrchestratorOpikTracer"] = None
        self._memory_hooks: Optional[OrchestratorMemoryHooks] = None

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

    @property
    def memory_hooks(self) -> Optional[OrchestratorMemoryHooks]:
        """Lazy-load memory hooks (#883 PR B, mirrors health_score #879)."""
        if self._memory_hooks is None and self.enable_memory:
            try:
                self._memory_hooks = get_orchestrator_memory_hooks()
            except Exception as e:
                logger.warning(f"Failed to initialize memory hooks: {e}")
        return self._memory_hooks

    async def _load_conversation_history(
        self,
        session_id: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Hydrate ``conversation_history`` from working memory — BUDGETED, fail-open.

        #883 read-side deferral: PR #886 wired the WRITE side (every completed
        turn persists its conversation turn), but nothing consumed the read
        back, so a second turn in the same session always started blank —
        every production call site passes ``session_id`` but never
        ``conversation_history`` (a contract-documented optional input,
        CONTRACT_VALIDATION.md §1). This closes the loop: when the caller did
        not supply history, read it back under a hard
        ``MEMORY_READ_BUDGET_SECONDS`` budget (``asyncio.wait_for``). The
        consumer is the intent classifier's LLM fallback — ambiguous
        follow-ups ("what about the other brand?") get the prior turns as
        referent context, i.e. §10.3's "session context for routing".

        Deliberately NOT wired here (intent decision, not an omission):
        episodic + semantic ``get_context`` reads would put an embedding API
        call and FalkorDB round-trips on the <2s critical path with no graph
        node consuming the result — a decorative read; and
        ``get_routing_decisions``'s documented consumer is batch DSPy routing
        optimization (AgentRoutingSignature), not per-request routing, which
        is a deterministic intent→agent map prior decisions cannot honestly
        change. Either becomes worth wiring only with a real consumer.

        Returns the stored messages, or ``None`` (no fabricated context) when
        memory is disabled, no session is keyed, nothing is stored, the read
        exceeds the budget, or it errors — each failure logged once and never
        allowed to poison the turn.
        """
        if not self.enable_memory or not session_id:
            return None
        hooks = self.memory_hooks
        if hooks is None:
            return None
        try:
            history = await asyncio.wait_for(
                hooks.get_conversation_history(session_id=session_id, limit=10),
                timeout=self.MEMORY_READ_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Conversation-history read exceeded the %.1fs budget — "
                "continuing without context (fail-open)",
                self.MEMORY_READ_BUDGET_SECONDS,
            )
            return None
        except Exception as e:
            logger.warning(f"Conversation-history read failed — continuing without context: {e}")
            return None
        return history or None

    async def _contribute_to_memory(
        self,
        output: Dict[str, Any],
        final_state: Dict[str, Any],
        session_id: Optional[str],
    ) -> None:
        """Contribute a completed orchestration turn to memory — NON-BLOCKING.

        Caller-side try/except per the settled cross-agent posture (#879 /
        causal_impact / het / experiment_monitor; the migration-046 trap
        lesson): a memory failure must NEVER poison the turn's status/errors.
        ``contribute_to_memory`` itself skips failed-status turns and covers
        all four per-turn writes in one call: working-memory cache,
        conversation turn (user+assistant messages), the episodic
        orchestration record, and the routing-decision signal for DSPy
        optimization — each individually best-effort inside the hook.

        ``brand``/``region`` are deliberately NOT derived from free-form
        ``user_context`` values: episodic ``region`` is enum-typed (the #851
        lesson — an unvalidated string would 22P02 the whole insert).
        """
        if not self.enable_memory:
            return
        try:
            memory_stats = await contribute_to_memory(
                result=output,
                state=final_state,
                memory_hooks=self.memory_hooks,
                session_id=session_id,
            )
            logger.debug(
                f"Memory contribution complete: "
                f"episodic={memory_stats.get('episodic_stored', 0)}, "
                f"conversation={memory_stats.get('conversation_stored', 0)}, "
                f"routing={memory_stats.get('routing_tracked', 0)}, "
                f"cached={memory_stats.get('working_cached', 0)}"
            )
        except Exception as e:
            logger.warning(f"Memory contribution failed (non-blocking): {e}")

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

        # #883 read-side: the caller's history wins; only when it is
        # absent/None is it hydrated from working memory (budgeted,
        # fail-open — see _load_conversation_history). An explicit [] is a
        # caller statement of "no history" and is respected as-is.
        conversation_history = input_data.get("conversation_history")
        if conversation_history is None:
            conversation_history = await self._load_conversation_history(session_id)

        # Prepare initial state
        initial_state: OrchestratorState = {
            "query": query,
            "query_id": query_id,
            "user_id": user_id,
            "session_id": session_id,
            "user_context": input_data.get("user_context", {}),
            "conversation_history": conversation_history,
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
            output = self._build_output(final_state)
            # #883 PR B: SINGLE memory-contribution site, shared by the
            # opik/non-opik branches and keyed to the graph outcome — when the
            # graph raises, no output is built and nothing is stored (there is
            # no trustworthy turn to record). Non-blocking by contract.
            await self._contribute_to_memory(output, dict(final_state), session_id)
            return output

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
                    # #1451: the dispatcher's user-facing next step, when it
                    # authored one — chat surfaces render this instead of the
                    # internal ``error`` string.
                    "user_action": r.get("user_action"),
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
            # 4-stage ClassificationPipeline (shadow/active) — None when off.
            "classification": state.get("classification"),
            "routing_pattern": state.get("routing_pattern"),
            "used_llm_layer": state.get("used_llm_layer"),
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
            agent_registry=self.agent_registry,
            enable_checkpointing=False,
            allow_mock=self._allow_mock,
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
                agent_registry=self.agent_registry,
                enable_checkpointing=False,
                allow_mock=self._allow_mock,
            )
