"""
Audit Chain Mixin for Agent Integration.

Provides reusable audit chain functionality for LangGraph agents,
creating tamper-evident logging with SHA-256 hash-linked chains.

Usage:
    class MyAgent(AuditChainMixin):
        async def run(self, state):
            workflow_id = await self.start_audit_workflow(
                agent_name="my_agent",
                agent_tier=AgentTier.CAUSAL_ANALYTICS,
                action_type="initialization",
                input_data=state,
            )
            state["audit_workflow_id"] = workflow_id
            ...

Or use the decorator:
    @audited_traced_node("estimation", agent_tier=AgentTier.CAUSAL_ANALYTICS)
    async def estimation_node(state: MyState) -> Dict[str, Any]:
        ...

Version: 4.1
Date: December 2025
"""

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, TypeVar, cast
from uuid import UUID

from src.mlops.opik_connector import get_opik_connector
from src.utils.audit_chain import (
    AgentTier,
    AuditChainEntry,
    AuditChainService,
    ChainVerificationResult,
    RefutationResults,
)
from src.utils.stage_timing import record_stage_wall_time

logger = logging.getLogger(__name__)

# Type variable for node functions
F = TypeVar("F", bound=Callable[..., Any])

# Global service instance (lazy initialization)
_audit_service: Optional[AuditChainService] = None


def get_audit_chain_service() -> Optional[AuditChainService]:
    """Get the global audit chain service instance.

    Returns None if not initialized (e.g., during testing without DB).
    """
    global _audit_service
    return _audit_service


def set_audit_chain_service(service: AuditChainService) -> None:
    """Set the global audit chain service instance."""
    global _audit_service
    _audit_service = service


def init_audit_chain_service(supabase_url: str, supabase_key: str) -> AuditChainService:
    """Initialize the global audit chain service.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase anon/service key

    Returns:
        The initialized AuditChainService
    """
    from supabase import create_client

    client = create_client(supabase_url, supabase_key)
    service = AuditChainService(client)
    set_audit_chain_service(service)
    return service


class AuditChainMixin:
    """
    Mixin class providing audit chain functionality for agents.

    Provides methods to:
    - Start a new workflow audit chain (genesis block)
    - Add entries to existing workflows
    - Complete workflows with final status
    - Verify workflow chain integrity

    Agents using this mixin should call start_audit_workflow at the
    beginning of their execution and store the workflow_id in state.
    """

    def _get_audit_service(self) -> Optional[AuditChainService]:
        """Get the audit chain service, or None if not available."""
        return get_audit_chain_service()

    async def start_audit_workflow(
        self,
        agent_name: str,
        agent_tier: AgentTier,
        action_type: str,
        input_data: Optional[Any] = None,
        user_id: Optional[str] = None,
        session_id: Optional[UUID] = None,
        query_text: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> Optional[UUID]:
        """
        Start a new audit workflow chain (genesis block).

        Args:
            agent_name: Name of the agent starting the workflow
            agent_tier: Tier classification of the agent
            action_type: Type of initial action (e.g., "initialization")
            input_data: Optional input data to hash
            user_id: User who triggered the workflow
            session_id: Session reference
            query_text: Original user query
            brand: Brand context (Remibrutinib, Fabhalta, Kisqali)

        Returns:
            The workflow_id (UUID) or None if audit service unavailable
        """
        service = self._get_audit_service()
        if service is None:
            logger.debug(f"Audit chain service not available for {agent_name}")
            return None

        try:
            entry = service.start_workflow(
                agent_name=agent_name,
                agent_tier=agent_tier,
                action_type=action_type,
                input_data=input_data,
                user_id=user_id,
                session_id=session_id,
                query_text=query_text,
                brand=brand,
            )
            logger.debug(f"Started audit workflow {entry.workflow_id} for {agent_name}")
            return entry.workflow_id
        except Exception as e:
            logger.warning(f"Failed to start audit workflow for {agent_name}: {e}")
            return None

    async def add_audit_entry(
        self,
        workflow_id: UUID,
        agent_name: str,
        agent_tier: AgentTier,
        action_type: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        duration_ms: Optional[int] = None,
        validation_passed: Optional[bool] = None,
        confidence_score: Optional[float] = None,
        refutation_results: Optional[RefutationResults] = None,
    ) -> Optional[AuditChainEntry]:
        """
        Add an entry to an existing workflow chain.

        Args:
            workflow_id: ID of the workflow to add to
            agent_name: Name of the agent performing the action
            agent_tier: Tier classification of the agent
            action_type: Type of action being performed
            input_data: Optional input data to hash
            output_data: Optional output data to hash
            duration_ms: Execution time in milliseconds
            validation_passed: Whether validation tests passed
            confidence_score: Confidence level (0.0 to 1.0)
            refutation_results: DoWhy refutation test results

        Returns:
            The new AuditChainEntry or None if service unavailable
        """
        service = self._get_audit_service()
        if service is None:
            return None

        try:
            entry = service.add_entry(
                workflow_id=workflow_id,
                agent_name=agent_name,
                agent_tier=agent_tier,
                action_type=action_type,
                input_data=input_data,
                output_data=output_data,
                duration_ms=duration_ms,
                validation_passed=validation_passed,
                confidence_score=confidence_score,
                refutation_results=refutation_results,
            )
            logger.debug(f"Added audit entry {entry.entry_id} to workflow {workflow_id}")
            return entry
        except Exception as e:
            logger.warning(f"Failed to add audit entry to {workflow_id}: {e}")
            return None

    async def verify_audit_workflow(self, workflow_id: UUID) -> Optional[ChainVerificationResult]:
        """
        Verify the integrity of a workflow's audit chain.

        Args:
            workflow_id: The workflow to verify

        Returns:
            ChainVerificationResult or None if service unavailable
        """
        service = self._get_audit_service()
        if service is None:
            return None

        try:
            return service.verify_workflow(workflow_id)
        except Exception as e:
            logger.warning(f"Failed to verify workflow {workflow_id}: {e}")
            return None

    def get_workflow_entries(self, workflow_id: UUID) -> List[AuditChainEntry]:
        """
        Get all entries for a workflow chain.

        Args:
            workflow_id: The workflow to retrieve

        Returns:
            List of AuditChainEntry objects (empty if service unavailable)
        """
        service = self._get_audit_service()
        if service is None:
            return []

        try:
            # Query all entries for the workflow
            result = (
                service.db.table("audit_chain_entries")
                .select("*")
                .eq("workflow_id", str(workflow_id))
                .order("sequence_number")
                .execute()
            )

            # Convert to AuditChainEntry objects
            entries = []
            for row in result.data or []:
                entries.append(_row_to_entry(cast(Dict[str, Any], row)))
            return entries
        except Exception as e:
            logger.warning(f"Failed to get entries for workflow {workflow_id}: {e}")
            return []


def _row_to_entry(row: Dict[str, Any]) -> AuditChainEntry:
    """Convert a database row to AuditChainEntry."""
    from datetime import datetime

    return AuditChainEntry(
        entry_id=UUID(row["entry_id"]),
        workflow_id=UUID(row["workflow_id"]),
        sequence_number=row["sequence_number"],
        agent_name=row["agent_name"],
        agent_tier=row["agent_tier"],
        action_type=row["action_type"],
        created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        duration_ms=row.get("duration_ms"),
        input_hash=row.get("input_hash"),
        output_hash=row.get("output_hash"),
        validation_passed=row.get("validation_passed"),
        confidence_score=row.get("confidence_score"),
        refutation_results=row.get("refutation_results"),
        previous_entry_id=(
            UUID(row["previous_entry_id"]) if row.get("previous_entry_id") else None
        ),
        previous_hash=row.get("previous_hash"),
        entry_hash=row["entry_hash"],
        user_id=row.get("user_id"),
        session_id=UUID(row["session_id"]) if row.get("session_id") else None,
        brand=row.get("brand"),
    )


def audited_traced_node(
    node_name: str,
    agent_name: str,
    agent_tier: AgentTier,
) -> Callable[[F], F]:
    """
    Decorator combining Opik tracing with audit chain recording.

    This decorator wraps LangGraph node functions to:
    1. Create Opik trace spans for observability
    2. Record audit chain entries for tamper-evident logging
    3. Measure execution duration
    4. Track validation results and confidence scores

    The decorated function's state must have an 'audit_workflow_id' field
    for the audit chain to work. If missing, only Opik tracing occurs.

    Args:
        node_name: Name of the node (e.g., "estimation", "refutation")
        agent_name: Name of the agent (e.g., "causal_impact")
        agent_tier: Tier classification of the agent

    Returns:
        Decorated async function with tracing and audit

    Example:
        @audited_traced_node("estimation", "causal_impact", AgentTier.CAUSAL_ANALYTICS)
        async def estimation_node(state: CausalImpactState) -> Dict[str, Any]:
            # Node implementation
            return {"estimation_result": result}
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            opik = get_opik_connector()
            service = get_audit_chain_service()

            # Extract tracing context from state
            trace_id = state.get("query_id")
            parent_span_id = state.get("span_id")
            session_id = state.get("session_id")
            workflow_id = state.get("audit_workflow_id")

            # Prepare sanitized input for tracing
            sanitized_input = {
                "query": state.get("query"),
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
                "current_phase": state.get("current_phase"),
                "session_id": session_id,
            }

            # Prepare input hash for audit
            input_hash_data = {k: v for k, v in sanitized_input.items() if v is not None}

            # Metadata for tracing
            metadata = {
                "node_name": node_name,
                "agent_name": agent_name,
                "agent_tier": agent_tier.name,
                "session_id": session_id,
                "dispatch_id": state.get("dispatch_id"),
                "audit_workflow_id": str(workflow_id) if workflow_id else None,
            }

            start_time = time.time()

            async with opik.trace_agent(
                agent_name=agent_name,
                operation=node_name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                metadata=metadata,
                tags=[agent_name, node_name, "workflow_node", "audited"],
                input_data=sanitized_input,
            ) as span:
                try:
                    # Execute the actual node function
                    result = await func(state)

                    duration_ms = int((time.time() - start_time) * 1000)

                    # Extract output for tracing
                    output_summary = {
                        "current_phase": result.get("current_phase"),
                        "status": result.get("status"),
                        "has_error": bool(result.get(f"{node_name}_error")),
                    }

                    # Record audit entry if workflow exists
                    if workflow_id and service:
                        try:
                            # Extract validation info from result
                            validation_passed = None
                            confidence_score = None
                            refutation_results = None

                            # Handle refutation node specially
                            if node_name == "refutation":
                                ref = result.get("refutation_results", {})
                                validation_passed = ref.get("overall_robust")
                                individual = ref.get("individual_tests", {})
                                refutation_results = RefutationResults(
                                    placebo_treatment=individual.get("placebo_treatment", {}).get(
                                        "passed"
                                    ),
                                    random_common_cause=individual.get(
                                        "random_common_cause", {}
                                    ).get("passed"),
                                    data_subset=individual.get("data_subset", {}).get("passed"),
                                    unobserved_confound=individual.get(
                                        "unobserved_common_cause", {}
                                    ).get("passed"),
                                    # Issue #368: bootstrap is the only refutation that
                                    # runs in degraded DoWhy mode (causal_model None).
                                    # Without this kwarg it was silently dropped from
                                    # the audit chain entry persisted by add_entry().
                                    bootstrap=individual.get("bootstrap", {}).get("passed"),
                                )

                            # Extract confidence from estimation
                            if node_name == "estimation":
                                est = result.get("estimation_result", {})
                                confidence_score = est.get("energy_score")

                            # A node that caught its own failure (returned
                            # ``{node}_error`` or flipped ``status`` to failed)
                            # is recorded as ``<node>_error`` exactly like a
                            # raising node — see node_failed_closed.
                            if node_failed_closed(node_name, state, result):
                                action_type = f"{node_name}_error"
                                validation_passed = False
                                output_summary["has_error"] = True
                            else:
                                action_type = node_name

                            # Record the audit entry
                            service.add_entry(
                                workflow_id=workflow_id,
                                agent_name=agent_name,
                                agent_tier=agent_tier,
                                action_type=action_type,
                                input_data=input_hash_data,
                                output_data=output_summary,
                                duration_ms=duration_ms,
                                validation_passed=validation_passed,
                                confidence_score=confidence_score,
                                refutation_results=refutation_results,
                            )
                            logger.debug(f"Recorded audit entry for {node_name}")
                        except Exception as e:
                            logger.warning(f"Failed to record audit entry for {node_name}: {e}")

                    # Set span output
                    span.set_output(output_summary)

                    return cast(Dict[str, Any], result)

                except Exception as e:
                    # Log error to span
                    span.set_error(str(e))

                    # Record failed audit entry
                    duration_ms = int((time.time() - start_time) * 1000)
                    if workflow_id and service:
                        try:
                            service.add_entry(
                                workflow_id=workflow_id,
                                agent_name=agent_name,
                                agent_tier=agent_tier,
                                action_type=f"{node_name}_error",
                                input_data=input_hash_data,
                                output_data={"error": str(e)},
                                duration_ms=duration_ms,
                                validation_passed=False,
                            )
                        except Exception:
                            pass  # Don't fail on audit failure

                    raise

        return wrapper  # type: ignore

    return decorator


def create_workflow_initializer(
    agent_name: str,
    agent_tier: AgentTier,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a workflow initialization function for an agent.

    Returns a function that initializes audit workflow in state.

    Args:
        agent_name: Name of the agent
        agent_tier: Tier classification of the agent

    Returns:
        A sync function that adds audit_workflow_id to state

    Example:
        init_audit = create_workflow_initializer("causal_impact", AgentTier.CAUSAL_ANALYTICS)
        state = init_audit(state)  # Adds audit_workflow_id
    """

    def initializer(state: Dict[str, Any]) -> Dict[str, Any]:
        service = get_audit_chain_service()
        if service is None:
            return state

        try:
            entry = service.start_workflow(
                agent_name=agent_name,
                agent_tier=agent_tier,
                action_type="workflow_start",
                input_data={
                    "query": state.get("query"),
                    "treatment_var": state.get("treatment_var"),
                    "outcome_var": state.get("outcome_var"),
                },
                user_id=state.get("user_id"),
                session_id=state.get("session_id"),
                query_text=state.get("query"),
                brand=state.get("brand"),
            )
            return {**state, "audit_workflow_id": entry.workflow_id}
        except Exception as e:
            logger.warning(f"Failed to initialize audit workflow: {e}")
            return state

    return initializer


# The conventional result keys a node may set to report whether its own
# validation/robustness checks passed. We read these ONLY if present and never
# invent a value — a missing key means "unmeasured" (None), not a fabricated
# pass/fail. Mirrors the honest-null convention used by the analytics latency
# panel (avg_latency_ms null == unmeasured, not zero).
_VALIDATION_RESULT_KEYS = ("validation_passed", "validation_result", "validated")


def node_failed_closed(node_name: str, state: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """True when an audited node that did NOT raise still failed its run.

    Two fail-closed conventions exist in the graphs (2026-09-06):

    * the node returns a ``{node_name}_error`` key (causal_impact estimation /
      refutation, heterogeneous_optimizer hierarchical analysis, ...);
    * the node returns ``status="failed"`` (plus ``errors``/``error``) and the
      graph routes to an UNaudited error handler (resource_optimizer,
      gap_analyzer, explainer, feedback_learner, prediction_synthesizer, ...).

    Only the node that FLIPS the status to failed is the failure; a downstream
    node passing an already-failed state through (``if state["status"] ==
    "failed": return state``) did not fail, so a failed run is not spelled as
    N error rows. A ``validation_passed`` verdict on a completed node is never
    a failure. The audit row's ``output_data`` is persisted only as a hash, so
    the ``<node>_error`` action_type this drives is the one readable execution
    outcome the /system-health and /analytics readers can count
    (``src.api.utils.audit_outcomes``).
    """
    if result.get(f"{node_name}_error"):
        return True
    return result.get("status") == "failed" and state.get("status") != "failed"


def _elapsed_ms(start: float) -> int:
    """Whole-millisecond elapsed time since ``start`` (time.perf_counter), floored
    to 1 for any positive duration.

    The node DID execute, so a real sub-millisecond run should record 1ms, not 0.
    Recording 0 is indistinguishable downstream from "no measurement" — the
    analytics aggregator drops ``duration_ms`` that is falsy/<=0 and the UI treats
    0 as unmeasured — which would silently lose fast nodes from the latency panel.
    This is millisecond-resolution quantization of a REAL measurement, not a
    fabricated timing.
    """
    elapsed = time.perf_counter() - start
    ms = int(elapsed * 1000)
    if ms <= 0 and elapsed > 0:
        return 1
    return ms


def audited_node(
    func: Callable[..., Any],
    *,
    agent_name: str,
    agent_tier: AgentTier,
    node_name: str,
) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Wrap a LangGraph node so its execution emits a REAL timed audit entry.

    This is the shared, *node-agnostic* counterpart to causal_impact's inline
    ``traced_node`` (graph.py): it measures wall-clock ``duration_ms`` around the
    node, runs the node, and records an ``add_entry`` against the workflow's audit
    chain. The 11 non-causal_impact agent graphs only emitted a genesis
    ``workflow_start`` entry (no ``duration_ms``) via ``create_workflow_initializer``,
    so once they ran the analytics latency panel had nothing to average and showed
    a fake "0ms". Wrapping their nodes with ``audited_node`` makes them record real
    per-node latency.

    Behaviour contract (honest, fail-open on audit, never on the node):

    * Timing is REAL: ``duration_ms = int((perf_counter end - start) * 1000)``.
    * Audit is best-effort. If the audit service is unavailable, or the state
      carries no ``audit_workflow_id`` (the genesis ``audit_init`` node was a
      no-op because no service was configured), the node still runs and NO entry
      is fabricated.
    * ``validation_passed`` is read from a conventional result key only if the
      node actually set one; otherwise it stays ``None`` (unmeasured).
    * On exception — or when the node returns a ``{node_name}_error`` key, the
      fail-closed convention — the wrapper records a timed ``{node_name}_error``
      entry with ``validation_passed=False``; on exception it then re-raises —
      execution semantics are
      unchanged, telemetry is added.

    Accepts sync or async node callables and always returns an async node, so
    graph wiring is uniform regardless of how the underlying node is defined.

    Args:
        func: The node callable (``state -> dict`` or ``async state -> dict``).
        agent_name: Audit ``agent_name`` (e.g. "gap_analyzer").
        agent_tier: Audit ``AgentTier`` for the agent.
        node_name: Logical node name, recorded as ``action_type``.

    Returns:
        An async node ``state -> dict`` suitable for ``workflow.add_node``.
    """

    is_async = inspect.iscoroutinefunction(func)

    @functools.wraps(func)
    async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        service = get_audit_chain_service()
        workflow_id = state.get("audit_workflow_id")

        start = time.perf_counter()
        try:
            if is_async:
                result = await func(state)
            else:
                # Run sync node off the event loop so a slow node does not block
                # other concurrent agent graphs.
                result = await asyncio.to_thread(func, state)

            duration_ms = _elapsed_ms(start)
            result_dict = cast(Dict[str, Any], result if isinstance(result, dict) else {})

            if workflow_id and service is not None:
                validation_passed: Optional[bool] = None
                for key in _VALIDATION_RESULT_KEYS:
                    if key in result_dict:
                        raw = result_dict[key]
                        validation_passed = bool(raw) if raw is not None else None
                        break
                # Fail-closed node (returned ``{node}_error`` or flipped status
                # to failed instead of raising) -> recorded as ``<node>_error``
                # like a raise; see node_failed_closed.
                has_error = node_failed_closed(node_name, state, result_dict)
                action_type = f"{node_name}_error" if has_error else node_name
                if has_error:
                    validation_passed = False
                try:
                    service.add_entry(
                        workflow_id=workflow_id,
                        agent_name=agent_name,
                        agent_tier=agent_tier,
                        action_type=action_type,
                        input_data={"node": node_name},
                        output_data={
                            "status": result_dict.get("status"),
                            "has_error": has_error,
                        },
                        duration_ms=duration_ms,
                        validation_passed=validation_passed,
                    )
                except Exception as audit_err:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed to record audit entry for %s/%s: %s",
                        agent_name,
                        node_name,
                        audit_err,
                    )

            return cast(Dict[str, Any], result)

        except Exception as exc:
            duration_ms = _elapsed_ms(start)
            if workflow_id and service is not None:
                try:
                    service.add_entry(
                        workflow_id=workflow_id,
                        agent_name=agent_name,
                        agent_tier=agent_tier,
                        action_type=f"{node_name}_error",
                        input_data={"node": node_name},
                        output_data={"error": str(exc)},
                        duration_ms=duration_ms,
                        validation_passed=False,
                    )
                except Exception:  # pragma: no cover - defensive
                    pass  # never let audit failure mask the real error
            raise

        finally:
            # #1475: per-request stage attribution. No-op unless a stage
            # ledger is active in this context (the chatbot's orchestrator
            # node activates one around orchestrator.run). Recorded in
            # ``finally`` — same perf_counter start — so a failing node is
            # still attributed.
            record_stage_wall_time(
                f"{agent_name}.{node_name}", (time.perf_counter() - start) * 1000.0
            )

    # Structural marker for the #1475 coverage pin: a graph node without this
    # wrapper would silently vanish from orchestrator_stage_ms.
    wrapper.__stage_timed__ = True  # type: ignore[attr-defined]

    return wrapper


def add_audited_node(
    workflow: Any,
    name: str,
    func: Callable[..., Any],
    *,
    agent_name: str,
    agent_tier: AgentTier,
) -> None:
    """Register ``func`` on ``workflow`` as a node wrapped by :func:`audited_node`.

    Thin convenience so each agent graph can swap a bare
    ``workflow.add_node(name, func)`` for one timed call without repeating the
    wrapper boilerplate. ``name`` is used as both the LangGraph node name and the
    audit ``action_type`` (``node_name``).
    """
    workflow.add_node(
        name,
        audited_node(func, agent_name=agent_name, agent_tier=agent_tier, node_name=name),
    )


# Re-export key types for convenience
__all__ = [
    "AuditChainMixin",
    "AgentTier",
    "AuditChainEntry",
    "AuditChainService",
    "ChainVerificationResult",
    "RefutationResults",
    "audited_traced_node",
    "audited_node",
    "add_audited_node",
    "create_workflow_initializer",
    "get_audit_chain_service",
    "set_audit_chain_service",
    "init_audit_chain_service",
]
