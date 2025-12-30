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

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import UUID

from src.mlops.opik_connector import get_opik_connector
from src.utils.audit_chain import (
    AgentTier,
    AuditChainEntry,
    AuditChainService,
    ChainVerificationResult,
    RefutationResults,
)

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

    async def verify_audit_workflow(
        self, workflow_id: UUID
    ) -> Optional[ChainVerificationResult]:
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
                entries.append(_row_to_entry(row))
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
            input_hash_data = {
                k: v for k, v in sanitized_input.items()
                if v is not None
            }

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
                                    placebo_treatment=individual.get("placebo_treatment", {}).get("passed"),
                                    random_common_cause=individual.get("random_common_cause", {}).get("passed"),
                                    data_subset=individual.get("data_subset", {}).get("passed"),
                                    unobserved_confound=individual.get("unobserved_common_cause", {}).get("passed"),
                                )

                            # Extract confidence from estimation
                            if node_name == "estimation":
                                est = result.get("estimation_result", {})
                                confidence_score = est.get("energy_score")

                            # Record the audit entry
                            service.add_entry(
                                workflow_id=workflow_id,
                                agent_name=agent_name,
                                agent_tier=agent_tier,
                                action_type=node_name,
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

                    return result

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


# Re-export key types for convenience
__all__ = [
    "AuditChainMixin",
    "AgentTier",
    "AuditChainEntry",
    "AuditChainService",
    "ChainVerificationResult",
    "RefutationResults",
    "audited_traced_node",
    "create_workflow_initializer",
    "get_audit_chain_service",
    "set_audit_chain_service",
    "init_audit_chain_service",
]
