"""LangGraph workflow for Causal Impact Agent.

Contract-compliant workflow with conditional routing:
  graph_builder → estimation → [refutation|interpretation|error] → sensitivity → interpretation → END

Contract: .claude/contracts/tier2-contracts.md
Observability:
  - Per-node Opik tracing (CONTRACT_VALIDATION.md #12)
  - MLflow experiment tracking (CONTRACT_VALIDATION.md #13)
  - Audit chain recording for tamper-evident logging
"""

import functools
import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, Literal, Optional, TypeVar

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import (
    create_workflow_initializer,
    get_audit_chain_service,
)
from src.agents.causal_impact.nodes.adjustment_set_policy import (
    apply_adjustment_set_policy,
)
from src.agents.causal_impact.nodes.estimation import estimate_causal_effect
from src.agents.causal_impact.nodes.graph_builder import build_causal_graph
from src.agents.causal_impact.nodes.interpretation import interpret_results
from src.agents.causal_impact.nodes.refutation import refute_causal_estimate
from src.agents.causal_impact.nodes.sensitivity import analyze_sensitivity
from src.agents.causal_impact.state import CausalImpactState
from src.mlops.mlflow_connector import get_mlflow_connector
from src.mlops.opik_connector import get_opik_connector
from src.utils.audit_chain import AgentTier, RefutationResults

logger = logging.getLogger(__name__)

# Type variable for node functions
F = TypeVar("F", bound=Callable[..., Any])


def traced_node(node_name: str) -> Callable[[F], F]:
    """Decorator to add Opik tracing and audit chain recording to workflow nodes.

    Creates a span for each node execution with:
    - Node name and operation tracking
    - Input/output data (sanitized for large fields)
    - Latency measurement
    - Error tracking
    - Parent span linking via state.span_id
    - Audit chain entry for tamper-evident logging

    Args:
        node_name: Name of the node (e.g., "graph_builder", "estimation")

    Returns:
        Decorated async function with Opik tracing and audit recording

    Example:
        @traced_node("graph_builder")
        async def build_causal_graph(state: CausalImpactState) -> Dict:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(state: CausalImpactState) -> Dict[str, Any]:
            opik = get_opik_connector()
            audit_service = get_audit_chain_service()

            # Extract tracing context from state
            trace_id = state.get("query_id")  # Use query_id as trace correlation
            parent_span_id = state.get("span_id")  # Parent span from dispatcher
            session_id = state.get("session_id")
            workflow_id = state.get("audit_workflow_id")

            # Prepare sanitized input (exclude large data structures)
            sanitized_input = {
                "query": state.get("query"),
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
                "current_phase": state.get("current_phase"),
                "session_id": session_id,
            }

            # Metadata for the span
            metadata = {
                "node_name": node_name,
                "agent_name": "causal_impact",
                "session_id": session_id,
                "dispatch_id": state.get("dispatch_id"),
                "audit_workflow_id": str(workflow_id) if workflow_id else None,
            }

            start_time = time.time()

            async with opik.trace_agent(
                agent_name="causal_impact",
                operation=node_name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                metadata=metadata,
                tags=["causal_impact", node_name, "workflow_node", "audited"],
                input_data=sanitized_input,
            ) as span:
                try:
                    # Execute the actual node function
                    result = await func(state)

                    # Floor a real positive elapsed time to >=1ms: recording 0 is
                    # indistinguishable downstream from "unmeasured" (analytics
                    # drops falsy duration_ms), so a fast node would vanish from
                    # the latency panel. Honest ms-resolution quantization.
                    elapsed = time.time() - start_time
                    duration_ms = max(1, int(elapsed * 1000)) if elapsed > 0 else 0

                    # Set output data (sanitized)
                    output_summary = {
                        "current_phase": result.get("current_phase"),
                        "status": result.get("status"),
                        "has_error": bool(result.get(f"{node_name}_error")),
                    }

                    # Add node-specific output fields
                    validation_passed = None
                    confidence_score = None
                    refutation_results = None

                    if node_name == "graph_builder":
                        output_summary["graph_confidence"] = result.get("causal_graph", {}).get(
                            "confidence"
                        )
                    elif node_name == "estimation":
                        est = result.get("estimation_result", {})
                        output_summary["ate"] = est.get("ate")
                        output_summary["p_value"] = est.get("p_value")
                        output_summary["statistical_significance"] = est.get(
                            "statistical_significance"
                        )
                        confidence_score = est.get("confidence")
                    elif node_name == "refutation":
                        ref = result.get("refutation_results", {})
                        output_summary["tests_passed"] = ref.get("tests_passed")
                        output_summary["overall_robust"] = ref.get("overall_robust")
                        output_summary["gate_decision"] = ref.get("gate_decision")
                        validation_passed = ref.get("overall_robust")
                        # add_entry calls refutation_results.to_dict() internally
                        # (audit_chain.py:345); the refutation node persists a
                        # dict via RefutationSuite.to_legacy_format() so we wrap
                        # it in the dataclass here. Field mapping mirrors
                        # audit_chain_mixin.audited_traced_node (:388-403).
                        # When the refutation node failed early ref/individual_tests
                        # is empty — leave refutation_results=None so the audit row
                        # distinguishes "no refutation ran" from "all tests null".
                        individual = ref.get("individual_tests", {})
                        if individual:
                            refutation_results = RefutationResults(
                                placebo_treatment=individual.get("placebo_treatment", {}).get(
                                    "passed"
                                ),
                                random_common_cause=individual.get("random_common_cause", {}).get(
                                    "passed"
                                ),
                                data_subset=individual.get("data_subset", {}).get("passed"),
                                unobserved_confound=individual.get(
                                    "unobserved_common_cause", {}
                                ).get("passed"),
                                # Issue #368: bootstrap is the only refutation that runs
                                # in degraded DoWhy mode (causal_model None). Without
                                # this kwarg it was silently dropped from the audit
                                # chain, leaving tamper-evident logging blind to the
                                # only test that actually executed.
                                bootstrap=individual.get("bootstrap", {}).get("passed"),
                            )
                    elif node_name == "sensitivity":
                        sens = result.get("sensitivity_analysis", {})
                        output_summary["e_value"] = sens.get("e_value")
                        output_summary["robust_to_confounding"] = sens.get("robust_to_confounding")
                    elif node_name == "interpretation":
                        interp = result.get("interpretation", {})
                        output_summary["causal_confidence"] = interp.get("causal_confidence")
                        output_summary["depth_level"] = interp.get("depth_level")
                        confidence_score = interp.get("causal_confidence")

                    span.set_output(output_summary)

                    # Set latency attribute from node result
                    latency_key = f"{node_name}_latency_ms"
                    if latency_key in result:
                        span.set_attribute("node_latency_ms", result[latency_key])

                    # Record audit chain entry. add_entry hashes input_data /
                    # output_data internally via AuditChainService.hash_payload;
                    # user_id / session_id / brand are inherited from the
                    # workflow's genesis entry (see audit_chain.py:348-350).
                    if workflow_id and audit_service:
                        try:
                            audit_service.add_entry(
                                workflow_id=workflow_id,
                                agent_name="causal_impact",
                                agent_tier=AgentTier.CAUSAL_ANALYTICS,
                                action_type=node_name,
                                duration_ms=duration_ms,
                                input_data=sanitized_input,
                                output_data=output_summary,
                                validation_passed=validation_passed,
                                confidence_score=confidence_score,
                                refutation_results=refutation_results,
                            )
                            logger.debug(f"Recorded audit entry for {node_name}")
                        except Exception as ae:
                            logger.warning(f"Failed to record audit entry: {ae}")

                    return result  # type: ignore[no-any-return]

                except Exception as e:
                    # Log error details to span
                    span.set_attribute("error", str(e))
                    span.set_attribute("error_type", type(e).__name__)
                    logger.error(f"Node {node_name} failed: {e}")
                    raise

        return wrapper  # type: ignore

    return decorator


# Create traced versions of node functions
traced_build_causal_graph = traced_node("graph_builder")(build_causal_graph)
traced_estimate_causal_effect = traced_node("estimation")(estimate_causal_effect)
traced_refute_causal_estimate = traced_node("refutation")(refute_causal_estimate)
traced_analyze_sensitivity = traced_node("sensitivity")(analyze_sensitivity)
traced_interpret_results = traced_node("interpretation")(interpret_results)


# ----------------------------------------------------------------------
# Phase 2 (Issue #237) — adjustment_set_policy traced wrapper
# ----------------------------------------------------------------------
#
# This wrapper deliberately diverges from ``traced_node`` above on a
# single point: it passes ``output_data=...`` to ``audit_service.add_entry``
# instead of the pre-existing ``output_hash=...`` (which is not a valid
# kwarg on the service signature; see ``src/utils/audit_chain.py:288-301``).
# Plan §2.1 + case 8 of the forcing tests pin this contract.
async def traced_apply_adjustment_policy(state: CausalImpactState) -> Dict[str, Any]:
    """Opik-traced + audit-chained wrapper for the policy node."""

    opik = get_opik_connector()
    audit_service = get_audit_chain_service()

    trace_id = state.get("query_id")
    parent_span_id = state.get("span_id")
    session_id = state.get("session_id")
    workflow_id = state.get("audit_workflow_id")

    sanitized_input = {
        "query_id": state.get("query_id"),
        "treatment_var": state.get("treatment_var"),
        "outcome_var": state.get("outcome_var"),
        "role_attributions_n": len(state.get("role_attributions") or []),
        "current_phase": state.get("current_phase"),
        "session_id": session_id,
    }

    metadata = {
        "node_name": "adjustment_set_policy",
        "agent_name": "causal_impact",
        "session_id": session_id,
        "dispatch_id": state.get("dispatch_id"),
        "audit_workflow_id": str(workflow_id) if workflow_id else None,
    }

    start_time = time.time()

    async with opik.trace_agent(
        agent_name="causal_impact",
        operation="adjustment_set_policy",
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        metadata=metadata,
        tags=["causal_impact", "adjustment_set_policy", "workflow_node", "audited"],
        input_data=sanitized_input,
    ) as span:
        try:
            result = await apply_adjustment_set_policy(state)

            # Floor positive elapsed time to >=1ms (see traced_node) so a real
            # sub-ms node is not dropped by analytics as "unmeasured".
            elapsed = time.time() - start_time
            duration_ms = max(1, int(elapsed * 1000)) if elapsed > 0 else 0

            policy_log = result.get("policy_log") or []
            cg = result.get("causal_graph") or {}
            output_payload: Dict[str, Any] = {
                "policy": os.environ.get("CAUSAL_IMPACT_ADJUSTMENT_POLICY", "OFF").upper(),
                "mutated": cg.get("adjustment_set_hash")
                != cg.get("adjustment_set_hash_pre_policy"),
                "n_dropped": sum(1 for e in policy_log if e.get("kind", "").startswith("dropped_")),
                "n_warned": sum(1 for e in policy_log if e.get("kind", "").startswith("warning_")),
                "log_was_truncated": bool(result.get("policy_log_was_truncated")),
                "adjustment_set_hash": cg.get("adjustment_set_hash"),
                "adjustment_set_hash_pre_policy": cg.get("adjustment_set_hash_pre_policy"),
                "has_error": bool(result.get("adjustment_set_policy_error")),
            }

            span.set_output(output_payload)
            span.set_attribute(
                "node_latency_ms",
                result.get("adjustment_set_policy_latency_ms", 0.0),
            )

            if workflow_id and audit_service:
                try:
                    # CORRECT KWARG: output_data= (NOT output_hash=). The
                    # pre-existing traced_node wrapper uses output_hash;
                    # this Phase 2 wrapper uses output_data so the audit
                    # service can do its own canonical hashing.
                    audit_service.add_entry(
                        workflow_id=workflow_id,
                        agent_name="causal_impact",
                        agent_tier=AgentTier.CAUSAL_ANALYTICS,
                        action_type="adjustment_set_policy",
                        duration_ms=duration_ms,
                        input_data=sanitized_input,
                        output_data=output_payload,
                    )
                    logger.debug("Recorded audit entry for adjustment_set_policy")
                except Exception as ae:
                    logger.warning(f"Failed to record audit entry: {ae}")

            return result

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_attribute("error_type", type(e).__name__)
            logger.error(f"Node adjustment_set_policy failed: {e}")
            raise


def should_continue_after_estimation(
    state: CausalImpactState,
) -> Literal["refutation", "error_handler"]:
    """Conditional routing after estimation node.

    Contract: fail-closed. A partial success (estimation_error set but an ATE was
    still produced) must STILL be validated by refutation — it must NOT skip to
    interpretation, which would surface an unvalidated estimate as if validated.
    Only a total estimation failure (no ATE) routes to the error handler.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("estimation_error"):
        # Partial success: an ATE exists but estimation flagged a problem. Do NOT
        # skip to interpretation — route through refutation so the gate validates
        # (or blocks) the estimate. Only a total failure (no ATE) errors out.
        if state.get("estimation_result", {}).get("ate") is not None:
            return "refutation"
        return "error_handler"
    return "refutation"


def should_continue_after_refutation(
    state: CausalImpactState,
) -> Literal["sensitivity", "error_handler"]:
    """Conditional routing after refutation node.

    Contract: gate_decision determines flow, fail-CLOSED (H1).

    A refutation that ERRORED or FAILED sets ``refutation_error`` /
    ``status="failed"`` but NO ``refutation_results`` — previously the gate then
    defaulted to ``"proceed"`` and carried a never-validated estimate forward to
    sensitivity → interpretation → a "completed" result. Route those to the
    error handler instead. Only PROCEED/REVIEW continue to sensitivity; BLOCK
    and any error/failure stop at the error handler.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    # Fail-closed: an errored/failed refutation must not proceed as if validated.
    if state.get("refutation_error") or state.get("status") == "failed":
        return "error_handler"
    gate = state.get("gate_decision") or state.get("refutation_results", {}).get(
        "gate_decision", "proceed"
    )
    if gate == "block":
        return "error_handler"
    return "sensitivity"


def handle_workflow_error(state: CausalImpactState) -> CausalImpactState:
    """Handle workflow errors gracefully.

    Contract: Accumulate errors and mark workflow as failed.

    Args:
        state: Current workflow state

    Returns:
        Updated state with error status
    """
    error_msg = state.get("error_message") or "Unknown error occurred"

    # Accumulate error if not already present
    errors = list(state.get("errors", []))
    errors.append({"phase": state.get("current_phase", "unknown"), "message": error_msg})

    return {
        **state,
        "status": "failed",
        "errors": errors,
        "current_phase": "failed",
    }


def create_causal_impact_graph(enable_checkpointing: bool = False):
    """Create causal impact workflow graph with conditional routing.

    Contract-compliant pipeline with error handling:
    0. audit_init: Initialize audit chain workflow (genesis block)
    1. graph_builder: Construct causal DAG (Standard, <10s)
    2. estimation: Estimate causal effect (Standard, <30s)
       → conditional: refutation | error_handler (no ATE)
    3. refutation: Robustness tests (Standard, <15s)
       → conditional: sensitivity | error_handler (if blocked)
    4. sensitivity: E-value analysis (Standard, <5s)
    5. interpretation: Natural language output (Deep Reasoning, <30s)

    Total target: <120s (60s computation + 30s interpretation)

    Args:
        enable_checkpointing: Whether to enable state checkpointing

    Returns:
        Compiled LangGraph workflow
    """
    # Create workflow
    workflow = StateGraph(CausalImpactState)

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("causal_impact", AgentTier.CAUSAL_ANALYTICS)

    # Add nodes with Opik tracing wrappers (CONTRACT_VALIDATION.md #12)
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain
    workflow.add_node("graph_builder", traced_build_causal_graph)  # type: ignore[type-var,arg-type,call-overload]
    # Phase 2 (Issue #237): collider/mediator exclusion policy sits
    # between graph_builder and estimation. Default policy OFF makes
    # this a no-op until explicitly enabled.
    workflow.add_node("adjustment_set_policy", traced_apply_adjustment_policy)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("estimation", traced_estimate_causal_effect)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("refutation", traced_refute_causal_estimate)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("sensitivity", traced_analyze_sensitivity)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("interpretation", traced_interpret_results)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("error_handler", handle_workflow_error)  # type: ignore[type-var,arg-type,call-overload]  # Error handler not traced

    # Set entry point to audit initializer
    workflow.set_entry_point("audit_init")

    # Linear edge: audit_init → graph_builder → adjustment_set_policy → estimation
    # (Phase 2 wedge — OFF by default; see nodes/adjustment_set_policy.py.)
    workflow.add_edge("audit_init", "graph_builder")
    workflow.add_edge("graph_builder", "adjustment_set_policy")
    workflow.add_edge("adjustment_set_policy", "estimation")

    # Conditional edge after estimation (contract: partial success routing)
    workflow.add_conditional_edges(
        "estimation",
        should_continue_after_estimation,
        {
            "refutation": "refutation",
            "error_handler": "error_handler",
        },
    )

    # Conditional edge after refutation (contract: gate_decision routing)
    workflow.add_conditional_edges(
        "refutation",
        should_continue_after_refutation,
        {
            "sensitivity": "sensitivity",
            "error_handler": "error_handler",
        },
    )

    # Linear edges for remaining flow
    workflow.add_edge("sensitivity", "interpretation")
    workflow.add_edge("interpretation", END)
    workflow.add_edge("error_handler", END)

    # Compile. checkpointer=False (not the bare default None) — this graph
    # runs as a SUBGRAPH of the chatbot graph on the chat path, and LangGraph
    # propagates the parent's Redis checkpointer into bare-compiled children.
    # State carries the estimation DataFrame (data_cache['estimation_data']),
    # which the checkpoint serde (ormsgpack) cannot serialize: the live #1351
    # turn died in <1s with "Type is not msgpack serializable: DataFrame".
    if enable_checkpointing:
        # Would add memory/checkpointing here in production
        return workflow.compile()
    else:
        return workflow.compile(checkpointer=False)


# MLflow experiment tracking constants
MLFLOW_EXPERIMENT_NAME = "e2i_causal_impact"
MLFLOW_EXPERIMENT_TAGS = {
    "agent": "causal_impact",
    "tier": "2",
    "domain": "causal_analytics",
}


async def run_workflow_with_mlflow(
    workflow,
    initial_state: CausalImpactState,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute causal impact workflow with MLflow experiment tracking.

    Creates an MLflow run to track:
    - Parameters: treatment_var, outcome_var, confounders, estimation method
    - Metrics: ATE, p-value, standard_error, e_value, tests_passed, latency
    - Tags: session_id, query_id, dispatch_id
    - Artifacts: DAG DOT format (if available)

    Args:
        workflow: Compiled LangGraph workflow
        initial_state: Initial state with query and variables
        run_name: Optional custom run name (defaults to query_id)

    Returns:
        Final workflow state with MLflow run_id added

    Example:
        workflow = create_causal_impact_graph()
        state = {"query": "...", "treatment_var": "...", ...}
        result = await run_workflow_with_mlflow(workflow, state)
    """
    mlflow = get_mlflow_connector()
    start_time = time.time()

    # Generate run name from query_id if not provided
    query_id = initial_state.get("query_id", "unknown")
    run_name = run_name or f"causal_impact_{query_id}"

    # Get or create experiment
    experiment_id = await mlflow.get_or_create_experiment(
        name=MLFLOW_EXPERIMENT_NAME,
        tags=MLFLOW_EXPERIMENT_TAGS,
    )

    # Prepare run tags
    run_tags = {
        "query_id": query_id,
        "session_id": initial_state.get("session_id", ""),
        "dispatch_id": initial_state.get("dispatch_id", ""),
        "treatment_var": initial_state.get("treatment_var", ""),
        "outcome_var": initial_state.get("outcome_var", ""),
    }

    final_state = None
    mlflow_run_id = None

    try:
        async with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=run_tags,
            description=f"Causal impact analysis: {initial_state.get('query', '')[:100]}",
        ) as run:
            mlflow_run_id = run.run_id

            # Log input parameters
            params = {
                "treatment_var": initial_state.get("treatment_var", ""),
                "outcome_var": initial_state.get("outcome_var", ""),
                "confounders": ",".join(initial_state.get("confounders", [])),
                "data_source": initial_state.get("data_source", ""),
                "interpretation_depth": initial_state.get("interpretation_depth", "standard"),
            }
            await run.log_params(params)

            # Execute workflow
            final_state = await workflow.ainvoke(initial_state)

            # Calculate total latency
            total_latency_ms = (time.time() - start_time) * 1000

            # Log metrics from final state
            metrics = _extract_mlflow_metrics(final_state, total_latency_ms)
            await run.log_metrics(metrics)

            # Log additional tags based on results
            result_tags = _extract_mlflow_result_tags(final_state)
            await run.set_tags(result_tags)

            # Log DAG as artifact if available
            dag_dot = final_state.get("causal_graph", {}).get("dag_dot")
            if dag_dot:
                # Write DOT to temp file and log as artifact
                with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
                    f.write(dag_dot)
                    temp_path = f.name
                await run.log_artifact(temp_path, "causal_dag.dot")

            logger.info(
                f"MLflow run {mlflow_run_id} completed for query {query_id} "
                f"in {total_latency_ms:.1f}ms"
            )

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        # Re-raise to let caller handle
        raise

    # Add MLflow run_id to final state for traceability
    if final_state:
        final_state["mlflow_run_id"] = mlflow_run_id

    return final_state  # type: ignore[no-any-return]


def _extract_mlflow_metrics(state: Dict[str, Any], total_latency_ms: float) -> Dict[str, float]:
    """Extract metrics from workflow state for MLflow logging.

    Args:
        state: Final workflow state
        total_latency_ms: Total execution time in milliseconds

    Returns:
        Dictionary of metric name to value
    """
    metrics = {
        "total_latency_ms": total_latency_ms,
    }

    # Estimation metrics
    estimation = state.get("estimation_result", {})
    if estimation:
        if estimation.get("ate") is not None:
            metrics["ate"] = float(estimation["ate"])
        if estimation.get("p_value") is not None:
            metrics["p_value"] = float(estimation["p_value"])
        if estimation.get("standard_error") is not None:
            metrics["standard_error"] = float(estimation["standard_error"])
        if estimation.get("sample_size") is not None:
            metrics["sample_size"] = float(estimation["sample_size"])

        # V4.2 Enhancement: Energy Score metrics
        if estimation.get("energy_score") is not None:
            metrics["energy_score"] = float(estimation["energy_score"])
        if estimation.get("energy_score_gap") is not None:
            metrics["energy_score_gap"] = float(estimation["energy_score_gap"])
        if estimation.get("n_estimators_evaluated") is not None:
            metrics["n_estimators_evaluated"] = float(estimation["n_estimators_evaluated"])
        if estimation.get("n_estimators_succeeded") is not None:
            metrics["n_estimators_succeeded"] = float(estimation["n_estimators_succeeded"])

        # Per-estimator energy scores
        all_evaluated = estimation.get("all_estimators_evaluated", [])
        for est_result in all_evaluated:
            if isinstance(est_result, dict):
                est_name = est_result.get("estimator_type", "")
                es_data = est_result.get("energy_score_data", {})
                if est_name and es_data and es_data.get("score") is not None:
                    metrics[f"energy_score_{est_name}"] = float(es_data["score"])

    # Refutation metrics
    refutation = state.get("refutation_results", {})
    if refutation:
        if refutation.get("tests_passed") is not None:
            metrics["refutation_tests_passed"] = float(refutation["tests_passed"])
        if refutation.get("total_tests") is not None:
            metrics["refutation_tests_total"] = float(refutation["total_tests"])
        if refutation.get("confidence_adjustment") is not None:
            metrics["confidence_adjustment"] = float(refutation["confidence_adjustment"])

    # Sensitivity metrics
    sensitivity = state.get("sensitivity_analysis", {})
    if sensitivity:
        if sensitivity.get("e_value") is not None:
            metrics["e_value"] = float(sensitivity["e_value"])
        if sensitivity.get("e_value_ci") is not None:
            metrics["e_value_ci"] = float(sensitivity["e_value_ci"])

    # Node latencies
    for node in ["graph_builder", "estimation", "refutation", "sensitivity", "interpretation"]:
        latency_key = f"{node}_latency_ms"
        if state.get(latency_key) is not None:
            metrics[latency_key] = float(state[latency_key])

    # V4.2: Energy score computation latency
    if state.get("energy_score_latency_ms") is not None:
        metrics["energy_score_latency_ms"] = float(state["energy_score_latency_ms"])

    # Overall confidence
    interpretation = state.get("interpretation", {})
    if interpretation.get("causal_confidence"):
        # Map confidence levels to numeric values for tracking
        confidence_map = {"low": 0.33, "medium": 0.66, "high": 1.0}
        confidence_str = interpretation["causal_confidence"].lower()
        if confidence_str in confidence_map:
            metrics["causal_confidence"] = confidence_map[confidence_str]

    return metrics


def _extract_mlflow_result_tags(state: Dict[str, Any]) -> Dict[str, str]:
    """Extract result tags from workflow state for MLflow.

    Args:
        state: Final workflow state

    Returns:
        Dictionary of tag name to value
    """
    tags = {
        "status": state.get("status", "unknown"),
        "current_phase": state.get("current_phase", "unknown"),
    }

    # Estimation method
    estimation = state.get("estimation_result", {})
    if estimation.get("method"):
        tags["estimation_method"] = estimation["method"]

    # Statistical significance
    if estimation.get("statistical_significance") is not None:
        tags["statistically_significant"] = str(estimation["statistical_significance"])

    # Effect size
    if estimation.get("effect_size"):
        tags["effect_size"] = estimation["effect_size"]

    # V4.2 Enhancement: Energy Score tags
    if state.get("energy_score_enabled") is not None:
        tags["energy_score_enabled"] = str(state["energy_score_enabled"]).lower()
    if estimation.get("selection_strategy"):
        tags["selection_strategy"] = estimation["selection_strategy"]
    if estimation.get("selected_estimator"):
        tags["selected_estimator"] = estimation["selected_estimator"]
    if state.get("energy_score_quality_tier"):
        tags["energy_score_quality_tier"] = state["energy_score_quality_tier"]

    # Refutation gate decision
    refutation = state.get("refutation_results", {})
    if refutation.get("gate_decision"):
        tags["refutation_gate"] = refutation["gate_decision"]
    if refutation.get("overall_robust") is not None:
        tags["overall_robust"] = str(refutation["overall_robust"])

    # Sensitivity robustness
    sensitivity = state.get("sensitivity_analysis", {})
    if sensitivity.get("robust_to_confounding") is not None:
        tags["robust_to_confounding"] = str(sensitivity["robust_to_confounding"])

    # Interpretation depth
    interpretation = state.get("interpretation", {})
    if interpretation.get("depth_level"):
        tags["interpretation_depth"] = interpretation["depth_level"]
    if interpretation.get("causal_confidence"):
        tags["causal_confidence_level"] = interpretation["causal_confidence"]

    return tags
