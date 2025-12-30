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
import hashlib
import json
import logging
import tempfile
import time
from typing import Any, Callable, Dict, Literal, Optional, TypeVar

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import (
    create_workflow_initializer,
    get_audit_chain_service,
)
from src.agents.causal_impact.nodes.estimation import estimate_causal_effect
from src.agents.causal_impact.nodes.graph_builder import build_causal_graph
from src.agents.causal_impact.nodes.interpretation import interpret_results
from src.agents.causal_impact.nodes.refutation import refute_causal_estimate
from src.agents.causal_impact.nodes.sensitivity import analyze_sensitivity
from src.agents.causal_impact.state import CausalImpactState
from src.mlops.mlflow_connector import get_mlflow_connector
from src.mlops.opik_connector import get_opik_connector
from src.utils.audit_chain import AgentTier

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

                    duration_ms = int((time.time() - start_time) * 1000)

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
                        output_summary["graph_confidence"] = result.get(
                            "causal_graph", {}
                        ).get("confidence")
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
                        # Capture refutation results for audit
                        validation_passed = ref.get("overall_robust")
                        refutation_results = ref
                    elif node_name == "sensitivity":
                        sens = result.get("sensitivity_analysis", {})
                        output_summary["e_value"] = sens.get("e_value")
                        output_summary["robust_to_confounding"] = sens.get(
                            "robust_to_confounding"
                        )
                    elif node_name == "interpretation":
                        interp = result.get("interpretation", {})
                        output_summary["causal_confidence"] = interp.get(
                            "causal_confidence"
                        )
                        output_summary["depth_level"] = interp.get("depth_level")
                        confidence_score = interp.get("causal_confidence")

                    span.set_output(output_summary)

                    # Set latency attribute from node result
                    latency_key = f"{node_name}_latency_ms"
                    if latency_key in result:
                        span.set_attribute("node_latency_ms", result[latency_key])

                    # Record audit chain entry
                    if workflow_id and audit_service:
                        try:
                            # Compute input/output hashes
                            input_hash = hashlib.sha256(
                                json.dumps(sanitized_input, sort_keys=True, default=str).encode()
                            ).hexdigest()[:32]
                            output_hash = hashlib.sha256(
                                json.dumps(output_summary, sort_keys=True, default=str).encode()
                            ).hexdigest()[:32]

                            audit_service.add_entry(
                                workflow_id=workflow_id,
                                agent_name="causal_impact",
                                agent_tier=AgentTier.CAUSAL_ANALYTICS.value,
                                action_type=node_name,
                                duration_ms=duration_ms,
                                input_hash=input_hash,
                                output_hash=output_hash,
                                validation_passed=validation_passed,
                                confidence_score=confidence_score,
                                refutation_results=refutation_results,
                                user_id=state.get("user_id"),
                                session_id=state.get("session_id"),
                                brand=state.get("brand"),
                            )
                            logger.debug(f"Recorded audit entry for {node_name}")
                        except Exception as ae:
                            logger.warning(f"Failed to record audit entry: {ae}")

                    return result

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


def should_continue_after_estimation(
    state: CausalImpactState,
) -> Literal["refutation", "interpretation", "error_handler"]:
    """Conditional routing after estimation node.

    Contract: Partial success path - if ate_estimate exists, skip to interpretation on error.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("estimation_error"):
        # Partial success: if we have an ATE estimate, go to interpretation
        if state.get("estimation_result", {}).get("ate") is not None:
            return "interpretation"
        return "error_handler"
    return "refutation"


def should_continue_after_refutation(
    state: CausalImpactState,
) -> Literal["sensitivity", "error_handler"]:
    """Conditional routing after refutation node.

    Contract: gate_decision determines flow.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    gate = state.get("refutation_results", {}).get("gate_decision", "proceed")
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
       → conditional: refutation | interpretation (partial success) | error_handler
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
    workflow.add_node("audit_init", audit_initializer)  # Initialize audit chain
    workflow.add_node("graph_builder", traced_build_causal_graph)
    workflow.add_node("estimation", traced_estimate_causal_effect)
    workflow.add_node("refutation", traced_refute_causal_estimate)
    workflow.add_node("sensitivity", traced_analyze_sensitivity)
    workflow.add_node("interpretation", traced_interpret_results)
    workflow.add_node("error_handler", handle_workflow_error)  # Error handler not traced

    # Set entry point to audit initializer
    workflow.set_entry_point("audit_init")

    # Linear edge: audit_init → graph_builder → estimation
    workflow.add_edge("audit_init", "graph_builder")
    workflow.add_edge("graph_builder", "estimation")

    # Conditional edge after estimation (contract: partial success routing)
    workflow.add_conditional_edges(
        "estimation",
        should_continue_after_estimation,
        {
            "refutation": "refutation",
            "interpretation": "interpretation",
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

    # Compile
    if enable_checkpointing:
        # Would add memory/checkpointing here in production
        return workflow.compile()
    else:
        return workflow.compile()


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
                "interpretation_depth": initial_state.get(
                    "interpretation_depth", "standard"
                ),
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
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".dot", delete=False
                ) as f:
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

    return final_state


def _extract_mlflow_metrics(
    state: Dict[str, Any], total_latency_ms: float
) -> Dict[str, float]:
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
