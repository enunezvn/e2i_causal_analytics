"""LangGraph assembly for data_preparer agent.

This module assembles the data preparation pipeline using LangGraph.
"""

import logging
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from .nodes import (
    compute_baseline_metrics,
    detect_leakage,
    load_data,
    register_features_in_feast,
    review_and_remediate_qc,
    run_ge_validation,
    run_quality_checks,
    run_schema_validation,
    transform_data,
)
from .state import DataPreparerState

logger = logging.getLogger(__name__)

# Maximum remediation attempts before giving up
MAX_REMEDIATION_ATTEMPTS = 2


def create_data_preparer_graph() -> StateGraph:
    """Create the data_preparer LangGraph.

    The graph executes the following pipeline:
    1. load_data - Load and split data from Supabase using MLDataLoader
    2. run_schema_validation - Pandera schema validation (fast, ~10ms)
    3. run_quality_checks - Validate data quality (completeness, validity, etc.)
    4. run_ge_validation - Great Expectations validation (business rules)
    5. detect_leakage - Check for data leakage (temporal, target, train-test)
    6. transform_data - Encode, scale, and impute features
    7. register_features_in_feast - Register features in Feast feature store
    8. compute_baseline_metrics - Compute baseline metrics from train split
    9. finalize_output - Generate final output and QC gate decision
    10. qc_remediation - LLM-assisted review and remediation if QC fails

    QC Remediation Loop:
    - If QC gate fails, routes to LLM-assisted remediation review
    - Analyzes root causes using Claude
    - Attempts automatic fixes (imputation, type conversion, etc.)
    - Re-runs quality checks if fixes are applied
    - Maximum 2 remediation attempts before final failure

    Validation Pipeline Order:
    - Pandera: Fast schema checks (types, nullability, enums)
    - Quality Checker: 5 dimension scoring (completeness, validity, etc.)
    - Great Expectations: Business rules and statistical checks

    Feast Integration:
    - Features registered after transformation (ready for point-in-time retrieval)
    - Freshness check included in registration (QC validation)
    - Non-blocking: failures generate warnings, not errors

    Returns:
        StateGraph ready to be compiled
    """
    # Create the graph
    graph = StateGraph(DataPreparerState)

    # Add nodes
    graph.add_node("load_data", load_data)
    graph.add_node("run_schema_validation", run_schema_validation)
    graph.add_node("run_quality_checks", run_quality_checks)
    graph.add_node("run_ge_validation", run_ge_validation)
    graph.add_node("detect_leakage", detect_leakage)
    graph.add_node("transform_data", transform_data)
    graph.add_node("register_features_in_feast", register_features_in_feast)
    graph.add_node("compute_baseline_metrics", compute_baseline_metrics)
    graph.add_node("finalize_output", finalize_output)
    graph.add_node("qc_remediation", review_and_remediate_qc)

    # Define edges (sequential execution with QC remediation loop)
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", "detect_leakage")
    graph.add_edge("detect_leakage", "transform_data")
    graph.add_edge("transform_data", "register_features_in_feast")
    graph.add_edge("register_features_in_feast", "compute_baseline_metrics")
    graph.add_edge("compute_baseline_metrics", "finalize_output")

    # Conditional edge: after finalize_output, check if QC passed
    graph.add_conditional_edges(
        "finalize_output",
        _route_after_finalize,
        {
            "end": END,
            "remediate": "qc_remediation",
        },
    )

    # Conditional edge: after remediation, either retry validation or end
    graph.add_conditional_edges(
        "qc_remediation",
        _route_after_remediation,
        {
            "retry": "run_quality_checks",
            "end": END,
        },
    )

    return graph


def _route_after_finalize(state: DataPreparerState) -> Literal["end", "remediate"]:
    """Route after finalize_output based on QC gate result.

    Args:
        state: Current agent state

    Returns:
        "end" if QC passed, "remediate" if QC failed
    """
    gate_passed = state.get("gate_passed", False)
    qc_status = state.get("qc_status", "unknown")

    if gate_passed and qc_status == "passed":
        logger.info("QC gate passed, proceeding to end")
        return "end"
    else:
        logger.info(
            f"QC gate failed (status={qc_status}, passed={gate_passed}), "
            "routing to remediation review"
        )
        return "remediate"


def _route_after_remediation(state: DataPreparerState) -> Literal["retry", "end"]:
    """Route after remediation based on result.

    Args:
        state: Current agent state

    Returns:
        "retry" if remediation was applied and revalidation needed, "end" otherwise
    """
    remediation_status = state.get("remediation_status", "unknown")
    requires_revalidation = state.get("requires_revalidation", False)
    remediation_attempts = state.get("remediation_attempts", 0)

    if remediation_status == "applied" and requires_revalidation:
        if remediation_attempts < MAX_REMEDIATION_ATTEMPTS:
            logger.info(
                f"Remediation applied, retrying validation "
                f"(attempt {remediation_attempts + 1}/{MAX_REMEDIATION_ATTEMPTS})"
            )
            return "retry"

    logger.info(f"Remediation complete with status: {remediation_status}")
    return "end"


async def finalize_output(state: DataPreparerState) -> Dict[str, Any]:
    """Finalize output and make QC gate decision.

    This node:
    1. Aggregates all QC results
    2. Computes data readiness
    3. Makes the CRITICAL QC gate decision
    4. Prepares final output

    The QC gate blocks downstream training if:
    - QC status is "failed"
    - There are blocking issues
    - Overall QC score < 0.80

    Args:
        state: Current agent state

    Returns:
        Updated state with final outputs
    """
    logger.info(f"Finalizing output for experiment {state['experiment_id']}")

    try:
        # === QC GATE DECISION ===
        qc_status = state.get("qc_status", "unknown")
        overall_score = state.get("overall_score")
        blocking_issues = state.get("blocking_issues", [])

        # Apply gate logic (from tier0-contracts.md)
        # Gate ONLY passes if qc_status is explicitly "passed" AND score meets threshold
        gate_passed = True

        # CRITICAL: Gate fails if QC status is not explicitly "passed"
        # This prevents unknown/skipped/failed statuses from passing
        if qc_status != "passed":
            gate_passed = False
            logger.warning(f"QC gate BLOCKED: qc_status='{qc_status}' (must be 'passed')")

        if blocking_issues:
            gate_passed = False
            logger.warning(f"QC gate BLOCKED: {len(blocking_issues)} blocking issues")

        # CRITICAL: Gate fails if overall_score is None or below threshold
        if overall_score is None:
            gate_passed = False
            logger.warning("QC gate BLOCKED: overall_score is None (QC checks may not have run)")
        elif overall_score < 0.80:
            gate_passed = False
            logger.warning(f"QC gate BLOCKED: score {overall_score:.2f} < 0.80")

        # === DATA READINESS ===
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")
        holdout_df = state.get("holdout_df")

        train_samples = len(train_df) if train_df is not None else 0
        validation_samples = len(validation_df) if validation_df is not None else 0
        test_samples = len(test_df) if test_df is not None else 0
        holdout_samples = len(holdout_df) if holdout_df is not None else 0
        total_samples = train_samples + validation_samples + test_samples + holdout_samples

        # Available features
        available_features = list(train_df.columns) if train_df is not None else []

        # Missing required features
        scope_spec = state.get("scope_spec", {})
        required_features = scope_spec.get("required_features", [])
        missing_required_features = [f for f in required_features if f not in available_features]

        # Data is ready if QC passed and no missing required features
        qc_passed = gate_passed
        is_ready = qc_passed and len(missing_required_features) == 0

        # Blockers (same as blocking_issues)
        blockers = (blocking_issues or []).copy()
        if missing_required_features:
            blockers.append(f"Missing required features: {', '.join(missing_required_features)}")

        # Update state
        updates = {
            "gate_passed": gate_passed,
            "qc_passed": qc_passed,
            "qc_score": overall_score,
            "is_ready": is_ready,
            "total_samples": total_samples,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "holdout_samples": holdout_samples,
            "available_features": available_features,
            "missing_required_features": missing_required_features,
            "blockers": blockers,
        }

        logger.info(
            f"Data preparation completed: gate_passed={gate_passed}, "
            f"is_ready={is_ready}, total_samples={total_samples}"
        )

        return updates

    except Exception as e:
        logger.error(f"Finalize output failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "finalize_output_error",
            "gate_passed": False,
            "qc_passed": False,
            "is_ready": False,
            "blockers": [f"Finalization error: {str(e)}"],
        }
