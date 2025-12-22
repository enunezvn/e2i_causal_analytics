"""LangGraph assembly for data_preparer agent.

This module assembles the data preparation pipeline using LangGraph.
"""

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .nodes import (
    compute_baseline_metrics,
    detect_leakage,
    load_data,
    run_ge_validation,
    run_quality_checks,
    transform_data,
)
from .state import DataPreparerState

logger = logging.getLogger(__name__)


def create_data_preparer_graph() -> StateGraph:
    """Create the data_preparer LangGraph.

    The graph executes the following pipeline:
    1. load_data - Load and split data from Supabase using MLDataLoader
    2. run_quality_checks - Validate data quality (completeness, validity, etc.)
    3. run_ge_validation - Great Expectations validation (Phase 3)
    4. detect_leakage - Check for data leakage (temporal, target, train-test)
    5. transform_data - Encode, scale, and impute features
    6. compute_baseline_metrics - Compute baseline metrics from train split
    7. finalize_output - Generate final output and QC gate decision

    Returns:
        StateGraph ready to be compiled
    """
    # Create the graph
    graph = StateGraph(DataPreparerState)

    # Add nodes
    graph.add_node("load_data", load_data)
    graph.add_node("run_quality_checks", run_quality_checks)
    graph.add_node("run_ge_validation", run_ge_validation)
    graph.add_node("detect_leakage", detect_leakage)
    graph.add_node("transform_data", transform_data)
    graph.add_node("compute_baseline_metrics", compute_baseline_metrics)
    graph.add_node("finalize_output", finalize_output)

    # Define edges (sequential execution)
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", "detect_leakage")
    graph.add_edge("detect_leakage", "transform_data")
    graph.add_edge("transform_data", "compute_baseline_metrics")
    graph.add_edge("compute_baseline_metrics", "finalize_output")
    graph.add_edge("finalize_output", END)

    return graph


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
        qc_status = state.get("qc_status", "skipped")
        overall_score = state.get("overall_score", 0.0)
        blocking_issues = state.get("blocking_issues", [])

        # Apply gate logic (from tier0-contracts.md)
        gate_passed = True

        if qc_status == "failed":
            gate_passed = False
            logger.warning("QC gate BLOCKED: status=failed")

        if blocking_issues:
            gate_passed = False
            logger.warning(f"QC gate BLOCKED: {len(blocking_issues)} blocking issues")

        if overall_score < 0.80:
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
        blockers = blocking_issues.copy()
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
