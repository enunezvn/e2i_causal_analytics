"""QC gate checking for model_trainer.

This module validates that data quality checks passed before training.
"""

from typing import Any, Dict

from src.agents.ml_foundation.data_preparer.nodes.qc_threshold import (
    resolve_qc_min_overall_score,
)


async def check_qc_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify QC gate passed before allowing training.

    CRITICAL: This is a mandatory gate check. Training MUST NOT proceed
    if QC validation failed.

    The binding decision is the upstream ``qc_report["qc_passed"]`` boolean,
    which data_preparer already computed against the resolved minimum bar.
    This node resolves that same bar through the single source of truth
    (``resolve_qc_min_overall_score``) only to SURFACE it in the gate message,
    so the threshold reported here can never drift from the one data_preparer
    enforced.

    Args:
        state: ModelTrainerState with qc_report

    Returns:
        Dictionary with qc_gate_passed, qc_gate_message

    Raises:
        No exceptions - returns error in state if gate blocked
    """
    qc_report = state.get("qc_report", {})

    # Extract QC status
    qc_passed = qc_report.get("qc_passed", False)
    qc_score = qc_report.get("overall_score", 0.0)
    qc_errors = qc_report.get("qc_errors", [])

    # Resolve the effective minimum bar for messaging/consistency. data_preparer
    # carries the bar it ACTUALLY enforced on the qc_report, so prefer that; the
    # resolver coerces/validates it and falls back to the strict 0.80 default
    # when absent. Routed through the single source of truth so the threshold
    # reported here can never drift from the one data_preparer enforced.
    min_overall_score = resolve_qc_min_overall_score(
        {"qc_min_overall_score": qc_report.get("qc_min_overall_score")}
    )

    if not qc_passed:
        return {
            "qc_gate_passed": False,
            "qc_gate_message": (
                f"QC gate BLOCKED: Quality check failed with score {qc_score} "
                f"(min {min_overall_score:.2f}). "
                f"Errors: {', '.join(qc_errors[:3])}"
            ),
            "error": "QC gate blocked - cannot train with failed data quality",
            "error_type": "qc_gate_blocked_error",
        }

    # Check for critical warnings
    # Per data_preparer.schemas.QCReportSchema, qc_warnings is List[Dict[str, Any]]
    # — each item is a Great Expectations result dict (expectation_type, threshold, etc.).
    # Stringify by the most informative field before joining for the user-facing message.
    qc_warnings = qc_report.get("qc_warnings", [])
    if qc_warnings:

        def _summarize(w):
            if isinstance(w, dict):
                return str(w.get("expectation_type", w))
            return str(w)

        warning_message = (
            f"QC warnings present: {', '.join(_summarize(w) for w in qc_warnings[:2])}"
        )
    else:
        warning_message = "No QC warnings"

    return {
        "qc_gate_passed": True,
        "qc_gate_message": (
            f"QC gate PASSED: Quality score {qc_score} "
            f"(min {min_overall_score:.2f}). {warning_message}"
        ),
    }
