"""QC gate checking for model_trainer.

This module validates that data quality checks passed before training.
"""

from typing import Dict, Any


async def check_qc_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Verify QC gate passed before allowing training.

    CRITICAL: This is a mandatory gate check. Training MUST NOT proceed
    if QC validation failed.

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

    if not qc_passed:
        return {
            "qc_gate_passed": False,
            "qc_gate_message": (
                f"QC gate BLOCKED: Quality check failed with score {qc_score}. "
                f"Errors: {', '.join(qc_errors[:3])}"
            ),
            "error": "QC gate blocked - cannot train with failed data quality",
            "error_type": "qc_gate_blocked_error",
        }

    # Check for critical warnings
    qc_warnings = qc_report.get("qc_warnings", [])
    if qc_warnings:
        warning_message = f"QC warnings present: {', '.join(qc_warnings[:2])}"
    else:
        warning_message = "No QC warnings"

    return {
        "qc_gate_passed": True,
        "qc_gate_message": (
            f"QC gate PASSED: Quality score {qc_score}. {warning_message}"
        ),
    }
