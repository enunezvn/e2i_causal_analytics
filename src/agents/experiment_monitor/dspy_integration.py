"""
E2I Experiment Monitor Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy prompt optimization for experiment_monitor Recipient role

The Experiment Monitor agent is a DSPy Recipient agent that:
1. Consumes optimized prompts for alert message generation
2. Uses optimized prompt templates for summary generation
3. Does NOT generate training signals (Fast Path agent)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. OPTIMIZED PROMPT TEMPLATES
# =============================================================================


@dataclass
class ExperimentMonitorPrompts:
    """
    Optimized prompt templates for experiment monitoring.

    These prompts are consumed from feedback_learner after MIPROv2 optimization.
    The Experiment Monitor agent uses these templates for generating
    human-readable alert messages and summaries.
    """

    # Summary generation prompt
    summary_template: str = (
        "Generate a monitoring summary for {experiments_checked} experiments. "
        "Healthy: {healthy_count}, Warnings: {warning_count}, Critical: {critical_count}. "
        "Key issues: {issue_types}."
    )

    # Alert message prompt
    alert_template: str = (
        "Generate alert for experiment '{experiment_name}' ({experiment_id}). "
        "Issue type: {alert_type}. Severity: {severity}. "
        "Details: {details}. Recommend action."
    )

    # SRM detection prompt
    srm_template: str = (
        "Describe Sample Ratio Mismatch for experiment '{experiment_name}'. "
        "Chi-squared: {chi_squared:.2f}, p-value: {p_value:.4f}. "
        "Expected ratio: {expected_ratio}. Actual counts: {actual_counts}."
    )

    # Enrollment issue prompt
    enrollment_template: str = (
        "Describe enrollment issue for experiment '{experiment_name}'. "
        "Current rate: {current_rate:.1f}/day, expected: {expected_rate:.1f}/day. "
        "Below threshold for {days_below_threshold} days."
    )

    # Fidelity issue prompt
    fidelity_template: str = (
        "Describe Digital Twin fidelity issue for experiment '{experiment_name}'. "
        "Predicted effect: {predicted_effect:.3f}, actual: {actual_effect:.3f}. "
        "Error: {prediction_error:.1%}. Calibration needed: {calibration_needed}."
    )

    # Recommendation prompt
    recommendation_template: str = (
        "Given monitoring results: {healthy_count} healthy, {warning_count} warnings, "
        "{critical_count} critical experiments. Issues: {issue_summary}. "
        "Generate prioritized recommendations."
    )

    # Optimized by MIPROv2/GEPA
    version: str = "1.0"
    last_optimized: str = ""
    optimization_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary_template": self.summary_template,
            "alert_template": self.alert_template,
            "srm_template": self.srm_template,
            "enrollment_template": self.enrollment_template,
            "fidelity_template": self.fidelity_template,
            "recommendation_template": self.recommendation_template,
            "version": self.version,
            "last_optimized": self.last_optimized,
            "optimization_score": self.optimization_score,
        }


# =============================================================================
# 2. DSPy SIGNATURES (for feedback_learner optimization)
# =============================================================================

try:
    import dspy

    class MonitorSummarySignature(dspy.Signature):
        """
        Generate monitoring summary from check results.

        This signature is optimized by feedback_learner and consumed by experiment_monitor.
        """

        experiments_checked: int = dspy.InputField(desc="Number of experiments monitored")
        healthy_count: int = dspy.InputField(desc="Number of healthy experiments")
        warning_count: int = dspy.InputField(desc="Number with warnings")
        critical_count: int = dspy.InputField(desc="Number with critical issues")
        issue_types: str = dspy.InputField(desc="Types of issues detected")

        summary: str = dspy.OutputField(desc="Concise monitoring summary")
        priority_actions: list = dspy.OutputField(desc="Top priority actions")
        overall_status: str = dspy.OutputField(desc="Overall experiment health status")

    class AlertGenerationSignature(dspy.Signature):
        """
        Generate alert message for experiment issue.

        Creates human-readable alert messages with recommended actions.
        """

        experiment_name: str = dspy.InputField(desc="Experiment name")
        alert_type: str = dspy.InputField(desc="Type of alert (SRM, enrollment, etc.)")
        severity: str = dspy.InputField(desc="Alert severity level")
        details: str = dspy.InputField(desc="Issue details")

        message: str = dspy.OutputField(desc="Human-readable alert message")
        recommended_action: str = dspy.OutputField(desc="Specific recommended action")
        urgency_level: str = dspy.OutputField(desc="Urgency assessment")

    class SRMDescriptionSignature(dspy.Signature):
        """
        Describe Sample Ratio Mismatch issue.

        Generates clear explanation of SRM detection for stakeholders.
        """

        experiment_name: str = dspy.InputField(desc="Experiment name")
        chi_squared: float = dspy.InputField(desc="Chi-squared test statistic")
        p_value: float = dspy.InputField(desc="P-value from chi-squared test")
        expected_ratio: str = dspy.InputField(desc="Expected arm ratios")
        actual_counts: str = dspy.InputField(desc="Actual counts per arm")

        explanation: str = dspy.OutputField(desc="Plain language SRM explanation")
        potential_causes: list = dspy.OutputField(desc="Potential causes of mismatch")
        recommended_actions: list = dspy.OutputField(desc="Actions to investigate/fix")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Experiment Monitor agent (Recipient)")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using default monitoring templates")
    MonitorSummarySignature = None
    AlertGenerationSignature = None
    SRMDescriptionSignature = None


# =============================================================================
# 3. PROMPT CONSUMER
# =============================================================================


class ExperimentMonitorDSPyIntegration:
    """
    DSPy integration for Experiment Monitor agent (Recipient role).

    Consumes optimized prompts from feedback_learner but does not
    generate training signals (Fast Path computational agent).
    """

    def __init__(self):
        self.dspy_type: Literal["recipient"] = "recipient"
        self._prompts = ExperimentMonitorPrompts()
        self._prompt_versions: Dict[str, str] = {}

    @property
    def prompts(self) -> ExperimentMonitorPrompts:
        """Get current optimized prompts."""
        return self._prompts

    def update_optimized_prompts(
        self,
        prompts: Dict[str, str],
        optimization_score: float,
    ) -> None:
        """
        Update prompts with optimized versions from feedback_learner.

        Args:
            prompts: Dictionary of prompt_type -> optimized_prompt
            optimization_score: Quality score from optimization
        """
        if "summary_template" in prompts:
            self._prompts.summary_template = prompts["summary_template"]
        if "alert_template" in prompts:
            self._prompts.alert_template = prompts["alert_template"]
        if "srm_template" in prompts:
            self._prompts.srm_template = prompts["srm_template"]
        if "enrollment_template" in prompts:
            self._prompts.enrollment_template = prompts["enrollment_template"]
        if "fidelity_template" in prompts:
            self._prompts.fidelity_template = prompts["fidelity_template"]
        if "recommendation_template" in prompts:
            self._prompts.recommendation_template = prompts["recommendation_template"]

        self._prompts.last_optimized = datetime.now(timezone.utc).isoformat()
        self._prompts.optimization_score = optimization_score
        self._prompts.version = f"1.{len(self._prompt_versions) + 1}"

        logger.info(
            f"Experiment Monitor prompts updated: version={self._prompts.version}, "
            f"score={optimization_score:.4f}"
        )

    def get_summary_prompt(
        self,
        experiments_checked: int,
        healthy_count: int,
        warning_count: int,
        critical_count: int,
        issue_types: str,
    ) -> str:
        """Get formatted summary prompt with current optimized template."""
        return self._prompts.summary_template.format(
            experiments_checked=experiments_checked,
            healthy_count=healthy_count,
            warning_count=warning_count,
            critical_count=critical_count,
            issue_types=issue_types,
        )

    def get_alert_prompt(
        self,
        experiment_name: str,
        experiment_id: str,
        alert_type: str,
        severity: str,
        details: str,
    ) -> str:
        """Get formatted alert prompt."""
        return self._prompts.alert_template.format(
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            alert_type=alert_type,
            severity=severity,
            details=details,
        )

    def get_srm_prompt(
        self,
        experiment_name: str,
        chi_squared: float,
        p_value: float,
        expected_ratio: str,
        actual_counts: str,
    ) -> str:
        """Get formatted SRM description prompt."""
        return self._prompts.srm_template.format(
            experiment_name=experiment_name,
            chi_squared=chi_squared,
            p_value=p_value,
            expected_ratio=expected_ratio,
            actual_counts=actual_counts,
        )

    def get_enrollment_prompt(
        self,
        experiment_name: str,
        current_rate: float,
        expected_rate: float,
        days_below_threshold: int,
    ) -> str:
        """Get formatted enrollment issue prompt."""
        return self._prompts.enrollment_template.format(
            experiment_name=experiment_name,
            current_rate=current_rate,
            expected_rate=expected_rate,
            days_below_threshold=days_below_threshold,
        )

    def get_fidelity_prompt(
        self,
        experiment_name: str,
        predicted_effect: float,
        actual_effect: float,
        prediction_error: float,
        calibration_needed: bool,
    ) -> str:
        """Get formatted fidelity issue prompt."""
        return self._prompts.fidelity_template.format(
            experiment_name=experiment_name,
            predicted_effect=predicted_effect,
            actual_effect=actual_effect,
            prediction_error=prediction_error,
            calibration_needed=calibration_needed,
        )

    def get_recommendation_prompt(
        self,
        healthy_count: int,
        warning_count: int,
        critical_count: int,
        issue_summary: str,
    ) -> str:
        """Get formatted recommendation prompt."""
        return self._prompts.recommendation_template.format(
            healthy_count=healthy_count,
            warning_count=warning_count,
            critical_count=critical_count,
            issue_summary=issue_summary,
        )

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about current prompts."""
        return {
            "agent": "experiment_monitor",
            "dspy_type": self.dspy_type,
            "prompts": self._prompts.to_dict(),
            "prompt_count": 6,
            "dspy_available": DSPY_AVAILABLE,
        }


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_dspy_integration: Optional[ExperimentMonitorDSPyIntegration] = None


def get_experiment_monitor_dspy_integration() -> ExperimentMonitorDSPyIntegration:
    """Get or create DSPy integration singleton."""
    global _dspy_integration
    if _dspy_integration is None:
        _dspy_integration = ExperimentMonitorDSPyIntegration()
    return _dspy_integration


def reset_dspy_integration() -> None:
    """Reset singletons (for testing)."""
    global _dspy_integration
    _dspy_integration = None


# =============================================================================
# 5. EXPORTS
# =============================================================================

__all__ = [
    # Prompt Templates
    "ExperimentMonitorPrompts",
    # DSPy Signatures
    "MonitorSummarySignature",
    "AlertGenerationSignature",
    "SRMDescriptionSignature",
    "DSPY_AVAILABLE",
    # Integration
    "ExperimentMonitorDSPyIntegration",
    # Access
    "get_experiment_monitor_dspy_integration",
    "reset_dspy_integration",
]
