"""
E2I Health Score Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy prompt optimization for health_score Recipient role

The Health Score agent is a DSPy Recipient agent that:
1. Consumes optimized prompts for health reporting
2. Uses optimized prompt templates for summary generation
3. Does NOT generate training signals (Fast Path agent)
"""

from __future__ import annotations

import functools
import importlib.util
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Resolved lazily at runtime via PEP 562 ``__getattr__`` so importing dspy
    # stays off the Fast Path. Declaring them here lets static tools (ruff F822
    # on ``__all__``, mypy) see them without an eager import.
    HealthSummarySignature: Any
    HealthRecommendationSignature: Any


# =============================================================================
# 1. OPTIMIZED PROMPT TEMPLATES
# =============================================================================


@dataclass
class HealthReportPrompts:
    """
    Optimized prompt templates for health score generation.

    These prompts are consumed from feedback_learner after MIPROv2 optimization.
    The Health Score agent is primarily computational but uses optimized templates
    for generating human-readable summaries and recommendations.
    """

    # Summary generation prompt.
    #
    # NOTE: this template is the OPTIMIZABLE source of the human-readable health
    # summary that ``ScoreComposerNode._generate_summary`` emits. The default
    # value below renders BYTE-IDENTICALLY to the node's historical inline
    # construction so wiring the getter in is a pure drop-in. The
    # ``{components_suffix}`` placeholder is empty for score_composer (which
    # supplies no component names) and only renders when components are passed.
    summary_template: str = (
        "System health is {status} (Grade: {grade}, Score: {score:.1f}/100). "
        "{issue_clause}{components_suffix}"
    )

    # Recommendation prompt
    recommendation_template: str = (
        "Given health status: component={component_score}, model={model_score}, "
        "pipeline={pipeline_score}, agent={agent_score}. "
        "Critical issues: {critical_issues}. "
        "Generate prioritized recommendations."
    )

    # Issue description prompt
    issue_description_template: str = (
        "Describe health issue: {issue_type} in {component} with status {status}. "
        "Latency: {latency_ms}ms. Error: {error_message}."
    )

    # Optimized by MIPROv2
    version: str = "1.0"
    last_optimized: str = ""
    optimization_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary_template": self.summary_template,
            "recommendation_template": self.recommendation_template,
            "issue_description_template": self.issue_description_template,
            "version": self.version,
            "last_optimized": self.last_optimized,
            "optimization_score": self.optimization_score,
        }


# =============================================================================
# 2. DSPy SIGNATURES (for feedback_learner optimization)
# =============================================================================

# DSPy availability is probed WITHOUT importing dspy. ``import dspy`` loads
# ~714 MB; the Health Score agent is a Fast Path computational agent ("Zero LLM
# usage in critical path") so it must never pull dspy in at import time. The
# Signature subclasses below are needed ONLY by feedback_learner's MIPROv2/GEPA
# optimizer paths, so they are built lazily on first attribute access (PEP 562
# module ``__getattr__``) and ``dspy`` is imported only then.
DSPY_AVAILABLE: bool = importlib.util.find_spec("dspy") is not None
if DSPY_AVAILABLE:
    logger.info("DSPy detected for Health Score agent (Recipient); import deferred until needed")
else:
    logger.warning("DSPy not available - using default health templates")


@functools.lru_cache(maxsize=1)
def _get_health_signatures() -> Dict[str, Any]:
    """Build (once) and return the health_score DSPy Signature classes.

    Importing dspy here keeps the ~714 MB import off the fast-path module-import
    chain; it happens only when an optimizer actually requests these signatures.
    Returns an all-``None`` mapping when dspy is not installed (mirrors the
    historical placeholder behavior).
    """
    if not DSPY_AVAILABLE:
        return {
            "HealthSummarySignature": None,
            "HealthRecommendationSignature": None,
        }

    import dspy

    class HealthSummarySignature(dspy.Signature):
        """
        Generate health summary from metrics.

        This signature is optimized by feedback_learner and consumed by health_score.
        """

        overall_score: float = dspy.InputField(desc="Overall health score (0-100)")
        grade: str = dspy.InputField(desc="Health grade (A-F)")
        component_scores: str = dspy.InputField(desc="Scores per dimension")
        critical_issues: str = dspy.InputField(desc="List of critical issues")

        summary: str = dspy.OutputField(desc="Concise health summary")
        priority_actions: list = dspy.OutputField(desc="Top priority actions")
        status_description: str = dspy.OutputField(desc="Overall status description")

    class HealthRecommendationSignature(dspy.Signature):
        """
        Generate recommendations from health metrics.

        Creates actionable recommendations for improving system health.
        """

        health_metrics: str = dspy.InputField(desc="Current health metrics")
        issue_list: str = dspy.InputField(desc="Identified issues")
        historical_patterns: str = dspy.InputField(desc="Past health patterns")

        recommendations: list = dspy.OutputField(desc="Prioritized recommendations")
        urgency_assessment: str = dspy.OutputField(desc="Urgency level and rationale")
        expected_improvement: str = dspy.OutputField(desc="Expected improvement from actions")

    logger.info("DSPy signatures built for Health Score agent (Recipient)")
    return {
        "HealthSummarySignature": HealthSummarySignature,
        "HealthRecommendationSignature": HealthRecommendationSignature,
    }


# Names resolved lazily via PEP 562: ``from ...dspy_integration import
# HealthSummarySignature`` (and attribute access) builds the class on first use
# without importing dspy at module import time.
_LAZY_SIGNATURE_NAMES = frozenset({"HealthSummarySignature", "HealthRecommendationSignature"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SIGNATURE_NAMES:
        return _get_health_signatures()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# 3. PROMPT CONSUMER
# =============================================================================


class HealthScoreDSPyIntegration:
    """
    DSPy integration for Health Score agent (Recipient role).

    Consumes optimized prompts from feedback_learner but does not
    generate training signals (Fast Path computational agent).
    """

    def __init__(self) -> None:
        self.dspy_type: Literal["recipient"] = "recipient"
        self._prompts: HealthReportPrompts = HealthReportPrompts()
        self._prompt_versions: Dict[str, str] = {}

    @property
    def prompts(self) -> HealthReportPrompts:
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
        if "recommendation_template" in prompts:
            self._prompts.recommendation_template = prompts["recommendation_template"]
        if "issue_description_template" in prompts:
            self._prompts.issue_description_template = prompts["issue_description_template"]

        self._prompts.last_optimized = datetime.now(timezone.utc).isoformat()
        self._prompts.optimization_score = optimization_score
        self._prompts.version = f"1.{len(self._prompt_versions) + 1}"

        logger.info(
            f"Health Score prompts updated: version={self._prompts.version}, "
            f"score={optimization_score:.4f}"
        )

    # Maps a health grade to the human-readable status word used in the summary.
    # Mirrors ScoreComposerNode._generate_summary's historical mapping so the
    # optimizable template renders identically.
    _STATUS_BY_GRADE: Dict[str, str] = {
        "A": "excellent",
        "B": "good",
        "C": "fair",
        "D": "poor",
        "F": "critical",
    }

    def get_summary_prompt(
        self,
        grade: str,
        score: float,
        components: str,
        critical_count: int,
        warning_count: int,
    ) -> str:
        """Get the formatted summary via the current (optimizable) template.

        Drop-in replacement for ScoreComposerNode's inline summary construction:
        with ``components=""`` (as score_composer calls it) the output is
        byte-identical to the historical ``_generate_summary`` string. When
        component names ARE supplied, a ``Components: ...`` suffix is appended.
        ``status`` and ``issue_clause`` are derived here so the template stays a
        pure ``.format()`` string (no conditionals) and remains optimizable.
        """
        status = self._STATUS_BY_GRADE.get(grade, "unknown")
        if critical_count:
            issue_clause = f"{critical_count} critical issue(s) detected."
        else:
            issue_clause = "All systems operational."
        components_suffix = f" Components: {components}." if components else ""
        return self._prompts.summary_template.format(
            grade=grade,
            score=score,
            components=components,
            critical_count=critical_count,
            warning_count=warning_count,
            status=status,
            issue_clause=issue_clause,
            components_suffix=components_suffix,
        )

    def get_recommendation_prompt(
        self,
        component_score: float,
        model_score: float,
        pipeline_score: float,
        agent_score: float,
        critical_issues: str,
    ) -> str:
        """Get formatted recommendation prompt."""
        return self._prompts.recommendation_template.format(
            component_score=component_score,
            model_score=model_score,
            pipeline_score=pipeline_score,
            agent_score=agent_score,
            critical_issues=critical_issues,
        )

    def get_issue_description_prompt(
        self,
        issue_type: str,
        component: str,
        status: str,
        latency_ms: int,
        error_message: str,
    ) -> str:
        """Get formatted issue description prompt."""
        return self._prompts.issue_description_template.format(
            issue_type=issue_type,
            component=component,
            status=status,
            latency_ms=latency_ms,
            error_message=error_message or "None",
        )

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about current prompts."""
        return {
            "agent": "health_score",
            "dspy_type": self.dspy_type,
            "prompts": self._prompts.to_dict(),
            "prompt_count": 3,
            "dspy_available": DSPY_AVAILABLE,
        }


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_dspy_integration: Optional[HealthScoreDSPyIntegration] = None


def get_health_score_dspy_integration() -> HealthScoreDSPyIntegration:
    """Get or create DSPy integration singleton."""
    global _dspy_integration
    if _dspy_integration is None:
        _dspy_integration = HealthScoreDSPyIntegration()
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
    "HealthReportPrompts",
    # DSPy Signatures
    "HealthSummarySignature",
    "HealthRecommendationSignature",
    "DSPY_AVAILABLE",
    # Integration
    "HealthScoreDSPyIntegration",
    # Access
    "get_health_score_dspy_integration",
    "reset_dspy_integration",
]
