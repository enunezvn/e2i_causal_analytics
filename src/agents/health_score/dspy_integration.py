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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Sequence

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

    These prompts are consumed from feedback_learner after GEPA/MIPROv2 optimization.
    The Health Score agent is primarily computational but uses optimized templates
    for generating human-readable summaries and recommendations.
    """

    # Summary generation prompt (MEASURED / PARTIAL states).
    #
    # NOTE: this template is the OPTIMIZABLE source of the human-readable health
    # summary that ``ScoreComposerNode._generate_summary`` emits. With
    # ``{scope_label}`` -> "System" (the full-scope / scope-absent default) it
    # renders BYTE-IDENTICALLY to the node's historical inline construction, so
    # wiring the getter in stayed a pure drop-in. The ``{components_suffix}``
    # placeholder is empty for score_composer (which supplies no component
    # names) and only renders when components are passed.
    #
    # #1447: ``{scope_label}`` replaced the hardcoded literal "System" because a
    # models/pipelines/agents/quick-scoped check is NOT a whole-system verdict
    # (see ``_record_full_check`` in src/api/routes/health_score.py, which
    # already refuses to trend anything but a FULL check for that reason).
    summary_template: str = (
        "{scope_label} health is {status} (Grade: {grade}, Score: {score:.1f}/100). "
        "{issue_clause}{components_suffix}"
    )

    # Summary rendered when NOTHING was measured (``data_provenance ==
    # "unknown"``) — #1447.
    #
    # The composer's F1 anti-fabrication guard DELIBERATELY reports 0.0 / grade
    # "F" for a zero-measured check so no consumer can mistake an unmeasured
    # system for a healthy one. That payload is correct and unchanged; what was
    # wrong was the NARRATION: the measured template rendered the unmeasured
    # state byte-identically to a genuine grade-F catastrophe ("System health is
    # critical ... 1 critical issue(s) detected.") and dropped the explanatory
    # issue the node already builds. This template says UNKNOWN, reconciles the
    # 0.0/F placeholder so a reader comparing prose to the widget is not misled,
    # and surfaces the leading critical-issue TEXT via ``{issue_detail}``.
    #
    # No DSPy signature backs this field (``RECIPIENT_SIGNATURE_FIELDS`` lists
    # only signature-backed fields), so the optimizer never rewrites it — but it
    # still round-trips through to_dict()/update_optimized_prompts() with the
    # rest of the bundle.
    unknown_summary_template: str = (
        "{scope_label} health status is UNKNOWN - nothing was measured. "
        "The {score:.1f}/100 Grade-{grade} score is a fail-closed placeholder for "
        "UNMEASURED, not a measured failure. {issue_detail}"
    )

    # === MODEL QUALITY METRICS (#1450) ===
    #
    # A question naming a metric ("what is the ROC-AUC and calibration of the
    # current Kisqali model?") asks for the MEASUREMENT, not a composite grade.
    # These render the named evaluation metrics the model_health node carried
    # through from ``ml_performance_metrics``, always with the model version and
    # the cohort/date the numbers were measured on — a bare 0.77 with no cohort
    # or as-of date is not an auditable answer for a governance reviewer.
    #
    # Like ``unknown_summary_template`` these are NOT registered in
    # ``RECIPIENT_SIGNATURE_FIELDS`` (no backing DSPy signature), so the
    # optimizer never rewrites them; they still round-trip through
    # ``to_dict()``/``update_optimized_prompts()`` so a bundle install cannot
    # silently drop them.
    model_metrics_header_template: str = "Model quality metrics (requested: {requested}):"
    model_metrics_template: str = "- {model_label}: {metric_list} [{provenance_clause}]"
    model_metrics_missing_template: str = (
        "Requested but NOT recorded for any matched model: {missing}. No value is "
        "reported for it - reporting one without a measurement would be a fabrication."
    )
    model_metrics_unavailable_template: str = (
        "Model quality metrics (requested: {requested}) are UNKNOWN - {reason} "
        "No number is reported, because reporting one without a measurement "
        "would be a fabrication."
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

    # Optimized by GEPA/MIPROv2
    version: str = "1.0"
    last_optimized: str = ""
    optimization_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary_template": self.summary_template,
            "unknown_summary_template": self.unknown_summary_template,
            "model_metrics_header_template": self.model_metrics_header_template,
            "model_metrics_template": self.model_metrics_template,
            "model_metrics_missing_template": self.model_metrics_missing_template,
            "model_metrics_unavailable_template": self.model_metrics_unavailable_template,
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
        if "unknown_summary_template" in prompts:
            self._prompts.unknown_summary_template = prompts["unknown_summary_template"]
        for field in (
            "model_metrics_header_template",
            "model_metrics_template",
            "model_metrics_missing_template",
            "model_metrics_unavailable_template",
        ):
            if field in prompts:
                setattr(self._prompts, field, prompts[field])
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

    # Maps ``check_scope`` to the subject of the summary sentence (#1447). Only a
    # FULL check is a whole-"System" verdict; ``quick`` is the component-only
    # graph (audit_init -> component -> compose), and the single-dimension scopes
    # narrate that dimension. An absent/unrecognised scope keeps the historical
    # "System" label so the default rendering is unchanged.
    _SCOPE_LABELS: Dict[str, str] = {
        "full": "System",
        "quick": "Component",
        "models": "Model",
        "pipelines": "Pipeline",
        "agents": "Agent",
    }

    def scope_label(self, check_scope: Optional[str]) -> str:
        """Subject of the summary sentence for a given check scope."""
        return self._SCOPE_LABELS.get(check_scope or "full", "System")

    def get_summary_prompt(
        self,
        grade: str,
        score: float,
        components: str,
        critical_count: int,
        warning_count: int,
        data_provenance: str = "measured",
        check_scope: Optional[str] = None,
        critical_issues: Optional[List[str]] = None,
    ) -> str:
        """Get the formatted summary via the current (optimizable) template.

        Drop-in replacement for ScoreComposerNode's inline summary construction:
        with ``components=""`` and a full/absent ``check_scope`` (as
        score_composer historically called it) the output is byte-identical to
        the historical ``_generate_summary`` string. When component names ARE
        supplied, a ``Components: ...`` suffix is appended. ``status``,
        ``issue_clause``, ``scope_label`` and ``issue_detail`` are derived here
        so the templates stay pure ``.format()`` strings (no conditionals) and
        remain optimizable.

        ``data_provenance == "unknown"`` (NOTHING measured) selects
        ``unknown_summary_template`` instead — see #1447: the composer's
        deliberate 0.0/grade-"F" fail-closed payload must not be NARRATED as a
        measured critical failure. The three new parameters default to the
        historical behaviour so existing 5-argument callers are unaffected.
        """
        label = self.scope_label(check_scope)
        if data_provenance == "unknown":
            # Surface the leading critical issue's TEXT (the composer builds an
            # explanatory one for exactly this state), not merely its count.
            issue_detail = (
                critical_issues[0]
                if critical_issues
                else "No health backends are wired for this scope."
            )
            return self._prompts.unknown_summary_template.format(
                grade=grade,
                score=score,
                scope_label=label,
                issue_detail=issue_detail,
                critical_count=critical_count,
                warning_count=warning_count,
            )

        status = self._STATUS_BY_GRADE.get(grade, "unknown")
        if critical_count:
            issue_clause = f"{critical_count} critical issue(s) detected."
        elif label == "System":
            # Full (or absent/unrecognised, historical-default) scope: every
            # dimension was evaluated, so the whole-system claim is earned.
            issue_clause = "All systems operational."
        else:
            # #1460: a scoped check measured ONLY its own dimension (e.g.
            # scope="models" runs model_health alone), so "All systems
            # operational." would assert health for dimensions never evaluated.
            # Name what was actually checked, mirroring the #1447 scope_label.
            issue_clause = f"No {label.lower()} health issues detected."
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
            scope_label=label,
        )

    # Display names for the recorded evaluation metrics (#1450). Keys are the
    # ``ml_performance_metrics.metric_name`` values; "psi" has no key there and
    # is listed so a question naming it can be answered "not recorded" BY NAME.
    _EVAL_METRIC_LABELS: Dict[str, str] = {
        "auc_roc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "brier_score": "Brier score",
        "calibration_slope": "calibration slope",
        "accuracy": "accuracy",
        "f1": "F1",
        "precision": "precision",
        "recall": "recall",
        "psi": "PSI",
    }

    # Always reported when recorded, even if the question named only one of
    # them: discrimination WITHOUT calibration is the classic misleading model
    # answer, and Brier is the proper score that reconciles the two.
    _CORE_QUALITY_METRICS = ("auc_roc", "calibration_slope", "brier_score")

    def metric_label(self, metric_key: str) -> str:
        """Human-readable name for a metric key (falls back to the key)."""
        return self._EVAL_METRIC_LABELS.get(metric_key, metric_key)

    def get_model_metrics_prompt(
        self,
        requested: Sequence[str],
        models: Sequence[Mapping[str, Any]],
        unavailable_reason: str = "",
    ) -> str:
        """Render the model-quality answer for a metric-naming question (#1450).

        ``models`` are the ``ModelMetrics`` entries the composer already matched
        to the question. Each contributes ONE line naming the model, its version
        and stage, the metric values, and the cohort/size/date they were
        measured on. Metrics that were requested but are not recorded are named
        explicitly — never silently dropped and never substituted with a
        different metric's value. #1460: the disclosure is PER MODEL LINE (a
        model's own recorded set decides its own line), because a single global
        "missing" set made model B's silent omission indistinguishable from
        "fine" whenever model A recorded the metric. A metric recorded by NO
        matched model additionally gets the global missing line.

        ``unavailable_reason`` (non-empty) short-circuits to the unavailable
        template: nothing was measured, so no number may be printed at all.
        """
        requested_label = (
            ", ".join(self.metric_label(m) for m in requested) if requested else "model quality"
        )
        if unavailable_reason:
            return self._prompts.model_metrics_unavailable_template.format(
                requested=requested_label,
                reason=unavailable_reason,
            )

        lines: List[str] = [
            self._prompts.model_metrics_header_template.format(requested=requested_label)
        ]
        reported: set[str] = set()
        for model in models:
            eval_metrics = dict(model.get("eval_metrics") or {})
            name = model.get("model_name") or model.get("model_id") or "unknown model"
            version = model.get("model_version")
            stage = model.get("model_stage")
            model_label = str(name)
            if version:
                model_label += f" v{version}"
            if stage:
                model_label += f" ({stage})"

            if not eval_metrics:
                lines.append(
                    self._prompts.model_metrics_template.format(
                        model_label=model_label,
                        metric_list="no evaluation metrics recorded",
                        provenance_clause="nothing to attribute - no evaluation on record",
                    )
                )
                continue

            # Requested-and-recorded first (in the order asked), then the core
            # quality trio so discrimination is never reported without
            # calibration.
            ordered: List[str] = [k for k in requested if k in eval_metrics]
            ordered += [
                k for k in self._CORE_QUALITY_METRICS if k in eval_metrics and k not in ordered
            ]
            reported.update(ordered)
            metric_list = ", ".join(
                f"{self.metric_label(k)} {eval_metrics[k]:.3f}" for k in ordered
            )
            # #1460: THIS model's requested-but-unrecorded metrics are disclosed
            # on THIS model's line — computed from its own eval_metrics only, so
            # another model recording the metric can never mask the omission.
            missing_here = [k for k in requested if k not in eval_metrics]
            if missing_here:
                metric_list += "; " + ", ".join(
                    f"{self.metric_label(k)} not recorded" for k in missing_here
                )
            lines.append(
                self._prompts.model_metrics_template.format(
                    model_label=model_label,
                    metric_list=metric_list,
                    provenance_clause=self._provenance_clause(model),
                )
            )

        missing = [m for m in requested if m not in reported]
        if missing:
            lines.append(
                self._prompts.model_metrics_missing_template.format(
                    missing=", ".join(self.metric_label(m) for m in missing)
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _provenance_clause(model: Mapping[str, Any]) -> str:
        """ "holdout cohort, n=1000, as of 2026-06-01" — with honest gaps.

        Every part is Optional at the source; an absent part is named as not
        recorded rather than defaulted, so a reader can never mistake an
        unlabelled number for a labelled one.
        """
        cohort = model.get("eval_cohort")
        sample_size = model.get("eval_sample_size")
        as_of = model.get("eval_as_of")
        parts = [
            f"{cohort} cohort" if cohort else "evaluation cohort not recorded",
            f"n={sample_size}" if sample_size is not None else "cohort size not recorded",
            f"as of {str(as_of)[:10]}" if as_of else "measurement date not recorded",
        ]
        return ", ".join(parts)

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
            # summary, unknown_summary, the 4 model-quality templates (#1450),
            # recommendation, issue_description.
            "prompt_count": 8,
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
