"""
E2I Resource Optimizer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy prompt optimization for resource_optimizer Recipient role

The Resource Optimizer agent is a DSPy Recipient agent that:
1. Consumes optimized prompts for optimization summaries
2. Uses optimized prompt templates for recommendation generation
3. Does NOT generate training signals (computational optimization agent)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. OPTIMIZED PROMPT TEMPLATES
# =============================================================================


@dataclass
class ResourceOptimizationPrompts:
    """
    Optimized prompt templates for resource optimization outputs.

    These prompts are consumed from feedback_learner after MIPROv2 optimization.
    The Resource Optimizer uses these for generating natural language summaries
    and recommendations from mathematical optimization results.
    """

    # Optimization summary prompt
    summary_template: str = (
        "Summarize optimization results for {resource_type} allocation. "
        "Objective: {objective}. Solver: {solver_type}. "
        "Optimal value: {objective_value}. Projected ROI: {projected_roi}. "
        "Entities optimized: {entity_count}. Changes: {increase_count} increases, {decrease_count} decreases."
    )

    # Allocation recommendation prompt
    recommendation_template: str = (
        "Generate recommendations for {entity_id} ({entity_type}). "
        "Current: {current}. Optimized: {optimized}. Change: {change_pct}%. "
        "Expected impact: {expected_impact}."
    )

    # Scenario comparison prompt
    scenario_comparison_template: str = (
        "Compare scenarios: {scenario_names}. "
        "Best scenario: {best_scenario} with ROI {best_roi}. "
        "Constraint violations in alternatives: {violations}."
    )

    # Constraint warning prompt
    constraint_warning_template: str = (
        "Warning for constraint {constraint_type}: {description}. "
        "Value: {value}. Scope: {scope}. Impact on optimization: {impact}."
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
            "scenario_comparison_template": self.scenario_comparison_template,
            "constraint_warning_template": self.constraint_warning_template,
            "version": self.version,
            "last_optimized": self.last_optimized,
            "optimization_score": self.optimization_score,
        }


# =============================================================================
# 2. DSPy SIGNATURES (for feedback_learner optimization)
# =============================================================================

try:
    import dspy

    class OptimizationSummarySignature(dspy.Signature):
        """
        Generate optimization summary from results.

        This signature is optimized by feedback_learner and consumed by resource_optimizer.
        """

        optimization_results: str = dspy.InputField(desc="Mathematical optimization output")
        allocation_changes: str = dspy.InputField(desc="List of allocation changes")
        constraints_used: str = dspy.InputField(desc="Active constraints")
        objective_value: float = dspy.InputField(desc="Optimal objective value")

        executive_summary: str = dspy.OutputField(desc="Executive-friendly summary")
        key_changes: list = dspy.OutputField(desc="Most impactful allocation changes")
        implementation_priority: list = dspy.OutputField(desc="Order to implement changes")

    class AllocationRecommendationSignature(dspy.Signature):
        """
        Generate allocation recommendations.

        Creates actionable recommendations from optimization results.
        """

        entity_allocations: str = dspy.InputField(desc="Optimized allocations per entity")
        impact_projections: str = dspy.InputField(desc="Expected impact per entity")
        constraints: str = dspy.InputField(desc="Business constraints applied")

        recommendations: list = dspy.OutputField(desc="Prioritized allocation recommendations")
        risk_assessment: str = dspy.OutputField(desc="Risks of implementing changes")
        alternative_actions: list = dspy.OutputField(desc="If constraints prevent optimal")

    class ScenarioNarrativeSignature(dspy.Signature):
        """
        Generate narrative comparing optimization scenarios.

        Creates comparison narrative for what-if analysis.
        """

        scenarios: str = dspy.InputField(desc="Scenario results to compare")
        baseline: str = dspy.InputField(desc="Current baseline allocation")
        objective: str = dspy.InputField(desc="Optimization objective")

        comparison_narrative: str = dspy.OutputField(desc="Scenario comparison story")
        recommended_scenario: str = dspy.OutputField(desc="Best scenario with rationale")
        tradeoff_analysis: str = dspy.OutputField(desc="Key tradeoffs between scenarios")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Resource Optimizer agent (Recipient)")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using default optimization templates")
    OptimizationSummarySignature = None  # type: ignore[assignment, misc]
    AllocationRecommendationSignature = None  # type: ignore[assignment, misc]
    ScenarioNarrativeSignature = None  # type: ignore[assignment, misc]


# =============================================================================
# 3. PROMPT CONSUMER
# =============================================================================


class ResourceOptimizerDSPyIntegration:
    """
    DSPy integration for Resource Optimizer agent (Recipient role).

    Consumes optimized prompts from feedback_learner but does not
    generate training signals (mathematical optimization agent).
    """

    def __init__(self):
        self.dspy_type: Literal["recipient"] = "recipient"
        self._prompts = ResourceOptimizationPrompts()
        self._prompt_versions: Dict[str, str] = {}

    @property
    def prompts(self) -> ResourceOptimizationPrompts:
        """Get current optimized prompts."""
        return self._prompts  # type: ignore[no-any-return]

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
        if "scenario_comparison_template" in prompts:
            self._prompts.scenario_comparison_template = prompts["scenario_comparison_template"]
        if "constraint_warning_template" in prompts:
            self._prompts.constraint_warning_template = prompts["constraint_warning_template"]

        self._prompts.last_optimized = datetime.now(timezone.utc).isoformat()
        self._prompts.optimization_score = optimization_score
        self._prompts.version = f"1.{len(self._prompt_versions) + 1}"

        logger.info(
            f"Resource Optimizer prompts updated: version={self._prompts.version}, "
            f"score={optimization_score:.4f}"
        )

    def get_summary_prompt(
        self,
        resource_type: str,
        objective: str,
        solver_type: str,
        objective_value: float,
        projected_roi: float,
        entity_count: int,
        increase_count: int,
        decrease_count: int,
    ) -> str:
        """Get formatted summary prompt with current optimized template."""
        return self._prompts.summary_template.format(  # type: ignore[no-any-return]
            resource_type=resource_type,
            objective=objective,
            solver_type=solver_type,
            objective_value=objective_value,
            projected_roi=projected_roi,
            entity_count=entity_count,
            increase_count=increase_count,
            decrease_count=decrease_count,
        )

    def get_recommendation_prompt(
        self,
        entity_id: str,
        entity_type: str,
        current: float,
        optimized: float,
        change_pct: float,
        expected_impact: float,
    ) -> str:
        """Get formatted recommendation prompt."""
        return self._prompts.recommendation_template.format(  # type: ignore[no-any-return]
            entity_id=entity_id,
            entity_type=entity_type,
            current=current,
            optimized=optimized,
            change_pct=change_pct,
            expected_impact=expected_impact,
        )

    def get_scenario_comparison_prompt(
        self,
        scenario_names: str,
        best_scenario: str,
        best_roi: float,
        violations: str,
    ) -> str:
        """Get formatted scenario comparison prompt."""
        return self._prompts.scenario_comparison_template.format(  # type: ignore[no-any-return]
            scenario_names=scenario_names,
            best_scenario=best_scenario,
            best_roi=best_roi,
            violations=violations,
        )

    def get_constraint_warning_prompt(
        self,
        constraint_type: str,
        description: str,
        value: float,
        scope: str,
        impact: str,
    ) -> str:
        """Get formatted constraint warning prompt."""
        return self._prompts.constraint_warning_template.format(  # type: ignore[no-any-return]
            constraint_type=constraint_type,
            description=description,
            value=value,
            scope=scope,
            impact=impact,
        )

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about current prompts."""
        return {
            "agent": "resource_optimizer",
            "dspy_type": self.dspy_type,
            "prompts": self._prompts.to_dict(),
            "prompt_count": 4,
            "dspy_available": DSPY_AVAILABLE,
        }


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_dspy_integration: Optional[ResourceOptimizerDSPyIntegration] = None


def get_resource_optimizer_dspy_integration() -> ResourceOptimizerDSPyIntegration:
    """Get or create DSPy integration singleton."""
    global _dspy_integration
    if _dspy_integration is None:
        _dspy_integration = ResourceOptimizerDSPyIntegration()
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
    "ResourceOptimizationPrompts",
    # DSPy Signatures
    "OptimizationSummarySignature",
    "AllocationRecommendationSignature",
    "ScenarioNarrativeSignature",
    "DSPY_AVAILABLE",
    # Integration
    "ResourceOptimizerDSPyIntegration",
    # Access
    "get_resource_optimizer_dspy_integration",
    "reset_dspy_integration",
]
