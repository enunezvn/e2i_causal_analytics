"""
E2I Explainer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy prompt optimization for explainer Recipient role

The Explainer agent is a DSPy Recipient agent that:
1. Consumes optimized prompts for explanation generation
2. Uses QueryRewriteSignature-optimized templates
3. Does NOT generate training signals (consumes from other agents)

Note: Although the Explainer uses LLM for deep reasoning, it is a Recipient
in the DSPy architecture because it consumes optimized prompts rather than
generating training signals for optimization.
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
class ExplanationPrompts:
    """
    Optimized prompt templates for explanation generation.

    These prompts are consumed from feedback_learner after MIPROv2 optimization.
    The Explainer uses LLM for generation but the prompts themselves are
    optimized via the DSPy Recipient pattern.
    """

    # Context assembly prompt
    context_assembly_template: str = (
        "Assemble context from {source_count} agent outputs for {user_expertise} audience. "
        "Analysis types: {analysis_types}. Focus areas: {focus_areas}. "
        "Output format: {output_format}."
    )

    # Executive summary prompt
    executive_summary_template: str = (
        "Generate an executive summary for {user_expertise} audience. "
        "Key findings: {key_findings_count}. Confidence: {avg_confidence}. "
        "Focus on business impact and actionable insights."
    )

    # Detailed explanation prompt
    detailed_explanation_template: str = (
        "Create detailed explanation of {analysis_type} results. "
        "Insights: {insights}. Themes: {themes}. "
        "Adapt technical depth for {user_expertise} audience."
    )

    # Insight extraction prompt
    insight_extraction_template: str = (
        "Extract insights from analysis: {analysis_summary}. "
        "Categorize as: finding, recommendation, warning, or opportunity. "
        "Assign priority and actionability levels."
    )

    # Narrative section prompt
    narrative_section_template: str = (
        "Generate narrative section: {section_type}. "
        "Title: {title}. Supporting data: {has_data}. "
        "Connect to previous sections: {connection_points}."
    )

    # Follow-up questions prompt
    followup_questions_template: str = (
        "Generate follow-up questions based on: {explanation_summary}. "
        "Gaps in analysis: {gaps}. User expertise: {user_expertise}. "
        "Suggest questions that deepen understanding."
    )

    # Optimized by MIPROv2
    version: str = "1.0"
    last_optimized: str = ""
    optimization_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_assembly_template": self.context_assembly_template,
            "executive_summary_template": self.executive_summary_template,
            "detailed_explanation_template": self.detailed_explanation_template,
            "insight_extraction_template": self.insight_extraction_template,
            "narrative_section_template": self.narrative_section_template,
            "followup_questions_template": self.followup_questions_template,
            "version": self.version,
            "last_optimized": self.last_optimized,
            "optimization_score": self.optimization_score,
        }


# =============================================================================
# 2. DSPy SIGNATURES (for feedback_learner optimization)
# =============================================================================

try:
    import dspy

    class ExplanationSynthesisSignature(dspy.Signature):
        """
        Synthesize explanations from multi-agent analysis.

        This signature is optimized by feedback_learner and consumed by explainer.
        """

        analysis_results: str = dspy.InputField(desc="Results from multiple agents")
        user_expertise: str = dspy.InputField(desc="executive, analyst, or data_scientist")
        focus_areas: str = dspy.InputField(desc="Areas user wants to focus on")
        output_format: str = dspy.InputField(desc="narrative, structured, presentation, brief")

        executive_summary: str = dspy.OutputField(desc="Brief overview for executives")
        detailed_explanation: str = dspy.OutputField(desc="Full narrative explanation")
        key_insights: list = dspy.OutputField(desc="Extracted actionable insights")

    class InsightExtractionSignature(dspy.Signature):
        """
        Extract structured insights from analysis.

        Creates categorized, prioritized insights from raw results.
        """

        analysis_data: str = dspy.InputField(desc="Raw analysis output")
        context: str = dspy.InputField(desc="Business and user context")

        insights: list = dspy.OutputField(desc="Categorized insights with priority")
        themes: list = dspy.OutputField(desc="Key themes across insights")
        recommendations: list = dspy.OutputField(desc="Actionable recommendations")

    class NarrativeStructureSignature(dspy.Signature):
        """
        Plan narrative structure for explanation.

        Determines optimal structure and flow for explanation.
        """

        insights: str = dspy.InputField(desc="Extracted insights")
        audience: str = dspy.InputField(desc="Target audience expertise level")
        format: str = dspy.InputField(desc="Output format requested")

        sections: list = dspy.OutputField(desc="Ordered narrative sections")
        transitions: list = dspy.OutputField(desc="Transition points between sections")
        emphasis_points: list = dspy.OutputField(desc="Key points to emphasize")

    class QueryRewriteForExplanationSignature(dspy.Signature):
        """
        Rewrite queries for explanation retrieval.

        Optimizes queries for retrieving relevant context for explanations.
        """

        original_query: str = dspy.InputField(desc="User's original question")
        analysis_type: str = dspy.InputField(desc="Type of analysis being explained")
        user_expertise: str = dspy.InputField(desc="User's expertise level")

        rewritten_query: str = dspy.OutputField(desc="Optimized query for retrieval")
        context_keywords: list = dspy.OutputField(desc="Key terms to include")
        scope_refinement: str = dspy.OutputField(desc="How to scope the explanation")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Explainer agent (Recipient)")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using default explanation templates")
    ExplanationSynthesisSignature = None
    InsightExtractionSignature = None
    NarrativeStructureSignature = None
    QueryRewriteForExplanationSignature = None


# =============================================================================
# 3. PROMPT CONSUMER
# =============================================================================


class ExplainerDSPyIntegration:
    """
    DSPy integration for Explainer agent (Recipient role).

    Consumes optimized prompts from feedback_learner. Although the Explainer
    uses LLM for deep reasoning, it receives optimized prompts rather than
    generating training signals.
    """

    def __init__(self):
        self.dspy_type: Literal["recipient"] = "recipient"
        self._prompts = ExplanationPrompts()
        self._prompt_versions: Dict[str, str] = {}

    @property
    def prompts(self) -> ExplanationPrompts:
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
        if "context_assembly_template" in prompts:
            self._prompts.context_assembly_template = prompts["context_assembly_template"]
        if "executive_summary_template" in prompts:
            self._prompts.executive_summary_template = prompts["executive_summary_template"]
        if "detailed_explanation_template" in prompts:
            self._prompts.detailed_explanation_template = prompts["detailed_explanation_template"]
        if "insight_extraction_template" in prompts:
            self._prompts.insight_extraction_template = prompts["insight_extraction_template"]
        if "narrative_section_template" in prompts:
            self._prompts.narrative_section_template = prompts["narrative_section_template"]
        if "followup_questions_template" in prompts:
            self._prompts.followup_questions_template = prompts["followup_questions_template"]

        self._prompts.last_optimized = datetime.now(timezone.utc).isoformat()
        self._prompts.optimization_score = optimization_score
        self._prompts.version = f"1.{len(self._prompt_versions) + 1}"

        logger.info(
            f"Explainer prompts updated: version={self._prompts.version}, "
            f"score={optimization_score:.4f}"
        )

    def get_context_assembly_prompt(
        self,
        source_count: int,
        user_expertise: str,
        analysis_types: str,
        focus_areas: str,
        output_format: str,
    ) -> str:
        """Get formatted context assembly prompt."""
        return self._prompts.context_assembly_template.format(
            source_count=source_count,
            user_expertise=user_expertise,
            analysis_types=analysis_types,
            focus_areas=focus_areas or "all",
            output_format=output_format,
        )

    def get_executive_summary_prompt(
        self,
        user_expertise: str,
        key_findings_count: int,
        avg_confidence: float,
    ) -> str:
        """Get formatted executive summary prompt."""
        return self._prompts.executive_summary_template.format(
            user_expertise=user_expertise,
            key_findings_count=key_findings_count,
            avg_confidence=avg_confidence,
        )

    def get_detailed_explanation_prompt(
        self,
        analysis_type: str,
        insights: str,
        themes: str,
        user_expertise: str,
    ) -> str:
        """Get formatted detailed explanation prompt."""
        return self._prompts.detailed_explanation_template.format(
            analysis_type=analysis_type,
            insights=insights,
            themes=themes,
            user_expertise=user_expertise,
        )

    def get_insight_extraction_prompt(
        self,
        analysis_summary: str,
    ) -> str:
        """Get formatted insight extraction prompt."""
        return self._prompts.insight_extraction_template.format(
            analysis_summary=analysis_summary,
        )

    def get_narrative_section_prompt(
        self,
        section_type: str,
        title: str,
        has_data: bool,
        connection_points: str,
    ) -> str:
        """Get formatted narrative section prompt."""
        return self._prompts.narrative_section_template.format(
            section_type=section_type,
            title=title,
            has_data="yes" if has_data else "no",
            connection_points=connection_points,
        )

    def get_followup_questions_prompt(
        self,
        explanation_summary: str,
        gaps: str,
        user_expertise: str,
    ) -> str:
        """Get formatted follow-up questions prompt."""
        return self._prompts.followup_questions_template.format(
            explanation_summary=explanation_summary,
            gaps=gaps or "none identified",
            user_expertise=user_expertise,
        )

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about current prompts."""
        return {
            "agent": "explainer",
            "dspy_type": self.dspy_type,
            "prompts": self._prompts.to_dict(),
            "prompt_count": 6,
            "dspy_available": DSPY_AVAILABLE,
        }


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_dspy_integration: Optional[ExplainerDSPyIntegration] = None


def get_explainer_dspy_integration() -> ExplainerDSPyIntegration:
    """Get or create DSPy integration singleton."""
    global _dspy_integration
    if _dspy_integration is None:
        _dspy_integration = ExplainerDSPyIntegration()
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
    "ExplanationPrompts",
    # DSPy Signatures
    "ExplanationSynthesisSignature",
    "InsightExtractionSignature",
    "NarrativeStructureSignature",
    "QueryRewriteForExplanationSignature",
    "DSPY_AVAILABLE",
    # Integration
    "ExplainerDSPyIntegration",
    # Access
    "get_explainer_dspy_integration",
    "reset_dspy_integration",
]
