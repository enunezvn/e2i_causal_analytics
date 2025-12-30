"""
Tests for Explainer DSPy Integration.

Tests the Recipient role implementation including:
- Optimized prompt templates
- Prompt consumer functionality
- DSPy signature availability
- Singleton pattern for integration
- Prompt formatting and updates
"""

import pytest
from unittest.mock import patch, MagicMock

# Mark all tests in this module as dspy_integration to group them
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


class TestExplanationPrompts:
    """Test ExplanationPrompts dataclass."""

    def test_default_initialization(self):
        """Test prompts initialize with default templates."""
        from src.agents.explainer.dspy_integration import ExplanationPrompts

        prompts = ExplanationPrompts()

        assert "{source_count}" in prompts.context_assembly_template
        assert "{user_expertise}" in prompts.executive_summary_template
        assert "{analysis_type}" in prompts.detailed_explanation_template
        assert "{analysis_summary}" in prompts.insight_extraction_template
        assert "{section_type}" in prompts.narrative_section_template
        assert "{explanation_summary}" in prompts.followup_questions_template
        assert prompts.version == "1.0"
        assert prompts.last_optimized == ""
        assert prompts.optimization_score == 0.0

    def test_to_dict_structure(self):
        """Test to_dict produces correct structure."""
        from src.agents.explainer.dspy_integration import ExplanationPrompts

        prompts = ExplanationPrompts(
            version="2.0",
            last_optimized="2025-12-30T12:00:00Z",
            optimization_score=0.92,
        )

        result = prompts.to_dict()

        assert "context_assembly_template" in result
        assert "executive_summary_template" in result
        assert "detailed_explanation_template" in result
        assert "insight_extraction_template" in result
        assert "narrative_section_template" in result
        assert "followup_questions_template" in result
        assert result["version"] == "2.0"
        assert result["last_optimized"] == "2025-12-30T12:00:00Z"
        assert result["optimization_score"] == 0.92


class TestExplainerDSPyIntegration:
    """Test ExplainerDSPyIntegration class (Recipient pattern)."""

    def test_initialization(self):
        """Test integration initializes correctly."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        assert integration.dspy_type == "recipient"
        assert integration.prompts is not None
        assert integration._prompt_versions == {}

    def test_prompts_property(self):
        """Test prompts property returns ExplanationPrompts."""
        from src.agents.explainer.dspy_integration import (
            ExplainerDSPyIntegration,
            ExplanationPrompts,
        )

        integration = ExplainerDSPyIntegration()

        assert isinstance(integration.prompts, ExplanationPrompts)

    def test_update_optimized_prompts(self):
        """Test updating prompts with optimized versions."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        new_prompts = {
            "context_assembly_template": "OPTIMIZED: Assemble from {source_count} sources for {user_expertise}. Types: {analysis_types}. Focus: {focus_areas}. Format: {output_format}.",
            "executive_summary_template": "OPTIMIZED: Summary for {user_expertise}. Findings: {key_findings_count}. Confidence: {avg_confidence}.",
        }

        integration.update_optimized_prompts(
            prompts=new_prompts,
            optimization_score=0.89,
        )

        assert "OPTIMIZED:" in integration.prompts.context_assembly_template
        assert "OPTIMIZED:" in integration.prompts.executive_summary_template
        assert integration.prompts.optimization_score == 0.89
        assert integration.prompts.last_optimized != ""
        assert integration.prompts.version == "1.1"

    def test_update_all_prompts(self):
        """Test updating all prompts."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        new_prompts = {
            "context_assembly_template": "A: {source_count} {user_expertise} {analysis_types} {focus_areas} {output_format}",
            "executive_summary_template": "B: {user_expertise} {key_findings_count} {avg_confidence}",
            "detailed_explanation_template": "C: {analysis_type} {insights} {themes} {user_expertise}",
            "insight_extraction_template": "D: {analysis_summary}",
            "narrative_section_template": "E: {section_type} {title} {has_data} {connection_points}",
            "followup_questions_template": "F: {explanation_summary} {gaps} {user_expertise}",
        }

        integration.update_optimized_prompts(prompts=new_prompts, optimization_score=0.95)

        assert integration.prompts.context_assembly_template.startswith("A:")
        assert integration.prompts.executive_summary_template.startswith("B:")
        assert integration.prompts.detailed_explanation_template.startswith("C:")
        assert integration.prompts.insight_extraction_template.startswith("D:")
        assert integration.prompts.narrative_section_template.startswith("E:")
        assert integration.prompts.followup_questions_template.startswith("F:")

    def test_get_context_assembly_prompt(self):
        """Test context assembly prompt formatting."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_context_assembly_prompt(
            source_count=5,
            user_expertise="executive",
            analysis_types="causal, prediction",
            focus_areas="churn reduction",
            output_format="narrative",
        )

        assert "5" in prompt
        assert "executive" in prompt
        assert "causal, prediction" in prompt
        assert "churn reduction" in prompt
        assert "narrative" in prompt

    def test_get_context_assembly_prompt_empty_focus(self):
        """Test context assembly with empty focus areas."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_context_assembly_prompt(
            source_count=3,
            user_expertise="analyst",
            analysis_types="gap",
            focus_areas="",  # Empty
            output_format="structured",
        )

        assert "all" in prompt  # Should default to "all"

    def test_get_executive_summary_prompt(self):
        """Test executive summary prompt formatting."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_executive_summary_prompt(
            user_expertise="data_scientist",
            key_findings_count=7,
            avg_confidence=0.85,
        )

        assert "data_scientist" in prompt
        assert "7" in prompt
        assert "0.85" in prompt

    def test_get_detailed_explanation_prompt(self):
        """Test detailed explanation prompt formatting."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_detailed_explanation_prompt(
            analysis_type="causal_impact",
            insights="High conversion in Northeast",
            themes="regional performance",
            user_expertise="analyst",
        )

        assert "causal_impact" in prompt
        assert "High conversion in Northeast" in prompt
        assert "regional performance" in prompt
        assert "analyst" in prompt

    def test_get_insight_extraction_prompt(self):
        """Test insight extraction prompt formatting."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_insight_extraction_prompt(
            analysis_summary="Analysis shows 23% improvement opportunity",
        )

        assert "23% improvement opportunity" in prompt

    def test_get_narrative_section_prompt(self):
        """Test narrative section prompt formatting."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_narrative_section_prompt(
            section_type="findings",
            title="Key Findings",
            has_data=True,
            connection_points="Introduction, Methodology",
        )

        assert "findings" in prompt
        assert "Key Findings" in prompt
        assert "yes" in prompt
        assert "Introduction, Methodology" in prompt

    def test_get_narrative_section_prompt_no_data(self):
        """Test narrative section with no data."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_narrative_section_prompt(
            section_type="recommendations",
            title="Recommendations",
            has_data=False,
            connection_points="Findings",
        )

        assert "no" in prompt

    def test_get_followup_questions_prompt(self):
        """Test follow-up questions prompt formatting."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_followup_questions_prompt(
            explanation_summary="Northeast region shows highest potential",
            gaps="Competitor analysis missing",
            user_expertise="executive",
        )

        assert "Northeast region" in prompt
        assert "Competitor analysis missing" in prompt
        assert "executive" in prompt

    def test_get_followup_questions_prompt_no_gaps(self):
        """Test follow-up questions with no gaps."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        prompt = integration.get_followup_questions_prompt(
            explanation_summary="Complete analysis",
            gaps="",  # Empty
            user_expertise="analyst",
        )

        assert "none identified" in prompt

    def test_get_prompt_metadata(self):
        """Test getting prompt metadata."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        metadata = integration.get_prompt_metadata()

        assert metadata["agent"] == "explainer"
        assert metadata["dspy_type"] == "recipient"
        assert metadata["prompt_count"] == 6
        assert "prompts" in metadata
        assert "dspy_available" in metadata


class TestSingletonAccess:
    """Test singleton pattern for DSPy integration."""

    def test_get_integration_creates_singleton(self):
        """Test that getter creates singleton."""
        from src.agents.explainer.dspy_integration import (
            get_explainer_dspy_integration,
            reset_dspy_integration,
        )

        reset_dspy_integration()

        integration1 = get_explainer_dspy_integration()
        integration2 = get_explainer_dspy_integration()

        assert integration1 is integration2

    def test_reset_clears_singleton(self):
        """Test that reset clears singleton."""
        from src.agents.explainer.dspy_integration import (
            get_explainer_dspy_integration,
            reset_dspy_integration,
        )

        integration1 = get_explainer_dspy_integration()
        reset_dspy_integration()
        integration2 = get_explainer_dspy_integration()

        assert integration1 is not integration2


class TestDSPySignatures:
    """Test DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE flag."""
        from src.agents.explainer.dspy_integration import DSPY_AVAILABLE

        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_explanation_synthesis_signature(self):
        """Test ExplanationSynthesisSignature is valid DSPy signature."""
        from src.agents.explainer.dspy_integration import (
            ExplanationSynthesisSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(ExplanationSynthesisSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_insight_extraction_signature(self):
        """Test InsightExtractionSignature is valid DSPy signature."""
        from src.agents.explainer.dspy_integration import (
            InsightExtractionSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(InsightExtractionSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_narrative_structure_signature(self):
        """Test NarrativeStructureSignature is valid DSPy signature."""
        from src.agents.explainer.dspy_integration import (
            NarrativeStructureSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(NarrativeStructureSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_query_rewrite_signature(self):
        """Test QueryRewriteForExplanationSignature is valid DSPy signature."""
        from src.agents.explainer.dspy_integration import (
            QueryRewriteForExplanationSignature,
            DSPY_AVAILABLE,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(QueryRewriteForExplanationSignature, dspy.Signature)


class TestExplanationWorkflow:
    """Test the explanation generation workflow."""

    def test_full_explanation_prompt_workflow(self):
        """Test generating prompts for a full explanation workflow."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        # Step 1: Context assembly
        context_prompt = integration.get_context_assembly_prompt(
            source_count=4,
            user_expertise="executive",
            analysis_types="causal, prediction, gap",
            focus_areas="Northeast region performance",
            output_format="narrative",
        )

        assert len(context_prompt) > 0

        # Step 2: Executive summary
        summary_prompt = integration.get_executive_summary_prompt(
            user_expertise="executive",
            key_findings_count=5,
            avg_confidence=0.87,
        )

        assert len(summary_prompt) > 0

        # Step 3: Detailed explanation
        detail_prompt = integration.get_detailed_explanation_prompt(
            analysis_type="causal",
            insights="Territory shows 23% improvement potential",
            themes="Regional performance, HCP engagement",
            user_expertise="executive",
        )

        assert len(detail_prompt) > 0

        # Step 4: Follow-up questions
        followup_prompt = integration.get_followup_questions_prompt(
            explanation_summary="Analysis complete for Northeast",
            gaps="Competitor data limited",
            user_expertise="executive",
        )

        assert len(followup_prompt) > 0

    def test_audience_adaptation(self):
        """Test prompts adapt to different audiences."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        audiences = ["executive", "analyst", "data_scientist"]

        for audience in audiences:
            prompt = integration.get_executive_summary_prompt(
                user_expertise=audience,
                key_findings_count=3,
                avg_confidence=0.8,
            )

            assert audience in prompt
