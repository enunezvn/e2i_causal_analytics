"""Tests for NarrativeGeneratorNode."""

import pytest

from src.agents.explainer.nodes.context_assembler import ContextAssemblerNode
from src.agents.explainer.nodes.deep_reasoner import DeepReasonerNode
from src.agents.explainer.nodes.narrative_generator import NarrativeGeneratorNode


class TestNarrativeGeneratorNode:
    """Tests for the NarrativeGeneratorNode."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_init_deterministic_mode(self):
        """Test node initialization in deterministic mode."""
        node = NarrativeGeneratorNode(use_llm=False)
        assert node.use_llm is False
        assert node.llm is None

    def test_init_llm_mode(self, mock_llm):
        """Test node initialization in LLM mode."""
        node = NarrativeGeneratorNode(use_llm=True, llm=mock_llm)
        assert node.use_llm is True
        assert node.llm is mock_llm

    # ========================================================================
    # HELPER: Get reasoned state
    # ========================================================================

    async def _get_reasoned_state(self, base_state, analysis_results):
        """Helper to get a reasoned state for testing generation."""
        assembler = ContextAssemblerNode()
        state = {**base_state, "analysis_results": analysis_results}
        assembled = await assembler.execute(state)

        reasoner = DeepReasonerNode(use_llm=False)
        return await reasoner.execute(assembled)

    # ========================================================================
    # BASIC EXECUTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_generates_executive_summary(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that executive summary is generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["status"] == "completed"
        assert result["executive_summary"] is not None
        assert len(result["executive_summary"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_detailed_explanation(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that detailed explanation is generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["detailed_explanation"] is not None
        assert len(result["detailed_explanation"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_narrative_sections(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that narrative sections are generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["narrative_sections"] is not None
        assert len(result["narrative_sections"]) > 0

    # ========================================================================
    # OUTPUT FORMAT TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_narrative_format(self, base_explainer_state, sample_causal_analysis):
        """Test narrative output format."""
        state = {**base_explainer_state, "output_format": "narrative"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Narrative should be prose
        assert result["detailed_explanation"] is not None
        assert len(result["detailed_explanation"]) > 50  # Substantial content

    @pytest.mark.asyncio
    async def test_brief_format(self, base_explainer_state, sample_causal_analysis):
        """Test brief output format."""
        state = {**base_explainer_state, "output_format": "brief"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Brief should have summary
        assert result["executive_summary"] is not None
        assert len(result["executive_summary"]) > 0

    @pytest.mark.asyncio
    async def test_structured_format(self, base_explainer_state, sample_causal_analysis):
        """Test structured output format."""
        state = {**base_explainer_state, "output_format": "structured"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Structured should have sections or explanation
        assert result["detailed_explanation"] is not None

    @pytest.mark.asyncio
    async def test_presentation_format(self, base_explainer_state, sample_causal_analysis):
        """Test presentation output format."""
        state = {**base_explainer_state, "output_format": "presentation"}
        reasoned = await self._get_reasoned_state(state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Presentation should have slides/sections
        assert result["narrative_sections"] is not None
        assert len(result["narrative_sections"]) > 0

    # ========================================================================
    # SECTION GENERATION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_sections_have_required_fields(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that narrative sections have required fields."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        for section in result["narrative_sections"]:
            assert "title" in section
            assert "content" in section

    # ========================================================================
    # VISUAL SUGGESTIONS TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generates_visual_suggestions(self, base_explainer_state, sample_causal_analysis):
        """Test that visual suggestions are generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["visual_suggestions"] is not None
        assert len(result["visual_suggestions"]) > 0

    @pytest.mark.asyncio
    async def test_visual_suggestions_have_type(self, base_explainer_state, sample_causal_analysis):
        """Test that visual suggestions have type field."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        for visual in result["visual_suggestions"]:
            assert "type" in visual

    # ========================================================================
    # FOLLOW-UP QUESTIONS TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_generates_follow_up_questions(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that follow-up questions are generated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["follow_up_questions"] is not None
        assert len(result["follow_up_questions"]) > 0

    @pytest.mark.asyncio
    async def test_follow_up_questions_are_questions(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that follow-ups are actual questions."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        for question in result["follow_up_questions"]:
            # Questions should end with ? or be interrogative
            is_question = (
                question.endswith("?")
                or question.lower().startswith("what")
                or question.lower().startswith("how")
                or question.lower().startswith("why")
                or question.lower().startswith("when")
                or question.lower().startswith("can")
            )
            assert is_question, f"Not a question: {question}"

    # ========================================================================
    # EDGE CASE TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_handles_failed_status(self, base_explainer_state):
        """Test that already-failed state is passed through."""
        node = NarrativeGeneratorNode(use_llm=False)
        state = {**base_explainer_state, "status": "failed"}

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_handles_empty_insights(self, base_explainer_state, sample_causal_analysis):
        """Test handling of empty insights list."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])
        # Override with empty insights
        reasoned["extracted_insights"] = []

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Should still complete
        assert result["status"] == "completed"
        assert result["executive_summary"] is not None

    # ========================================================================
    # LATENCY TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_tracks_generation_latency(self, base_explainer_state, sample_causal_analysis):
        """Test that generation latency is tracked."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["generation_latency_ms"] >= 0
        assert isinstance(result["generation_latency_ms"], int)

    @pytest.mark.asyncio
    async def test_calculates_total_latency(self, base_explainer_state, sample_causal_analysis):
        """Test that total latency is calculated."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Total should include all phases
        total = result["total_latency_ms"]
        assert total >= result["generation_latency_ms"]

    # ========================================================================
    # MODEL TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_preserves_model_used(self, base_explainer_state, sample_causal_analysis):
        """Test that model used is preserved from reasoning phase."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        assert result["model_used"] == "deterministic"

    # ========================================================================
    # CONTENT QUALITY TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_summary_is_reasonable_length(self, base_explainer_state, sample_causal_analysis):
        """Test that executive summary has reasonable length."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Summary should be concise but informative
        summary_length = len(result["executive_summary"])
        assert 10 < summary_length < 5000  # Reasonable bounds

    @pytest.mark.asyncio
    async def test_explanation_has_substance(self, base_explainer_state, sample_causal_analysis):
        """Test that detailed explanation has substance."""
        reasoned = await self._get_reasoned_state(base_explainer_state, [sample_causal_analysis])

        node = NarrativeGeneratorNode(use_llm=False)
        result = await node.execute(reasoned)

        # Explanation should have meaningful content
        assert len(result["detailed_explanation"]) > 20
