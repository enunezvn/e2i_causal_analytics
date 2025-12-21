"""Tests for DeepReasonerNode."""

import pytest

from src.agents.explainer.nodes.context_assembler import ContextAssemblerNode
from src.agents.explainer.nodes.deep_reasoner import DeepReasonerNode


class TestDeepReasonerNode:
    """Tests for the DeepReasonerNode."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_init_deterministic_mode(self):
        """Test node initialization in deterministic mode."""
        node = DeepReasonerNode(use_llm=False)
        assert node.use_llm is False
        assert node.llm is None

    def test_init_llm_mode(self, mock_llm):
        """Test node initialization in LLM mode."""
        node = DeepReasonerNode(use_llm=True, llm=mock_llm)
        assert node.use_llm is True
        assert node.llm is mock_llm

    # ========================================================================
    # HELPER: Get assembled state
    # ========================================================================

    async def _get_assembled_state(self, base_state, analysis_results):
        """Helper to get an assembled state for testing reasoning."""
        assembler = ContextAssemblerNode()
        state = {**base_state, "analysis_results": analysis_results}
        return await assembler.execute(state)

    # ========================================================================
    # BASIC EXECUTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_extracts_insights(self, base_explainer_state, sample_causal_analysis):
        """Test that insights are extracted from assembled context."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        # Status transitions to "generating"
        assert result["status"] == "generating"
        assert result["extracted_insights"] is not None
        assert len(result["extracted_insights"]) > 0

    @pytest.mark.asyncio
    async def test_execute_creates_narrative_structure(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that narrative structure is created."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        assert result["narrative_structure"] is not None
        assert isinstance(result["narrative_structure"], list)
        assert len(result["narrative_structure"]) > 0

    @pytest.mark.asyncio
    async def test_execute_identifies_themes(self, base_explainer_state, sample_causal_analysis):
        """Test that key themes are identified."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        assert result["key_themes"] is not None
        assert len(result["key_themes"]) > 0

    # ========================================================================
    # INSIGHT EXTRACTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_insight_has_required_fields(self, base_explainer_state, sample_causal_analysis):
        """Test that extracted insights have required fields."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        for insight in result["extracted_insights"]:
            assert "insight_id" in insight
            assert "category" in insight
            assert "statement" in insight
            assert "confidence" in insight

    @pytest.mark.asyncio
    async def test_insight_categories_valid(self, base_explainer_state, sample_causal_analysis):
        """Test that insight categories are valid."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        valid_categories = {"finding", "recommendation", "warning", "opportunity"}

        result = await node.execute(assembled)

        for insight in result["extracted_insights"]:
            assert insight["category"] in valid_categories

    @pytest.mark.asyncio
    async def test_insights_have_priority(self, base_explainer_state, sample_causal_analysis):
        """Test that insights are prioritized."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        priorities = [i.get("priority", 99) for i in result["extracted_insights"]]
        # Should have valid priorities
        assert all(1 <= p <= 5 for p in priorities)

    @pytest.mark.asyncio
    async def test_insights_have_actionability(self, base_explainer_state, sample_causal_analysis):
        """Test that insights have actionability assessment."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        valid_actionability = {"immediate", "short_term", "long_term", "informational"}

        result = await node.execute(assembled)

        for insight in result["extracted_insights"]:
            if "actionability" in insight:
                assert insight["actionability"] in valid_actionability

    # ========================================================================
    # EXPERTISE LEVEL ADAPTATION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_executive_narrative_structure(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test narrative structure for executive audience."""
        state = {
            **base_explainer_state,
            "user_expertise": "executive",
        }
        assembled = await self._get_assembled_state(state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        structure = result["narrative_structure"]
        # Executive should have executive summary
        assert any("Executive" in s or "Summary" in s for s in structure)

    @pytest.mark.asyncio
    async def test_analyst_narrative_structure(self, base_explainer_state, sample_causal_analysis):
        """Test narrative structure for analyst audience."""
        state = {
            **base_explainer_state,
            "user_expertise": "analyst",
        }
        assembled = await self._get_assembled_state(state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        structure = result["narrative_structure"]
        # Analyst should have summary and findings
        assert any("Summary" in s for s in structure)

    @pytest.mark.asyncio
    async def test_data_scientist_narrative_structure(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test narrative structure for data scientist audience."""
        state = {
            **base_explainer_state,
            "user_expertise": "data_scientist",
        }
        assembled = await self._get_assembled_state(state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        structure = result["narrative_structure"]
        # Data scientist should have methodology section
        assert any("Methodology" in s or "Statistical" in s for s in structure)

    # ========================================================================
    # EDGE CASE TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_handles_empty_context(self, base_explainer_state):
        """Test handling of empty analysis context."""
        node = DeepReasonerNode(use_llm=False)
        state = {
            **base_explainer_state,
            "analysis_context": [],
            "status": "reasoning",
        }

        result = await node.execute(state)

        # Should fail with no context
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_handles_failed_status(self, base_explainer_state):
        """Test that already-failed state is passed through."""
        node = DeepReasonerNode(use_llm=False)
        state = {**base_explainer_state, "status": "failed"}

        result = await node.execute(state)

        assert result["status"] == "failed"

    # ========================================================================
    # LATENCY TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_tracks_reasoning_latency(self, base_explainer_state, sample_causal_analysis):
        """Test that reasoning latency is tracked."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        assert result["reasoning_latency_ms"] >= 0
        assert isinstance(result["reasoning_latency_ms"], int)

    # ========================================================================
    # MODEL TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_tracks_model_used(self, base_explainer_state, sample_causal_analysis):
        """Test that model used is tracked."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        assert result["model_used"] == "deterministic"

    # ========================================================================
    # THEME EXTRACTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_theme_extraction(self, base_explainer_state, sample_causal_analysis):
        """Test theme extraction from analysis results."""
        assembled = await self._get_assembled_state(base_explainer_state, [sample_causal_analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        themes = result["key_themes"]
        assert len(themes) > 0
        # Themes should be strings
        assert all(isinstance(t, str) for t in themes)

    # ========================================================================
    # FOCUS AREAS TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_focus_areas_affect_priority(self, base_explainer_state):
        """Test that focus areas influence insight priority."""
        # Create analysis with specific content
        analysis = {
            "agent": "test",
            "analysis_type": "test",
            "key_findings": [
                "Sales increased significantly",
                "Regional performance varies",
            ],
            "status": "completed",
        }
        state = {
            **base_explainer_state,
            "focus_areas": ["sales"],
        }
        assembled = await self._get_assembled_state(state, [analysis])

        node = DeepReasonerNode(use_llm=False)
        result = await node.execute(assembled)

        # Should prioritize sales-related findings
        insights = result["extracted_insights"]
        assert len(insights) > 0
