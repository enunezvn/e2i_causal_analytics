"""Tests for ContextAssemblerNode."""

import pytest

from src.agents.explainer.nodes.context_assembler import ContextAssemblerNode


class TestContextAssemblerNode:
    """Tests for the ContextAssemblerNode."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_init_without_store(self):
        """Test node initialization without conversation store."""
        node = ContextAssemblerNode()
        assert node.conversation_store is None

    def test_init_with_store(self, mock_conversation_store):
        """Test node initialization with conversation store."""
        node = ContextAssemblerNode(mock_conversation_store)
        assert node.conversation_store is mock_conversation_store

    # ========================================================================
    # BASIC EXECUTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_with_single_result(self, base_explainer_state, sample_causal_analysis):
        """Test context assembly with single analysis result."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": [sample_causal_analysis]}

        result = await node.execute(state)

        # Status transitions to "reasoning" for next phase
        assert result["status"] == "reasoning"
        assert result["analysis_context"] is not None
        assert isinstance(result["analysis_context"], list)
        assert len(result["analysis_context"]) == 1
        assert result["assembly_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_with_multiple_results(
        self, base_explainer_state, sample_analysis_results
    ):
        """Test context assembly with multiple analysis results."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": sample_analysis_results}

        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert isinstance(result["analysis_context"], list)
        assert len(result["analysis_context"]) == 3

    @pytest.mark.asyncio
    async def test_execute_extracts_context_fields(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that context fields are correctly extracted from analysis results."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": [sample_causal_analysis]}

        result = await node.execute(state)

        context = result["analysis_context"][0]
        assert "source_agent" in context
        assert context["source_agent"] == "causal_impact"
        assert "analysis_type" in context
        assert context["analysis_type"] == "effect_estimation"
        assert "key_findings" in context

    # ========================================================================
    # EDGE CASE TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_with_empty_results(self, base_explainer_state):
        """Test context assembly with no analysis results."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": []}

        result = await node.execute(state)

        # Should fail with no results
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_execute_with_minimal_result(self, base_explainer_state, minimal_analysis_result):
        """Test context assembly with minimal analysis result."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": [minimal_analysis_result]}

        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert len(result["analysis_context"]) == 1

    @pytest.mark.asyncio
    async def test_execute_handles_failed_status(self, base_explainer_state):
        """Test that already-failed state is passed through."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "status": "failed"}

        result = await node.execute(state)

        assert result["status"] == "failed"

    # ========================================================================
    # CONVERSATION HISTORY TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_loads_conversation_history(
        self, base_explainer_state, sample_causal_analysis, mock_conversation_store
    ):
        """Test that conversation history is loaded when store is provided."""
        node = ContextAssemblerNode(mock_conversation_store)
        state = {**base_explainer_state, "analysis_results": [sample_causal_analysis]}

        result = await node.execute(state)

        assert result["conversation_history"] is not None
        assert len(result["conversation_history"]) > 0

    @pytest.mark.asyncio
    async def test_execute_without_conversation_history(
        self, base_explainer_state, sample_causal_analysis
    ):
        """Test that execution works without conversation store."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": [sample_causal_analysis]}

        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert result["conversation_history"] == []

    # ========================================================================
    # USER CONTEXT TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_extracts_user_context(self, base_explainer_state, sample_causal_analysis):
        """Test that user context is extracted."""
        node = ContextAssemblerNode()
        state = {
            **base_explainer_state,
            "analysis_results": [sample_causal_analysis],
            "user_expertise": "executive",
            "output_format": "brief",
        }

        result = await node.execute(state)

        assert result["user_context"] is not None
        assert result["user_context"]["expertise"] == "executive"
        assert result["user_context"]["output_format"] == "brief"

    @pytest.mark.asyncio
    async def test_preserves_user_expertise(self, base_explainer_state, sample_causal_analysis):
        """Test that user expertise level is preserved in state."""
        node = ContextAssemblerNode()
        state = {
            **base_explainer_state,
            "analysis_results": [sample_causal_analysis],
            "user_expertise": "executive",
        }

        result = await node.execute(state)

        assert result["user_expertise"] == "executive"

    @pytest.mark.asyncio
    async def test_preserves_focus_areas(self, base_explainer_state, sample_causal_analysis):
        """Test that focus areas are preserved."""
        node = ContextAssemblerNode()
        focus = ["sales", "regional"]
        state = {
            **base_explainer_state,
            "analysis_results": [sample_causal_analysis],
            "focus_areas": focus,
        }

        result = await node.execute(state)

        assert result["focus_areas"] == focus

    # ========================================================================
    # LATENCY TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_tracks_assembly_latency(self, base_explainer_state, sample_causal_analysis):
        """Test that assembly latency is tracked."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": [sample_causal_analysis]}

        result = await node.execute(state)

        assert result["assembly_latency_ms"] >= 0
        assert isinstance(result["assembly_latency_ms"], int)

    # ========================================================================
    # KEY FINDINGS EXTRACTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_extracts_dict_key_findings(self, base_explainer_state, sample_causal_analysis):
        """Test extraction of dictionary-style key findings."""
        node = ContextAssemblerNode()
        state = {**base_explainer_state, "analysis_results": [sample_causal_analysis]}

        result = await node.execute(state)

        context = result["analysis_context"][0]
        # key_findings from dict should be converted to list of strings
        assert isinstance(context["key_findings"], list)
        assert len(context["key_findings"]) > 0

    @pytest.mark.asyncio
    async def test_handles_list_key_findings(self, base_explainer_state):
        """Test handling of list-style key findings."""
        node = ContextAssemblerNode()
        analysis = {
            "agent": "test",
            "analysis_type": "test",
            "key_findings": ["Finding 1", "Finding 2"],
            "status": "completed",
        }
        state = {**base_explainer_state, "analysis_results": [analysis]}

        result = await node.execute(state)

        context = result["analysis_context"][0]
        assert context["key_findings"] == ["Finding 1", "Finding 2"]
