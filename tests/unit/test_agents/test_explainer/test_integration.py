"""Integration tests for Explainer Agent."""

import pytest

from src.agents.explainer import (
    ExplainerAgent,
    ExplainerInput,
    ExplainerOutput,
    build_explainer_graph,
    build_simple_explainer_graph,
    explain_analysis,
)


class TestExplainerAgentIntegration:
    """Integration tests for the complete Explainer agent pipeline."""

    # ========================================================================
    # END-TO-END PIPELINE TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_full_pipeline_single_result(self, sample_causal_analysis):
        """Test complete pipeline with single analysis result."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
            query="What is the causal effect?",
        )

        assert result.status == "completed"
        assert result.executive_summary != ""
        assert result.detailed_explanation != ""
        assert len(result.extracted_insights) > 0
        assert result.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_full_pipeline_multiple_results(self, sample_analysis_results):
        """Test complete pipeline with multiple analysis results."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Summarize all findings",
        )

        assert result.status == "completed"
        assert result.executive_summary != ""
        assert len(result.extracted_insights) > 0
        assert len(result.narrative_sections) > 0
        assert len(result.follow_up_questions) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_empty_results(self):
        """Test pipeline handles empty results."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[],
            query="What happened?",
        )

        # Should fail gracefully with empty results
        assert result.status == "failed"

    # ========================================================================
    # EXPERTISE LEVEL TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_executive_audience(self, sample_analysis_results):
        """Test explanation for executive audience."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="What should I know?",
            user_expertise="executive",
            output_format="brief",
        )

        assert result.status == "completed"
        # Executive should have concise summary
        assert len(result.executive_summary) > 0

    @pytest.mark.asyncio
    async def test_analyst_audience(self, sample_analysis_results):
        """Test explanation for analyst audience."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Give me the analysis details",
            user_expertise="analyst",
            output_format="narrative",
        )

        assert result.status == "completed"
        assert len(result.detailed_explanation) > 0
        assert len(result.narrative_sections) > 0

    @pytest.mark.asyncio
    async def test_data_scientist_audience(self, sample_analysis_results):
        """Test explanation for data scientist audience."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Show me the methodology",
            user_expertise="data_scientist",
            output_format="structured",
        )

        assert result.status == "completed"
        assert len(result.detailed_explanation) > 0

    # ========================================================================
    # OUTPUT FORMAT TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_narrative_format(self, sample_causal_analysis):
        """Test narrative output format."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
            output_format="narrative",
        )

        assert result.status == "completed"
        assert len(result.detailed_explanation) > 50

    @pytest.mark.asyncio
    async def test_structured_format(self, sample_causal_analysis):
        """Test structured output format."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
            output_format="structured",
        )

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_presentation_format(self, sample_causal_analysis):
        """Test presentation output format."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
            output_format="presentation",
        )

        assert result.status == "completed"
        assert len(result.narrative_sections) > 0

    @pytest.mark.asyncio
    async def test_brief_format(self, sample_causal_analysis):
        """Test brief output format."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
            output_format="brief",
        )

        assert result.status == "completed"
        assert len(result.executive_summary) > 0

    # ========================================================================
    # CONVENIENCE METHOD TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_summarize_method(self, sample_analysis_results):
        """Test summarize convenience method."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.summarize(
            analysis_results=sample_analysis_results,
            query="Quick summary please",
        )

        assert result.status == "completed"
        # Summarize uses executive + brief
        assert len(result.executive_summary) > 0

    @pytest.mark.asyncio
    async def test_explain_for_audience_method(self, sample_analysis_results):
        """Test explain_for_audience convenience method."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain_for_audience(
            analysis_results=sample_analysis_results,
            audience="executive",
            query="What's the bottom line?",
        )

        assert result.status == "completed"

    # ========================================================================
    # HANDOFF TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_get_handoff(self, sample_analysis_results):
        """Test handoff generation for orchestrator."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Analyze this",
        )

        handoff = agent.get_handoff(result)

        assert handoff["agent"] == "explainer"
        assert handoff["analysis_type"] == "explanation"
        assert "key_findings" in handoff
        assert "outputs" in handoff
        assert "suggestions" in handoff
        assert handoff["suggested_next_agent"] == "feedback_learner"

    @pytest.mark.asyncio
    async def test_handoff_on_failure(self):
        """Test handoff when explanation fails."""
        agent = ExplainerAgent(use_llm=False)

        # Create a failed output
        output = ExplainerOutput(status="failed")

        handoff = agent.get_handoff(output)

        assert handoff["requires_further_analysis"] is True
        assert handoff["suggested_next_agent"] is None

    # ========================================================================
    # CONVENIENCE FUNCTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_explain_analysis_function(self, sample_causal_analysis):
        """Test explain_analysis convenience function."""
        result = await explain_analysis(
            analysis_results=[sample_causal_analysis],
            query="Explain this",
            user_expertise="analyst",
            output_format="narrative",
        )

        assert result.status == "completed"
        assert result.executive_summary != ""

    # ========================================================================
    # GRAPH BUILDER TESTS
    # ========================================================================

    def test_build_explainer_graph(self):
        """Test explainer graph construction."""
        graph = build_explainer_graph(use_llm=False)
        assert graph is not None

    def test_build_simple_explainer_graph(self):
        """Test simple explainer graph construction."""
        graph = build_simple_explainer_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_graph_direct_invocation(self, sample_causal_analysis):
        """Test direct graph invocation."""
        graph = build_simple_explainer_graph()

        initial_state = {
            "query": "Test query",
            "analysis_results": [sample_causal_analysis],
            "user_expertise": "analyst",
            "output_format": "narrative",
            "focus_areas": None,
            # Memory integration fields
            "session_id": "test_session",
            "memory_config": {},
            "episodic_context": None,
            "semantic_context": None,
            "working_memory_messages": None,
            # Context fields
            "analysis_context": None,
            "user_context": None,
            "conversation_history": None,
            "extracted_insights": None,
            "narrative_structure": None,
            "key_themes": None,
            "executive_summary": None,
            "detailed_explanation": None,
            "narrative_sections": None,
            "visual_suggestions": None,
            "follow_up_questions": None,
            "related_analyses": None,
            "assembly_latency_ms": 0,
            "reasoning_latency_ms": 0,
            "generation_latency_ms": 0,
            "total_latency_ms": 0,
            "model_used": None,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        # Provide config with thread_id for checkpointer
        config = {"configurable": {"thread_id": "test_graph_direct"}}
        result = await graph.ainvoke(initial_state, config=config)

        assert result["status"] == "completed"
        assert result["executive_summary"] is not None

    # ========================================================================
    # PYDANTIC CONTRACT TESTS
    # ========================================================================

    def test_explainer_input_defaults(self):
        """Test ExplainerInput default values."""
        input_obj = ExplainerInput()

        assert input_obj.query == ""
        assert input_obj.analysis_results == []
        assert input_obj.user_expertise == "analyst"
        assert input_obj.output_format == "narrative"
        assert input_obj.focus_areas is None

    def test_explainer_input_with_values(self):
        """Test ExplainerInput with provided values."""
        input_obj = ExplainerInput(
            query="Test query",
            analysis_results=[{"agent": "test"}],
            user_expertise="executive",
            output_format="brief",
            focus_areas=["sales"],
        )

        assert input_obj.query == "Test query"
        assert len(input_obj.analysis_results) == 1
        assert input_obj.user_expertise == "executive"
        assert input_obj.output_format == "brief"
        assert input_obj.focus_areas == ["sales"]

    def test_explainer_output_defaults(self):
        """Test ExplainerOutput default values."""
        output = ExplainerOutput()

        assert output.executive_summary == ""
        assert output.detailed_explanation == ""
        assert output.narrative_sections == []
        assert output.extracted_insights == []
        assert output.status == "pending"

    # ========================================================================
    # INSIGHT QUALITY TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_insights_are_valid(self, sample_analysis_results):
        """Test that insights have valid structure."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Analyze findings",
        )

        for insight in result.extracted_insights:
            assert "insight_id" in insight
            assert "category" in insight
            assert "statement" in insight

    @pytest.mark.asyncio
    async def test_insights_are_prioritized(self, sample_analysis_results):
        """Test that insights are prioritized."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="What's most important?",
        )

        if len(result.extracted_insights) > 1:
            priorities = [i.get("priority", 99) for i in result.extracted_insights]
            # Should have valid priorities
            assert all(1 <= p <= 5 for p in priorities)

    # ========================================================================
    # VISUAL SUGGESTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_visual_suggestions_generated(self, sample_causal_analysis):
        """Test that visual suggestions are generated."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
            query="Show me the results",
        )

        assert len(result.visual_suggestions) > 0
        for visual in result.visual_suggestions:
            assert "type" in visual

    # ========================================================================
    # LATENCY TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_latency_tracking(self, sample_analysis_results):
        """Test that all latency phases are tracked."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Analyze this",
        )

        assert result.total_latency_ms >= 0
        # Latency is tracked (may be 0 for fast operations)
        assert isinstance(result.total_latency_ms, int)

    # ========================================================================
    # TIMESTAMP TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_timestamp_generated(self, sample_causal_analysis):
        """Test that timestamp is generated."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
        )

        assert result.timestamp != ""
        # Should be ISO format
        assert "T" in result.timestamp

    # ========================================================================
    # MODEL TRACKING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_model_used_tracked(self, sample_causal_analysis):
        """Test that model used is tracked."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[sample_causal_analysis],
        )

        assert result.model_used == "deterministic"

    # ========================================================================
    # FOCUS AREAS TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_focus_areas_applied(self, sample_analysis_results):
        """Test that focus areas influence output."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=sample_analysis_results,
            query="Focus on opportunities",
            focus_areas=["opportunities", "recommendations"],
        )

        assert result.status == "completed"

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_handles_malformed_results(self):
        """Test handling of malformed analysis results."""
        agent = ExplainerAgent(use_llm=False)

        result = await agent.explain(
            analysis_results=[{"malformed": True}],  # Missing required fields
            query="Explain this",
        )

        # Should complete (may create empty/minimal context)
        assert result.status in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_errors_captured(self):
        """Test that errors are captured in output."""
        agent = ExplainerAgent(use_llm=False)

        # Empty input should fail
        result = await agent.explain(
            analysis_results=[],
            query="",
        )

        # Output should exist and indicate failure
        assert result is not None
        assert result.status == "failed"
        # Errors list should exist
        assert result.errors is not None
