"""
Integration tests for Feedback Learner agent.
"""

import pytest
from unittest.mock import AsyncMock

from src.agents.feedback_learner import (
    FeedbackLearnerAgent,
    FeedbackLearnerInput,
    FeedbackLearnerOutput,
    build_feedback_learner_graph,
    build_simple_feedback_learner_graph,
    process_feedback_batch,
)


class TestFeedbackLearnerAgent:
    """Integration tests for FeedbackLearnerAgent."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_feedback(self, sample_feedback_items):
        """Test full pipeline with feedback items."""
        agent = FeedbackLearnerAgent()

        result = await agent.process_feedback(sample_feedback_items)

        assert isinstance(result, FeedbackLearnerOutput)
        assert result.status in ["completed", "failed"]
        assert result.batch_id.startswith("batch_")
        assert result.timestamp != ""

    @pytest.mark.asyncio
    async def test_full_pipeline_empty(self):
        """Test full pipeline with no feedback."""
        agent = FeedbackLearnerAgent()

        result = await agent.process_feedback([])

        assert isinstance(result, FeedbackLearnerOutput)
        assert result.feedback_count == 0
        assert result.pattern_count == 0
        assert result.recommendation_count == 0

    @pytest.mark.asyncio
    async def test_learn_method(self):
        """Test learn method with time range."""
        agent = FeedbackLearnerAgent()

        result = await agent.learn(
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-01-31T23:59:59Z",
            batch_id="test_batch",
        )

        assert isinstance(result, FeedbackLearnerOutput)
        assert result.batch_id == "test_batch"

    @pytest.mark.asyncio
    async def test_learn_with_focus_agents(self):
        """Test learn with focus agents filter."""
        agent = FeedbackLearnerAgent()

        result = await agent.learn(
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-01-31T23:59:59Z",
            focus_agents=["causal_impact", "gap_analyzer"],
        )

        assert isinstance(result, FeedbackLearnerOutput)

    @pytest.mark.asyncio
    async def test_learn_auto_generates_batch_id(self):
        """Test that learn auto-generates batch ID if not provided."""
        agent = FeedbackLearnerAgent()

        result = await agent.learn(
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-01-31T23:59:59Z",
        )

        assert result.batch_id.startswith("batch_")
        assert len(result.batch_id) > 6

    @pytest.mark.asyncio
    async def test_agent_with_stores(self, mock_feedback_store, mock_outcome_store):
        """Test agent with feedback and outcome stores."""
        agent = FeedbackLearnerAgent(
            feedback_store=mock_feedback_store,
            outcome_store=mock_outcome_store,
        )

        result = await agent.learn(
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-01-31T23:59:59Z",
        )

        assert isinstance(result, FeedbackLearnerOutput)

    @pytest.mark.asyncio
    async def test_agent_with_llm(self, mock_llm):
        """Test agent with LLM enabled."""
        agent = FeedbackLearnerAgent(use_llm=True, llm=mock_llm)

        result = await agent.process_feedback([
            {
                "feedback_id": "F1",
                "source_agent": "test",
                "query": "test query",
                "agent_response": "test response",
                "user_feedback": 3,
                "feedback_type": "rating",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ])

        assert isinstance(result, FeedbackLearnerOutput)

    @pytest.mark.asyncio
    async def test_agent_with_knowledge_stores(self, mock_knowledge_stores):
        """Test agent with knowledge stores."""
        agent = FeedbackLearnerAgent(knowledge_stores=mock_knowledge_stores)

        result = await agent.process_feedback([
            {
                "feedback_id": "F1",
                "source_agent": "test",
                "query": "test query",
                "agent_response": "test response",
                "user_feedback": 1,
                "feedback_type": "rating",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ])

        assert isinstance(result, FeedbackLearnerOutput)


class TestFeedbackLearnerOutput:
    """Tests for FeedbackLearnerOutput contract."""

    @pytest.mark.asyncio
    async def test_output_fields(self, sample_feedback_items):
        """Test that output has all required fields."""
        agent = FeedbackLearnerAgent()
        result = await agent.process_feedback(sample_feedback_items)

        # Check all required fields exist
        assert hasattr(result, "batch_id")
        assert hasattr(result, "detected_patterns")
        assert hasattr(result, "learning_recommendations")
        assert hasattr(result, "priority_improvements")
        assert hasattr(result, "proposed_updates")
        assert hasattr(result, "applied_updates")
        assert hasattr(result, "learning_summary")
        assert hasattr(result, "feedback_count")
        assert hasattr(result, "pattern_count")
        assert hasattr(result, "recommendation_count")
        assert hasattr(result, "total_latency_ms")
        assert hasattr(result, "model_used")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "status")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    @pytest.mark.asyncio
    async def test_output_types(self, sample_feedback_items):
        """Test that output fields have correct types."""
        agent = FeedbackLearnerAgent()
        result = await agent.process_feedback(sample_feedback_items)

        assert isinstance(result.batch_id, str)
        assert isinstance(result.detected_patterns, list)
        assert isinstance(result.learning_recommendations, list)
        assert isinstance(result.priority_improvements, list)
        assert isinstance(result.proposed_updates, list)
        assert isinstance(result.applied_updates, list)
        assert isinstance(result.learning_summary, str)
        assert isinstance(result.feedback_count, int)
        assert isinstance(result.pattern_count, int)
        assert isinstance(result.recommendation_count, int)
        assert isinstance(result.total_latency_ms, int)
        assert isinstance(result.model_used, str)
        assert isinstance(result.timestamp, str)
        assert isinstance(result.status, str)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


class TestHandoff:
    """Tests for handoff generation."""

    @pytest.mark.asyncio
    async def test_handoff_structure(self, sample_feedback_items):
        """Test handoff has correct structure."""
        agent = FeedbackLearnerAgent()
        result = await agent.process_feedback(sample_feedback_items)

        handoff = agent.get_handoff(result)

        assert "agent" in handoff
        assert handoff["agent"] == "feedback_learner"
        assert "analysis_type" in handoff
        assert "key_findings" in handoff
        assert "patterns" in handoff
        assert "top_recommendations" in handoff
        assert "summary" in handoff
        assert "requires_further_analysis" in handoff
        assert "suggested_next_agent" in handoff

    @pytest.mark.asyncio
    async def test_handoff_key_findings(self, sample_feedback_items):
        """Test handoff key findings."""
        agent = FeedbackLearnerAgent()
        result = await agent.process_feedback(sample_feedback_items)

        handoff = agent.get_handoff(result)
        findings = handoff["key_findings"]

        assert "feedback_processed" in findings
        assert "patterns_detected" in findings
        assert "recommendations" in findings
        assert "updates_applied" in findings

    @pytest.mark.asyncio
    async def test_handoff_next_agent_on_success(self, sample_feedback_items):
        """Test handoff suggests next agent on success."""
        agent = FeedbackLearnerAgent()
        result = await agent.process_feedback(sample_feedback_items)

        if result.status == "completed":
            handoff = agent.get_handoff(result)
            assert handoff["suggested_next_agent"] == "experiment_designer"
            assert handoff["requires_further_analysis"] is False


class TestGraphBuilder:
    """Tests for graph building functions."""

    def test_build_feedback_learner_graph(self):
        """Test building feedback learner graph."""
        graph = build_feedback_learner_graph()
        assert graph is not None

    def test_build_feedback_learner_graph_with_stores(
        self, mock_feedback_store, mock_outcome_store, mock_knowledge_stores
    ):
        """Test building graph with stores."""
        graph = build_feedback_learner_graph(
            feedback_store=mock_feedback_store,
            outcome_store=mock_outcome_store,
            knowledge_stores=mock_knowledge_stores,
        )
        assert graph is not None

    def test_build_feedback_learner_graph_with_llm(self, mock_llm):
        """Test building graph with LLM."""
        graph = build_feedback_learner_graph(use_llm=True, llm=mock_llm)
        assert graph is not None

    def test_build_simple_feedback_learner_graph(self):
        """Test building simple graph."""
        graph = build_simple_feedback_learner_graph()
        assert graph is not None


class TestConvenienceFunction:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_process_feedback_batch(self):
        """Test process_feedback_batch convenience function."""
        result = await process_feedback_batch(
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-01-31T23:59:59Z",
        )

        assert isinstance(result, FeedbackLearnerOutput)

    @pytest.mark.asyncio
    async def test_process_feedback_batch_with_focus(self):
        """Test process_feedback_batch with focus agents."""
        result = await process_feedback_batch(
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-01-31T23:59:59Z",
            focus_agents=["agent1", "agent2"],
        )

        assert isinstance(result, FeedbackLearnerOutput)


class TestLatencyTracking:
    """Tests for latency tracking."""

    @pytest.mark.asyncio
    async def test_latency_tracked(self, sample_feedback_items):
        """Test that latency is tracked."""
        agent = FeedbackLearnerAgent()

        result = await agent.process_feedback(sample_feedback_items)

        assert result.total_latency_ms >= 0
        assert isinstance(result.total_latency_ms, int)


class TestModelUsed:
    """Tests for model_used field."""

    @pytest.mark.asyncio
    async def test_model_used_deterministic(self, sample_feedback_items):
        """Test model_used is set for deterministic mode."""
        agent = FeedbackLearnerAgent(use_llm=False)

        result = await agent.process_feedback(sample_feedback_items)

        assert result.model_used == "deterministic"

    @pytest.mark.asyncio
    async def test_model_used_never_none(self, sample_feedback_items):
        """Test model_used is never None."""
        agent = FeedbackLearnerAgent()

        result = await agent.process_feedback(sample_feedback_items)

        assert result.model_used is not None
        assert result.model_used != ""


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_errors_captured(self):
        """Test that errors are captured in output."""
        agent = FeedbackLearnerAgent()

        # Process invalid feedback that may cause warnings
        result = await agent.process_feedback([])

        # Even with empty input, should complete
        assert isinstance(result, FeedbackLearnerOutput)
        assert result.status in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_graph_lazy_loading(self):
        """Test that graph is lazily loaded."""
        agent = FeedbackLearnerAgent()

        # Graph should be None initially
        assert agent._graph is None

        # Access graph property
        _ = agent.graph

        # Now graph should be loaded
        assert agent._graph is not None

    @pytest.mark.asyncio
    async def test_graph_reuses_instance(self):
        """Test that graph is reused on multiple accesses."""
        agent = FeedbackLearnerAgent()

        graph1 = agent.graph
        graph2 = agent.graph

        assert graph1 is graph2
