"""
Tests for Pattern Analyzer node.
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode


class TestPatternAnalyzerNode:
    """Tests for PatternAnalyzerNode."""

    @pytest.mark.asyncio
    async def test_execute_with_feedback(self, state_with_feedback):
        """Test execution with feedback items."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        assert result["status"] == "extracting"
        assert result["detected_patterns"] is not None
        assert result["pattern_clusters"] is not None
        assert result["analysis_latency_ms"] >= 0
        assert result["model_used"] == "deterministic"

    @pytest.mark.asyncio
    async def test_execute_empty_feedback(self, base_state):
        """Test execution with no feedback items."""
        state = {**base_state, "feedback_items": [], "status": "analyzing"}
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        assert result["status"] == "extracting"
        assert result["detected_patterns"] == []
        assert result["pattern_clusters"] == {}

    @pytest.mark.asyncio
    async def test_skip_if_already_failed(self, base_state):
        """Test that node skips execution if already failed."""
        state = {**base_state, "status": "failed"}
        node = PatternAnalyzerNode()

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_detect_low_rating_pattern(self, base_state, low_rating_feedback):
        """Test detection of low rating pattern."""
        state = {
            **base_state,
            "feedback_items": low_rating_feedback,
            "feedback_summary": {
                "total_count": len(low_rating_feedback),
                "by_type": {"rating": len(low_rating_feedback)},
                "by_agent": {"explainer": len(low_rating_feedback)},
                "average_rating": 1.5,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        assert len(patterns) > 0
        # Should detect accuracy issue from low ratings
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) > 0
        assert accuracy_patterns[0]["severity"] in ["medium", "high"]

    @pytest.mark.asyncio
    async def test_detect_correction_pattern(self, base_state, correction_heavy_feedback):
        """Test detection of correction-heavy pattern."""
        state = {
            **base_state,
            "feedback_items": correction_heavy_feedback,
            "feedback_summary": {
                "total_count": len(correction_heavy_feedback),
                "by_type": {"correction": len(correction_heavy_feedback)},
                "by_agent": {"causal_impact": len(correction_heavy_feedback)},
                "average_rating": None,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        assert len(patterns) > 0
        # Should detect accuracy issue from corrections
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) > 0

    @pytest.mark.asyncio
    async def test_detect_outcome_error_pattern(self, base_state, outcome_error_feedback):
        """Test detection of outcome error pattern."""
        state = {
            **base_state,
            "feedback_items": outcome_error_feedback,
            "feedback_summary": {
                "total_count": len(outcome_error_feedback),
                "by_type": {"outcome": len(outcome_error_feedback)},
                "by_agent": {"prediction_synthesizer": len(outcome_error_feedback)},
                "average_rating": None,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        assert len(patterns) > 0
        # Should detect accuracy issue from prediction errors
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) > 0

    @pytest.mark.asyncio
    async def test_detect_agent_specific_pattern(self, base_state):
        """Test detection of agent-specific high negative feedback rate."""
        # Create feedback where one agent has many negative ratings
        feedback_items = [
            {
                "feedback_id": f"F{i:03d}",
                "source_agent": "problematic_agent",
                "query": f"Query {i}",
                "agent_response": f"Response {i}",
                "user_feedback": 1 if i < 6 else 4,  # 6 low, 4 high
                "feedback_type": "rating",
                "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
            }
            for i in range(10)
        ]

        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 10,
                "by_type": {"rating": 10},
                "by_agent": {"problematic_agent": 10},
                "average_rating": 2.2,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        # Should detect relevance issue for high negative rate agent
        relevance_patterns = [p for p in patterns if p["pattern_type"] == "relevance_issue"]
        assert len(relevance_patterns) > 0
        assert "problematic_agent" in relevance_patterns[0]["affected_agents"]

    @pytest.mark.asyncio
    async def test_pattern_clustering(self, state_with_feedback):
        """Test that patterns are properly clustered by type."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        clusters = result["pattern_clusters"]
        patterns = result["detected_patterns"]

        # All pattern IDs should be in clusters
        all_ids_in_clusters = set()
        for ids in clusters.values():
            all_ids_in_clusters.update(ids)

        for pattern in patterns:
            assert pattern["pattern_id"] in all_ids_in_clusters

    @pytest.mark.asyncio
    async def test_llm_mode_fallback(self, state_with_feedback, mock_llm):
        """Test LLM mode falls back to deterministic on error."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        node = PatternAnalyzerNode(use_llm=True, llm=mock_llm)

        result = await node.execute(state_with_feedback)

        # Should fall back to deterministic mode
        assert result["status"] == "extracting"
        assert result["model_used"] == "deterministic"

    @pytest.mark.asyncio
    async def test_llm_mode_success(self, state_with_feedback, mock_llm):
        """Test LLM mode when successful."""
        node = PatternAnalyzerNode(use_llm=True, llm=mock_llm)

        result = await node.execute(state_with_feedback)

        assert result["status"] == "extracting"
        assert len(result["detected_patterns"]) > 0
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_mode_without_llm_instance(self, state_with_feedback):
        """Test LLM mode without LLM instance falls back to deterministic."""
        node = PatternAnalyzerNode(use_llm=True, llm=None)

        result = await node.execute(state_with_feedback)

        assert result["status"] == "extracting"
        assert result["model_used"] == "deterministic"

    @pytest.mark.asyncio
    async def test_pattern_structure(self, state_with_feedback):
        """Test that detected patterns have correct structure."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        for pattern in result["detected_patterns"]:
            assert "pattern_id" in pattern
            assert "pattern_type" in pattern
            assert "description" in pattern
            assert "frequency" in pattern
            assert "severity" in pattern
            assert "affected_agents" in pattern
            assert "example_feedback_ids" in pattern
            assert "root_cause_hypothesis" in pattern

    @pytest.mark.asyncio
    async def test_error_handling(self, base_state):
        """Test error handling in pattern analysis."""
        # Create invalid state that would cause an error
        state = {
            **base_state,
            "feedback_items": "invalid",  # Should be a list
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "pattern_analyzer"

    @pytest.mark.asyncio
    async def test_latency_tracking(self, state_with_feedback):
        """Test that latency is properly tracked."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        assert "analysis_latency_ms" in result
        assert isinstance(result["analysis_latency_ms"], int)
        assert result["analysis_latency_ms"] >= 0
