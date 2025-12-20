"""
Tests for Learning Extractor node.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.feedback_learner.nodes.learning_extractor import LearningExtractorNode


class TestLearningExtractorNode:
    """Tests for LearningExtractorNode."""

    @pytest.mark.asyncio
    async def test_execute_with_patterns(self, state_with_patterns):
        """Test execution with detected patterns."""
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state_with_patterns)

        assert result["status"] == "updating"
        assert result["learning_recommendations"] is not None
        assert result["priority_improvements"] is not None
        assert result["extraction_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_empty_patterns(self, state_with_feedback):
        """Test execution with no patterns."""
        state = {**state_with_feedback, "detected_patterns": [], "status": "extracting"}
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        assert result["status"] == "updating"
        assert result["learning_recommendations"] == []
        assert result["priority_improvements"] == []

    @pytest.mark.asyncio
    async def test_skip_if_already_failed(self, base_state):
        """Test that node skips execution if already failed."""
        state = {**base_state, "status": "failed"}
        node = LearningExtractorNode()

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_accuracy_issue_generates_data_update(self, base_state):
        """Test that accuracy issues generate data_update recommendations."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "accuracy_issue",
                    "description": "Low accuracy detected",
                    "frequency": 10,
                    "severity": "high",
                    "affected_agents": ["test_agent"],
                    "example_feedback_ids": ["F1", "F2"],
                    "root_cause_hypothesis": "Outdated data",
                }
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        recommendations = result["learning_recommendations"]
        assert len(recommendations) >= 1
        # High severity accuracy issue should generate data_update
        data_update_recs = [r for r in recommendations if r["category"] == "data_update"]
        assert len(data_update_recs) >= 1

    @pytest.mark.asyncio
    async def test_critical_accuracy_issue_generates_retrain(self, base_state):
        """Test that critical accuracy issues generate model_retrain recommendations."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "accuracy_issue",
                    "description": "Critical accuracy issue",
                    "frequency": 20,
                    "severity": "critical",
                    "affected_agents": ["test_agent"],
                    "example_feedback_ids": ["F1"],
                    "root_cause_hypothesis": "Model needs retraining",
                }
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        recommendations = result["learning_recommendations"]
        # Critical severity should also generate model_retrain recommendation
        retrain_recs = [r for r in recommendations if r["category"] == "model_retrain"]
        assert len(retrain_recs) >= 1

    @pytest.mark.asyncio
    async def test_latency_issue_generates_config_change(self, base_state):
        """Test that latency issues generate config_change recommendations."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "latency_issue",
                    "description": "Slow response times",
                    "frequency": 5,
                    "severity": "medium",
                    "affected_agents": ["slow_agent"],
                    "example_feedback_ids": ["F1"],
                    "root_cause_hypothesis": "Query processing too slow",
                }
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        recommendations = result["learning_recommendations"]
        config_recs = [r for r in recommendations if r["category"] == "config_change"]
        assert len(config_recs) >= 1

    @pytest.mark.asyncio
    async def test_relevance_issue_generates_prompt_update(self, base_state):
        """Test that relevance issues generate prompt_update recommendations."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "relevance_issue",
                    "description": "Irrelevant responses",
                    "frequency": 8,
                    "severity": "high",
                    "affected_agents": ["agent1"],
                    "example_feedback_ids": ["F1"],
                    "root_cause_hypothesis": "Prompt not specific enough",
                }
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        recommendations = result["learning_recommendations"]
        prompt_recs = [r for r in recommendations if r["category"] == "prompt_update"]
        assert len(prompt_recs) >= 1

    @pytest.mark.asyncio
    async def test_format_issue_generates_prompt_update(self, base_state):
        """Test that format issues generate prompt_update recommendations."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "format_issue",
                    "description": "Poor formatting",
                    "frequency": 3,
                    "severity": "low",
                    "affected_agents": ["agent1"],
                    "example_feedback_ids": ["F1"],
                    "root_cause_hypothesis": "No formatting guidelines",
                }
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        recommendations = result["learning_recommendations"]
        prompt_recs = [r for r in recommendations if r["category"] == "prompt_update"]
        assert len(prompt_recs) >= 1

    @pytest.mark.asyncio
    async def test_coverage_gap_generates_new_capability(self, base_state):
        """Test that coverage gaps generate new_capability recommendations."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "coverage_gap",
                    "description": "Missing knowledge area",
                    "frequency": 15,
                    "severity": "medium",
                    "affected_agents": ["agent1"],
                    "example_feedback_ids": ["F1"],
                    "root_cause_hypothesis": "Topic not covered",
                }
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        recommendations = result["learning_recommendations"]
        capability_recs = [r for r in recommendations if r["category"] == "new_capability"]
        assert len(capability_recs) >= 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self, base_state):
        """Test that recommendations are prioritized correctly."""
        state = {
            **base_state,
            "detected_patterns": [
                {
                    "pattern_id": "P1",
                    "pattern_type": "accuracy_issue",
                    "description": "High priority accuracy",
                    "frequency": 20,
                    "severity": "high",
                    "affected_agents": ["agent1"],
                    "example_feedback_ids": ["F1"],
                    "root_cause_hypothesis": "Data issues",
                },
                {
                    "pattern_id": "P2",
                    "pattern_type": "format_issue",
                    "description": "Low priority format",
                    "frequency": 2,
                    "severity": "low",
                    "affected_agents": ["agent2"],
                    "example_feedback_ids": ["F2"],
                    "root_cause_hypothesis": "Formatting",
                },
            ],
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        priorities = result["priority_improvements"]
        # High priority should come before low priority
        assert len(priorities) >= 2
        # Priority 1 items should be at top
        recommendations = result["learning_recommendations"]
        high_priority = [r for r in recommendations if r["priority"] == 1]
        if high_priority:
            assert priorities[0] == high_priority[0]["description"]

    @pytest.mark.asyncio
    async def test_recommendation_structure(self, state_with_patterns):
        """Test that recommendations have correct structure."""
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state_with_patterns)

        for rec in result["learning_recommendations"]:
            assert "recommendation_id" in rec
            assert "category" in rec
            assert "description" in rec
            assert "affected_agents" in rec
            assert "expected_impact" in rec
            assert "implementation_effort" in rec
            assert "priority" in rec
            assert "proposed_change" in rec

    @pytest.mark.asyncio
    async def test_llm_mode_fallback(self, state_with_patterns):
        """Test LLM mode falls back to deterministic on error."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        node = LearningExtractorNode(use_llm=True, llm=mock_llm)

        result = await node.execute(state_with_patterns)

        # Should fall back to deterministic mode
        assert result["status"] == "updating"
        assert len(result["learning_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_llm_mode_success(self, state_with_patterns):
        """Test LLM mode when successful."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""```json
{
  "recommendations": [
    {
      "recommendation_id": "R1",
      "category": "data_update",
      "description": "LLM recommendation",
      "affected_agents": ["agent1"],
      "expected_impact": "Improved accuracy",
      "implementation_effort": "low",
      "priority": 1,
      "proposed_change": "Update training data"
    }
  ]
}
```"""
            )
        )

        node = LearningExtractorNode(use_llm=True, llm=mock_llm)

        result = await node.execute(state_with_patterns)

        assert result["status"] == "updating"
        assert len(result["learning_recommendations"]) > 0
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_mode_without_llm_instance(self, state_with_patterns):
        """Test LLM mode without LLM instance falls back to deterministic."""
        node = LearningExtractorNode(use_llm=True, llm=None)

        result = await node.execute(state_with_patterns)

        assert result["status"] == "updating"
        assert len(result["learning_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, base_state):
        """Test error handling in learning extraction."""
        # Create invalid state
        state = {
            **base_state,
            "detected_patterns": "invalid",  # Should be a list
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "learning_extractor"

    @pytest.mark.asyncio
    async def test_latency_tracking(self, state_with_patterns):
        """Test that latency is properly tracked."""
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state_with_patterns)

        assert "extraction_latency_ms" in result
        assert isinstance(result["extraction_latency_ms"], int)
        assert result["extraction_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_max_priorities(self, base_state):
        """Test that priority improvements are capped at 5."""
        # Create many patterns
        patterns = [
            {
                "pattern_id": f"P{i}",
                "pattern_type": "accuracy_issue",
                "description": f"Pattern {i}",
                "frequency": 10,
                "severity": "high",
                "affected_agents": [f"agent{i}"],
                "example_feedback_ids": [f"F{i}"],
                "root_cause_hypothesis": f"Hypothesis {i}",
            }
            for i in range(10)
        ]
        state = {
            **base_state,
            "detected_patterns": patterns,
            "status": "extracting",
        }
        node = LearningExtractorNode(use_llm=False)

        result = await node.execute(state)

        # Should be capped at 5 priorities
        assert len(result["priority_improvements"]) <= 5
