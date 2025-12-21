"""
Tests for Knowledge Updater node.
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.feedback_learner.nodes.knowledge_updater import KnowledgeUpdaterNode


class TestKnowledgeUpdaterNode:
    """Tests for KnowledgeUpdaterNode."""

    @pytest.mark.asyncio
    async def test_execute_with_recommendations(
        self, state_with_recommendations, mock_knowledge_stores
    ):
        """Test execution with learning recommendations."""
        node = KnowledgeUpdaterNode(knowledge_stores=mock_knowledge_stores)

        result = await node.execute(state_with_recommendations)

        assert result["status"] == "completed"
        assert result["proposed_updates"] is not None
        assert result["applied_updates"] is not None
        assert result["learning_summary"] is not None
        assert result["update_latency_ms"] >= 0
        assert result["total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_without_stores(self, state_with_recommendations):
        """Test execution without knowledge stores."""
        node = KnowledgeUpdaterNode(knowledge_stores=None)

        result = await node.execute(state_with_recommendations)

        assert result["status"] == "completed"
        # Updates proposed but not applied (no stores)
        assert result["proposed_updates"] is not None
        assert len(result["applied_updates"]) == 0

    @pytest.mark.asyncio
    async def test_execute_empty_recommendations(self, state_with_patterns):
        """Test execution with no recommendations."""
        state = {
            **state_with_patterns,
            "learning_recommendations": [],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        assert result["status"] == "completed"
        assert result["proposed_updates"] == []
        assert result["applied_updates"] == []

    @pytest.mark.asyncio
    async def test_skip_if_already_failed(self, base_state):
        """Test that node skips execution if already failed."""
        state = {**base_state, "status": "failed"}
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_data_update_generates_baseline_update(self, base_state):
        """Test that data_update recommendations generate baseline updates."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "data_update",
                    "description": "Update baseline data",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better accuracy",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New baseline values",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        updates = result["proposed_updates"]
        assert len(updates) >= 1
        baseline_updates = [u for u in updates if u["knowledge_type"] == "baseline"]
        assert len(baseline_updates) >= 1
        assert baseline_updates[0]["key"] == "agent1"

    @pytest.mark.asyncio
    async def test_config_change_generates_agent_config_update(self, base_state):
        """Test that config_change recommendations generate agent_config updates."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "config_change",
                    "description": "Update agent config",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better performance",
                    "implementation_effort": "low",
                    "priority": 2,
                    "proposed_change": "New config values",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        updates = result["proposed_updates"]
        config_updates = [u for u in updates if u["knowledge_type"] == "agent_config"]
        assert len(config_updates) >= 1

    @pytest.mark.asyncio
    async def test_prompt_update_generates_prompt_update(self, base_state):
        """Test that prompt_update recommendations generate prompt updates."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "prompt_update",
                    "description": "Update prompts",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better relevance",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New prompt text",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        updates = result["proposed_updates"]
        prompt_updates = [u for u in updates if u["knowledge_type"] == "prompt"]
        assert len(prompt_updates) >= 1

    @pytest.mark.asyncio
    async def test_threshold_generates_threshold_update(self, base_state):
        """Test that threshold recommendations generate threshold updates."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "threshold",
                    "description": "Update threshold",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better detection",
                    "implementation_effort": "low",
                    "priority": 2,
                    "proposed_change": 0.85,
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        updates = result["proposed_updates"]
        threshold_updates = [u for u in updates if u["knowledge_type"] == "threshold"]
        assert len(threshold_updates) >= 1

    @pytest.mark.asyncio
    async def test_updates_applied_to_stores(self, base_state, mock_knowledge_stores):
        """Test that updates are applied to knowledge stores."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "data_update",
                    "description": "Update baseline",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better accuracy",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New value",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode(knowledge_stores=mock_knowledge_stores)

        result = await node.execute(state)

        # Should have applied the update
        assert len(result["applied_updates"]) >= 1
        mock_knowledge_stores["baseline"].update.assert_called()

    @pytest.mark.asyncio
    async def test_store_update_failure_handled(self, base_state):
        """Test that store update failures are handled gracefully."""
        mock_stores = {
            "baseline": AsyncMock(update=AsyncMock(side_effect=Exception("Store error")))
        }
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "data_update",
                    "description": "Update baseline",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better accuracy",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New value",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode(knowledge_stores=mock_stores)

        result = await node.execute(state)

        # Should complete but with no applied updates
        assert result["status"] == "completed"
        assert len(result["applied_updates"]) == 0
        assert len(result["proposed_updates"]) >= 1

    @pytest.mark.asyncio
    async def test_update_structure(self, state_with_recommendations):
        """Test that knowledge updates have correct structure."""
        node = KnowledgeUpdaterNode()

        result = await node.execute(state_with_recommendations)

        for update in result["proposed_updates"]:
            assert "update_id" in update
            assert "knowledge_type" in update
            assert "key" in update
            assert "old_value" in update  # Can be None
            assert "new_value" in update
            assert "justification" in update
            assert "effective_date" in update

    @pytest.mark.asyncio
    async def test_learning_summary_generation(self, state_with_recommendations):
        """Test that learning summary is generated correctly."""
        node = KnowledgeUpdaterNode()

        result = await node.execute(state_with_recommendations)

        summary = result["learning_summary"]
        assert "Learning cycle complete" in summary
        assert "feedback" in summary.lower() or "Processed" in summary
        assert "pattern" in summary.lower()
        assert "recommendation" in summary.lower()

    @pytest.mark.asyncio
    async def test_total_latency_calculation(self, base_state):
        """Test that total latency is calculated correctly."""
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [],
            "priority_improvements": [],
            "collection_latency_ms": 10,
            "analysis_latency_ms": 20,
            "extraction_latency_ms": 15,
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        # Total should include all previous latencies plus update latency
        assert result["total_latency_ms"] >= 45  # 10 + 20 + 15

    @pytest.mark.asyncio
    async def test_error_handling(self, base_state):
        """Test error handling in knowledge update."""
        # Create invalid state
        state = {
            **base_state,
            "learning_recommendations": "invalid",  # Should be a list
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "knowledge_updater"

    @pytest.mark.asyncio
    async def test_latency_tracking(self, state_with_recommendations):
        """Test that latency is properly tracked."""
        node = KnowledgeUpdaterNode()

        result = await node.execute(state_with_recommendations)

        assert "update_latency_ms" in result
        assert isinstance(result["update_latency_ms"], int)
        assert result["update_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_priority_in_summary(self, base_state):
        """Test that top priority is included in summary."""
        state = {
            **base_state,
            "feedback_items": [{"feedback_id": "F1"}],
            "detected_patterns": [{"pattern_id": "P1"}],
            "learning_recommendations": [{"recommendation_id": "R1"}],
            "priority_improvements": ["First priority", "Second priority"],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode()

        result = await node.execute(state)

        summary = result["learning_summary"]
        assert "First priority" in summary

    @pytest.mark.asyncio
    async def test_missing_store_type_handled(self, base_state):
        """Test that missing store type is handled gracefully."""
        # Only have baseline store, but try to apply prompt update
        mock_stores = {"baseline": AsyncMock(update=AsyncMock())}
        state = {
            **base_state,
            "feedback_items": [],
            "detected_patterns": [],
            "learning_recommendations": [
                {
                    "recommendation_id": "R1",
                    "category": "prompt_update",  # No prompt store
                    "description": "Update prompts",
                    "affected_agents": ["agent1"],
                    "expected_impact": "Better relevance",
                    "implementation_effort": "medium",
                    "priority": 1,
                    "proposed_change": "New prompt",
                }
            ],
            "priority_improvements": [],
            "status": "updating",
        }
        node = KnowledgeUpdaterNode(knowledge_stores=mock_stores)

        result = await node.execute(state)

        # Should complete but not apply the update
        assert result["status"] == "completed"
        assert len(result["proposed_updates"]) >= 1
        assert len(result["applied_updates"]) == 0
