"""
Unit tests for Discovery Feedback Node.
Version: 4.4

Tests the discovery feedback loop for causal discovery results.
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.agents.feedback_learner.nodes.discovery_feedback_node import (
    DiscoveryFeedbackNode,
    create_discovery_feedback_node,
)
from src.agents.feedback_learner.state import (
    DiscoveryFeedbackItem,
    FeedbackLearnerState,
)


def create_discovery_feedback(
    algorithm: str = "pc",
    user_decision: str = "accept",
    accuracy_score: float = 0.8,
    gate_decision: str = "accept",
    edge_corrections: list = None,
) -> DiscoveryFeedbackItem:
    """Helper to create discovery feedback items."""
    return DiscoveryFeedbackItem(
        feedback_id=f"df_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        feedback_type="expert_review",
        discovery_run_id=f"run_{uuid.uuid4().hex[:8]}",
        algorithm_used=algorithm,
        dag_adjacency={"A": ["B", "C"], "B": ["D"]},
        dag_nodes=["A", "B", "C", "D"],
        original_gate_decision=gate_decision,
        user_decision=user_decision,
        edge_corrections=edge_corrections,
        comments=None,
        accuracy_score=accuracy_score,
        metadata={},
    )


def create_state_with_discovery_feedback(
    items: list,
) -> FeedbackLearnerState:
    """Helper to create state with discovery feedback."""
    return FeedbackLearnerState(
        batch_id="test_batch",
        time_range_start="2025-01-01T00:00:00Z",
        time_range_end="2025-01-02T00:00:00Z",
        focus_agents=None,
        cognitive_context=None,
        training_signal=None,
        feedback_items=None,
        feedback_summary=None,
        discovery_feedback_items=items,
        discovery_accuracy_tracking=None,
        discovery_parameter_recommendations=None,
        has_discovery_feedback=len(items) > 0,
        validation_outcomes=None,
        detected_patterns=[],
        pattern_clusters=None,
        learning_recommendations=None,
        priority_improvements=None,
        proposed_updates=None,
        applied_updates=None,
        rubric_evaluation_context=None,
        rubric_evaluation=None,
        rubric_weighted_score=None,
        rubric_decision=None,
        rubric_pattern_flags=None,
        rubric_improvement_suggestion=None,
        rubric_latency_ms=None,
        rubric_error=None,
        learning_summary=None,
        metrics_before=None,
        metrics_after=None,
        collection_latency_ms=0,
        analysis_latency_ms=0,
        extraction_latency_ms=0,
        update_latency_ms=0,
        total_latency_ms=0,
        model_used=None,
        errors=[],
        warnings=[],
        status="analyzing",
        audit_workflow_id=None,
    )


class TestDiscoveryFeedbackNode:
    """Tests for DiscoveryFeedbackNode class."""

    @pytest.fixture
    def node(self):
        """Create default node."""
        return DiscoveryFeedbackNode()

    @pytest.fixture
    def node_low_threshold(self):
        """Create node with low thresholds for testing."""
        return DiscoveryFeedbackNode(
            min_runs_for_recommendation=3,
            accuracy_threshold=0.8,
        )

    # Basic functionality tests
    @pytest.mark.asyncio
    async def test_no_discovery_feedback(self, node):
        """Node should handle empty discovery feedback."""
        state = create_state_with_discovery_feedback([])

        result = await node.execute(state)

        # Should return state unchanged
        assert result.get("discovery_accuracy_tracking") is None
        assert result.get("discovery_parameter_recommendations") is None

    @pytest.mark.asyncio
    async def test_single_feedback_item(self, node_low_threshold):
        """Node should process single feedback item."""
        items = [create_discovery_feedback(algorithm="pc")]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        # Should have tracking
        tracking = result.get("discovery_accuracy_tracking")
        assert tracking is not None
        assert "pc" in tracking
        assert tracking["pc"]["total_runs"] == 1
        assert tracking["pc"]["accepted_runs"] == 1

    @pytest.mark.asyncio
    async def test_multiple_algorithms(self, node_low_threshold):
        """Node should track multiple algorithms separately."""
        items = [
            create_discovery_feedback(algorithm="pc", user_decision="accept"),
            create_discovery_feedback(algorithm="pc", user_decision="accept"),
            create_discovery_feedback(algorithm="fci", user_decision="reject"),
            create_discovery_feedback(algorithm="fci", user_decision="reject"),
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        tracking = result.get("discovery_accuracy_tracking")
        assert "pc" in tracking
        assert "fci" in tracking
        assert tracking["pc"]["accepted_runs"] == 2
        assert tracking["fci"]["rejected_runs"] == 2

    # Accuracy tracking tests
    @pytest.mark.asyncio
    async def test_accuracy_calculation(self, node_low_threshold):
        """Node should correctly calculate accuracy metrics."""
        items = [
            create_discovery_feedback(algorithm="pc", user_decision="accept", accuracy_score=0.9),
            create_discovery_feedback(algorithm="pc", user_decision="accept", accuracy_score=0.8),
            create_discovery_feedback(algorithm="pc", user_decision="modify", accuracy_score=0.7),
            create_discovery_feedback(algorithm="pc", user_decision="reject", accuracy_score=0.5),
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        tracking = result.get("discovery_accuracy_tracking")["pc"]
        assert tracking["total_runs"] == 4
        assert tracking["accepted_runs"] == 2
        assert tracking["rejected_runs"] == 1
        assert tracking["modified_runs"] == 1
        # Average accuracy: (0.9 + 0.8 + 0.7 + 0.5) / 4 = 0.725
        assert abs(tracking["average_accuracy"] - 0.725) < 0.01

    @pytest.mark.asyncio
    async def test_edge_corrections_tracking(self, node_low_threshold):
        """Node should track edge corrections."""
        items = [
            create_discovery_feedback(
                algorithm="pc",
                user_decision="modify",
                edge_corrections=[
                    {"action": "confirm", "edge": ("A", "B")},
                    {"action": "confirm", "edge": ("A", "C")},
                    {"action": "remove", "edge": ("B", "D")},  # False positive
                ],
            ),
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        tracking = result.get("discovery_accuracy_tracking")["pc"]
        assert tracking["modified_runs"] == 1
        # 2 confirmed edges, 1 removed (was predicted but wrong)
        # Precision: 2 correct / 3 predicted = 0.67

    # Recommendation tests
    @pytest.mark.asyncio
    async def test_high_rejection_rate_recommendation(self, node_low_threshold):
        """High rejection rate should trigger recommendations."""
        # Create 10 items with 5 rejected (50% rejection rate)
        items = [
            create_discovery_feedback(algorithm="pc", user_decision="reject", accuracy_score=0.4)
            for _ in range(5)
        ] + [
            create_discovery_feedback(algorithm="pc", user_decision="accept", accuracy_score=0.6)
            for _ in range(5)
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        recommendations = result.get("discovery_parameter_recommendations")
        assert recommendations is not None
        # Should have recommendation for significance threshold
        rec_params = [r["parameter_name"] for r in recommendations]
        assert any("threshold" in p for p in rec_params)

    @pytest.mark.asyncio
    async def test_no_recommendation_for_good_accuracy(self, node_low_threshold):
        """Good accuracy should not trigger recommendations."""
        items = [
            create_discovery_feedback(algorithm="pc", user_decision="accept", accuracy_score=0.9)
            for _ in range(10)
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        recommendations = result.get("discovery_parameter_recommendations")
        assert recommendations == []

    @pytest.mark.asyncio
    async def test_minimum_runs_for_recommendation(self, node):
        """Should require minimum runs before recommending."""
        # Create only 5 items (below default threshold of 10)
        items = [
            create_discovery_feedback(algorithm="pc", user_decision="reject", accuracy_score=0.3)
            for _ in range(5)
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node.execute(state)

        recommendations = result.get("discovery_parameter_recommendations")
        assert recommendations == []

    # Pattern detection tests
    @pytest.mark.asyncio
    async def test_low_acceptance_pattern_detected(self, node_low_threshold):
        """Low acceptance rate should be detected as pattern."""
        items = [
            create_discovery_feedback(algorithm="pc", user_decision="reject", accuracy_score=0.3)
            for _ in range(8)
        ] + [
            create_discovery_feedback(algorithm="pc", user_decision="accept", accuracy_score=0.8)
            for _ in range(2)
        ]
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        patterns = result.get("detected_patterns")
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) >= 1
        assert "pc" in accuracy_patterns[0]["description"]

    @pytest.mark.asyncio
    async def test_gate_override_pattern_detected(self, node_low_threshold):
        """High gate override rate should be detected."""
        # Create items with gate overrides
        override_items = [
            DiscoveryFeedbackItem(
                feedback_id=f"df_{i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                feedback_type="gate_override",
                discovery_run_id=f"run_{i}",
                algorithm_used="pc",
                dag_adjacency={"A": ["B"]},
                dag_nodes=["A", "B"],
                original_gate_decision="reject",
                user_decision="accept",
                edge_corrections=None,
                comments="Override accepted",
                accuracy_score=0.7,
                metadata={},
            )
            for i in range(6)
        ]
        # Add some normal items
        normal_items = [
            create_discovery_feedback(algorithm="pc", user_decision="accept") for _ in range(4)
        ]
        items = override_items + normal_items
        state = create_state_with_discovery_feedback(items)

        result = await node_low_threshold.execute(state)

        patterns = result.get("detected_patterns")
        coverage_patterns = [p for p in patterns if p["pattern_type"] == "coverage_gap"]
        # 6/10 = 60% override rate > 20% threshold
        assert len(coverage_patterns) >= 1

    # Error handling tests
    @pytest.mark.asyncio
    async def test_error_handling(self, node):
        """Node should handle errors gracefully."""
        # Create state with invalid data
        state = create_state_with_discovery_feedback([])
        state["discovery_feedback_items"] = [{"invalid": "data"}]  # Missing required fields

        result = await node.execute(state)

        # Should have error recorded
        errors = result.get("errors", [])
        assert len(errors) >= 1 or len(result.get("warnings", [])) >= 0


class TestDiscoveryFeedbackNodeFactory:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Factory should create node with defaults."""
        node = create_discovery_feedback_node()

        assert isinstance(node, DiscoveryFeedbackNode)
        assert node._min_runs == 10
        assert node._accuracy_threshold == 0.7

    def test_create_with_custom_config(self):
        """Factory should accept custom configuration."""
        node = create_discovery_feedback_node(
            min_runs_for_recommendation=5,
            accuracy_threshold=0.8,
        )

        assert node._min_runs == 5
        assert node._accuracy_threshold == 0.8


class TestDiscoveryFeedbackIntegration:
    """Integration tests for discovery feedback processing."""

    @pytest.mark.asyncio
    async def test_full_feedback_cycle(self):
        """Test complete feedback processing cycle."""
        node = DiscoveryFeedbackNode(
            min_runs_for_recommendation=5,
            accuracy_threshold=0.7,
        )

        # Simulate realistic feedback cycle
        items = []

        # Good PC algorithm results
        for i in range(6):
            items.append(
                create_discovery_feedback(
                    algorithm="pc",
                    user_decision="accept",
                    accuracy_score=0.85 + (i * 0.01),
                )
            )

        # Problematic FCI algorithm results
        for i in range(6):
            items.append(
                create_discovery_feedback(
                    algorithm="fci",
                    user_decision="reject" if i < 4 else "modify",
                    accuracy_score=0.45 + (i * 0.02),
                )
            )

        state = create_state_with_discovery_feedback(items)
        result = await node.execute(state)

        # Check tracking
        tracking = result.get("discovery_accuracy_tracking")
        assert tracking["pc"]["average_accuracy"] > 0.8
        assert tracking["fci"]["average_accuracy"] < 0.6

        # Check recommendations - should have recommendations for FCI
        recommendations = result.get("discovery_parameter_recommendations")
        fci_recs = [r for r in recommendations if r["algorithm"] == "fci"]
        assert len(fci_recs) > 0

        # Check patterns - should detect FCI issues
        patterns = result.get("detected_patterns")
        fci_patterns = [p for p in patterns if "fci" in p["description"].lower()]
        assert len(fci_patterns) > 0
