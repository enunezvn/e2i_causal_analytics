"""
E2I Resource Optimizer Agent - Impact Projector Node Tests
"""

import pytest

from src.agents.resource_optimizer.nodes.impact_projector import (
    ImpactProjectorNode,
)


class TestImpactProjectorNode:
    """Tests for ImpactProjectorNode."""

    @pytest.mark.asyncio
    async def test_project_impact(self, optimized_state):
        """Test impact projection."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        assert result["status"] == "completed"
        assert result["projected_total_outcome"] is not None
        assert result["projected_roi"] is not None
        assert result["optimization_summary"] is not None
        assert result["total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_project_total_outcome(self, optimized_state):
        """Test total outcome calculation."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        expected_outcome = sum(a["expected_impact"] for a in optimized_state["optimal_allocations"])
        assert result["projected_total_outcome"] == pytest.approx(expected_outcome, rel=0.01)

    @pytest.mark.asyncio
    async def test_project_roi(self, optimized_state):
        """Test ROI calculation."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        # The implementation calculates incremental ROI when allocation_targets are present:
        # ROI = (projected_outcome - current_outcome) / optimized_total
        allocation_targets = optimized_state.get("allocation_targets", [])

        # Build response map - handle both AllocationTarget objects and dicts
        response_by_entity = {}
        for t in allocation_targets:
            if hasattr(t, 'entity_id'):
                # AllocationTarget object
                response_by_entity[t.entity_id] = t.expected_response
            else:
                # Dictionary
                response_by_entity[t.get("entity_id")] = t.get("expected_response", 0)

        projected_outcome = sum(
            response_by_entity.get(a["entity_id"], 0) * a["optimized_allocation"]
            for a in optimized_state["optimal_allocations"]
        )

        # Calculate current outcome
        current_outcome = 0
        for t in allocation_targets:
            if hasattr(t, 'entity_id'):
                current_outcome += t.expected_response * t.current_allocation
            else:
                current_outcome += t.get("expected_response", 0) * t.get("current_allocation", 0)

        total_allocation = sum(
            a["optimized_allocation"] for a in optimized_state["optimal_allocations"]
        )
        expected_roi = (projected_outcome - current_outcome) / total_allocation

        assert result["projected_roi"] == pytest.approx(expected_roi, rel=0.01)

    @pytest.mark.asyncio
    async def test_project_impact_by_segment(self, optimized_state):
        """Test impact by segment calculation."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        assert result["impact_by_segment"] is not None
        assert "territory" in result["impact_by_segment"]

    @pytest.mark.asyncio
    async def test_project_summary_generation(self, optimized_state):
        """Test summary generation."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        summary = result["optimization_summary"]
        assert "Optimization complete" in summary
        assert "ROI" in summary

    @pytest.mark.asyncio
    async def test_project_recommendations(self, optimized_state):
        """Test recommendations generation."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        recommendations = result["recommendations"]
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_project_increase_recommendations(self, optimized_state):
        """Test increase recommendations."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        recommendations = result["recommendations"]
        increase_recs = [r for r in recommendations if "Increase" in r]
        assert len(increase_recs) > 0

    @pytest.mark.asyncio
    async def test_project_decrease_recommendations(self, optimized_state):
        """Test decrease recommendations."""
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        recommendations = result["recommendations"]
        decrease_recs = [r for r in recommendations if "Reduce" in r]
        assert len(decrease_recs) > 0

    @pytest.mark.asyncio
    async def test_project_no_allocations_fails(self, base_state):
        """Test failure when no allocations."""
        base_state["status"] = "projecting"
        base_state["optimal_allocations"] = []
        node = ImpactProjectorNode()
        result = await node.execute(base_state)

        assert result["status"] == "failed"
        assert any("No allocations" in e["error"] for e in result["errors"])

    @pytest.mark.asyncio
    async def test_project_already_failed_passthrough(self, optimized_state):
        """Test that already failed state passes through."""
        optimized_state["status"] = "failed"
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_project_total_latency(self, optimized_state):
        """Test total latency calculation."""
        optimized_state["formulation_latency_ms"] = 5
        optimized_state["optimization_latency_ms"] = 10
        node = ImpactProjectorNode()
        result = await node.execute(optimized_state)

        # Total latency should include formulation + optimization + projection
        assert result["total_latency_ms"] >= 15
