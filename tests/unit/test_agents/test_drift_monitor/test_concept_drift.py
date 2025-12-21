"""Tests for Concept Drift Detection Node.

NOTE: These are placeholder tests as concept drift detection is not fully implemented.
"""

import pytest

from src.agents.drift_monitor.nodes.concept_drift import ConceptDriftNode
from src.agents.drift_monitor.state import DriftMonitorState


class TestConceptDriftNode:
    """Test ConceptDriftNode (placeholder)."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state with defaults."""
        state: DriftMonitorState = {
            "query": "test",
            "features_to_monitor": ["feature1"],
            "time_window": "7d",
            "significance_level": 0.05,
            "psi_threshold": 0.1,
            "check_data_drift": True,
            "check_model_drift": True,
            "check_concept_drift": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "concept_drift_results" in result
        assert isinstance(result["concept_drift_results"], list)

    @pytest.mark.asyncio
    async def test_returns_empty_results(self):
        """Test returns empty results (placeholder)."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Placeholder implementation returns empty results
        assert result["concept_drift_results"] == []

    @pytest.mark.asyncio
    async def test_adds_warning(self):
        """Test adds warning about not implemented."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should add warning about not implemented
        assert any("not yet implemented" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "detection_latency_ms" in result
        assert result["detection_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_concept_drift_disabled(self):
        """Test concept drift check can be disabled."""
        node = ConceptDriftNode()
        state = self._create_test_state(check_concept_drift=False)

        result = await node.execute(state)

        assert result["concept_drift_results"] == []
        assert any("skipped" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = ConceptDriftNode()
        state = self._create_test_state(status="failed")

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result["concept_drift_results"] == []
