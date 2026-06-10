"""
Tests for Component Health Node
"""

from datetime import datetime

import pytest

from src.agents.health_score.nodes.component_health import ComponentHealthNode


class TestComponentHealthNode:
    """Tests for ComponentHealthNode"""

    @pytest.mark.asyncio
    async def test_healthy_components(self, mock_health_client, initial_state):
        """Test all healthy components"""
        node = ComponentHealthNode(health_client=mock_health_client)
        result = await node.execute(initial_state)

        assert result["component_health_score"] == 1.0
        assert len(result["component_statuses"]) == 5
        for status in result["component_statuses"]:
            assert status["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_mixed_health_components(self, unhealthy_health_client, initial_state):
        """Test mixed healthy/unhealthy components"""
        node = ComponentHealthNode(health_client=unhealthy_health_client)
        result = await node.execute(initial_state)

        # 3 healthy + 1 degraded (0.5) + 1 unhealthy (0) = 3.5/5 = 0.7
        assert result["component_health_score"] == 0.7

        # Verify individual statuses
        status_map = {s["component_name"]: s["status"] for s in result["component_statuses"]}
        assert status_map["database"] == "unhealthy"
        assert status_map["vector_store"] == "degraded"

    @pytest.mark.asyncio
    async def test_quick_check_measures_components(self, mock_health_client, quick_check_state):
        """Codex F1.2 regression: quick check IS the component-focused check
        (graph.py quick graph = component -> compose). With a real health_client
        the component node MUST measure components on quick scope, NOT skip them
        (skipping made quick yield 0/F/unknown)."""
        node = ComponentHealthNode(health_client=mock_health_client)
        result = await node.execute(quick_check_state)

        assert result["component_health_measured"] is True
        assert result["component_health_score"] == 1.0
        assert len(result["component_statuses"]) == 5
        assert mock_health_client.call_count > 0

    @pytest.mark.asyncio
    async def test_skips_components_for_models_only_scope(self, mock_health_client, initial_state):
        """A scope that excludes components (models/pipelines/agents-only) skips
        the component check => UNMEASURED (no fail-open 1.0 score). Quick scope is
        explicitly NOT in this set."""
        initial_state["check_scope"] = "models"
        node = ComponentHealthNode(health_client=mock_health_client)
        result = await node.execute(initial_state)

        assert result["component_health_measured"] is False
        assert result.get("component_health_score") is None
        assert result["component_statuses"] == []
        assert mock_health_client.call_count == 0

    @pytest.mark.asyncio
    async def test_no_client_fails_closed_to_unknown(self, initial_state):
        """F1 (was test_no_client_uses_mock_statuses): with NO real health_client
        the component dimension is UNKNOWN, not fabricated-healthy. The node must
        emit per-component 'unknown' statuses (so the dashboard still lists them)
        and mark the dimension unmeasured with NO fail-open 1.0 score."""
        node = ComponentHealthNode(health_client=None)
        result = await node.execute(initial_state)

        assert result["component_health_measured"] is False
        assert result.get("component_health_score") is None  # no fail-open score written
        # The per-component status list is still emitted, but as 'unknown'.
        assert len(result["component_statuses"]) == 5
        for status in result["component_statuses"]:
            assert status["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_custom_components(self, mock_health_client, initial_state):
        """Test with custom component list"""
        custom_components = [
            {"name": "custom_db", "endpoint": "/health/custom_db"},
            {"name": "custom_api", "endpoint": "/health/custom_api"},
        ]
        node = ComponentHealthNode(
            health_client=mock_health_client,
            components=custom_components,
        )
        result = await node.execute(initial_state)

        assert len(result["component_statuses"]) == 2
        names = [s["component_name"] for s in result["component_statuses"]]
        assert "custom_db" in names
        assert "custom_api" in names

    @pytest.mark.asyncio
    async def test_records_latency(self, mock_health_client, initial_state):
        """Test that latency is recorded"""
        node = ComponentHealthNode(health_client=mock_health_client)
        result = await node.execute(initial_state)

        assert result["total_latency_ms"] >= 0
        for status in result["component_statuses"]:
            assert status["latency_ms"] is not None
            assert status["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_records_timestamp(self, mock_health_client, initial_state):
        """Test that timestamps are recorded"""
        node = ComponentHealthNode(health_client=mock_health_client)
        result = await node.execute(initial_state)

        for status in result["component_statuses"]:
            assert status["last_check"] is not None
            # Verify it's a valid ISO timestamp
            datetime.fromisoformat(status["last_check"])

    @pytest.mark.asyncio
    async def test_status_progression(self, mock_health_client, initial_state):
        """Test that status is set to checking"""
        node = ComponentHealthNode(health_client=mock_health_client)
        result = await node.execute(initial_state)

        assert result["status"] == "checking"


class TestComponentHealthTimeout:
    """Tests for timeout handling"""

    @pytest.mark.asyncio
    async def test_custom_timeout(self, mock_health_client, initial_state):
        """Test custom timeout configuration"""
        node = ComponentHealthNode(
            health_client=mock_health_client,
            timeout_ms=5000,
        )
        assert node.timeout_ms == 5000
