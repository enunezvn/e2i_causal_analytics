"""
Tests for Agent Health Node
"""

import pytest

from src.agents.health_score.nodes.agent_health import AgentHealthNode


class TestAgentHealthNode:
    """Tests for AgentHealthNode"""

    @pytest.mark.asyncio
    async def test_healthy_agents(self, mock_agent_registry, initial_state):
        """Test all healthy agents"""
        node = AgentHealthNode(agent_registry=mock_agent_registry)
        result = await node.execute(initial_state)

        assert result["agent_health_score"] == 1.0
        assert len(result["agent_statuses"]) == 2
        for status in result["agent_statuses"]:
            assert status["available"] is True

    @pytest.mark.asyncio
    async def test_unavailable_agents(self, unavailable_agent_registry, initial_state):
        """Test with unavailable and low success agents"""
        node = AgentHealthNode(agent_registry=unavailable_agent_registry)
        result = await node.execute(initial_state)

        # 1 healthy (1.0) + 1 unavailable (0) + 1 low success (0.5) = 1.5/3 = 0.5
        assert result["agent_health_score"] == 0.5

        # Check statuses
        status_map = {a["agent_name"]: a for a in result["agent_statuses"]}
        assert status_map["available_agent"]["available"] is True
        assert status_map["available_agent"]["success_rate"] >= 0.9
        assert status_map["unavailable_agent"]["available"] is False
        assert status_map["low_success_agent"]["available"] is True
        assert status_map["low_success_agent"]["success_rate"] < 0.9

    @pytest.mark.asyncio
    async def test_skips_for_non_agent_scope(self, mock_agent_registry, initial_state):
        """Test that non-agent scope skips agent check.

        F1: a skipped dimension is UNMEASURED (no fail-open 1.0 score).
        """
        initial_state["check_scope"] = "models"
        node = AgentHealthNode(agent_registry=mock_agent_registry)
        result = await node.execute(initial_state)

        assert result["agent_health_measured"] is False
        assert result.get("agent_health_score") is None  # no fail-open score written
        assert result["agent_statuses"] == []

    @pytest.mark.asyncio
    async def test_includes_for_agent_scope(self, mock_agent_registry, initial_state):
        """Test that agents scope includes agent check"""
        initial_state["check_scope"] = "agents"
        node = AgentHealthNode(agent_registry=mock_agent_registry)
        result = await node.execute(initial_state)

        assert len(result["agent_statuses"]) == 2

    @pytest.mark.asyncio
    async def test_no_registry_fails_closed_to_unmeasured(self, initial_state):
        """F1 (was test_no_registry_returns_healthy): with NO real agent_registry
        the agent dimension is UNKNOWN, not 'healthy by default'. The node must
        mark it unmeasured and emit NO fail-open 1.0 score."""
        node = AgentHealthNode(agent_registry=None)
        result = await node.execute(initial_state)

        assert result["agent_health_measured"] is False
        assert result.get("agent_health_score") is None  # no fail-open score written
        assert result["agent_statuses"] == []

    @pytest.mark.asyncio
    async def test_accumulates_latency(self, mock_agent_registry, initial_state):
        """Test that latency is accumulated"""
        initial_state["total_latency_ms"] = 100
        node = AgentHealthNode(agent_registry=mock_agent_registry)
        result = await node.execute(initial_state)

        assert result["total_latency_ms"] >= 100


class TestAgentStatusFields:
    """Tests for agent status field population"""

    @pytest.mark.asyncio
    async def test_all_fields_populated(self, mock_agent_registry, initial_state):
        """Test that all status fields are populated"""
        node = AgentHealthNode(agent_registry=mock_agent_registry)
        result = await node.execute(initial_state)

        status = result["agent_statuses"][0]
        assert status["agent_name"] is not None
        assert status["tier"] >= 0
        assert isinstance(status["available"], bool)
        assert status["avg_latency_ms"] >= 0
        assert 0.0 <= status["success_rate"] <= 1.0
        assert status["last_invocation"] is not None


class TestAgentHealthThresholds:
    """Tests for threshold-based scoring"""

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, mock_agent_registry, initial_state):
        """Test custom success rate threshold"""
        node = AgentHealthNode(
            agent_registry=mock_agent_registry,
            min_success_rate=0.99,  # Very high threshold
        )
        result = await node.execute(initial_state)

        # With default metrics at 0.95, should be degraded
        assert result["agent_health_score"] < 1.0


@pytest.mark.asyncio
async def test_agent_health_exception_fails_closed_to_unmeasured(initial_state):
    """codex R3: a FAILED agent check measured nothing. The node must mark the
    dimension UNMEASURED (so the composer excludes it), NOT emit a fabricated
    0.5 score with measured=True presented to the dashboard as real."""
    from unittest.mock import AsyncMock, MagicMock

    store = MagicMock()
    store.get_all_agents = AsyncMock(side_effect=RuntimeError("boom"))
    node = AgentHealthNode(agent_registry=store)
    result = await node.execute(initial_state)

    assert result["agent_health_measured"] is False
    assert result.get("agent_health_score") is None
