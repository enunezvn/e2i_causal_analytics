"""
Tests for src/api/routes/agents.py

Covers:
- GET /agents/status endpoint
- AgentInfo model validation
- AgentStatusResponse model validation
- Agent tier and status enums
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.routes.agents import (
    AgentInfo,
    AgentStatusEnum,
    AgentStatusResponse,
    AgentTierEnum,
    AGENT_REGISTRY,
    router,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_client():
    """Create a FastAPI test client with the agents router."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# =============================================================================
# AgentStatusEnum Tests
# =============================================================================


class TestAgentStatusEnum:
    """Tests for AgentStatusEnum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert AgentStatusEnum.ACTIVE.value == "active"
        assert AgentStatusEnum.IDLE.value == "idle"
        assert AgentStatusEnum.PROCESSING.value == "processing"
        assert AgentStatusEnum.ERROR.value == "error"

    def test_status_count(self):
        """Test that we have exactly 4 status values."""
        assert len(AgentStatusEnum) == 4


# =============================================================================
# AgentTierEnum Tests
# =============================================================================


class TestAgentTierEnum:
    """Tests for AgentTierEnum."""

    def test_tier_values(self):
        """Test that all expected tier values exist."""
        assert AgentTierEnum.ML_FOUNDATION.value == 0
        assert AgentTierEnum.ORCHESTRATION.value == 1
        assert AgentTierEnum.CAUSAL_ANALYTICS.value == 2
        assert AgentTierEnum.MONITORING.value == 3
        assert AgentTierEnum.ML_PREDICTIONS.value == 4
        assert AgentTierEnum.SELF_IMPROVEMENT.value == 5

    def test_tier_count(self):
        """Test that we have exactly 6 tiers."""
        assert len(AgentTierEnum) == 6


# =============================================================================
# AgentInfo Model Tests
# =============================================================================


class TestAgentInfo:
    """Tests for AgentInfo model."""

    def test_create_valid_agent(self):
        """Test creating a valid AgentInfo."""
        agent = AgentInfo(
            id="test-agent",
            name="Test Agent",
            tier=0,
            status=AgentStatusEnum.ACTIVE,
            capabilities=["testing"],
        )
        assert agent.id == "test-agent"
        assert agent.name == "Test Agent"
        assert agent.tier == 0
        assert agent.status == AgentStatusEnum.ACTIVE
        assert agent.capabilities == ["testing"]

    def test_agent_with_last_activity(self):
        """Test creating an agent with last_activity timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        agent = AgentInfo(
            id="test-agent",
            name="Test Agent",
            tier=1,
            status=AgentStatusEnum.PROCESSING,
            last_activity=timestamp,
        )
        assert agent.last_activity == timestamp

    def test_agent_default_capabilities(self):
        """Test that capabilities defaults to empty list."""
        agent = AgentInfo(
            id="test-agent",
            name="Test Agent",
            tier=2,
            status=AgentStatusEnum.IDLE,
        )
        assert agent.capabilities == []

    def test_agent_tier_validation(self):
        """Test that tier must be 0-5."""
        # Valid tiers
        for tier in range(6):
            agent = AgentInfo(
                id="test",
                name="Test",
                tier=tier,
                status=AgentStatusEnum.ACTIVE,
            )
            assert agent.tier == tier

        # Invalid tier
        with pytest.raises(ValueError):
            AgentInfo(
                id="test",
                name="Test",
                tier=6,
                status=AgentStatusEnum.ACTIVE,
            )


# =============================================================================
# AgentStatusResponse Model Tests
# =============================================================================


class TestAgentStatusResponse:
    """Tests for AgentStatusResponse model."""

    def test_create_valid_response(self):
        """Test creating a valid response."""
        agents = [
            AgentInfo(
                id="agent-1",
                name="Agent 1",
                tier=0,
                status=AgentStatusEnum.ACTIVE,
            ),
            AgentInfo(
                id="agent-2",
                name="Agent 2",
                tier=1,
                status=AgentStatusEnum.IDLE,
            ),
        ]
        response = AgentStatusResponse(
            agents=agents,
            total_agents=2,
            active_count=1,
            processing_count=0,
            error_count=0,
            timestamp=datetime.now(timezone.utc),
        )
        assert response.total_agents == 2
        assert response.active_count == 1
        assert len(response.agents) == 2


# =============================================================================
# AGENT_REGISTRY Tests
# =============================================================================


class TestAgentRegistry:
    """Tests for the agent registry."""

    def test_registry_has_20_agents(self):
        """Test that registry contains 20 agents."""
        assert len(AGENT_REGISTRY) == 20

    def test_tier_0_has_8_agents(self):
        """Test that Tier 0 (ML Foundation) has 8 agents."""
        tier_0 = [a for a in AGENT_REGISTRY if a.tier == 0]
        assert len(tier_0) == 8

    def test_tier_1_has_2_agents(self):
        """Test that Tier 1 (Orchestration) has 2 agents."""
        tier_1 = [a for a in AGENT_REGISTRY if a.tier == 1]
        assert len(tier_1) == 2

    def test_tier_2_has_3_agents(self):
        """Test that Tier 2 (Causal Analytics) has 3 agents."""
        tier_2 = [a for a in AGENT_REGISTRY if a.tier == 2]
        assert len(tier_2) == 3

    def test_tier_3_has_3_agents(self):
        """Test that Tier 3 (Monitoring) has 3 agents."""
        tier_3 = [a for a in AGENT_REGISTRY if a.tier == 3]
        assert len(tier_3) == 3

    def test_tier_4_has_2_agents(self):
        """Test that Tier 4 (ML Predictions) has 2 agents."""
        tier_4 = [a for a in AGENT_REGISTRY if a.tier == 4]
        assert len(tier_4) == 2

    def test_tier_5_has_2_agents(self):
        """Test that Tier 5 (Self-Improvement) has 2 agents."""
        tier_5 = [a for a in AGENT_REGISTRY if a.tier == 5]
        assert len(tier_5) == 2

    def test_all_agents_have_unique_ids(self):
        """Test that all agents have unique IDs."""
        ids = [a.id for a in AGENT_REGISTRY]
        assert len(ids) == len(set(ids))

    def test_all_agents_have_capabilities(self):
        """Test that all agents have at least one capability."""
        for agent in AGENT_REGISTRY:
            assert len(agent.capabilities) > 0, f"{agent.id} has no capabilities"

    def test_known_agents_exist(self):
        """Test that specific known agents exist."""
        agent_ids = {a.id for a in AGENT_REGISTRY}
        expected_agents = [
            "scope-definer",
            "data-preparer",
            "feature-analyzer",
            "model-selector",
            "model-trainer",
            "model-deployer",
            "observability-connector",
            "cohort-constructor",
            "orchestrator",
            "tool-composer",
            "causal-impact",
            "gap-analyzer",
            "heterogeneous-optimizer",
            "drift-monitor",
            "experiment-designer",
            "health-score",
            "prediction-synthesizer",
            "resource-optimizer",
            "explainer",
            "feedback-learner",
        ]
        for agent_id in expected_agents:
            assert agent_id in agent_ids, f"Agent {agent_id} not found"


# =============================================================================
# GET /agents/status Endpoint Tests
# =============================================================================


class TestGetAgentStatusEndpoint:
    """Tests for GET /agents/status endpoint."""

    def test_get_status_success(self, test_client):
        """Test successful status retrieval."""
        response = test_client.get("/agents/status")
        assert response.status_code == 200

    def test_response_has_required_fields(self, test_client):
        """Test that response contains all required fields."""
        response = test_client.get("/agents/status")
        data = response.json()

        assert "agents" in data
        assert "total_agents" in data
        assert "active_count" in data
        assert "processing_count" in data
        assert "error_count" in data
        assert "timestamp" in data

    def test_response_agent_count(self, test_client):
        """Test that response contains 20 agents."""
        response = test_client.get("/agents/status")
        data = response.json()

        assert data["total_agents"] == 20
        assert len(data["agents"]) == 20

    def test_response_counts_are_valid(self, test_client):
        """Test that status counts are non-negative and sum correctly."""
        response = test_client.get("/agents/status")
        data = response.json()

        assert data["active_count"] >= 0
        assert data["processing_count"] >= 0
        assert data["error_count"] >= 0

        # Total should match the breakdown
        total = data["active_count"] + data["processing_count"] + data["error_count"]
        idle_count = sum(1 for a in data["agents"] if a["status"] == "idle")
        assert total + idle_count == data["total_agents"]

    def test_response_timestamp_format(self, test_client):
        """Test that timestamp is a valid ISO format."""
        response = test_client.get("/agents/status")
        data = response.json()

        timestamp = data["timestamp"]
        # Should be parseable as datetime
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed is not None

    def test_each_agent_has_required_fields(self, test_client):
        """Test that each agent has all required fields."""
        response = test_client.get("/agents/status")
        data = response.json()

        for agent in data["agents"]:
            assert "id" in agent
            assert "name" in agent
            assert "tier" in agent
            assert "status" in agent
            assert "capabilities" in agent

    def test_agent_tiers_are_valid(self, test_client):
        """Test that all agent tiers are between 0-5."""
        response = test_client.get("/agents/status")
        data = response.json()

        for agent in data["agents"]:
            assert 0 <= agent["tier"] <= 5

    def test_agent_statuses_are_valid(self, test_client):
        """Test that all agent statuses are valid enum values."""
        response = test_client.get("/agents/status")
        data = response.json()

        valid_statuses = {"active", "idle", "processing", "error"}
        for agent in data["agents"]:
            assert agent["status"] in valid_statuses


# =============================================================================
# Integration Tests
# =============================================================================


class TestAgentStatusIntegration:
    """Integration tests for agent status functionality."""

    def test_multiple_requests_consistent(self, test_client):
        """Test that multiple requests return consistent data."""
        response1 = test_client.get("/agents/status")
        response2 = test_client.get("/agents/status")

        data1 = response1.json()
        data2 = response2.json()

        # Agents should be the same
        assert data1["total_agents"] == data2["total_agents"]
        assert len(data1["agents"]) == len(data2["agents"])

        # Agent IDs should match
        ids1 = {a["id"] for a in data1["agents"]}
        ids2 = {a["id"] for a in data2["agents"]}
        assert ids1 == ids2

    def test_agent_capabilities_not_empty(self, test_client):
        """Test that all agents have capabilities."""
        response = test_client.get("/agents/status")
        data = response.json()

        for agent in data["agents"]:
            assert len(agent["capabilities"]) > 0
