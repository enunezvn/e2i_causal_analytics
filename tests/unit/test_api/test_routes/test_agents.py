"""
Tests for src/api/routes/agents.py

Covers:
- GET /agents/status endpoint
- AgentInfo model validation
- AgentStatusResponse model validation
- Agent tier and status enums
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies.auth import require_auth
from src.api.routes.agents import (
    AGENT_REGISTRY,
    AgentInfo,
    AgentStatusEnum,
    AgentStatusResponse,
    AgentTierEnum,
    _apply_live_statuses,
    _derive_live_statuses,
    _humanize_slug,
    _normalize_agent_name,
    _row_status,
    _row_to_activity,
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

    def test_registry_has_22_agents(self):
        """Test that registry contains 22 agents."""
        assert len(AGENT_REGISTRY) == 22

    def test_tier_0_has_9_agents(self):
        """Test that Tier 0 (ML Foundation) has 9 agents."""
        tier_0 = [a for a in AGENT_REGISTRY if a.tier == 0]
        assert len(tier_0) == 9

    def test_tier_1_has_2_agents(self):
        """Test that Tier 1 (Orchestration) has 2 agents."""
        tier_1 = [a for a in AGENT_REGISTRY if a.tier == 1]
        assert len(tier_1) == 2

    def test_tier_2_has_3_agents(self):
        """Test that Tier 2 (Causal Analytics) has 3 agents."""
        tier_2 = [a for a in AGENT_REGISTRY if a.tier == 2]
        assert len(tier_2) == 3

    def test_tier_3_has_4_agents(self):
        """Test that Tier 3 (Monitoring) has 4 agents."""
        tier_3 = [a for a in AGENT_REGISTRY if a.tier == 3]
        assert len(tier_3) == 4

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
            "cohort-profiler",
            "orchestrator",
            "tool-composer",
            "causal-impact",
            "gap-analyzer",
            "heterogeneous-optimizer",
            "drift-monitor",
            "experiment-designer",
            "experiment-monitor",
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
        """Test that response contains 22 agents."""
        response = test_client.get("/agents/status")
        data = response.json()

        assert data["total_agents"] == 22
        assert len(data["agents"]) == 22

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


# =============================================================================
# Live-derivation helper tests (audit_chain_entries -> status / activity)
# =============================================================================


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class TestNameNormalization:
    """audit_chain_entries.agent_name (snake) -> registry id (kebab)."""

    def test_snake_to_kebab(self):
        assert _normalize_agent_name("gap_analyzer") == "gap-analyzer"
        assert _normalize_agent_name("experiment_monitor") == "experiment-monitor"

    def test_poller_aliases_to_canonical_agent(self):
        # The automated poller is the health-score agent under the hood.
        assert _normalize_agent_name("health_score_quick") == "health-score"

    def test_normalized_names_hit_the_registry(self):
        ids = {a.id for a in AGENT_REGISTRY}
        for raw in ("gap_analyzer", "resource_optimizer", "health_score_quick"):
            assert _normalize_agent_name(raw) in ids


class TestHumanizeSlug:
    def test_basic_titlecase(self):
        assert _humanize_slug("model_training") == "Model Training"

    def test_acronyms_preserved(self):
        assert _humanize_slug("estimate_cate") == "Estimate CATE"
        assert _humanize_slug("srm_detector") == "SRM Detector"

    def test_empty_slug(self):
        assert _humanize_slug("") == "Activity"


class TestRowStatus:
    """NULL validation == completed; only explicit False / *_error == failed."""

    def test_null_is_completed(self):
        assert _row_status(None, "agent") == "completed"

    def test_true_is_completed(self):
        assert _row_status(True, "agent") == "completed"

    def test_false_is_failed(self):
        assert _row_status(False, "agent") == "failed"

    def test_error_action_is_failed(self):
        assert _row_status(None, "estimate_error") == "failed"


class TestDeriveLiveStatuses:
    def test_recent_row_is_active(self):
        now = datetime.now(timezone.utc)
        rows = [{"agent_name": "health_score", "created_at": _iso(now), "action_type": "agent"}]
        live = _derive_live_statuses(rows, now)
        assert live["health-score"]["status"] == AgentStatusEnum.ACTIVE

    def test_old_row_is_idle(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=6)
        rows = [
            {"agent_name": "gap_analyzer", "created_at": _iso(old), "action_type": "gap_detector"}
        ]
        live = _derive_live_statuses(rows, now)
        assert live["gap-analyzer"]["status"] == AgentStatusEnum.IDLE
        assert live["gap-analyzer"]["last_activity"] == _iso(old)

    def test_failed_latest_is_error(self):
        now = datetime.now(timezone.utc)
        rows = [
            {
                "agent_name": "drift_monitor",
                "created_at": _iso(now),
                "validation_passed": False,
                "action_type": "agent",
            }
        ]
        live = _derive_live_statuses(rows, now)
        assert live["drift-monitor"]["status"] == AgentStatusEnum.ERROR

    def test_newest_first_dedup_keeps_latest(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=6)
        # rows are newest-first: the recent one must win -> ACTIVE
        rows = [
            {"agent_name": "health_score", "created_at": _iso(now), "action_type": "agent"},
            {"agent_name": "health_score", "created_at": _iso(old), "action_type": "agent"},
        ]
        live = _derive_live_statuses(rows, now)
        assert live["health-score"]["status"] == AgentStatusEnum.ACTIVE
        assert live["health-score"]["last_activity"] == _iso(now)

    def test_poller_counts_for_health_score_status(self):
        now = datetime.now(timezone.utc)
        rows = [
            {
                "agent_name": "health_score_quick",
                "created_at": _iso(now),
                "action_type": "component",
            }
        ]
        live = _derive_live_statuses(rows, now)
        assert live["health-score"]["status"] == AgentStatusEnum.ACTIVE


class TestApplyLiveStatuses:
    def test_returns_all_22_agents(self):
        agents = _apply_live_statuses({})
        assert len(agents) == 22

    def test_unmapped_agents_are_idle_with_no_activity(self):
        agents = _apply_live_statuses({})
        for a in agents:
            assert a.status == AgentStatusEnum.IDLE
            assert a.last_activity is None

    def test_mapped_agent_gets_overlaid(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        live = {"health-score": {"status": AgentStatusEnum.ACTIVE, "last_activity": now_iso}}
        agents = {a.id: a for a in _apply_live_statuses(live)}
        assert agents["health-score"].status == AgentStatusEnum.ACTIVE
        assert agents["health-score"].last_activity == now_iso

    def test_does_not_mutate_registry(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        live = {"health-score": {"status": AgentStatusEnum.ERROR, "last_activity": now_iso}}
        _apply_live_statuses(live)
        registry_health = next(a for a in AGENT_REGISTRY if a.id == "health-score")
        # The module-level registry must be untouched (last_activity stays None).
        assert registry_health.last_activity is None


class TestRowToActivity:
    def test_maps_registry_name_and_tier(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        item = _row_to_activity(
            {
                "entry_id": "abc",
                "agent_name": "gap_analyzer",
                "agent_tier": 2,
                "action_type": "gap_detector",
                "created_at": now_iso,
                "duration_ms": 42,
                "validation_passed": None,
                "query_text": "find gaps",
            }
        )
        assert item is not None
        assert item.agent_id == "gap-analyzer"
        assert item.agent_name == "Gap Analyzer"  # from registry
        assert item.tier == 2
        assert item.action == "Gap Detector"
        assert item.status == "completed"  # NULL validation
        assert item.duration_ms == 42
        assert item.details == "find gaps"

    def test_missing_timestamp_returns_none(self):
        assert _row_to_activity({"agent_name": "gap_analyzer", "created_at": None}) is None

    def test_unmapped_agent_falls_back_to_row_tier(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        item = _row_to_activity(
            {
                "entry_id": "x",
                "agent_name": "ml_foundation_pipeline",
                "agent_tier": 0,
                "action_type": "pipeline_start",
                "created_at": now_iso,
            }
        )
        assert item is not None
        assert item.tier == 0
        assert item.agent_name == "ML Foundation Pipeline"


# =============================================================================
# GET /agents/activity endpoint + live /agents/status
# =============================================================================


@pytest.fixture
def auth_client():
    """Test client with require_auth overridden (the activity feed is gated)."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = lambda: {"user_id": "test-user"}
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestGetAgentActivityEndpoint:
    # NOTE: auth enforcement (Depends(require_auth)) is declarative and is
    # exercised by the auth dependency's own tests; in this unit env require_auth
    # is bypassed, so a "rejects without auth" assertion here would be testing
    # the env, not the route.

    def test_returns_mapped_activities(self, auth_client, monkeypatch):
        now_iso = datetime.now(timezone.utc).isoformat()
        captured = {}

        def fake_fetch(since, *, feed_view, limit):
            captured["feed_view"] = feed_view
            return [
                {
                    "entry_id": "e1",
                    "agent_name": "gap_analyzer",
                    "agent_tier": 2,
                    "action_type": "gap_detector",
                    "created_at": now_iso,
                    "duration_ms": 12,
                    "validation_passed": None,
                    "query_text": "",
                }
            ]

        monkeypatch.setattr("src.api.routes.agents._fetch_audit_rows", fake_fetch)
        resp = auth_client.get("/agents/activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["window_hours"] == 24
        assert data["activities"][0]["agent_name"] == "Gap Analyzer"
        assert data["activities"][0]["status"] == "completed"
        # The feed MUST use feed-view filtering (poller + scaffolding excluded).
        assert captured["feed_view"] is True

    def test_empty_is_honest_empty(self, auth_client, monkeypatch):
        monkeypatch.setattr(
            "src.api.routes.agents._fetch_audit_rows",
            lambda since, *, feed_view, limit: [],
        )
        resp = auth_client.get("/agents/activity")
        assert resp.status_code == 200
        assert resp.json()["activities"] == []
        assert resp.json()["total"] == 0


class TestLiveAgentStatus:
    def test_recent_activity_marks_agent_active(self, test_client, monkeypatch):
        now_iso = datetime.now(timezone.utc).isoformat()
        monkeypatch.setattr(
            "src.api.routes.agents._fetch_audit_rows",
            lambda since, *, feed_view, limit: [
                {"agent_name": "health_score", "created_at": now_iso, "action_type": "agent"}
            ],
        )
        data = test_client.get("/agents/status").json()
        by_id = {a["id"]: a for a in data["agents"]}
        assert by_id["health-score"]["status"] == "active"
        assert by_id["health-score"]["last_activity"] == now_iso
        # An agent with no telemetry stays idle — never fabricated active.
        assert by_id["orchestrator"]["status"] == "idle"
        assert data["active_count"] >= 1

    def test_no_telemetry_all_idle(self, test_client, monkeypatch):
        monkeypatch.setattr(
            "src.api.routes.agents._fetch_audit_rows",
            lambda since, *, feed_view, limit: [],
        )
        data = test_client.get("/agents/status").json()
        assert data["active_count"] == 0
        assert all(a["status"] == "idle" for a in data["agents"])
        assert data["total_agents"] == 22
