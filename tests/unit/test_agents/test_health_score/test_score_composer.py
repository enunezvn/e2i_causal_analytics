"""
Tests for Score Composer Node
"""

from datetime import datetime

import pytest

from src.agents.health_score.metrics import ScoreWeights
from src.agents.health_score.nodes.score_composer import ScoreComposerNode


class TestScoreComposerNode:
    """Tests for ScoreComposerNode"""

    @pytest.fixture
    def full_state(self):
        """State with all health scores"""
        return {
            "query": "",
            "check_scope": "full",
            "component_statuses": [],
            "component_health_score": 0.9,
            "model_metrics": [],
            "model_health_score": 0.8,
            "pipeline_statuses": [],
            "pipeline_health_score": 0.85,
            "agent_statuses": [],
            "agent_health_score": 0.95,
            "overall_health_score": None,
            "health_grade": None,
            "critical_issues": None,
            "warnings": None,
            "health_summary": None,
            "total_latency_ms": 100,
            "timestamp": "",
            "errors": [],
            "status": "checking",
        }

    @pytest.mark.asyncio
    async def test_weighted_score_calculation(self, full_state):
        """Test weighted average calculation"""
        node = ScoreComposerNode()
        result = await node.execute(full_state)

        # Expected: 0.9*0.30 + 0.8*0.30 + 0.85*0.25 + 0.95*0.15 = 0.865
        expected_score = 0.865 * 100
        assert abs(result["overall_health_score"] - expected_score) < 0.01

    @pytest.mark.asyncio
    async def test_grade_a(self, full_state):
        """Test grade A (>=90%)"""
        full_state["component_health_score"] = 1.0
        full_state["model_health_score"] = 1.0
        full_state["pipeline_health_score"] = 1.0
        full_state["agent_health_score"] = 1.0

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "A"
        assert result["overall_health_score"] == 100.0

    @pytest.mark.asyncio
    async def test_grade_b(self, full_state):
        """Test grade B (>=80%, <90%)"""
        # Set scores to get ~85%
        full_state["component_health_score"] = 0.85
        full_state["model_health_score"] = 0.85
        full_state["pipeline_health_score"] = 0.85
        full_state["agent_health_score"] = 0.85

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "B"

    @pytest.mark.asyncio
    async def test_grade_c(self, full_state):
        """Test grade C (>=70%, <80%)"""
        full_state["component_health_score"] = 0.75
        full_state["model_health_score"] = 0.75
        full_state["pipeline_health_score"] = 0.75
        full_state["agent_health_score"] = 0.75

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "C"

    @pytest.mark.asyncio
    async def test_grade_d(self, full_state):
        """Test grade D (>=60%, <70%)"""
        full_state["component_health_score"] = 0.65
        full_state["model_health_score"] = 0.65
        full_state["pipeline_health_score"] = 0.65
        full_state["agent_health_score"] = 0.65

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "D"

    @pytest.mark.asyncio
    async def test_grade_f(self, full_state):
        """Test grade F (<60%)"""
        full_state["component_health_score"] = 0.5
        full_state["model_health_score"] = 0.5
        full_state["pipeline_health_score"] = 0.5
        full_state["agent_health_score"] = 0.5

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "F"

    @pytest.mark.asyncio
    async def test_status_completed(self, full_state):
        """Test that status is set to completed"""
        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_timestamp_recorded(self, full_state):
        """Test that timestamp is recorded"""
        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["timestamp"] is not None
        # Should be valid ISO format
        datetime.fromisoformat(result["timestamp"])

    @pytest.mark.asyncio
    async def test_missing_scores_default_to_healthy(self):
        """Test that missing scores default to 1.0"""
        minimal_state = {
            "query": "",
            "check_scope": "full",
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(minimal_state)

        # All default to 1.0, so should be grade A
        assert result["health_grade"] == "A"
        assert result["overall_health_score"] == 100.0


class TestIssueIdentification:
    """Tests for issue and warning identification"""

    @pytest.mark.asyncio
    async def test_identifies_unhealthy_components(self):
        """Test identification of unhealthy components"""
        state = {
            "component_statuses": [
                {"component_name": "db", "status": "unhealthy"},
                {"component_name": "cache", "status": "healthy"},
            ],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Component 'db' is unhealthy" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_degraded_components(self):
        """Test identification of degraded components as warnings"""
        state = {
            "component_statuses": [
                {"component_name": "cache", "status": "degraded"},
            ],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Component 'cache' is degraded" in result["warnings"]

    @pytest.mark.asyncio
    async def test_identifies_unhealthy_models(self):
        """Test identification of unhealthy models"""
        state = {
            "component_statuses": [],
            "model_metrics": [
                {"model_id": "model_1", "status": "unhealthy"},
            ],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Model 'model_1' is unhealthy" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_failed_pipelines(self):
        """Test identification of failed pipelines"""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [
                {"pipeline_name": "etl", "status": "failed"},
            ],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Pipeline 'etl' has failed" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_unavailable_agents(self):
        """Test identification of unavailable agents"""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [
                {"agent_name": "agent_1", "available": False, "success_rate": 0.0},
            ],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Agent 'agent_1' is unavailable" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_low_success_rate_agents(self):
        """Test identification of low success rate agents as warnings"""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [
                {"agent_name": "agent_1", "available": True, "success_rate": 0.7},
            ],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert any("low success rate" in w for w in result["warnings"])


class TestSummaryGeneration:
    """Tests for summary generation"""

    @pytest.mark.asyncio
    async def test_excellent_summary(self):
        """Test excellent health summary"""
        state = {
            "component_health_score": 1.0,
            "model_health_score": 1.0,
            "pipeline_health_score": 1.0,
            "agent_health_score": 1.0,
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "excellent" in result["health_summary"]
        assert "Grade: A" in result["health_summary"]
        assert "All systems operational" in result["health_summary"]

    @pytest.mark.asyncio
    async def test_critical_summary(self):
        """Test critical health summary"""
        state = {
            "component_health_score": 0.3,
            "model_health_score": 0.3,
            "pipeline_health_score": 0.3,
            "agent_health_score": 0.3,
            "component_statuses": [
                {"component_name": "db", "status": "unhealthy"},
            ],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "critical" in result["health_summary"]
        assert "Grade: F" in result["health_summary"]
        assert "critical issue" in result["health_summary"]


class TestCustomWeightsAndGrades:
    """Tests for custom weights and grade thresholds"""

    @pytest.mark.asyncio
    async def test_custom_weights(self):
        """Test with custom score weights"""
        state = {
            "component_health_score": 1.0,
            "model_health_score": 0.5,
            "pipeline_health_score": 0.5,
            "agent_health_score": 0.5,
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        # Custom weights: component = 1.0, others = 0
        custom_weights = ScoreWeights(
            component=1.0,
            model=0.0,
            pipeline=0.0,
            agent=0.0,
        )

        node = ScoreComposerNode(weights=custom_weights)
        result = await node.execute(state)

        # Should be 100% since only component matters
        assert result["overall_health_score"] == 100.0
        assert result["health_grade"] == "A"
