"""
Integration Tests for Health Score Agent
"""

from datetime import datetime, timezone

import pytest

from src.agents.health_score.agent import (
    HealthScoreAgent,
    HealthScoreInput,
    HealthScoreOutput,
    check_system_health,
)
from src.agents.health_score.graph import (
    build_health_score_graph,
    build_quick_check_graph,
)


class TestHealthScoreAgentInit:
    """Tests for HealthScoreAgent initialization"""

    def test_initialization_without_clients(self):
        """Test initialization without any clients"""
        agent = HealthScoreAgent()
        assert agent.health_client is None
        assert agent.metrics_store is None
        assert agent.pipeline_store is None
        assert agent.agent_registry is None

    def test_initialization_with_clients(
        self,
        mock_health_client,
        mock_metrics_store,
        mock_pipeline_store,
        mock_agent_registry,
    ):
        """Test initialization with all clients"""
        agent = HealthScoreAgent(
            health_client=mock_health_client,
            metrics_store=mock_metrics_store,
            pipeline_store=mock_pipeline_store,
            agent_registry=mock_agent_registry,
        )
        assert agent.health_client is mock_health_client
        assert agent.metrics_store is mock_metrics_store
        assert agent.pipeline_store is mock_pipeline_store
        assert agent.agent_registry is mock_agent_registry


class TestHealthScoreAgentCheckHealth:
    """Tests for check_health method"""

    @pytest.mark.asyncio
    async def test_full_check(self):
        """Test full health check"""
        agent = HealthScoreAgent()
        result = await agent.check_health(scope="full")

        assert isinstance(result, HealthScoreOutput)
        assert 0 <= result.overall_health_score <= 100
        assert result.health_grade in ["A", "B", "C", "D", "F"]
        assert result.total_latency_ms >= 0
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_quick_check(self):
        """Test quick health check"""
        agent = HealthScoreAgent()
        result = await agent.quick_check()

        assert isinstance(result, HealthScoreOutput)
        # Quick check should still return valid output
        assert 0 <= result.overall_health_score <= 100

    @pytest.mark.asyncio
    async def test_full_check_method(self):
        """Test full_check convenience method"""
        agent = HealthScoreAgent()
        result = await agent.full_check()

        assert isinstance(result, HealthScoreOutput)

    @pytest.mark.asyncio
    async def test_with_all_healthy_clients(
        self,
        mock_health_client,
        mock_metrics_store,
        mock_pipeline_store,
        mock_agent_registry,
    ):
        """Test with all healthy mock clients"""
        agent = HealthScoreAgent(
            health_client=mock_health_client,
            metrics_store=mock_metrics_store,
            pipeline_store=mock_pipeline_store,
            agent_registry=mock_agent_registry,
        )
        result = await agent.check_health(scope="full")

        # All healthy should give grade A
        assert result.health_grade == "A"
        assert result.overall_health_score == 100.0
        assert len(result.critical_issues) == 0

    @pytest.mark.asyncio
    async def test_with_degraded_clients(
        self,
        unhealthy_health_client,
        degraded_metrics_store,
        stale_pipeline_store,
        unavailable_agent_registry,
    ):
        """Test with degraded mock clients"""
        agent = HealthScoreAgent(
            health_client=unhealthy_health_client,
            metrics_store=degraded_metrics_store,
            pipeline_store=stale_pipeline_store,
            agent_registry=unavailable_agent_registry,
        )
        result = await agent.check_health(scope="full")

        # Should have lower score with degraded services
        assert result.overall_health_score < 100.0
        assert result.health_grade in ["C", "D", "F"]
        assert len(result.critical_issues) > 0


class TestHealthScoreFailClosed:
    """F1 anti-fabrication guard: absence of a real backend must NOT be
    encoded as a healthy 100/grade-A result tagged ``measured``.

    Reality: there are no real ``metrics_store``/``pipeline_store``/
    ``agent_registry`` implementations, and a default ``HealthScoreAgent()``
    wires no health_client either. The honest result is therefore
    ``unknown`` (no dimensions measured), never a fabricated perfect score.
    """

    @pytest.mark.asyncio
    async def test_no_stores_does_not_fabricate_grade_a(self):
        """With NO stores wired, the agent must not report a perfect, measured
        health (the live-dashboard 100/A/measured bug)."""
        agent = HealthScoreAgent()
        result = await agent.check_health(scope="full")

        # The core anti-fabrication invariants:
        assert result.data_provenance != "measured"
        assert result.health_grade != "A"
        assert result.overall_health_score != 100.0

    @pytest.mark.asyncio
    async def test_no_stores_reports_unknown_provenance(self):
        """With zero measurable dimensions the provenance is ``unknown`` and the
        result surfaces a critical issue explaining why."""
        agent = HealthScoreAgent()
        result = await agent.check_health(scope="full")

        assert result.data_provenance == "unknown"
        assert result.overall_health_score == 0.0
        assert any("could be measured" in issue.lower() for issue in result.critical_issues)

    @pytest.mark.asyncio
    async def test_component_only_reports_partial(self, mock_health_client):
        """A real health_client (component dimension) with no other stores must
        report ``partial`` provenance computed over only the measured dim."""
        agent = HealthScoreAgent(health_client=mock_health_client)
        result = await agent.check_health(scope="full")

        assert result.data_provenance == "partial"
        # All 5 mock components healthy => component dim 1.0 => overall 100 over
        # the single measured dimension. Grade A is honest HERE because the one
        # measured dimension really is healthy; provenance discloses it's partial.
        assert result.component_health_score == 1.0

    @pytest.mark.asyncio
    async def test_quick_check_measures_components(self, mock_health_client):
        """Codex F1.2: quick_check() with a real health_client MUST measure
        components (provenance not 'unknown', component score present), not skip
        them and collapse to 0/F/unknown."""
        agent = HealthScoreAgent(health_client=mock_health_client)
        result = await agent.quick_check()

        assert result.data_provenance != "unknown"
        assert result.component_health_score == 1.0
        assert result.overall_health_score > 0.0

    @pytest.mark.asyncio
    async def test_unmeasured_per_dim_scores_are_none_not_one(self, mock_health_client):
        """Codex F1.1: with only the component backend wired, the OUTPUT must NOT
        expose a fabricated 1.0 for the unmeasured model/pipeline/agent dims.
        Unmeasured == None."""
        agent = HealthScoreAgent(health_client=mock_health_client)
        result = await agent.check_health(scope="full")

        assert result.component_health_score == 1.0  # measured
        assert result.model_health_score is None  # unmeasured, NOT 1.0
        assert result.pipeline_health_score is None
        assert result.agent_health_score is None

    @pytest.mark.asyncio
    async def test_recommendations_do_not_claim_healthy_when_unmeasured(self, mock_health_client):
        """Codex F1.1: an unmeasured dimension must yield a 'wire a real backend'
        recommendation and must NOT produce the 'system is healthy' message."""
        agent = HealthScoreAgent(health_client=mock_health_client)
        result = await agent.check_health(scope="full")

        recs = " ".join(result.recommendations).lower()
        assert "system is healthy" not in recs
        assert "unmeasured" in recs

    @pytest.mark.asyncio
    async def test_handoff_does_not_crash_on_unmeasured_model_score(self, mock_health_client):
        """Codex F1.1: get_handoff reads model_health_score < 0.8; a None
        (unmeasured) dim must not crash and must not read as healthy."""
        agent = HealthScoreAgent(health_client=mock_health_client)
        result = await agent.check_health(scope="full")
        handoff = agent.get_handoff(result)  # must not raise

        assert handoff["component_scores"]["model"] is None


class TestHealthScoreAgentHandoff:
    """Tests for handoff generation"""

    @pytest.mark.asyncio
    async def test_handoff_format(self):
        """Test handoff output format"""
        agent = HealthScoreAgent()
        result = await agent.check_health(scope="full")
        handoff = agent.get_handoff(result)

        assert handoff["agent"] == "health_score"
        assert handoff["analysis_type"] == "system_health"
        assert "key_findings" in handoff
        assert "component_scores" in handoff
        assert "issues" in handoff
        assert "warnings" in handoff
        assert "recommendations" in handoff
        assert "requires_further_analysis" in handoff
        assert "suggested_next_agent" in handoff

    @pytest.mark.asyncio
    async def test_handoff_recommendations_healthy(self):
        """Test recommendations for healthy system"""
        agent = HealthScoreAgent()
        output = HealthScoreOutput(
            overall_health_score=95.0,
            health_grade="A",
            component_health_score=0.95,
            model_health_score=0.95,
            pipeline_health_score=0.95,
            agent_health_score=0.95,
            critical_issues=[],
            warnings=[],
            health_summary="System healthy",
            total_latency_ms=100,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        handoff = agent.get_handoff(output)

        assert "Continue monitoring" in handoff["recommendations"][0]
        assert handoff["requires_further_analysis"] is False

    @pytest.mark.asyncio
    async def test_handoff_recommendations_unhealthy(self):
        """Test recommendations for unhealthy system"""
        agent = HealthScoreAgent()
        output = HealthScoreOutput(
            overall_health_score=50.0,
            health_grade="F",
            component_health_score=0.5,
            model_health_score=0.5,
            pipeline_health_score=0.5,
            agent_health_score=0.5,
            critical_issues=["Component down"],
            warnings=[],
            health_summary="System critical",
            total_latency_ms=100,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        handoff = agent.get_handoff(output)

        # Should have specific recommendations
        assert len(handoff["recommendations"]) >= 4
        assert handoff["requires_further_analysis"] is True


class TestGraphBuilding:
    """Tests for graph building functions"""

    def test_build_full_graph(self):
        """Test building full health score graph"""
        graph = build_health_score_graph()
        assert graph is not None

    def test_build_quick_graph(self):
        """Test building quick check graph"""
        graph = build_quick_check_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_run_full_graph(self):
        """Test running full graph"""
        graph = build_health_score_graph()
        initial_state = {
            "query": "",
            "check_scope": "full",
            "component_statuses": None,
            "component_health_score": None,
            "model_metrics": None,
            "model_health_score": None,
            "pipeline_statuses": None,
            "pipeline_health_score": None,
            "agent_statuses": None,
            "agent_health_score": None,
            "overall_health_score": None,
            "health_grade": None,
            "critical_issues": None,
            "warnings": None,
            "health_summary": None,
            "total_latency_ms": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": [],
            "status": "pending",
        }
        result = await graph.ainvoke(initial_state)

        assert result["status"] == "completed"
        assert result["overall_health_score"] is not None
        assert result["health_grade"] is not None


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    @pytest.mark.asyncio
    async def test_check_system_health_quick(self):
        """Test quick system health check"""
        result = await check_system_health(scope="quick")
        assert isinstance(result, HealthScoreOutput)

    @pytest.mark.asyncio
    async def test_check_system_health_full(self):
        """Test full system health check"""
        result = await check_system_health(scope="full")
        assert isinstance(result, HealthScoreOutput)


class TestInputOutputContracts:
    """Tests for input/output contract models"""

    def test_input_contract_defaults(self):
        """Test input contract default values"""
        input_model = HealthScoreInput()
        assert input_model.query == ""
        assert input_model.check_scope == "full"

    def test_input_contract_custom_scope(self):
        """Test input contract with custom scope"""
        input_model = HealthScoreInput(check_scope="quick")
        assert input_model.check_scope == "quick"

    def test_output_contract_fields(self):
        """Test output contract has all required fields"""
        output = HealthScoreOutput(
            overall_health_score=85.0,
            health_grade="B",
            component_health_score=0.9,
            model_health_score=0.8,
            pipeline_health_score=0.85,
            agent_health_score=0.9,
            critical_issues=[],
            warnings=["Warning 1"],
            health_summary="System healthy",
            total_latency_ms=100,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert output.overall_health_score == 85.0
        assert output.health_grade == "B"
        assert len(output.warnings) == 1


class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully"""

        # Create agent with clients that will fail
        class FailingClient:
            async def check(self, endpoint):
                raise RuntimeError("Connection failed")

        agent = HealthScoreAgent(health_client=FailingClient())
        result = await agent.check_health(scope="full")

        # Should still return a valid output
        assert isinstance(result, HealthScoreOutput)
        # Will have lower score due to unknown component statuses
        assert result.overall_health_score >= 0

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test with empty query"""
        agent = HealthScoreAgent()
        result = await agent.check_health(scope="full", query="")
        assert isinstance(result, HealthScoreOutput)

    @pytest.mark.asyncio
    async def test_with_query_text(self):
        """Test with query text"""
        agent = HealthScoreAgent()
        result = await agent.check_health(
            scope="full",
            query="What is the system health?",
        )
        assert isinstance(result, HealthScoreOutput)


class TestPerformance:
    """Tests for performance requirements"""

    @pytest.mark.asyncio
    async def test_quick_check_latency(self):
        """Test that quick check is fast"""
        agent = HealthScoreAgent()
        result = await agent.quick_check()

        # Quick check should be under 1000ms with no external calls
        assert result.total_latency_ms < 1000

    @pytest.mark.asyncio
    async def test_full_check_latency(self):
        """Test that full check completes in reasonable time"""
        agent = HealthScoreAgent()
        result = await agent.full_check()

        # Full check should be under 5000ms with no external calls
        assert result.total_latency_ms < 5000
