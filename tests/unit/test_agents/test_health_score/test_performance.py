"""
Performance tests for Health Score Agent - SLA Compliance.

Validates documented SLAs:
- Quick check: <1s
- Full check: <5s
- Models/Pipelines/Agents only: <2s

These tests use mocked dependencies to measure pure agent latency
without external service calls.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mark all tests in this module as slow and group for worker isolation
pytestmark = [
    pytest.mark.slow,
    pytest.mark.xdist_group(name="performance_tests"),
]


def _create_mock_state(scope: str = "full") -> Dict[str, Any]:
    """Create mock HealthScoreState for testing."""
    return {
        "run_id": "test-perf-run",
        "check_scope": scope,
        "component_statuses": [],
        "model_metrics": [],
        "pipeline_statuses": [],
        "agent_statuses": [],
        "component_health_score": None,
        "model_health_score": None,
        "pipeline_health_score": None,
        "agent_health_score": None,
        "overall_health_score": None,
        "health_grade": None,
        "critical_issues": [],
        "warnings": [],
        "health_summary": None,
        "errors": [],
        "total_latency_ms": None,
    }


class MockHealthClient:
    """Mock health client for testing."""

    def __init__(self, latency_ms: int = 10, healthy: bool = True):
        self.latency_ms = latency_ms
        self.healthy = healthy

    async def check(self, endpoint: str) -> Dict[str, Any]:
        """Simulate health check with configurable latency."""
        await asyncio.sleep(self.latency_ms / 1000)
        return {"ok": self.healthy, "degraded": False}


class MockMetricsStore:
    """Mock metrics store for testing."""

    def __init__(self, models: List[str] = None):
        self.models = models or ["model_a", "model_b"]

    async def get_active_models(self) -> List[str]:
        return self.models

    async def get_model_metrics(self, model_id: str, time_window: str) -> Dict[str, Any]:
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
            "auc_roc": 0.92,
            "latency_p50": 25,
            "latency_p99": 100,
            "prediction_count": 10000,
            "error_rate": 0.001,
        }


class MockPipelineStore:
    """Mock pipeline store for testing."""

    def __init__(self, pipelines: List[str] = None):
        self.pipelines = pipelines or ["pipeline_a", "pipeline_b"]

    async def get_all_pipelines(self) -> List[str]:
        return self.pipelines

    async def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        from datetime import datetime, timezone

        return {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "last_success": datetime.now(timezone.utc).isoformat(),
            "rows_processed": 50000,
            "failed": False,
        }


class MockAgentRegistry:
    """Mock agent registry for testing."""

    def __init__(self, agents: List[Dict[str, Any]] = None):
        self.agents = agents or [
            {"name": "orchestrator", "tier": 1},
            {"name": "causal_impact", "tier": 2},
        ]

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        return self.agents

    async def get_agent_metrics(self, agent_name: str) -> Dict[str, Any]:
        return {
            "available": True,
            "avg_latency_ms": 500,
            "success_rate": 0.95,
            "last_invocation": "2024-01-01T00:00:00Z",
        }


@pytest.mark.xdist_group(name="performance_tests")
class TestHealthScoreSLA:
    """Validate Health Score meets <5s full, <1s quick SLAs."""

    @pytest.mark.asyncio
    async def test_quick_check_under_1s(self):
        """Test quick scope completes under 1 second."""
        from src.agents.health_score.nodes.component_health import ComponentHealthNode

        # Quick scope skips component checks entirely
        node = ComponentHealthNode()
        state = _create_mock_state(scope="quick")

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Quick check took {elapsed:.3f}s, exceeds 1s SLA"
        assert result.get("component_health_score") == 1.0

    @pytest.mark.asyncio
    async def test_full_check_under_5s(self):
        """Test full scope completes under 5 seconds."""
        from src.agents.health_score.nodes.score_composer import ScoreComposerNode

        node = ScoreComposerNode()

        # Pre-populated state with all health checks complete
        state = _create_mock_state(scope="full")
        state["component_health_score"] = 0.9
        state["model_health_score"] = 0.85
        state["pipeline_health_score"] = 0.8
        state["agent_health_score"] = 0.95

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Full check took {elapsed:.3f}s, exceeds 5s SLA"
        assert result.get("overall_health_score") is not None

    @pytest.mark.asyncio
    async def test_models_only_under_2s(self):
        """Test models-only scope completes under 2 seconds."""
        from src.agents.health_score.nodes.model_health import ModelHealthNode

        # Use mock metrics store
        mock_store = MockMetricsStore(models=["model_a", "model_b", "model_c"])
        node = ModelHealthNode(metrics_store=mock_store)
        state = _create_mock_state(scope="models")

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Models check took {elapsed:.3f}s, exceeds 2s SLA"
        assert result.get("model_health_score") is not None

    @pytest.mark.asyncio
    async def test_pipelines_only_under_2s(self):
        """Test pipelines-only scope completes under 2 seconds."""
        from src.agents.health_score.nodes.pipeline_health import PipelineHealthNode

        mock_store = MockPipelineStore(pipelines=["data_ingestion", "feature_pipeline"])
        node = PipelineHealthNode(pipeline_store=mock_store)
        state = _create_mock_state(scope="pipelines")

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Pipelines check took {elapsed:.3f}s, exceeds 2s SLA"
        assert result.get("pipeline_health_score") is not None

    @pytest.mark.asyncio
    async def test_agents_only_under_2s(self):
        """Test agents-only scope completes under 2 seconds."""
        from src.agents.health_score.nodes.agent_health import AgentHealthNode

        mock_registry = MockAgentRegistry(
            agents=[
                {"name": "orchestrator", "tier": 1},
                {"name": "causal_impact", "tier": 2},
            ]
        )
        node = AgentHealthNode(agent_registry=mock_registry)
        state = _create_mock_state(scope="agents")

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Agents check took {elapsed:.3f}s, exceeds 2s SLA"
        assert result.get("agent_health_score") is not None


@pytest.mark.xdist_group(name="performance_tests")
class TestHealthScoreParallelism:
    """Test parallel check performance."""

    @pytest.mark.asyncio
    async def test_parallel_checks_faster_than_sequential(self):
        """Test parallel execution is faster than sequential."""
        from src.agents.health_score.nodes.component_health import ComponentHealthNode

        # Create health client with 100ms simulated latency
        mock_client = MockHealthClient(latency_ms=100)
        node = ComponentHealthNode(health_client=mock_client)
        state = _create_mock_state(scope="full")

        start = time.perf_counter()
        await node.execute(state)
        elapsed = time.perf_counter() - start

        # 5 components x 100ms = 500ms if sequential
        # Should be ~100ms if parallel (allow overhead up to 300ms)
        assert elapsed < 0.3, f"Parallel checks took {elapsed:.3f}s, not parallelized"

    @pytest.mark.asyncio
    async def test_parallel_check_isolation(self):
        """Test parallel checks don't interfere with each other."""
        from src.agents.health_score.nodes.component_health import ComponentHealthNode

        node = ComponentHealthNode(health_client=MockHealthClient(latency_ms=5))

        # Run multiple concurrent state checks
        states = [_create_mock_state(scope="full") for _ in range(5)]

        # Run all concurrently
        results = await asyncio.gather(*[node.execute(s) for s in states])

        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert "component_statuses" in result
            assert result.get("component_health_score") is not None

    @pytest.mark.asyncio
    async def test_degraded_component_latency(self):
        """Test degraded components don't cause excessive latency."""
        from src.agents.health_score.nodes.component_health import ComponentHealthNode

        # Create slow health client (simulating timeout behavior)
        class SlowHealthClient:
            async def check(self, endpoint: str) -> Dict[str, Any]:
                await asyncio.sleep(0.5)  # 500ms simulated timeout
                return {"ok": False, "degraded": True}

        node = ComponentHealthNode(health_client=SlowHealthClient())
        state = _create_mock_state(scope="full")

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        # Should still complete reasonably fast due to parallelism
        # 5 components x 500ms = 2.5s if sequential, ~500ms if parallel
        assert elapsed < 1.0, f"Degraded check took {elapsed:.3f}s, should handle timeouts"
