"""
Test fixtures for Health Score agent tests.

Provides mock clients, sample data, and shared utilities.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock


# ============================================================================
# MOCK HEALTH CLIENT
# ============================================================================


class MockHealthClient:
    """Mock client for component health checks"""

    def __init__(self, responses: Dict[str, Dict[str, Any]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.call_history = []

    async def check(self, endpoint: str) -> Dict[str, Any]:
        """Mock health check"""
        self.call_count += 1
        self.call_history.append(endpoint)

        if endpoint in self.responses:
            return self.responses[endpoint]

        # Default healthy response
        return {"ok": True}

    def set_response(self, endpoint: str, response: Dict[str, Any]) -> None:
        """Set response for an endpoint"""
        self.responses[endpoint] = response

    def set_unhealthy(self, endpoint: str, error: str = "Service unavailable") -> None:
        """Set endpoint as unhealthy"""
        self.responses[endpoint] = {"ok": False, "error": error}


@pytest.fixture
def mock_health_client():
    """Create a mock health client with all healthy components"""
    return MockHealthClient()


@pytest.fixture
def unhealthy_health_client():
    """Create a mock health client with some unhealthy components"""
    client = MockHealthClient()
    client.set_unhealthy("/health/db", "Connection refused")
    client.set_response("/health/cache", {"ok": True})
    client.set_response("/health/vectors", {"ok": False, "degraded": True})
    client.set_response("/health/api", {"ok": True})
    client.set_response("/health/queue", {"ok": True})
    return client


# ============================================================================
# MOCK METRICS STORE
# ============================================================================


class MockMetricsStore:
    """Mock store for model metrics"""

    def __init__(self, models: List[str] = None, metrics: Dict[str, Dict] = None):
        self.models = models or ["model_1", "model_2"]
        self.metrics = metrics or {}

    async def get_active_models(self) -> List[str]:
        """Get list of active model IDs"""
        return self.models

    async def get_model_metrics(
        self, model_id: str, time_window: str
    ) -> Dict[str, Any]:
        """Get metrics for a specific model"""
        if model_id in self.metrics:
            return self.metrics[model_id]

        # Default healthy metrics
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
            "auc_roc": 0.9,
            "latency_p50": 50,
            "latency_p99": 200,
            "prediction_count": 1000,
            "error_rate": 0.01,
        }

    def set_model_metrics(self, model_id: str, metrics: Dict[str, Any]) -> None:
        """Set metrics for a model"""
        self.metrics[model_id] = metrics


@pytest.fixture
def mock_metrics_store():
    """Create a mock metrics store with healthy models"""
    return MockMetricsStore()


@pytest.fixture
def degraded_metrics_store():
    """Create a mock metrics store with degraded models"""
    store = MockMetricsStore(models=["healthy_model", "degraded_model", "unhealthy_model"])
    store.set_model_metrics(
        "healthy_model",
        {
            "accuracy": 0.9,
            "auc_roc": 0.92,
            "latency_p99": 150,
            "prediction_count": 500,
            "error_rate": 0.01,
        },
    )
    store.set_model_metrics(
        "degraded_model",
        {
            "accuracy": 0.65,  # Below 0.7 threshold - triggers degraded
            "auc_roc": 0.85,
            "latency_p99": 200,
            "prediction_count": 500,
            "error_rate": 0.01,
        },
    )
    store.set_model_metrics(
        "unhealthy_model",
        {
            "accuracy": 0.5,  # Way below threshold
            "auc_roc": 0.55,
            "latency_p99": 2000,  # Slow
            "prediction_count": 50,
            "error_rate": 0.1,  # High error rate
        },
    )
    return store


# ============================================================================
# MOCK PIPELINE STORE
# ============================================================================


class MockPipelineStore:
    """Mock store for pipeline status"""

    def __init__(self, pipelines: List[str] = None, statuses: Dict[str, Dict] = None):
        self.pipelines = pipelines or ["etl_daily", "feature_pipeline"]
        self.statuses = statuses or {}

    async def get_all_pipelines(self) -> List[str]:
        """Get list of all pipeline names"""
        return self.pipelines

    async def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get status for a specific pipeline"""
        if pipeline_name in self.statuses:
            return self.statuses[pipeline_name]

        # Default healthy status
        now = datetime.now(timezone.utc)
        return {
            "last_run": now.isoformat(),
            "last_success": now.isoformat(),
            "rows_processed": 10000,
            "failed": False,
        }

    def set_pipeline_status(self, pipeline_name: str, status: Dict[str, Any]) -> None:
        """Set status for a pipeline"""
        self.statuses[pipeline_name] = status


@pytest.fixture
def mock_pipeline_store():
    """Create a mock pipeline store with healthy pipelines"""
    return MockPipelineStore()


@pytest.fixture
def stale_pipeline_store():
    """Create a mock pipeline store with stale/failed pipelines"""
    store = MockPipelineStore(
        pipelines=["healthy_pipeline", "stale_pipeline", "failed_pipeline"]
    )
    now = datetime.now(timezone.utc)

    store.set_pipeline_status(
        "healthy_pipeline",
        {
            "last_run": now.isoformat(),
            "last_success": now.isoformat(),
            "rows_processed": 5000,
            "failed": False,
        },
    )
    store.set_pipeline_status(
        "stale_pipeline",
        {
            "last_run": (now - timedelta(hours=18)).isoformat(),
            "last_success": (now - timedelta(hours=18)).isoformat(),
            "rows_processed": 3000,
            "failed": False,
        },
    )
    store.set_pipeline_status(
        "failed_pipeline",
        {
            "last_run": now.isoformat(),
            "last_success": (now - timedelta(hours=48)).isoformat(),
            "rows_processed": 0,
            "failed": True,
        },
    )
    return store


# ============================================================================
# MOCK AGENT REGISTRY
# ============================================================================


class MockAgentRegistry:
    """Mock registry for agents"""

    def __init__(
        self, agents: List[Dict[str, Any]] = None, metrics: Dict[str, Dict] = None
    ):
        self.agents = agents or [
            {"name": "orchestrator", "tier": 1},
            {"name": "causal_impact", "tier": 2},
        ]
        self.metrics = metrics or {}

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get list of all registered agents"""
        return self.agents

    async def get_agent_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Get metrics for a specific agent"""
        if agent_name in self.metrics:
            return self.metrics[agent_name]

        # Default available agent
        return {
            "available": True,
            "avg_latency_ms": 500,
            "success_rate": 0.95,
            "last_invocation": datetime.now(timezone.utc).isoformat(),
        }

    def set_agent_metrics(self, agent_name: str, metrics: Dict[str, Any]) -> None:
        """Set metrics for an agent"""
        self.metrics[agent_name] = metrics


@pytest.fixture
def mock_agent_registry():
    """Create a mock agent registry with available agents"""
    return MockAgentRegistry()


@pytest.fixture
def unavailable_agent_registry():
    """Create a mock agent registry with some unavailable agents"""
    registry = MockAgentRegistry(
        agents=[
            {"name": "available_agent", "tier": 1},
            {"name": "unavailable_agent", "tier": 2},
            {"name": "low_success_agent", "tier": 3},
        ]
    )
    registry.set_agent_metrics(
        "available_agent",
        {
            "available": True,
            "avg_latency_ms": 200,
            "success_rate": 0.98,
            "last_invocation": datetime.now(timezone.utc).isoformat(),
        },
    )
    registry.set_agent_metrics(
        "unavailable_agent",
        {
            "available": False,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "last_invocation": "",
        },
    )
    registry.set_agent_metrics(
        "low_success_agent",
        {
            "available": True,
            "avg_latency_ms": 1500,
            "success_rate": 0.7,  # Below threshold
            "last_invocation": datetime.now(timezone.utc).isoformat(),
        },
    )
    return registry


# ============================================================================
# STATE FIXTURES
# ============================================================================


@pytest.fixture
def initial_state():
    """Create initial health score state"""
    return {
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
        "check_latency_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "status": "pending",
    }


@pytest.fixture
def quick_check_state():
    """Create state for quick check"""
    return {
        "query": "",
        "check_scope": "quick",
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
        "check_latency_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "status": "pending",
    }
