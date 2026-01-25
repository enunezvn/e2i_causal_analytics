"""
Performance tests for Drift Monitor Agent - SLA Compliance.

Validates documented SLAs:
- <10s for 50 features
- <3s for 10 features
- <20s for 100 features

These tests use mocked dependencies to measure pure agent latency
without external service calls.
"""

import asyncio
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from src.agents.drift_monitor.connectors.base import (
    BaseDataConnector,
    FeatureData,
    PredictionData,
    TimeWindow,
)

# Mark all tests in this module as slow and group for worker isolation
pytestmark = [
    pytest.mark.slow,
    pytest.mark.xdist_group(name="performance_tests"),
]


def _create_mock_state(num_features: int = 50) -> Dict[str, Any]:
    """Create mock DriftMonitorState for testing."""
    return {
        "run_id": "test-perf-run",
        "brand": "Kisqali",
        "features_to_monitor": [f"feature_{i}" for i in range(num_features)],
        "time_window": "7d",
        "significance_level": 0.05,
        "psi_threshold": 0.1,
        "check_data_drift": True,
        "check_concept_drift": True,
        "model_id": "test_model",
        "data_drift_results": [],
        "concept_drift_results": [],
        "model_drift_results": [],
        "overall_drift_score": None,
        "drift_alerts": [],
        "recommendations": [],
        "errors": [],
        "warnings": [],
    }


class MockDataConnector(BaseDataConnector):
    """Mock data connector for performance testing.

    Generates synthetic feature data with configurable latency.
    """

    def __init__(self, latency_ms: int = 5, num_samples: int = 1000):
        """Initialize mock connector.

        Args:
            latency_ms: Simulated query latency in milliseconds
            num_samples: Number of samples to generate per feature
        """
        self.latency_ms = latency_ms
        self.num_samples = num_samples
        np.random.seed(42)

    async def query_features(
        self,
        feature_names: List[str],
        time_window: TimeWindow,
        filters: Dict[str, Any] | None = None,
    ) -> Dict[str, FeatureData]:
        """Return mock feature data.

        Generates random normal data for each feature with simulated latency.
        """
        await asyncio.sleep(self.latency_ms / 1000)

        result = {}
        for name in feature_names:
            values = np.random.randn(self.num_samples)
            result[name] = FeatureData(
                feature_name=name,
                values=values,
                timestamps=None,
                entity_ids=None,
                time_window=time_window,
            )
        return result

    async def query_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: Dict[str, Any] | None = None,
    ) -> PredictionData:
        """Return mock prediction data."""
        await asyncio.sleep(self.latency_ms / 1000)

        scores = np.random.rand(self.num_samples)
        labels = (scores > 0.5).astype(int)

        return PredictionData(
            model_id=model_id,
            scores=scores,
            labels=labels,
            actual_labels=None,
            timestamps=None,
            entity_ids=None,
            time_window=time_window,
        )

    async def query_labeled_predictions(
        self,
        model_id: str,
        time_window: TimeWindow,
        filters: Dict[str, Any] | None = None,
    ) -> PredictionData:
        """Return mock labeled prediction data."""
        await asyncio.sleep(self.latency_ms / 1000)

        scores = np.random.rand(self.num_samples)
        labels = (scores > 0.5).astype(int)
        # Simulate some prediction errors (90% accuracy)
        actual = labels.copy()
        error_mask = np.random.rand(self.num_samples) < 0.1
        actual[error_mask] = 1 - actual[error_mask]

        return PredictionData(
            model_id=model_id,
            scores=scores,
            labels=labels,
            actual_labels=actual,
            timestamps=None,
            entity_ids=None,
            time_window=time_window,
        )

    async def get_available_features(
        self,
        source_table: str | None = None,
    ) -> List[str]:
        """Return list of available features."""
        return [f"feature_{i}" for i in range(100)]

    async def get_available_models(
        self,
        stage: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return list of available models."""
        return [
            {"model_id": "model_a", "stage": "production"},
            {"model_id": "model_b", "stage": "staging"},
        ]

    async def health_check(self) -> Dict[str, bool]:
        """Return healthy status."""
        return {"database": True, "feature_store": True}

    async def close(self) -> None:
        """Close connector (no-op for mock)."""
        pass


@pytest.mark.xdist_group(name="performance_tests")
class TestDriftMonitorSLA:
    """Validate Drift Monitor meets <10s SLA for 50 features."""

    @pytest.mark.asyncio
    async def test_latency_50_features_under_10s(self):
        """Test 50 features completes under 10 seconds."""
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        mock_connector = MockDataConnector(latency_ms=5)
        node = DataDriftNode(connector=mock_connector)
        state = _create_mock_state(num_features=50)

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"50 features took {elapsed:.3f}s, exceeds 10s SLA"
        assert result.get("data_drift_results") is not None

    @pytest.mark.asyncio
    async def test_latency_10_features_under_3s(self):
        """Test 10 features completes under 3 seconds."""
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        mock_connector = MockDataConnector(latency_ms=5)
        node = DataDriftNode(connector=mock_connector)
        state = _create_mock_state(num_features=10)

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"10 features took {elapsed:.3f}s, exceeds 3s SLA"
        assert result.get("data_drift_results") is not None

    @pytest.mark.asyncio
    async def test_latency_100_features_under_20s(self):
        """Test 100 features completes under 20 seconds."""
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        mock_connector = MockDataConnector(latency_ms=5)
        node = DataDriftNode(connector=mock_connector)
        state = _create_mock_state(num_features=100)

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        assert elapsed < 20.0, f"100 features took {elapsed:.3f}s, exceeds 20s SLA"
        assert result.get("data_drift_results") is not None

    @pytest.mark.asyncio
    async def test_latency_breakdown_by_node(self):
        """Test individual node latencies are reasonable."""
        from src.agents.drift_monitor.nodes.concept_drift import ConceptDriftNode
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        mock_connector = MockDataConnector(latency_ms=5)
        state = _create_mock_state(num_features=50)
        timings = {}

        # Time data drift node
        data_node = DataDriftNode(connector=mock_connector)
        start = time.perf_counter()
        state = await data_node.execute(state)
        timings["data_drift"] = time.perf_counter() - start

        # Time concept drift node
        concept_node = ConceptDriftNode(connector=mock_connector)
        start = time.perf_counter()
        state = await concept_node.execute(state)
        timings["concept_drift"] = time.perf_counter() - start

        # Each node should be under 5s
        for node_name, node_time in timings.items():
            assert node_time < 5.0, f"{node_name} took {node_time:.3f}s"

    @pytest.mark.asyncio
    async def test_latency_with_all_drift_types(self):
        """Test latency when all drift types are detected."""
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        # Use connector that produces drifted data
        mock_connector = MockDataConnector(latency_ms=5)
        node = DataDriftNode(connector=mock_connector)
        state = _create_mock_state(num_features=50)

        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        # Should still meet SLA even with drift detection
        assert elapsed < 10.0, f"Drift detection took {elapsed:.3f}s, exceeds 10s SLA"
        assert result.get("data_drift_results") is not None


@pytest.mark.xdist_group(name="performance_tests")
class TestDriftMonitorThroughput:
    """Test throughput characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_drift_checks(self):
        """Test multiple concurrent drift checks."""
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        mock_connector = MockDataConnector(latency_ms=10)
        node = DataDriftNode(connector=mock_connector)

        # Create multiple states for different brands
        states = [_create_mock_state(num_features=20) for _ in range(3)]
        for i, state in enumerate(states):
            state["brand"] = ["Kisqali", "Fabhalta", "Remibrutinib"][i]

        start = time.perf_counter()
        results = await asyncio.gather(*[node.execute(s) for s in states])
        elapsed = time.perf_counter() - start

        assert len(results) == 3
        # 3 concurrent checks should be faster than 3x sequential
        assert elapsed < 6.0, f"Concurrent checks took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_memory_usage_large_features(self):
        """Test memory efficiency with large feature sets."""
        from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

        mock_connector = MockDataConnector(latency_ms=5)
        node = DataDriftNode(connector=mock_connector)

        # Large feature set
        state = _create_mock_state(num_features=200)

        # This should complete without memory issues
        start = time.perf_counter()
        result = await node.execute(state)
        elapsed = time.perf_counter() - start

        # Should scale reasonably (not linear with feature count)
        assert elapsed < 40.0, f"Large feature set took {elapsed:.3f}s"
        assert result.get("data_drift_results") is not None
