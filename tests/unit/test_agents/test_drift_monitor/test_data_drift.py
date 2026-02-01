"""Tests for Data Drift Detection Node.

Tests PSI calculation, KS test, severity determination, and edge cases.
"""

import numpy as np
import pytest

from src.agents.drift_monitor.nodes.data_drift import DataDriftNode
from src.agents.drift_monitor.state import DriftMonitorState


class TestDataDriftNode:
    """Test DataDriftNode."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state with defaults."""
        state: DriftMonitorState = {
            "query": "test",
            "features_to_monitor": ["feature1", "feature2"],
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
        node = DataDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "data_drift_results" in result
        assert isinstance(result["data_drift_results"], list)
        assert result["status"] == "detecting"

    @pytest.mark.asyncio
    async def test_data_drift_results_structure(self):
        """Test structure of drift results."""
        node = DataDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        if result["data_drift_results"]:
            drift_result = result["data_drift_results"][0]
            assert "feature" in drift_result
            assert "drift_type" in drift_result
            assert "test_statistic" in drift_result
            assert "p_value" in drift_result
            assert "drift_detected" in drift_result
            assert "severity" in drift_result
            assert "baseline_period" in drift_result
            assert "current_period" in drift_result

    @pytest.mark.asyncio
    async def test_drift_type_is_data(self):
        """Test drift type is 'data'."""
        node = DataDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for drift_result in result["data_drift_results"]:
            assert drift_result["drift_type"] == "data"

    @pytest.mark.asyncio
    async def test_features_checked_count(self):
        """Test features_checked count."""
        node = DataDriftNode()
        state = self._create_test_state(features_to_monitor=["f1", "f2", "f3"])

        result = await node.execute(state)

        assert result["features_checked"] == 3

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement."""
        node = DataDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "total_latency_ms" in result
        assert result["total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_timestamps_set(self):
        """Test timestamps are set."""
        node = DataDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "baseline_timestamp" in result
        assert "timestamp" in result
        assert len(result["baseline_timestamp"]) > 0
        assert len(result["timestamp"]) > 0

    @pytest.mark.asyncio
    async def test_check_data_drift_disabled(self):
        """Test data drift check can be disabled."""
        node = DataDriftNode()
        state = self._create_test_state(check_data_drift=False)

        result = await node.execute(state)

        assert result["data_drift_results"] == []
        assert any("skipped" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = DataDriftNode()
        state = self._create_test_state(status="failed")

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result["data_drift_results"] == []


class TestPSICalculation:
    """Test PSI calculation."""

    def test_psi_identical_distributions(self):
        """Test PSI for identical distributions."""
        node = DataDriftNode()
        np.random.seed(42)

        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        psi = node._calculate_psi(baseline, current)

        # PSI should be low for similar distributions
        assert psi < 0.1

    def test_psi_shifted_distributions(self):
        """Test PSI for shifted distributions."""
        node = DataDriftNode()
        np.random.seed(42)

        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)  # Shifted mean

        psi = node._calculate_psi(baseline, current)

        # PSI should be higher for shifted distributions
        assert psi > 0.1

    def test_psi_different_variance(self):
        """Test PSI for different variance."""
        node = DataDriftNode()
        np.random.seed(42)

        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 2, 1000)  # Different variance

        psi = node._calculate_psi(baseline, current)

        # PSI should detect variance changes
        assert psi > 0.05

    def test_psi_range(self):
        """Test PSI is always non-negative."""
        node = DataDriftNode()
        np.random.seed(42)

        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        psi = node._calculate_psi(baseline, current)

        assert psi >= 0


class TestSeverityDetermination:
    """Test severity determination."""

    def test_critical_severity_high_psi(self):
        """Test critical severity for high PSI."""
        node = DataDriftNode()

        psi = 0.3  # Above 0.25 threshold
        p_value = 0.1
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_critical_severity_low_pvalue(self):
        """Test critical severity for very low p-value."""
        node = DataDriftNode()

        psi = 0.05
        p_value = 0.001  # < significance / 10
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_high_severity(self):
        """Test high severity."""
        node = DataDriftNode()

        psi = 0.22  # Between 0.2 and 0.25
        p_value = 0.03  # Significant
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "high"
        assert drift_detected is True

    def test_medium_severity(self):
        """Test medium severity."""
        node = DataDriftNode()

        psi = 0.15  # Between 0.1 and 0.2
        p_value = 0.03  # Significant
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "medium"
        assert drift_detected is True

    def test_low_severity(self):
        """Test low severity."""
        node = DataDriftNode()

        psi = 0.07  # Between 0.05 and 0.1
        p_value = 0.1  # Not significant
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "low"
        assert drift_detected is True

    def test_no_severity(self):
        """Test no drift detected."""
        node = DataDriftNode()

        psi = 0.03  # Below 0.05
        p_value = 0.5  # Not significant
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "none"
        assert drift_detected is False


class TestEdgeCases:
    """Test edge cases."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state."""
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
    async def test_empty_features_list(self):
        """Test with empty features list."""
        node = DataDriftNode()
        state = self._create_test_state(features_to_monitor=[])

        result = await node.execute(state)

        assert result["data_drift_results"] == []

    @pytest.mark.asyncio
    async def test_single_feature(self):
        """Test with single feature."""
        node = DataDriftNode()
        state = self._create_test_state(features_to_monitor=["feature1"])

        result = await node.execute(state)

        assert len(result["data_drift_results"]) <= 1

    @pytest.mark.asyncio
    async def test_many_features(self):
        """Test with many features."""
        node = DataDriftNode()
        features = [f"feature_{i}" for i in range(50)]
        state = self._create_test_state(features_to_monitor=features)

        result = await node.execute(state)

        # Should handle many features
        assert result["features_checked"] == 50

    @pytest.mark.asyncio
    async def test_brand_filter(self):
        """Test with brand filter."""
        node = DataDriftNode()
        state = self._create_test_state(brand="Remibrutinib")

        result = await node.execute(state)

        # Should execute without error
        assert "data_drift_results" in result


class TestTimeWindows:
    """Test different time windows."""

    def _create_test_state(self, time_window: str) -> DriftMonitorState:
        """Create test state with time window."""
        return {
            "query": "test",
            "features_to_monitor": ["feature1"],
            "time_window": time_window,
            "significance_level": 0.05,
            "psi_threshold": 0.1,
            "check_data_drift": True,
            "check_model_drift": True,
            "check_concept_drift": True,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

    @pytest.mark.asyncio
    async def test_7_day_window(self):
        """Test 7 day window."""
        node = DataDriftNode()
        state = self._create_test_state(time_window="7d")

        result = await node.execute(state)

        assert "baseline_timestamp" in result

    @pytest.mark.asyncio
    async def test_30_day_window(self):
        """Test 30 day window."""
        node = DataDriftNode()
        state = self._create_test_state(time_window="30d")

        result = await node.execute(state)

        assert "baseline_timestamp" in result
