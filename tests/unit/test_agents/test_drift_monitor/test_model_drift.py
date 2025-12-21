"""Tests for Model Drift Detection Node.

Tests prediction score drift, prediction class drift, and edge cases.
"""

import numpy as np
import pytest

from src.agents.drift_monitor.nodes.model_drift import ModelDriftNode
from src.agents.drift_monitor.state import DriftMonitorState


class TestModelDriftNode:
    """Test ModelDriftNode."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state with defaults."""
        state: DriftMonitorState = {
            "query": "test",
            "model_id": "model_v1",
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
    async def test_execute_basic(self):
        """Test basic execution."""
        node = ModelDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "model_drift_results" in result
        assert isinstance(result["model_drift_results"], list)

    @pytest.mark.asyncio
    async def test_model_drift_results_structure(self):
        """Test structure of drift results."""
        node = ModelDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        if result["model_drift_results"]:
            drift_result = result["model_drift_results"][0]
            assert "feature" in drift_result
            assert "drift_type" in drift_result
            assert "test_statistic" in drift_result
            assert "p_value" in drift_result
            assert "drift_detected" in drift_result
            assert "severity" in drift_result

    @pytest.mark.asyncio
    async def test_drift_type_is_model(self):
        """Test drift type is 'model'."""
        node = ModelDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for drift_result in result["model_drift_results"]:
            assert drift_result["drift_type"] == "model"

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement."""
        node = ModelDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "detection_latency_ms" in result
        assert result["detection_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_model_drift_disabled(self):
        """Test model drift check can be disabled."""
        node = ModelDriftNode()
        state = self._create_test_state(check_model_drift=False)

        result = await node.execute(state)

        assert result["model_drift_results"] == []
        assert any("skipped" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_no_model_id(self):
        """Test without model_id."""
        node = ModelDriftNode()
        state = self._create_test_state(model_id=None)

        result = await node.execute(state)

        assert result["model_drift_results"] == []
        assert any("no model_id" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = ModelDriftNode()
        state = self._create_test_state(status="failed")

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result["model_drift_results"] == []


class TestPredictionScoreDrift:
    """Test prediction score drift detection."""

    def test_detect_score_drift_identical(self):
        """Test score drift for identical distributions."""
        node = ModelDriftNode()
        np.random.seed(42)

        baseline = np.random.beta(2, 5, 1000)
        current = np.random.beta(2, 5, 1000)

        result = node._detect_score_drift(baseline, current, 0.05, 0.1)

        # Should detect minimal or no drift
        assert result is not None
        assert result["feature"] == "prediction_scores"
        assert result["drift_type"] == "model"

    def test_detect_score_drift_shifted(self):
        """Test score drift for shifted distributions."""
        node = ModelDriftNode()
        np.random.seed(42)

        baseline = np.random.beta(2, 5, 1000)
        current = np.random.beta(2, 2, 1000)  # Higher predictions

        result = node._detect_score_drift(baseline, current, 0.05, 0.1)

        # Should detect drift
        assert result is not None
        assert result["drift_detected"] is True

    def test_detect_score_drift_insufficient_data(self):
        """Test score drift with insufficient data."""
        node = ModelDriftNode()

        baseline = np.array([0.1, 0.2])  # Only 2 samples
        current = np.array([0.3, 0.4])

        result = node._detect_score_drift(baseline, current, 0.05, 0.1)

        # Should return None due to insufficient samples
        assert result is None


class TestPredictionClassDrift:
    """Test prediction class drift detection."""

    def test_detect_class_drift_identical(self):
        """Test class drift for identical distributions."""
        node = ModelDriftNode()
        np.random.seed(42)

        baseline = np.random.binomial(1, 0.5, 1000)
        current = np.random.binomial(1, 0.5, 1000)

        result = node._detect_class_drift(baseline, current, 0.05)

        # Should detect minimal or no drift
        assert result is not None
        assert result["feature"] == "prediction_classes"
        assert result["drift_type"] == "model"

    def test_detect_class_drift_shifted(self):
        """Test class drift for shifted distributions."""
        node = ModelDriftNode()
        np.random.seed(42)

        baseline = np.random.binomial(1, 0.3, 1000)  # 30% positive
        current = np.random.binomial(1, 0.7, 1000)  # 70% positive

        result = node._detect_class_drift(baseline, current, 0.05)

        # Should detect drift
        assert result is not None
        assert result["drift_detected"] is True

    def test_detect_class_drift_insufficient_data(self):
        """Test class drift with insufficient data."""
        node = ModelDriftNode()

        baseline = np.array([0, 1])  # Only 2 samples
        current = np.array([1, 0])

        result = node._detect_class_drift(baseline, current, 0.05)

        # Should return None due to insufficient samples
        assert result is None

    def test_detect_class_drift_multiclass(self):
        """Test class drift with multiple classes."""
        node = ModelDriftNode()
        np.random.seed(42)

        # 3 classes
        baseline = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
        current = np.random.choice([0, 1, 2], size=1000, p=[0.2, 0.3, 0.5])

        result = node._detect_class_drift(baseline, current, 0.05)

        # Should detect drift in class distribution
        assert result is not None


class TestSeverityDetermination:
    """Test severity determination for model drift."""

    def test_critical_severity(self):
        """Test critical severity."""
        node = ModelDriftNode()

        psi = 0.3
        p_value = 0.001
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_high_severity(self):
        """Test high severity."""
        node = ModelDriftNode()

        psi = 0.22
        p_value = 0.03
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "high"
        assert drift_detected is True

    def test_medium_severity(self):
        """Test medium severity."""
        node = ModelDriftNode()

        psi = 0.15
        p_value = 0.03
        significance = 0.05
        psi_threshold = 0.1

        severity, drift_detected = node._determine_severity(
            psi, p_value, significance, psi_threshold
        )

        assert severity == "medium"
        assert drift_detected is True


class TestEdgeCases:
    """Test edge cases."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state."""
        state: DriftMonitorState = {
            "query": "test",
            "model_id": "model_v1",
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
    async def test_brand_filter(self):
        """Test with brand filter."""
        node = ModelDriftNode()
        state = self._create_test_state(brand="Remibrutinib")

        result = await node.execute(state)

        # Should execute without error
        assert "model_drift_results" in result

    @pytest.mark.asyncio
    async def test_different_time_windows(self):
        """Test with different time windows."""
        node = ModelDriftNode()

        for window in ["7d", "14d", "30d"]:
            state = self._create_test_state(time_window=window)
            result = await node.execute(state)

            assert "model_drift_results" in result
