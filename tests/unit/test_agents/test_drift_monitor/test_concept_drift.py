"""Tests for Concept Drift Detection Node.

Tests performance degradation detection, correlation drift, Fisher Z-test,
severity determination, and edge cases.
"""

import numpy as np
import pytest

from src.agents.drift_monitor.nodes.concept_drift import ConceptDriftNode
from src.agents.drift_monitor.state import DriftMonitorState


class TestConceptDriftNode:
    """Test ConceptDriftNode basic functionality."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state with defaults."""
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
    async def test_execute_basic(self):
        """Test basic execution."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "concept_drift_results" in result
        assert isinstance(result["concept_drift_results"], list)

    @pytest.mark.asyncio
    async def test_returns_empty_results(self):
        """Test returns empty results (placeholder)."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Placeholder implementation returns empty results
        assert result["concept_drift_results"] == []

    @pytest.mark.asyncio
    async def test_adds_warning(self):
        """Test adds warning when model_id is not provided."""
        node = ConceptDriftNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should add warning about skipping concept drift (no model_id)
        assert any("skipped" in w.lower() or "no model_id" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement."""
        node = ConceptDriftNode()
        # Must provide model_id so node runs full path and measures latency
        state = self._create_test_state(model_id="test_model_v1")

        result = await node.execute(state)

        assert "total_latency_ms" in result
        assert result["total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_concept_drift_disabled(self):
        """Test concept drift check can be disabled."""
        node = ConceptDriftNode()
        state = self._create_test_state(check_concept_drift=False)

        result = await node.execute(state)

        assert result["concept_drift_results"] == []
        assert any("skipped" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = ConceptDriftNode()
        state = self._create_test_state(status="failed")

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result["concept_drift_results"] == []


class TestPerformanceDegradation:
    """Test _detect_performance_degradation method."""

    def test_detects_critical_accuracy_drop(self):
        """Test detection of critical accuracy drop (>20%)."""
        node = ConceptDriftNode()
        np.random.seed(42)

        # Baseline: 90% accuracy (90 correct out of 100)
        baseline_predicted = np.array([1] * 90 + [0] * 10)
        baseline_actual = np.array([1] * 100)

        # Current: 65% accuracy (>20% drop)
        current_predicted = np.array([1] * 65 + [0] * 35)
        current_actual = np.array([1] * 100)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        assert result is not None
        assert result["drift_detected"] is True
        assert result["severity"] == "critical"
        assert result["feature"] == "model_accuracy"
        assert result["drift_type"] == "concept"

    def test_detects_high_accuracy_drop(self):
        """Test detection of high accuracy drop (>10%)."""
        node = ConceptDriftNode()
        np.random.seed(42)

        # Baseline: 90% accuracy
        baseline_predicted = np.array([1] * 90 + [0] * 10)
        baseline_actual = np.array([1] * 100)

        # Current: 78% accuracy (~12% drop)
        current_predicted = np.array([1] * 78 + [0] * 22)
        current_actual = np.array([1] * 100)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        assert result is not None
        assert result["drift_detected"] is True
        assert result["severity"] == "high"

    def test_detects_medium_accuracy_drop(self):
        """Test detection of medium accuracy drop (>5%)."""
        node = ConceptDriftNode()
        np.random.seed(42)

        # Use larger samples to ensure statistical significance
        # Baseline: 90% accuracy (450 correct out of 500)
        baseline_predicted = np.array([1] * 450 + [0] * 50)
        baseline_actual = np.array([1] * 500)

        # Current: 83% accuracy (~7% drop, 415 correct out of 500)
        current_predicted = np.array([1] * 415 + [0] * 85)
        current_actual = np.array([1] * 500)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        assert result is not None
        assert result["drift_detected"] is True
        assert result["severity"] == "medium"

    def test_no_detection_stable_accuracy(self):
        """Test no detection when accuracy is stable."""
        node = ConceptDriftNode()
        np.random.seed(42)

        # Both periods: ~90% accuracy
        baseline_predicted = np.array([1] * 90 + [0] * 10)
        baseline_actual = np.array([1] * 100)

        current_predicted = np.array([1] * 89 + [0] * 11)
        current_actual = np.array([1] * 100)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        # Minimal drop should not be significant
        assert result is None or result["drift_detected"] is False

    def test_handles_empty_actual_labels(self):
        """Test handling of empty actual labels."""
        node = ConceptDriftNode()

        baseline_predicted = np.array([1, 0, 1])
        baseline_actual = np.array([])  # Empty
        current_predicted = np.array([1, 0, 1])
        current_actual = np.array([1, 0, 1])

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        assert result is None

    def test_handles_none_actual_labels(self):
        """Test handling of None actual labels."""
        node = ConceptDriftNode()

        baseline_predicted = np.array([1, 0, 1])
        current_predicted = np.array([1, 0, 1])

        result = node._detect_performance_degradation(
            baseline_predicted, None,
            current_predicted, np.array([1, 0, 1]),
            significance=0.05
        )

        assert result is None

    def test_handles_minimum_samples_threshold(self):
        """Test handling of samples below minimum threshold (50)."""
        node = ConceptDriftNode()

        # Less than 50 samples
        baseline_predicted = np.array([1] * 30)
        baseline_actual = np.array([1] * 30)
        current_predicted = np.array([1] * 30)
        current_actual = np.array([1] * 30)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        assert result is None

    def test_handles_perfect_accuracy(self):
        """Test handling of perfect accuracy (p_pooled = 1)."""
        node = ConceptDriftNode()

        # Perfect accuracy in both periods
        baseline_predicted = np.array([1] * 100)
        baseline_actual = np.array([1] * 100)
        current_predicted = np.array([1] * 100)
        current_actual = np.array([1] * 100)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        # Should handle p_pooled = 1 edge case
        assert result is None

    def test_handles_zero_accuracy(self):
        """Test handling of zero accuracy (p_pooled = 0)."""
        node = ConceptDriftNode()

        # Zero accuracy in both periods
        baseline_predicted = np.array([0] * 100)
        baseline_actual = np.array([1] * 100)
        current_predicted = np.array([0] * 100)
        current_actual = np.array([1] * 100)

        result = node._detect_performance_degradation(
            baseline_predicted, baseline_actual,
            current_predicted, current_actual,
            significance=0.05
        )

        # Should handle p_pooled = 0 edge case
        assert result is None


class TestFisherZTest:
    """Test _fisher_z_test method."""

    def test_identical_correlations(self):
        """Test identical correlations return z_stat ~0."""
        node = ConceptDriftNode()

        z_stat, p_value = node._fisher_z_test(0.5, 0.5, 100, 100)

        assert abs(z_stat) < 0.01  # Near zero
        assert p_value > 0.9  # Not significant

    def test_different_correlations(self):
        """Test different correlations return significant result."""
        node = ConceptDriftNode()

        # Large correlation difference
        z_stat, p_value = node._fisher_z_test(0.8, 0.2, 100, 100)

        assert abs(z_stat) > 2.0  # Significant difference
        assert p_value < 0.05

    def test_edge_correlation_values_high(self):
        """Test edge correlation values near +1."""
        node = ConceptDriftNode()

        # Values at +0.999 should be clipped
        z_stat, p_value = node._fisher_z_test(0.999, 0.998, 100, 100)

        # Should not raise error and return valid result
        assert isinstance(z_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_edge_correlation_values_low(self):
        """Test edge correlation values near -1."""
        node = ConceptDriftNode()

        # Values at -0.999 should be clipped
        z_stat, p_value = node._fisher_z_test(-0.999, -0.998, 100, 100)

        assert isinstance(z_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_opposite_sign_correlations(self):
        """Test correlations with opposite signs."""
        node = ConceptDriftNode()

        # Positive to negative is a big change
        z_stat, p_value = node._fisher_z_test(0.5, -0.5, 100, 100)

        assert abs(z_stat) > 3.0  # Very significant
        assert p_value < 0.01

    def test_small_sample_sizes(self):
        """Test with small sample sizes (minimum valid is n > 3)."""
        node = ConceptDriftNode()

        # Small but valid sample sizes
        z_stat, p_value = node._fisher_z_test(0.5, 0.3, 10, 10)

        assert isinstance(z_stat, float)
        assert isinstance(p_value, float)

    def test_asymmetric_sample_sizes(self):
        """Test with asymmetric sample sizes."""
        node = ConceptDriftNode()

        z_stat, p_value = node._fisher_z_test(0.6, 0.4, 50, 200)

        assert isinstance(z_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1


class TestPerformanceSeverityDetermination:
    """Test _determine_performance_severity method."""

    def test_performance_severity_critical(self):
        """Test critical severity for >20% drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.25, p_value=0.01, significance=0.05
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_performance_severity_high(self):
        """Test high severity for >10% drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.15, p_value=0.01, significance=0.05
        )

        assert severity == "high"
        assert drift_detected is True

    def test_performance_severity_medium(self):
        """Test medium severity for >5% drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.07, p_value=0.01, significance=0.05
        )

        assert severity == "medium"
        assert drift_detected is True

    def test_performance_severity_low(self):
        """Test low severity for small but significant drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.03, p_value=0.01, significance=0.05
        )

        assert severity == "low"
        assert drift_detected is True

    def test_performance_severity_none_not_significant(self):
        """Test no severity when p-value not significant."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.15, p_value=0.10, significance=0.05
        )

        assert severity == "none"
        assert drift_detected is False

    def test_performance_severity_none_no_drop(self):
        """Test no severity when no accuracy drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.0, p_value=0.01, significance=0.05
        )

        assert severity == "none"
        assert drift_detected is False

    def test_performance_severity_boundary_20_percent(self):
        """Test boundary at exactly 20% drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.20, p_value=0.01, significance=0.05
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_performance_severity_boundary_10_percent(self):
        """Test boundary at exactly 10% drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.10, p_value=0.01, significance=0.05
        )

        assert severity == "high"
        assert drift_detected is True

    def test_performance_severity_boundary_5_percent(self):
        """Test boundary at exactly 5% drop."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_performance_severity(
            accuracy_drop=0.05, p_value=0.01, significance=0.05
        )

        assert severity == "medium"
        assert drift_detected is True


class TestCorrelationSeverityDetermination:
    """Test _determine_correlation_severity method."""

    def test_correlation_severity_critical(self):
        """Test critical severity for >0.5 change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.6, p_value=0.01, significance=0.05
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_correlation_severity_high(self):
        """Test high severity for >0.3 change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.4, p_value=0.01, significance=0.05
        )

        assert severity == "high"
        assert drift_detected is True

    def test_correlation_severity_medium(self):
        """Test medium severity for >0.2 change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.25, p_value=0.01, significance=0.05
        )

        assert severity == "medium"
        assert drift_detected is True

    def test_correlation_severity_low(self):
        """Test low severity for >0.1 change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.15, p_value=0.01, significance=0.05
        )

        assert severity == "low"
        assert drift_detected is True

    def test_correlation_severity_none_small_change(self):
        """Test no severity for small change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.05, p_value=0.01, significance=0.05
        )

        assert severity == "none"
        assert drift_detected is False

    def test_correlation_severity_none_not_significant(self):
        """Test no severity when not statistically significant."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.4, p_value=0.10, significance=0.05
        )

        assert severity == "none"
        assert drift_detected is False

    def test_correlation_severity_boundary_0_5(self):
        """Test boundary at exactly 0.5 change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.5, p_value=0.01, significance=0.05
        )

        assert severity == "critical"
        assert drift_detected is True

    def test_correlation_severity_boundary_0_3(self):
        """Test boundary at exactly 0.3 change."""
        node = ConceptDriftNode()

        severity, drift_detected = node._determine_correlation_severity(
            correlation_change=0.3, p_value=0.01, significance=0.05
        )

        assert severity == "high"
        assert drift_detected is True


class TestConceptDriftEdgeCases:
    """Test edge cases for concept drift detection."""

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state with defaults."""
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
    async def test_single_feature_scenario(self):
        """Test with single feature."""
        node = ConceptDriftNode()
        state = self._create_test_state(
            features_to_monitor=["feature1"],
            model_id="test_model"
        )

        result = await node.execute(state)

        assert "concept_drift_results" in result
        assert isinstance(result["concept_drift_results"], list)

    @pytest.mark.asyncio
    async def test_high_dimensional_features(self):
        """Test with many features (100+)."""
        node = ConceptDriftNode()
        features = [f"feature_{i}" for i in range(100)]
        state = self._create_test_state(
            features_to_monitor=features,
            model_id="test_model"
        )

        result = await node.execute(state)

        assert "concept_drift_results" in result
        # Implementation limits to 10 features for performance
        assert isinstance(result["concept_drift_results"], list)

    @pytest.mark.asyncio
    async def test_brand_filter_applied(self):
        """Test brand filter is applied."""
        node = ConceptDriftNode()
        state = self._create_test_state(
            brand="Remibrutinib",
            model_id="test_model"
        )

        result = await node.execute(state)

        # Should execute without error
        assert "concept_drift_results" in result

    @pytest.mark.asyncio
    async def test_custom_significance_level(self):
        """Test with custom significance level."""
        node = ConceptDriftNode()
        state = self._create_test_state(
            significance_level=0.01,  # More strict
            model_id="test_model"
        )

        result = await node.execute(state)

        assert "concept_drift_results" in result

    @pytest.mark.asyncio
    async def test_30_day_time_window(self):
        """Test with 30 day time window."""
        node = ConceptDriftNode()
        state = self._create_test_state(
            time_window="30d",
            model_id="test_model"
        )

        result = await node.execute(state)

        assert "concept_drift_results" in result

    @pytest.mark.asyncio
    async def test_error_handling_graceful(self):
        """Test graceful error handling."""
        node = ConceptDriftNode()
        # Use a malformed time_window to trigger error path
        state = self._create_test_state(
            time_window="invalid",
            model_id="test_model"
        )

        # Should not raise, should handle gracefully
        result = await node.execute(state)

        # Either returns empty results or has error recorded
        assert "concept_drift_results" in result
        # Error should be recorded
        assert len(result.get("errors", [])) > 0 or len(result.get("warnings", [])) > 0
