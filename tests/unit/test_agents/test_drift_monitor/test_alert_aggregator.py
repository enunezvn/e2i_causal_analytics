"""Tests for Alert Aggregator Node.

Tests alert generation, drift score calculation, and summary generation.
"""

import pytest

from src.agents.drift_monitor.nodes.alert_aggregator import AlertAggregatorNode
from src.agents.drift_monitor.state import DriftMonitorState, DriftResult


class TestAlertAggregatorNode:
    """Test AlertAggregatorNode."""

    def _create_drift_result(
        self, feature: str, drift_type: str, severity: str, drift_detected: bool = True
    ) -> DriftResult:
        """Create test drift result."""
        return {
            "feature": feature,
            "drift_type": drift_type,  # type: ignore
            "test_statistic": 0.5,
            "p_value": 0.01,
            "drift_detected": drift_detected,
            "severity": severity,  # type: ignore
            "baseline_period": "baseline",
            "current_period": "current",
        }

    def _create_test_state(self, **overrides) -> DriftMonitorState:
        """Create test state with drift results."""
        state: DriftMonitorState = {
            "query": "test",
            "features_to_monitor": ["f1", "f2"],
            "time_window": "7d",
            "significance_level": 0.05,
            "psi_threshold": 0.1,
            "check_data_drift": True,
            "check_model_drift": True,
            "check_concept_drift": True,
            "data_drift_results": [],
            "model_drift_results": [],
            "concept_drift_results": [],
            "errors": [],
            "warnings": [],
            "status": "detecting",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = AlertAggregatorNode()
        state = self._create_test_state(
            data_drift_results=[self._create_drift_result("f1", "data", "high")]
        )

        result = await node.execute(state)

        assert "overall_drift_score" in result
        assert "features_with_drift" in result
        assert "alerts" in result
        assert "drift_summary" in result
        assert "recommended_actions" in result
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency measurement."""
        node = AlertAggregatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "total_latency_ms" in result
        assert result["total_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = AlertAggregatorNode()
        state = self._create_test_state(status="failed")

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result["overall_drift_score"] == 0.0
        assert result["drift_summary"] == "Drift detection failed"


class TestDriftScoreCalculation:
    """Test composite drift score calculation."""

    def test_calculate_drift_score_no_results(self):
        """Test drift score with no results."""
        node = AlertAggregatorNode()
        results = []

        drift_score = node._calculate_drift_score(results)

        assert drift_score == 0.0

    def test_calculate_drift_score_all_none(self):
        """Test drift score with all none severity."""
        node = AlertAggregatorNode()
        results = [{"severity": "none"} for _ in range(5)]

        drift_score = node._calculate_drift_score(results)  # type: ignore

        assert drift_score == 0.0

    def test_calculate_drift_score_all_low(self):
        """Test drift score with all low severity."""
        node = AlertAggregatorNode()
        results = [{"severity": "low"} for _ in range(5)]

        drift_score = node._calculate_drift_score(results)  # type: ignore

        # All low = 0.25 average
        assert drift_score == 0.25

    def test_calculate_drift_score_all_critical(self):
        """Test drift score with all critical severity."""
        node = AlertAggregatorNode()
        results = [{"severity": "critical"} for _ in range(5)]

        drift_score = node._calculate_drift_score(results)  # type: ignore

        # All critical = 1.0 average
        assert drift_score == 1.0

    def test_calculate_drift_score_mixed(self):
        """Test drift score with mixed severities."""
        node = AlertAggregatorNode()
        results = [
            {"severity": "critical"},  # 1.0
            {"severity": "high"},  # 0.75
            {"severity": "medium"},  # 0.5
            {"severity": "low"},  # 0.25
            {"severity": "none"},  # 0.0
        ]

        drift_score = node._calculate_drift_score(results)  # type: ignore

        # Average = (1.0 + 0.75 + 0.5 + 0.25 + 0.0) / 5 = 0.5
        assert drift_score == 0.5


class TestFeatureIdentification:
    """Test feature identification."""

    def _create_drift_result(self, feature: str, drift_detected: bool) -> DriftResult:
        """Create test drift result."""
        return {
            "feature": feature,
            "drift_type": "data",
            "test_statistic": 0.5,
            "p_value": 0.01,
            "drift_detected": drift_detected,
            "severity": "high",
            "baseline_period": "baseline",
            "current_period": "current",
        }

    def test_identify_drifted_features_none(self):
        """Test with no drifted features."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f1", False),
            self._create_drift_result("f2", False),
        ]

        features = node._identify_drifted_features(results)

        assert len(features) == 0

    def test_identify_drifted_features_some(self):
        """Test with some drifted features."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f1", True),
            self._create_drift_result("f2", False),
            self._create_drift_result("f3", True),
        ]

        features = node._identify_drifted_features(results)

        assert len(features) == 2
        assert "f1" in features
        assert "f3" in features

    def test_identify_drifted_features_duplicates(self):
        """Test removes duplicate features."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f1", True),
            self._create_drift_result("f1", True),  # Duplicate
            self._create_drift_result("f2", True),
        ]

        features = node._identify_drifted_features(results)

        # Should remove duplicates
        assert len(features) == 2
        assert features.count("f1") == 1

    def test_identify_drifted_features_sorted(self):
        """Test features are sorted."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f3", True),
            self._create_drift_result("f1", True),
            self._create_drift_result("f2", True),
        ]

        features = node._identify_drifted_features(results)

        assert features == ["f1", "f2", "f3"]


class TestAlertGeneration:
    """Test alert generation."""

    def _create_drift_result(self, feature: str, drift_type: str, severity: str) -> DriftResult:
        """Create test drift result."""
        return {
            "feature": feature,
            "drift_type": drift_type,  # type: ignore
            "test_statistic": 0.5,
            "p_value": 0.01,
            "drift_detected": True,
            "severity": severity,  # type: ignore
            "baseline_period": "baseline",
            "current_period": "current",
        }

    def test_generate_alerts_critical_data_drift(self):
        """Test critical alert for data drift."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "data", "critical")]

        alerts = node._generate_alerts(results)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["severity"] == "critical"
        assert alert["drift_type"] == "data"
        assert "f1" in alert["affected_features"]

    def test_generate_alerts_high_severity(self):
        """Test warning alert for high severity."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "data", "high")]

        alerts = node._generate_alerts(results)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["severity"] == "warning"
        assert alert["drift_type"] == "data"

    def test_generate_alerts_medium_severity(self):
        """Test no alert for medium severity."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "data", "medium")]

        alerts = node._generate_alerts(results)

        # Medium severity doesn't generate alerts
        assert len(alerts) == 0

    def test_generate_alerts_multiple_types(self):
        """Test alerts for multiple drift types."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f1", "data", "critical"),
            self._create_drift_result("pred", "model", "critical"),
        ]

        alerts = node._generate_alerts(results)

        assert len(alerts) == 2
        drift_types = [a["drift_type"] for a in alerts]
        assert "data" in drift_types
        assert "model" in drift_types

    def test_generate_alerts_multiple_features(self):
        """Test alert with multiple affected features."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f1", "data", "critical"),
            self._create_drift_result("f2", "data", "critical"),
            self._create_drift_result("f3", "data", "critical"),
        ]

        alerts = node._generate_alerts(results)

        assert len(alerts) == 1
        alert = alerts[0]
        assert len(alert["affected_features"]) == 3


class TestDriftSummary:
    """Test drift summary generation."""

    def _create_drift_result(
        self, feature: str, severity: str, drift_type: str = "data", drift_detected: bool = True
    ) -> DriftResult:
        """Create test drift result."""
        return {
            "feature": feature,
            "drift_type": drift_type,  # type: ignore
            "test_statistic": 0.5,
            "p_value": 0.01,
            "drift_detected": drift_detected,
            "severity": severity,  # type: ignore
            "baseline_period": "baseline",
            "current_period": "current",
        }

    def test_create_drift_summary_critical(self):
        """Test summary for critical drift."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "critical")]
        drift_score = 0.9
        features_with_drift = ["f1"]

        summary = node._create_drift_summary(results, drift_score, features_with_drift)

        assert "CRITICAL DRIFT" in summary
        assert "0.9" in summary or "0.900" in summary

    def test_create_drift_summary_no_drift(self):
        """Test summary for no drift."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "none", drift_detected=False)]
        drift_score = 0.0
        features_with_drift = []

        summary = node._create_drift_summary(results, drift_score, features_with_drift)

        assert "NO SIGNIFICANT DRIFT" in summary

    def test_create_drift_summary_severity_breakdown(self):
        """Test summary includes severity breakdown."""
        node = AlertAggregatorNode()
        results = [
            self._create_drift_result("f1", "critical"),
            self._create_drift_result("f2", "high"),
            self._create_drift_result("f3", "medium"),
        ]
        drift_score = 0.75
        features_with_drift = ["f1", "f2", "f3"]

        summary = node._create_drift_summary(results, drift_score, features_with_drift)

        assert "critical: 1" in summary.lower()
        assert "high: 1" in summary.lower()
        assert "medium: 1" in summary.lower()


class TestRecommendations:
    """Test recommendation generation."""

    def _create_drift_result(
        self, feature: str, severity: str, drift_type: str = "data"
    ) -> DriftResult:
        """Create test drift result."""
        return {
            "feature": feature,
            "drift_type": drift_type,  # type: ignore
            "test_statistic": 0.5,
            "p_value": 0.01,
            "drift_detected": True,
            "severity": severity,  # type: ignore
            "baseline_period": "baseline",
            "current_period": "current",
        }

    def test_generate_recommendations_critical_drift(self):
        """Test recommendations for critical drift."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "critical")]
        drift_score = 0.9

        recommendations = node._generate_recommendations(results, drift_score)

        assert len(recommendations) > 0
        assert any("URGENT" in r or "retraining" in r.lower() for r in recommendations)

    def test_generate_recommendations_no_drift(self):
        """Test recommendations for no drift."""
        node = AlertAggregatorNode()
        results = []
        drift_score = 0.0

        recommendations = node._generate_recommendations(results, drift_score)

        assert len(recommendations) > 0
        assert any("stable" in r.lower() for r in recommendations)

    def test_generate_recommendations_data_drift(self):
        """Test data drift specific recommendations."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("f1", "medium", "data")]
        drift_score = 0.5

        recommendations = node._generate_recommendations(results, drift_score)

        assert any("data pipeline" in r.lower() for r in recommendations)

    def test_generate_recommendations_model_drift(self):
        """Test model drift specific recommendations."""
        node = AlertAggregatorNode()
        results = [self._create_drift_result("pred", "medium", "model")]
        drift_score = 0.5

        recommendations = node._generate_recommendations(results, drift_score)

        assert any("prediction" in r.lower() for r in recommendations)
