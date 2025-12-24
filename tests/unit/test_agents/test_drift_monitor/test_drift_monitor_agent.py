"""Integration Tests for Drift Monitor Agent.

Tests complete end-to-end workflows and input/output validation.
"""

import pytest
from pydantic import ValidationError

from src.agents.drift_monitor import DriftMonitorAgent, DriftMonitorInput, DriftMonitorOutput


class TestDriftMonitorAgent:
    """Test DriftMonitorAgent integration."""

    def test_create_agent(self):
        """Test agent creation."""
        agent = DriftMonitorAgent()

        assert agent is not None
        assert agent.graph is not None

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic agent execution."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check for drift", features_to_monitor=["feature1", "feature2"]
        )

        result = await agent.run(input_data)

        assert isinstance(result, DriftMonitorOutput)
        assert result.overall_drift_score >= 0.0
        assert result.overall_drift_score <= 1.0

    @pytest.mark.asyncio
    async def test_run_with_model_id(self):
        """Test execution with model_id."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check model drift", features_to_monitor=["feature1"], model_id="model_v1"
        )

        result = await agent.run(input_data)

        # Should include model drift results
        assert "model_drift_results" in result.model_dump()

    @pytest.mark.asyncio
    async def test_run_with_brand(self):
        """Test execution with brand filter."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check drift for Remibrutinib",
            features_to_monitor=["feature1"],
            brand="Remibrutinib",
        )

        result = await agent.run(input_data)

        assert isinstance(result, DriftMonitorOutput)

    @pytest.mark.asyncio
    async def test_run_custom_time_window(self):
        """Test execution with custom time window."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check drift over 30 days", features_to_monitor=["feature1"], time_window="30d"
        )

        result = await agent.run(input_data)

        assert isinstance(result, DriftMonitorOutput)

    @pytest.mark.asyncio
    async def test_run_multiple_features(self):
        """Test execution with many features."""
        agent = DriftMonitorAgent()
        features = [f"feature_{i}" for i in range(20)]
        input_data = DriftMonitorInput(
            query="Check drift in all features", features_to_monitor=features
        )

        result = await agent.run(input_data)

        assert result.features_checked <= 20


class TestDriftMonitorInput:
    """Test DriftMonitorInput validation."""

    def test_valid_input_minimal(self):
        """Test valid minimal input."""
        input_data = DriftMonitorInput(query="test", features_to_monitor=["feature1"])

        assert input_data.query == "test"
        assert input_data.features_to_monitor == ["feature1"]
        assert input_data.time_window == "7d"  # Default
        assert input_data.significance_level == 0.05  # Default
        assert input_data.psi_threshold == 0.1  # Default

    def test_valid_input_full(self):
        """Test valid input with all fields."""
        input_data = DriftMonitorInput(
            query="test",
            features_to_monitor=["feature1", "feature2"],
            model_id="model_v1",
            time_window="14d",
            brand="Remibrutinib",
            significance_level=0.01,
            psi_threshold=0.15,
            check_data_drift=True,
            check_model_drift=False,
            check_concept_drift=True,
        )

        assert input_data.model_id == "model_v1"
        assert input_data.time_window == "14d"
        assert input_data.brand == "Remibrutinib"
        assert input_data.significance_level == 0.01
        assert input_data.psi_threshold == 0.15
        assert input_data.check_data_drift is True
        assert input_data.check_model_drift is False

    def test_invalid_empty_features(self):
        """Test invalid empty features list."""
        with pytest.raises(ValidationError):
            DriftMonitorInput(query="test", features_to_monitor=[])  # Empty list

    def test_invalid_time_window_no_d(self):
        """Test invalid time window without 'd'."""
        with pytest.raises(ValidationError):
            DriftMonitorInput(
                query="test", features_to_monitor=["f1"], time_window="7"  # Missing 'd'
            )

    def test_invalid_time_window_too_large(self):
        """Test invalid time window too large."""
        with pytest.raises(ValidationError):
            DriftMonitorInput(query="test", features_to_monitor=["f1"], time_window="500d")  # > 365

    def test_invalid_significance_level_too_low(self):
        """Test invalid significance level too low."""
        with pytest.raises(ValidationError):
            DriftMonitorInput(
                query="test", features_to_monitor=["f1"], significance_level=0.005  # < 0.01
            )

    def test_invalid_significance_level_too_high(self):
        """Test invalid significance level too high."""
        with pytest.raises(ValidationError):
            DriftMonitorInput(
                query="test", features_to_monitor=["f1"], significance_level=0.15  # > 0.10
            )

    def test_invalid_psi_threshold_negative(self):
        """Test invalid PSI threshold negative."""
        with pytest.raises(ValidationError):
            DriftMonitorInput(
                query="test", features_to_monitor=["f1"], psi_threshold=-0.1  # Negative
            )


class TestDriftMonitorOutput:
    """Test DriftMonitorOutput structure."""

    @pytest.mark.asyncio
    async def test_output_structure(self):
        """Test output structure."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(query="test", features_to_monitor=["feature1"])

        result = await agent.run(input_data)

        # Check all required fields exist
        assert hasattr(result, "data_drift_results")
        assert hasattr(result, "model_drift_results")
        assert hasattr(result, "concept_drift_results")
        assert hasattr(result, "overall_drift_score")
        assert hasattr(result, "features_with_drift")
        assert hasattr(result, "alerts")
        assert hasattr(result, "drift_summary")
        assert hasattr(result, "recommended_actions")
        assert hasattr(result, "detection_latency_ms")
        assert hasattr(result, "features_checked")
        assert hasattr(result, "baseline_timestamp")
        assert hasattr(result, "current_timestamp")
        assert hasattr(result, "warnings")

    @pytest.mark.asyncio
    async def test_output_drift_score_range(self):
        """Test drift score is in valid range."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(query="test", features_to_monitor=["feature1"])

        result = await agent.run(input_data)

        assert 0.0 <= result.overall_drift_score <= 1.0

    @pytest.mark.asyncio
    async def test_output_alerts_structure(self):
        """Test alerts structure."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(query="test", features_to_monitor=["feature1"])

        result = await agent.run(input_data)

        assert isinstance(result.alerts, list)
        if result.alerts:
            alert = result.alerts[0]
            assert "alert_id" in alert
            assert "severity" in alert
            assert "drift_type" in alert
            assert "affected_features" in alert
            assert "message" in alert
            assert "recommended_action" in alert
            assert "timestamp" in alert

    @pytest.mark.asyncio
    async def test_output_drift_summary_exists(self):
        """Test drift summary is generated."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(query="test", features_to_monitor=["feature1"])

        result = await agent.run(input_data)

        assert len(result.drift_summary) > 0

    @pytest.mark.asyncio
    async def test_output_recommended_actions_exists(self):
        """Test recommended actions are generated."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(query="test", features_to_monitor=["feature1"])

        result = await agent.run(input_data)

        assert len(result.recommended_actions) > 0


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_data_drift_only(self):
        """Test with only data drift enabled."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check data drift only",
            features_to_monitor=["feature1", "feature2"],
            check_data_drift=True,
            check_model_drift=False,
            check_concept_drift=False,
        )

        result = await agent.run(input_data)

        assert len(result.data_drift_results) >= 0
        assert len(result.model_drift_results) == 0
        assert len(result.concept_drift_results) == 0

    @pytest.mark.asyncio
    async def test_model_drift_only(self):
        """Test with only model drift enabled."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check model drift only",
            features_to_monitor=["feature1"],
            model_id="model_v1",
            check_data_drift=False,
            check_model_drift=True,
            check_concept_drift=False,
        )

        result = await agent.run(input_data)

        assert len(result.data_drift_results) == 0
        # Model drift results depend on mock data
        assert len(result.concept_drift_results) == 0

    @pytest.mark.asyncio
    async def test_all_drift_types(self):
        """Test with all drift types enabled."""
        agent = DriftMonitorAgent()
        input_data = DriftMonitorInput(
            query="Check all drift types",
            features_to_monitor=["feature1", "feature2"],
            model_id="model_v1",
            check_data_drift=True,
            check_model_drift=True,
            check_concept_drift=True,
        )

        result = await agent.run(input_data)

        # All three drift types should have results (even if empty)
        assert "data_drift_results" in result.model_dump()
        assert "model_drift_results" in result.model_dump()
        assert "concept_drift_results" in result.model_dump()

    @pytest.mark.asyncio
    async def test_multiple_brands(self):
        """Test with different brands."""
        agent = DriftMonitorAgent()

        for brand in ["Remibrutinib", "Fabhalta", "Kisqali"]:
            input_data = DriftMonitorInput(
                query=f"Check drift for {brand}", features_to_monitor=["feature1"], brand=brand
            )

            result = await agent.run(input_data)

            assert isinstance(result, DriftMonitorOutput)

    @pytest.mark.asyncio
    async def test_different_time_windows(self):
        """Test with different time windows."""
        agent = DriftMonitorAgent()

        for window in ["7d", "14d", "30d"]:
            input_data = DriftMonitorInput(
                query=f"Check drift over {window}",
                features_to_monitor=["feature1"],
                time_window=window,
            )

            result = await agent.run(input_data)

            assert isinstance(result, DriftMonitorOutput)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_latency_under_target(self):
        """Test latency is under 30s for 50 features (lenient for CI)."""
        agent = DriftMonitorAgent()
        features = [f"feature_{i}" for i in range(50)]

        input_data = DriftMonitorInput(query="Latency test", features_to_monitor=features)

        result = await agent.run(input_data)

        # Should be under 30,000ms (30s) for 50 features (lenient for CI environments)
        assert result.detection_latency_ms < 30_000
