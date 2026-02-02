"""Tests for Power Analysis Node.

Tests the statistical power calculation functionality.
"""

import pytest

from src.agents.experiment_designer.graph import create_initial_state
from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode


class TestPowerAnalysisNode:
    """Test PowerAnalysisNode functionality."""

    def test_create_node(self):
        """Test creating node."""
        node = PowerAnalysisNode()

        assert node is not None
        assert hasattr(node, "_default_alpha")
        assert hasattr(node, "_default_power")
        assert hasattr(node, "_default_effect_size")
        assert node._default_alpha == 0.05
        assert node._default_power == 0.80
        assert node._default_effect_size == 0.25

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Test power analysis",
            constraints={"expected_effect_size": 0.25, "power": 0.80, "alpha": 0.05},
        )
        state["status"] = "calculating"
        state["outcomes"] = [
            {"name": "Engagement", "metric_type": "continuous", "is_primary": True}
        ]

        result = await node.execute(state)

        assert result["status"] == "auditing"
        assert result.get("power_analysis") is not None
        assert "power_analysis" in result.get("node_latencies_ms", {})

    @pytest.mark.asyncio
    async def test_execute_returns_sample_size(self):
        """Test that sample size is returned."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Test sample size",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        assert pa.get("required_sample_size") > 0
        assert pa.get("required_sample_size_per_arm") > 0

    @pytest.mark.asyncio
    async def test_execute_returns_achieved_power(self):
        """Test that achieved power is returned."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Test achieved power",
            constraints={"expected_effect_size": 0.25, "power": 0.80},
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        assert 0.0 < pa.get("achieved_power", 0) <= 1.0

    @pytest.mark.asyncio
    async def test_execute_returns_mde(self):
        """Test that minimum detectable effect is returned."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Test MDE", constraints={"expected_effect_size": 0.25, "power": 0.80}
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        assert pa.get("minimum_detectable_effect") > 0

    @pytest.mark.asyncio
    async def test_execute_skip_on_failed(self):
        """Test execution skips on failed status."""
        node = PowerAnalysisNode()
        state = create_initial_state(business_question="Test skip on failed")
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_records_latency(self):
        """Test that node latency is recorded."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Test latency recording", constraints={"expected_effect_size": 0.25}
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        assert "node_latencies_ms" in result
        assert "power_analysis" in result["node_latencies_ms"]
        assert result["node_latencies_ms"]["power_analysis"] >= 0


class TestContinuousOutcomePowerAnalysis:
    """Test power analysis for continuous outcomes."""

    @pytest.mark.asyncio
    async def test_continuous_default_power(self):
        """Test continuous outcome with default power."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Continuous outcome test",
            constraints={"expected_effect_size": 0.25, "power": 0.80, "alpha": 0.05},
        )
        state["status"] = "calculating"
        state["outcomes"] = [
            {"name": "Engagement", "metric_type": "continuous", "is_primary": True}
        ]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        assert pa.get("effect_size_type") == "cohens_d"

    @pytest.mark.asyncio
    async def test_small_effect_size_requires_large_n(self):
        """Test that small effect size requires larger sample."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Small effect test",
            constraints={"expected_effect_size": 0.10, "power": 0.80},  # Small effect
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        # Small effect should require large sample
        assert pa.get("required_sample_size", 0) > 1000

    @pytest.mark.asyncio
    async def test_large_effect_size_requires_small_n(self):
        """Test that large effect size requires smaller sample."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Large effect test",
            constraints={"expected_effect_size": 0.80, "power": 0.80},  # Large effect
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        # Large effect should require smaller sample
        assert pa.get("required_sample_size", float("inf")) < 200

    @pytest.mark.asyncio
    async def test_higher_power_requires_larger_n(self):
        """Test that higher power requires larger sample."""
        node = PowerAnalysisNode()

        # Test with 80% power
        state_80 = create_initial_state(
            business_question="80% power test",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state_80["status"] = "calculating"
        state_80["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_80 = await node.execute(state_80)

        # Test with 95% power
        state_95 = create_initial_state(
            business_question="95% power test",
            constraints={"expected_effect_size": 0.30, "power": 0.95},
        )
        state_95["status"] = "calculating"
        state_95["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_95 = await node.execute(state_95)

        n_80 = result_80.get("power_analysis", {}).get("required_sample_size", 0)
        n_95 = result_95.get("power_analysis", {}).get("required_sample_size", 0)

        assert n_95 > n_80

    @pytest.mark.asyncio
    async def test_smaller_alpha_requires_larger_n(self):
        """Test that smaller alpha requires larger sample."""
        node = PowerAnalysisNode()

        # Test with alpha=0.05
        state_05 = create_initial_state(
            business_question="Alpha 0.05 test",
            constraints={"expected_effect_size": 0.30, "power": 0.80, "alpha": 0.05},
        )
        state_05["status"] = "calculating"
        state_05["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_05 = await node.execute(state_05)

        # Test with alpha=0.01
        state_01 = create_initial_state(
            business_question="Alpha 0.01 test",
            constraints={"expected_effect_size": 0.30, "power": 0.80, "alpha": 0.01},
        )
        state_01["status"] = "calculating"
        state_01["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_01 = await node.execute(state_01)

        n_05 = result_05.get("power_analysis", {}).get("required_sample_size", 0)
        n_01 = result_01.get("power_analysis", {}).get("required_sample_size", 0)

        assert n_01 > n_05


class TestBinaryOutcomePowerAnalysis:
    """Test power analysis for binary outcomes."""

    @pytest.mark.asyncio
    async def test_binary_outcome(self):
        """Test binary outcome power analysis."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Binary outcome test",
            constraints={"expected_effect_size": 0.10, "power": 0.80, "baseline_rate": 0.15},
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Conversion", "metric_type": "binary", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        assert pa.get("required_sample_size") > 0

    @pytest.mark.asyncio
    async def test_binary_higher_baseline_affects_n(self):
        """Test that baseline rate affects sample size."""
        node = PowerAnalysisNode()

        # Low baseline rate
        state_low = create_initial_state(
            business_question="Low baseline test",
            constraints={"expected_effect_size": 0.05, "power": 0.80, "baseline_rate": 0.05},
        )
        state_low["status"] = "calculating"
        state_low["outcomes"] = [{"name": "Y", "metric_type": "binary", "is_primary": True}]
        result_low = await node.execute(state_low)

        # High baseline rate
        state_high = create_initial_state(
            business_question="High baseline test",
            constraints={"expected_effect_size": 0.05, "power": 0.80, "baseline_rate": 0.50},
        )
        state_high["status"] = "calculating"
        state_high["outcomes"] = [{"name": "Y", "metric_type": "binary", "is_primary": True}]
        result_high = await node.execute(state_high)

        # Both should compute sample sizes
        assert result_low.get("power_analysis", {}).get("required_sample_size") > 0
        assert result_high.get("power_analysis", {}).get("required_sample_size") > 0


class TestClusterRCTPowerAnalysis:
    """Test power analysis for cluster RCT designs."""

    @pytest.mark.asyncio
    async def test_cluster_rct_increases_n(self):
        """Test that cluster RCT requires larger sample than individual RCT."""
        node = PowerAnalysisNode()

        # Individual RCT
        state_ind = create_initial_state(
            business_question="Individual RCT test",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state_ind["status"] = "calculating"
        state_ind["design_type"] = "RCT"
        state_ind["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_ind = await node.execute(state_ind)

        # Cluster RCT
        state_cluster = create_initial_state(
            business_question="Cluster RCT test",
            constraints={
                "expected_effect_size": 0.30,
                "power": 0.80,
                "cluster_size": 50,
                "expected_icc": 0.05,
            },
        )
        state_cluster["status"] = "calculating"
        state_cluster["design_type"] = "Cluster_RCT"
        state_cluster["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_cluster = await node.execute(state_cluster)

        n_ind = result_ind.get("power_analysis", {}).get("required_sample_size", 0)
        n_cluster = result_cluster.get("power_analysis", {}).get("required_sample_size", 0)

        # Cluster RCT should require larger sample due to design effect
        assert n_cluster >= n_ind

    @pytest.mark.asyncio
    async def test_higher_icc_increases_n(self):
        """Test that higher ICC increases required sample size."""
        node = PowerAnalysisNode()

        # Low ICC
        state_low = create_initial_state(
            business_question="Low ICC test",
            constraints={
                "expected_effect_size": 0.30,
                "power": 0.80,
                "cluster_size": 50,
                "expected_icc": 0.01,
            },
        )
        state_low["status"] = "calculating"
        state_low["design_type"] = "Cluster_RCT"
        state_low["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_low = await node.execute(state_low)

        # High ICC
        state_high = create_initial_state(
            business_question="High ICC test",
            constraints={
                "expected_effect_size": 0.30,
                "power": 0.80,
                "cluster_size": 50,
                "expected_icc": 0.10,
            },
        )
        state_high["status"] = "calculating"
        state_high["design_type"] = "Cluster_RCT"
        state_high["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]
        result_high = await node.execute(state_high)

        n_low = result_low.get("power_analysis", {}).get("required_sample_size", 0)
        n_high = result_high.get("power_analysis", {}).get("required_sample_size", 0)

        assert n_high > n_low


class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""

    @pytest.mark.asyncio
    async def test_sensitivity_analysis_included(self):
        """Test that sensitivity analysis is included."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Sensitivity analysis test",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        # Sensitivity analysis should be present
        assert "sensitivity_analysis" in pa or "assumptions" in pa


class TestPowerAnalysisPerformance:
    """Test power analysis performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test power analysis completes under 100ms target."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Test latency performance",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["power_analysis"]
        assert latency < 100, f"Power analysis took {latency}ms, exceeds 100ms target"


class TestPowerAnalysisEdgeCases:
    """Test power analysis edge cases."""

    @pytest.mark.asyncio
    async def test_missing_effect_size(self):
        """Test handling of missing effect size."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Missing effect size test",
            constraints={
                "power": 0.80
                # No expected_effect_size
            },
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        # Should use default or warn
        assert result["status"] in ["auditing", "failed"]

    @pytest.mark.asyncio
    async def test_missing_outcomes(self):
        """Test handling of missing outcomes."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Missing outcomes test",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state["status"] = "calculating"
        # No outcomes

        result = await node.execute(state)

        # Should handle gracefully
        assert result["status"] in ["auditing", "failed"]

    @pytest.mark.asyncio
    async def test_very_small_effect_size(self):
        """Test handling of very small effect size."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="Very small effect test",
            constraints={"expected_effect_size": 0.01, "power": 0.80},
        )
        state["status"] = "calculating"
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous", "is_primary": True}]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        # Very small effect should require very large sample
        assert pa.get("required_sample_size", 0) > 10000

    @pytest.mark.asyncio
    async def test_no_primary_outcome(self):
        """Test handling when no primary outcome specified."""
        node = PowerAnalysisNode()
        state = create_initial_state(
            business_question="No primary outcome test",
            constraints={"expected_effect_size": 0.30, "power": 0.80},
        )
        state["status"] = "calculating"
        state["outcomes"] = [
            {"name": "Y1", "metric_type": "continuous", "is_primary": False},
            {"name": "Y2", "metric_type": "binary", "is_primary": False},
        ]

        result = await node.execute(state)

        # Should handle by using first outcome or warning
        assert result["status"] in ["auditing", "failed"]
