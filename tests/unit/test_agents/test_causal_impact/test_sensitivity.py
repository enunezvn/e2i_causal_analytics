"""Tests for sensitivity analysis node."""

import math

import pytest

from src.agents.causal_impact.nodes.sensitivity import SensitivityNode
from src.agents.causal_impact.state import CausalImpactState, EstimationResult


class TestSensitivityNode:
    """Test SensitivityNode."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

    @pytest.mark.asyncio
    async def test_calculate_e_value(self):
        """Test E-value calculation."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-1",
            "estimation_result": self._create_test_estimation(ate=0.5),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "sensitivity_analysis" in result
        sens = result["sensitivity_analysis"]

        assert "e_value" in sens
        assert sens["e_value"] >= 1.0  # E-value is always >= 1
        assert result["current_phase"] == "interpreting"

    @pytest.mark.asyncio
    async def test_e_value_for_ci(self):
        """Test E-value calculation for confidence interval."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-2",
            "estimation_result": self._create_test_estimation(ate=0.5),
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        assert "e_value_ci" in sens
        assert sens["e_value_ci"] >= 1.0

        # E-value for CI should be <= E-value for point estimate
        # (CI bound is closer to null)
        assert sens["e_value_ci"] <= sens["e_value"]

    @pytest.mark.asyncio
    async def test_robustness_classification(self):
        """Test robustness classification."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-3",
            "estimation_result": self._create_test_estimation(ate=0.5),
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        assert "robust_to_confounding" in sens
        assert isinstance(sens["robust_to_confounding"], bool)

        # Robust if E-value > 2.0
        expected_robust = sens["e_value"] > 2.0
        assert sens["robust_to_confounding"] == expected_robust

    @pytest.mark.asyncio
    async def test_confounder_strength_classification(self):
        """Test unmeasured confounder strength classification."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-4",
            "estimation_result": self._create_test_estimation(ate=0.5),
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        assert "unmeasured_confounder_strength" in sens
        assert sens["unmeasured_confounder_strength"] in ["weak", "moderate", "strong"]

        # Classification based on E-value
        e_value = sens["e_value"]
        if e_value < 1.5:
            assert sens["unmeasured_confounder_strength"] == "weak"
        elif e_value < 3.0:
            assert sens["unmeasured_confounder_strength"] == "moderate"
        else:
            assert sens["unmeasured_confounder_strength"] == "strong"

    @pytest.mark.asyncio
    async def test_interpretation_text(self):
        """Test that interpretation text is generated."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-5",
            "estimation_result": self._create_test_estimation(ate=0.5),
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        assert "interpretation" in sens
        assert len(sens["interpretation"]) > 0
        assert "E-value" in sens["interpretation"]

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that sensitivity latency is measured."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-6",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "sensitivity_latency_ms" in result
        assert result["sensitivity_latency_ms"] >= 0
        assert result["sensitivity_latency_ms"] < 5000  # Should be < 5s

    @pytest.mark.asyncio
    async def test_error_handling_missing_estimation(self):
        """Test error handling when estimation result is missing."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-7",
            "status": "pending",
        }

        result = await node.execute(state)

        assert "sensitivity_error" in result
        assert result["status"] == "failed"


class TestEValueCalculation:
    """Test E-value calculation formula."""

    def test_e_value_zero_effect(self):
        """Test E-value for zero effect."""
        node = SensitivityNode()

        e_value = node._calculate_e_value(0.0)

        # E-value should be 1.0 for zero effect
        assert e_value == 1.0

    def test_e_value_small_effect(self):
        """Test E-value for small effect."""
        node = SensitivityNode()

        e_value = node._calculate_e_value(0.1)

        # Small effect should have E-value close to 1
        assert 1.0 <= e_value < 1.5

    def test_e_value_large_effect(self):
        """Test E-value for large effect."""
        node = SensitivityNode()

        e_value = node._calculate_e_value(1.0)

        # Large effect should have high E-value
        assert e_value > 3.0

    def test_e_value_negative_effect(self):
        """Test E-value for negative effect."""
        node = SensitivityNode()

        e_value_pos = node._calculate_e_value(0.5)
        e_value_neg = node._calculate_e_value(-0.5)

        # E-value should be same for positive and negative effects
        assert math.isclose(e_value_pos, e_value_neg, rel_tol=1e-6)

    def test_e_value_formula(self):
        """Test E-value formula directly."""
        node = SensitivityNode()

        # E-value = RR + sqrt(RR * (RR - 1))
        # Where RR = exp(|effect|)

        effect = 0.5
        e_value = node._calculate_e_value(effect)

        import numpy as np

        rr = np.exp(abs(effect))
        expected_e_value = rr + np.sqrt(rr * (rr - 1))

        assert math.isclose(e_value, expected_e_value, rel_tol=1e-6)


class TestEValueInterpretation:
    """Test E-value interpretation logic."""

    def test_interpret_very_weak_confounding(self):
        """Test interpretation for very weak confounding."""
        node = SensitivityNode()

        interpretation = node._interpret_e_value(1.2)

        assert "very weak" in interpretation or "weak" in interpretation
        assert "caution" in interpretation.lower()

    def test_interpret_moderate_confounding(self):
        """Test interpretation for moderate confounding."""
        node = SensitivityNode()

        interpretation = node._interpret_e_value(1.8)

        assert "moderate" in interpretation
        assert "carefully" in interpretation or "caution" in interpretation

    def test_interpret_strong_confounding(self):
        """Test interpretation for strong confounding."""
        node = SensitivityNode()

        interpretation = node._interpret_e_value(2.5)

        assert "strong" in interpretation or "fairly strong" in interpretation
        assert "robustness" in interpretation.lower()

    def test_interpret_very_strong_confounding(self):
        """Test interpretation for very strong confounding."""
        node = SensitivityNode()

        interpretation = node._interpret_e_value(4.0)

        assert "very strong" in interpretation
        assert "robustness" in interpretation.lower()


class TestSensitivityWithDifferentEffects:
    """Test sensitivity analysis with different effect sizes."""

    @pytest.mark.asyncio
    async def test_small_effect_sensitivity(self):
        """Test sensitivity for small effect."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-8",
            "estimation_result": self._create_test_estimation(ate=0.1),  # Small
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        # Small effects should have low E-value
        assert sens["e_value"] < 2.0
        assert sens["unmeasured_confounder_strength"] in ["weak", "moderate"]

    @pytest.mark.asyncio
    async def test_large_effect_sensitivity(self):
        """Test sensitivity for large effect."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-9",
            "estimation_result": self._create_test_estimation(ate=0.8),  # Large
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        # Large effects should have high E-value
        assert sens["e_value"] > 2.0
        assert sens["robust_to_confounding"] is True

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }
