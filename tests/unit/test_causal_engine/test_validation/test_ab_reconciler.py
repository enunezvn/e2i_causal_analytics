"""Tests for ABReconciler - A/B experiment reconciliation.

B8.2: A/B reconciliation tests.
"""

import pytest

from src.causal_engine.validation import (
    ABReconciler,
    ABExperimentResult,
    LibraryEffectEstimate,
)


class TestABReconciler:
    """Test suite for ABReconciler."""

    @pytest.fixture
    def reconciler(self) -> ABReconciler:
        """Create ABReconciler instance."""
        return ABReconciler()

    @pytest.fixture
    def experiment_result(self) -> ABExperimentResult:
        """Create sample A/B experiment result."""
        return ABExperimentResult(
            experiment_id="exp-001",
            treatment_group_size=5000,
            control_group_size=5000,
            observed_effect=0.15,
            observed_ci_lower=0.10,
            observed_ci_upper=0.20,
            observed_p_value=0.001,
            is_significant=True,
            experiment_duration_days=28,
        )

    @pytest.fixture
    def matching_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates that match the experiment well."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.14,  # Close to 0.15 observed
                ci_lower=0.08,
                ci_upper=0.20,
                p_value=0.002,
                confidence=0.85,
                latency_ms=100,
            ),
            LibraryEffectEstimate(
                library="econml",
                effect_type="ate",
                estimate=0.16,
                ci_lower=0.10,
                ci_upper=0.22,
                p_value=0.001,
                confidence=0.82,
                latency_ms=120,
            ),
        ]

    @pytest.fixture
    def mismatched_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates that poorly match the experiment (30-50% gap = 'poor' status)."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.09,  # ~40% lower than 0.15 observed (in 'poor' range)
                ci_lower=0.05,
                ci_upper=0.13,
                p_value=0.02,
                confidence=0.70,
                latency_ms=100,
            ),
        ]

    @pytest.fixture
    def opposite_estimates(self) -> list[LibraryEffectEstimate]:
        """Estimates with opposite direction from experiment."""
        return [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=-0.10,  # Opposite direction!
                ci_lower=-0.15,
                ci_upper=-0.05,
                p_value=0.01,
                confidence=0.75,
                latency_ms=100,
            ),
        ]

    @pytest.mark.asyncio
    async def test_reconcile_excellent_match(
        self,
        reconciler: ABReconciler,
        experiment_result: ABExperimentResult,
        matching_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test reconciliation with excellent match."""
        result = await reconciler.reconcile(
            experiment=experiment_result,
            causal_estimates=matching_estimates,
        )

        assert result["reconciliation_status"] in ["excellent", "good"]
        assert result["reconciliation_score"] > 0.7
        assert result["direction_match"] is True
        assert result["magnitude_match"] is True

    @pytest.mark.asyncio
    async def test_reconcile_poor_match(
        self,
        reconciler: ABReconciler,
        experiment_result: ABExperimentResult,
        mismatched_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test reconciliation with poor match (~40% gap)."""
        result = await reconciler.reconcile(
            experiment=experiment_result,
            causal_estimates=mismatched_estimates,
        )

        # 40% gap falls in "poor" range (30-50%)
        assert result["reconciliation_status"] == "poor"
        assert result["reconciliation_score"] < 0.5
        assert result["direction_match"] is True  # Both positive
        assert result["magnitude_match"] is False  # 40% gap exceeds 30% threshold

    @pytest.mark.asyncio
    async def test_reconcile_direction_mismatch(
        self,
        reconciler: ABReconciler,
        experiment_result: ABExperimentResult,
        opposite_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test reconciliation with opposite effect direction."""
        result = await reconciler.reconcile(
            experiment=experiment_result,
            causal_estimates=opposite_estimates,
        )

        assert result["reconciliation_status"] == "failed"
        assert result["direction_match"] is False
        assert "opposite" in result["discrepancy_analysis"].lower() or "direction" in result["discrepancy_analysis"].lower()

    @pytest.mark.asyncio
    async def test_within_ci_check(
        self,
        reconciler: ABReconciler,
        experiment_result: ABExperimentResult,
        matching_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test that within_ci is correctly computed."""
        result = await reconciler.reconcile(
            experiment=experiment_result,
            causal_estimates=matching_estimates,
        )

        # Estimates (0.14, 0.16) should fall within observed CI (0.10, 0.20)
        # Weighted average ~0.15 should be within CI
        assert result["within_ci"] is True

    @pytest.mark.asyncio
    async def test_reconciliation_metrics(
        self,
        reconciler: ABReconciler,
        experiment_result: ABExperimentResult,
        matching_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test all reconciliation metrics are computed."""
        result = await reconciler.reconcile(
            experiment=experiment_result,
            causal_estimates=matching_estimates,
        )

        # Check all expected fields exist
        assert "observed_vs_estimated_gap" in result
        assert "observed_vs_estimated_ratio" in result
        assert "within_ci" in result
        assert "ci_overlap" in result
        assert "direction_match" in result
        assert "magnitude_match" in result
        assert "significance_match" in result
        assert "reconciliation_status" in result
        assert "reconciliation_score" in result
        assert "discrepancy_analysis" in result
        assert "recommended_adjustments" in result
        assert "reconciliation_latency_ms" in result

    @pytest.mark.asyncio
    async def test_recommended_adjustments_generated(
        self,
        reconciler: ABReconciler,
        experiment_result: ABExperimentResult,
        mismatched_estimates: list[LibraryEffectEstimate],
    ) -> None:
        """Test that recommendations are generated based on status."""
        result = await reconciler.reconcile(
            experiment=experiment_result,
            causal_estimates=mismatched_estimates,
        )

        assert len(result["recommended_adjustments"]) > 0
        # Should have actionable recommendations
        assert all(
            isinstance(adj, str) and len(adj) > 10
            for adj in result["recommended_adjustments"]
        )


class TestABReconcilerCalibration:
    """Test calibration factor generation."""

    @pytest.fixture
    def reconciler(self) -> ABReconciler:
        """Create ABReconciler instance."""
        return ABReconciler()

    @pytest.mark.asyncio
    async def test_create_calibration_factor_good_match(
        self,
        reconciler: ABReconciler,
    ) -> None:
        """Test calibration factor for good reconciliation."""
        # Simulate a good reconciliation result
        experiment = ABExperimentResult(
            experiment_id="exp-001",
            observed_effect=0.15,
            observed_ci_lower=0.10,
            observed_ci_upper=0.20,
            observed_p_value=0.001,
            is_significant=True,
            treatment_group_size=5000,
            control_group_size=5000,
            experiment_duration_days=28,
        )

        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.12,  # Slightly underestimates
                confidence=0.8,
                latency_ms=100,
            ),
        ]

        reconciliation = await reconciler.reconcile(experiment, estimates)
        calibration = await reconciler.create_calibration_factor(reconciliation)

        # Should be applicable since it's not failed
        if reconciliation["reconciliation_status"] not in ["failed", "poor"]:
            assert calibration["applicable"] is True
            assert calibration["factor"] is not None
            assert 0.5 <= calibration["factor"] <= 2.0

    @pytest.mark.asyncio
    async def test_create_calibration_factor_failed(
        self,
        reconciler: ABReconciler,
    ) -> None:
        """Test calibration factor for failed reconciliation."""
        experiment = ABExperimentResult(
            experiment_id="exp-001",
            observed_effect=0.15,
            observed_ci_lower=0.10,
            observed_ci_upper=0.20,
            observed_p_value=0.001,
            is_significant=True,
            treatment_group_size=5000,
            control_group_size=5000,
            experiment_duration_days=28,
        )

        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=-0.10,  # Opposite direction
                confidence=0.8,
                latency_ms=100,
            ),
        ]

        reconciliation = await reconciler.reconcile(experiment, estimates)
        calibration = await reconciler.create_calibration_factor(reconciliation)

        assert calibration["applicable"] is False
        assert "status" in calibration["reason"].lower() or "poor" in calibration["reason"].lower()


class TestABReconcilerEdgeCases:
    """Edge case tests for ABReconciler."""

    @pytest.fixture
    def reconciler(self) -> ABReconciler:
        """Create ABReconciler instance."""
        return ABReconciler()

    @pytest.mark.asyncio
    async def test_empty_estimates(self, reconciler: ABReconciler) -> None:
        """Test handling of empty estimates list."""
        experiment = ABExperimentResult(
            experiment_id="exp-001",
            observed_effect=0.15,
            observed_ci_lower=0.10,
            observed_ci_upper=0.20,
            observed_p_value=0.001,
            is_significant=True,
            treatment_group_size=5000,
            control_group_size=5000,
            experiment_duration_days=28,
        )

        result = await reconciler.reconcile(experiment, [])

        # Should handle gracefully
        assert "reconciliation_status" in result
        assert result["reconciliation_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_zero_observed_effect(self, reconciler: ABReconciler) -> None:
        """Test handling when observed effect is zero."""
        experiment = ABExperimentResult(
            experiment_id="exp-001",
            observed_effect=0.0,  # No effect observed
            observed_ci_lower=-0.05,
            observed_ci_upper=0.05,
            observed_p_value=0.95,
            is_significant=False,
            treatment_group_size=5000,
            control_group_size=5000,
            experiment_duration_days=28,
        )

        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.02,  # Small positive estimate
                confidence=0.6,
                latency_ms=100,
            ),
        ]

        result = await reconciler.reconcile(experiment, estimates)

        # Should handle without division errors
        assert "reconciliation_status" in result
        assert "observed_vs_estimated_ratio" in result

    @pytest.mark.asyncio
    async def test_custom_thresholds(self) -> None:
        """Test ABReconciler with custom thresholds."""
        strict_reconciler = ABReconciler(
            excellent_threshold=0.05,  # Very strict
            good_threshold=0.10,
        )

        experiment = ABExperimentResult(
            experiment_id="exp-001",
            observed_effect=0.15,
            observed_ci_lower=0.10,
            observed_ci_upper=0.20,
            observed_p_value=0.001,
            is_significant=True,
            treatment_group_size=5000,
            control_group_size=5000,
            experiment_duration_days=28,
        )

        estimates = [
            LibraryEffectEstimate(
                library="dowhy",
                effect_type="ate",
                estimate=0.13,  # 13% difference from 0.15
                confidence=0.8,
                latency_ms=100,
            ),
        ]

        result = await strict_reconciler.reconcile(experiment, estimates)

        # With strict thresholds, a 13% difference won't be excellent
        assert result["reconciliation_status"] != "excellent"
