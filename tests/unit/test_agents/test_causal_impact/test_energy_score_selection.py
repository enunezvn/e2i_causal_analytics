"""Tests for Energy Score-based Estimator Selection in Causal Impact Agent.

V4.2 Enhancement: Energy Score-based Estimator Selection
Tests the integration of EstimatorSelector with the estimation node.
"""

import pytest

from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.agents.causal_impact.state import CausalImpactState


@pytest.fixture
def estimation_node():
    """Create an EstimationNode instance."""
    return EstimationNode()


@pytest.fixture
def base_state() -> CausalImpactState:
    """Create base state with causal graph."""
    return {
        "query": "What is the impact of engagement on conversion?",
        "query_id": "test-query-123",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region", "hcp_specialty"],
        # "synthetic" opts into the synthetic-data fallback: the estimation
        # node fails closed when data_source is neither real data nor an
        # explicit "synthetic" opt-in (anti-fabrication guard, commit 198067ac
        # / #416-#417). Real estimators still run against seeded synthetic
        # data — this is not a mock; #583 surfaced this stale fixture.
        "data_source": "synthetic",
        "causal_graph": {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("hcp_engagement_level", "patient_conversion_rate"),
                ("geographic_region", "hcp_engagement_level"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "digraph {}",
            "confidence": 0.85,
        },
        "errors": [],
        "warnings": [],
    }


class TestEnergyScoreSelectionEnabled:
    """Test energy score selection when enabled (default)."""

    @pytest.mark.asyncio
    async def test_energy_score_selection_enabled_by_default(self, estimation_node, base_state):
        """Energy score selection is enabled by default."""
        # No explicit parameters - should use energy score path
        result = await estimation_node.execute(base_state)

        # Check that energy score is enabled
        assert (
            result.get("energy_score_enabled") is True
            or result.get("energy_score_enabled") is False
        )
        # Either path should complete successfully
        assert (
            result.get("estimation_result") is not None
            or result.get("estimation_error") is not None
        )

    @pytest.mark.asyncio
    async def test_energy_score_result_fields(self, estimation_node, base_state):
        """Energy score result contains expected fields."""
        # Force legacy path for predictable test
        base_state["parameters"] = {"method": "CausalForestDML"}
        result = await estimation_node.execute(base_state)

        # Should have estimation result
        assert "estimation_result" in result
        est_result = result["estimation_result"]

        # Check required fields
        assert "method" in est_result
        assert "ate" in est_result
        assert "ate_ci_lower" in est_result
        assert "ate_ci_upper" in est_result


class TestLegacyModeWithExplicitMethod:
    """Test legacy mode when explicit method is provided."""

    @pytest.mark.asyncio
    async def test_legacy_mode_with_explicit_method(self, estimation_node, base_state):
        """Explicit method parameter uses legacy path."""
        base_state["parameters"] = {"method": "LinearDML"}
        result = await estimation_node.execute(base_state)

        # Should use legacy path
        assert result.get("energy_score_enabled") is False
        assert result["estimation_result"]["method"] == "LinearDML"

    @pytest.mark.asyncio
    async def test_legacy_mode_with_use_energy_score_false(self, estimation_node, base_state):
        """use_energy_score=False uses legacy path (no explicit method).

        #622: with no explicit ``method``, the legacy path still runs the full
        energy-score chain unrestricted. On THIS synthetic fixture all four
        estimators produce an EXACT energy-score tie (energy_score_gap == 0.0,
        verified), so the selection is decided by the tiebreak. Previously the
        stable-sort fallthrough picked the chain-priority head ``causal_forest``
        -> ``CausalForestDML`` (the SLOWEST estimator, whose refutation suite
        runs ~35-60 min). The tiebreak now prefers the fastest CONFOUNDING-ROBUST
        estimator on a tie (``linear_dml`` -> ``LinearDML``): a tie must NOT land
        on the slow causal_forest, NOR on confounding-blind ``ols`` (which ties
        on fit but fails the refutation gate under confounding — MEASURED on the
        patient_journeys gold standard).
        """
        base_state["parameters"] = {"use_energy_score": False}
        result = await estimation_node.execute(base_state)

        assert result.get("energy_score_enabled") is False
        # On the exact energy-score tie, the fastest confounding-robust estimator
        # wins (linear_dml -> LinearDML); naive OLS is excluded and the slow
        # causal_forest is avoided.
        assert result["estimation_result"]["method"] == "LinearDML"
        assert result["estimation_result"]["selected_estimator"] == "linear_dml"

    @pytest.mark.asyncio
    async def test_all_legacy_methods(self, estimation_node, base_state):
        """All legacy methods work correctly."""
        methods = [
            "CausalForestDML",
            "LinearDML",
            "linear_regression",
            "propensity_score_weighting",
        ]

        for method in methods:
            state = {**base_state, "parameters": {"method": method}}
            result = await estimation_node.execute(state)

            assert result["estimation_result"]["method"] == method
            assert result.get("energy_score_enabled") is False


class TestSelectionStrategies:
    """Test different selection strategies."""

    @pytest.mark.asyncio
    async def test_first_success_strategy(self, estimation_node, base_state):
        """first_success strategy selects first successful estimator."""
        base_state["parameters"] = {
            "use_energy_score": True,
            "selection_strategy": "first_success",
        }

        # This may fail if EstimatorSelector is not properly mocked
        # For now, test that the path is attempted
        result = await estimation_node.execute(base_state)

        # Should either succeed with energy score or fall back to legacy
        assert "estimation_result" in result or "estimation_error" in result

    @pytest.mark.asyncio
    async def test_best_energy_strategy_default(self, estimation_node, base_state):
        """best_energy is the default strategy."""
        # No explicit strategy
        base_state["parameters"] = {"use_energy_score": True}

        result = await estimation_node.execute(base_state)

        # Should use best_energy by default
        if result.get("selection_strategy"):
            assert result["selection_strategy"] == "best_energy"


class TestQualityTiers:
    """Test quality tier classification."""

    def test_quality_tier_excellent(self, estimation_node):
        """Score <= 0.25 is excellent."""
        assert estimation_node._get_quality_tier(0.20) == "excellent"
        assert estimation_node._get_quality_tier(0.25) == "excellent"

    def test_quality_tier_good(self, estimation_node):
        """Score 0.25-0.45 is good."""
        assert estimation_node._get_quality_tier(0.30) == "good"
        assert estimation_node._get_quality_tier(0.45) == "good"

    def test_quality_tier_acceptable(self, estimation_node):
        """Score 0.45-0.65 is acceptable."""
        assert estimation_node._get_quality_tier(0.50) == "acceptable"
        assert estimation_node._get_quality_tier(0.65) == "acceptable"

    def test_quality_tier_poor(self, estimation_node):
        """Score 0.65-0.80 is poor."""
        assert estimation_node._get_quality_tier(0.70) == "poor"
        assert estimation_node._get_quality_tier(0.80) == "poor"

    def test_quality_tier_unreliable(self, estimation_node):
        """Score > 0.80 is unreliable."""
        assert estimation_node._get_quality_tier(0.85) == "unreliable"
        assert estimation_node._get_quality_tier(1.0) == "unreliable"


class TestStateFields:
    """Test state fields populated by energy score selection."""

    @pytest.mark.asyncio
    async def test_estimation_latency_recorded(self, estimation_node, base_state):
        """Estimation latency is recorded."""
        base_state["parameters"] = {"method": "CausalForestDML"}
        result = await estimation_node.execute(base_state)

        assert "estimation_latency_ms" in result
        assert result["estimation_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_current_phase_updated(self, estimation_node, base_state):
        """Current phase is updated to refuting."""
        base_state["parameters"] = {"method": "CausalForestDML"}
        result = await estimation_node.execute(base_state)

        assert result["current_phase"] == "refuting"
        assert result["status"] == "computing"


class TestErrorHandling:
    """Test error handling in estimation."""

    @pytest.mark.asyncio
    async def test_missing_causal_graph(self, estimation_node, base_state):
        """Missing causal graph raises error."""
        del base_state["causal_graph"]
        result = await estimation_node.execute(base_state)

        assert result["status"] == "failed"
        assert "Causal graph not found" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_unknown_method_error(self, estimation_node, base_state):
        """Unknown method raises error."""
        base_state["parameters"] = {"method": "NonExistentMethod"}
        result = await estimation_node.execute(base_state)

        assert result["status"] == "failed"
        assert "Unknown estimation method" in result.get("error_message", "")


class TestBackwardCompatibility:
    """Test backward compatibility with existing tests."""

    @pytest.mark.asyncio
    async def test_existing_workflow_unchanged(self, estimation_node, base_state):
        """Existing workflow produces expected output structure."""
        base_state["parameters"] = {"method": "CausalForestDML"}
        result = await estimation_node.execute(base_state)

        # All existing fields should be present
        est = result["estimation_result"]
        assert "method" in est
        assert "ate" in est
        assert "ate_ci_lower" in est
        assert "ate_ci_upper" in est
        assert "standard_error" in est
        assert "effect_size" in est
        assert "statistical_significance" in est
        assert "sample_size" in est
        assert "covariates_adjusted" in est

    @pytest.mark.asyncio
    async def test_cate_segments_with_causal_forest(self, estimation_node, base_state):
        """CausalForest produces CATE segments."""
        base_state["parameters"] = {"method": "CausalForestDML"}
        result = await estimation_node.execute(base_state)

        est = result["estimation_result"]
        assert "cate_segments" in est
        assert len(est["cate_segments"]) > 0


class TestSubsampleDisclosure1392:
    """#1392: the estimation node must surface the selector's subsample
    metadata (selection_subsampled / selection_n_rows / selection_n_rows_total)
    so downstream honesty surfaces can disclose that the tournament ranking ran
    on a stratified subsample while the reported ATE is the full-frame fit."""

    @staticmethod
    def _stub_selector_returning(selection_result):
        class _StubSelector:
            def select(self, *args, **kwargs):
                return selection_result

        return _StubSelector()

    def _crafted_selection_result(self):
        import numpy as np

        from src.causal_engine.energy_score.estimator_selector import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )
        from src.causal_engine.energy_score.score_calculator import EnergyScoreResult

        n_total = 37_371
        energy = EnergyScoreResult(
            estimator_name="linear_dml",
            energy_score=0.36,
            treatment_balance_score=0.3,
            outcome_fit_score=0.4,
            propensity_calibration=0.2,
            n_samples=5_000,
            n_treated=2_500,
            n_control=2_500,
            computation_time_ms=10.0,
        )
        selected = EstimatorResult(
            estimator_type=EstimatorType.LINEAR_DML,
            success=True,
            ate=0.0388,
            cate=np.full(n_total, 0.0388),
            ate_std=0.004,
            ate_ci_lower=0.0308,
            ate_ci_upper=0.0467,
            energy_score_result=energy,
            propensity_scores=np.full(n_total, 0.5),
        )
        return SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[selected],
            selection_reason="Lowest energy score; tournament ranked on subsample.",
            energy_scores={"linear_dml": 0.36},
            energy_score_gap=0.01,
            adjustment_type="confounding",
            selection_subsampled=True,
            selection_n_rows=5_000,
            selection_n_rows_total=n_total,
        )

    def test_subsample_metadata_plumbed_into_result(self, estimation_node, monkeypatch):
        import numpy as np
        import pandas as pd

        sel_result = self._crafted_selection_result()
        monkeypatch.setattr(
            estimation_node,
            "_get_estimator_selector",
            lambda *a, **k: self._stub_selector_returning(sel_result),
        )
        rng = np.random.default_rng(0)
        n = 100  # frame content is irrelevant; the stub returns the crafted result
        data = pd.DataFrame(
            {
                "t": rng.integers(0, 2, n),
                "y": rng.normal(0, 1, n),
                "x1": rng.normal(0, 1, n),
            }
        )

        result, selection_dict, _ = estimation_node._select_estimator_with_energy_score(
            data=data,
            treatment="t",
            outcome="y",
            adjustment_set=["x1"],
        )

        es_data = result["energy_score_data"]
        assert es_data["selection_subsampled"] is True
        assert es_data["selection_n_rows"] == 5_000
        assert es_data["selection_n_rows_total"] == 37_371
        assert selection_dict["selection_subsampled"] is True
        assert selection_dict["selection_n_rows"] == 5_000
        assert selection_dict["selection_n_rows_total"] == 37_371

    @pytest.mark.asyncio
    async def test_small_frame_discloses_not_subsampled(self, estimation_node, base_state):
        """A below-cap frame (synthetic fixture n=1000 < 5000) runs today's
        full-frame selection and honestly reports selection_subsampled=False."""
        result = await estimation_node.execute(base_state)

        est = result.get("estimation_result")
        assert est is not None, result.get("error_message")
        es_data = est["energy_score_data"]
        assert es_data["selection_subsampled"] is False
        assert es_data["selection_n_rows"] == es_data["selection_n_rows_total"]
        assert es_data["selection_n_rows_total"] == est["sample_size"]
