"""Tests for sensitivity analysis node.

Since the 2026-09-10 calibration the node owns no E-value math: it delegates every
number and word to ``src.causal_engine.evalue`` and reports a benchmarked READING
(spec §4.4/§4.6). The formula and interpretation tests that used to live here moved
to ``tests/unit/test_causal_engine/test_evalue.py``.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.sensitivity import SensitivityNode
from src.agents.causal_impact.state import CausalImpactState, EstimationResult


def _create_test_estimation(ate: float = 0.5) -> EstimationResult:
    """Test estimation result. The CI is ``ate ± 0.1`` so it always brackets the ATE
    (``evalue.classify`` raises on a CI that excludes the point estimate)."""
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


def _frame(n: int = 800, seed: int = 5) -> pd.DataFrame:
    """A frame with real confounding by ``disease_severity`` — the benchmark is measured
    on it, never faked."""
    rng = np.random.default_rng(seed)
    sev = rng.normal(5, 1, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * (sev - 5) - 0.3)))).astype(int)
    y = (rng.random(n) < 0.30 + 0.15 * t + 0.02 * (sev - 5)).astype(int)
    return pd.DataFrame({"treatment_arm": t, "treatment_initiated": y, "disease_severity": sev})


def _state_with_frame(ate, lo, hi, naive=None, **extra) -> CausalImpactState:
    est = _create_test_estimation(ate=ate)
    est.update(  # type: ignore[typeddict-item]
        {"ate_ci_lower": lo, "ate_ci_upper": hi, "covariates_adjusted": ["disease_severity"]}
    )
    if naive is not None:
        est["naive_ate"] = naive
    state: CausalImpactState = {
        "query": "q",
        "query_id": "q-1",
        "status": "pending",
        "estimation_result": est,
        "estimation_data": _frame(),
        "treatment_var": "treatment_arm",
        "outcome_var": "treatment_initiated",
    }
    state.update(extra)  # type: ignore[typeddict-item]
    return state


class TestSensitivityNode:
    """Test SensitivityNode."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        return _create_test_estimation(ate)

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

    @pytest.mark.asyncio
    async def test_beyond_reading_is_robust(self):
        result = await SensitivityNode().execute(_state_with_frame(0.15, 0.08, 0.22, naive=0.20))
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "beyond_measured_confounding"
        assert sens["robust_to_confounding"] is True
        assert sens["headline"] == "Robust to confounding at measured strength"
        assert sens["unmeasured_confounder_strength"] == "beyond_measured_confounding"
        assert sens["benchmark_basis"] == "joint_naive_vs_adjusted"
        assert sens["conversion"] == "risk_ratio"
        assert "stronger than all measured confounding" in sens["interpretation"]

    @pytest.mark.asyncio
    async def test_within_reading_is_not_robust(self):
        result = await SensitivityNode().execute(_state_with_frame(0.03, 0.01, 0.05, naive=0.12))
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "within_measured_confounding"
        assert sens["robust_to_confounding"] is False

    @pytest.mark.asyncio
    async def test_null_crossing_ci_is_a_null_finding(self):
        result = await SensitivityNode().execute(_state_with_frame(0.05, -0.02, 0.12, naive=0.08))
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "null_finding"
        assert sens["e_value_ci"] == 1.0
        assert sens["robust_to_confounding"] is False
        assert "includes zero" in sens["interpretation"]

    @pytest.mark.asyncio
    async def test_point_e_value_matches_the_shared_module_on_the_smd_path(self):
        """No frame → SMD path with the 0.91 factor: the node and the runner agree."""
        from src.causal_engine import evalue

        result = await SensitivityNode().execute(
            {
                "query": "q",
                "query_id": "q-2",
                "status": "pending",
                "estimation_result": self._create_test_estimation(ate=0.5),
            }
        )
        sens = result["sensitivity_analysis"]
        assert sens["e_value"] == pytest.approx(evalue.e_value_from_rr(evalue.rr_from_smd(0.5)))
        assert sens["reading"] == "unbenchmarked"
        assert sens["robust_to_confounding"] is False

    @pytest.mark.asyncio
    async def test_a_frame_without_the_outcome_column_reads_unbenchmarked(self):
        """MISSING is not a failure: no outcome column means nothing to benchmark
        against, so the run reads ``unbenchmarked`` rather than erroring (spec §4.3)."""
        state = _state_with_frame(0.15, 0.08, 0.22, naive=0.20)
        state["estimation_data"] = _frame().drop(columns=["treatment_initiated"])
        result = await SensitivityNode().execute(state)
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "unbenchmarked"
        assert sens["benchmark"] is None
        assert "sensitivity_error" not in result

    @pytest.mark.asyncio
    async def test_an_unusable_outcome_column_surfaces_as_sensitivity_error(self):
        """UNUSABLE is a failure: a PRESENT outcome column that cannot be converted to a
        number has no σ_Y, and a reading standardized by nothing would be
        plausible-wrong. It must surface as ``sensitivity_error`` (spec §5)."""
        state = _state_with_frame(0.15, 0.08, 0.22, naive=0.20)
        state["estimation_data"] = _frame().assign(treatment_initiated="yes")
        result = await SensitivityNode().execute(state)
        assert "sensitivity_error" in result
        assert result["status"] == "failed"
        assert "sensitivity_analysis" not in result

    @pytest.mark.asyncio
    async def test_a_ci_that_excludes_the_estimate_surfaces_as_sensitivity_error(self):
        """A computation failure is never dressed up as a reading (spec §5)."""
        estimation = _create_test_estimation(ate=0.5)
        estimation["ate_ci_lower"] = 0.6  # above the point estimate
        estimation["ate_ci_upper"] = 0.7
        result = await SensitivityNode().execute(
            {
                "query": "q",
                "query_id": "q-bad-ci",
                "status": "pending",
                "estimation_result": estimation,
            }
        )
        assert "sensitivity_error" in result
        assert result["status"] == "failed"
        assert "sensitivity_analysis" not in result


class TestSensitivityWithDifferentEffects:
    """Test sensitivity analysis with different effect sizes."""

    @pytest.mark.asyncio
    async def test_small_effect_sensitivity(self):
        """A small effect with NO frame: the E-value is small and, with nothing measured
        to benchmark it against, the run is unbenchmarked rather than robust."""
        from src.causal_engine import evalue

        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-8",
            "estimation_result": _create_test_estimation(ate=0.1),  # Small
            "status": "pending",
        }
        # ate ± 0.1 would touch zero at ate=0.1 (a null finding); keep the interval
        # strictly positive so this test is about the SMALL EFFECT, not the null.
        state["estimation_result"]["ate_ci_lower"] = 0.05
        state["estimation_result"]["ate_ci_upper"] = 0.15

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        # Small effects should have low E-value
        assert sens["e_value"] < 2.0
        assert sens["e_value"] == pytest.approx(evalue.e_value_from_rr(evalue.rr_from_smd(0.1)))
        assert sens["reading"] == "unbenchmarked"
        assert sens["robust_to_confounding"] is False

    @pytest.mark.asyncio
    async def test_large_effect_sensitivity(self):
        """A large effect with NO frame is still unbenchmarked: a big E-value on its own
        is not evidence of robustness — that is the defect this calibration fixed."""
        node = SensitivityNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-9",
            "estimation_result": _create_test_estimation(ate=0.8),  # Large
            "status": "pending",
        }

        result = await node.execute(state)

        sens = result["sensitivity_analysis"]

        # Large effects should have high E-value
        assert sens["e_value"] > 2.0
        assert sens["reading"] == "unbenchmarked"
        assert sens["robust_to_confounding"] is False

    @pytest.mark.asyncio
    async def test_large_effect_with_a_frame_is_beyond_measured_confounding(self):
        """The SAME large effect, benchmarked against the confounding the adjustment
        removed, earns the robust reading."""
        result = await SensitivityNode().execute(_state_with_frame(0.8, 0.7, 0.9, naive=0.20))
        sens = result["sensitivity_analysis"]
        assert sens["e_value"] > 2.0
        assert sens["reading"] == "beyond_measured_confounding"
        assert sens["robust_to_confounding"] is True
        assert sens["benchmark"] is not None


class TestOutcomeStdOnRealFrames:
    """``estimation_data`` is the RAW frame and the estimation node masks NaN rows
    before it fits, so a NaN outcome is an EXPECTED input — not a reason to turn a
    usable estimate into ``sensitivity_error``."""

    @pytest.mark.asyncio
    async def test_nan_rows_do_not_fail_the_node(self):
        from src.causal_engine import evalue

        frame = _frame()
        frame.loc[[1, 4, 9], "treatment_initiated"] = np.nan
        frame.loc[[2, 7], "treatment_arm"] = np.nan
        state = _state_with_frame(0.15, 0.08, 0.22, naive=0.20)
        state["estimation_data"] = frame

        result = await SensitivityNode().execute(state)

        assert "sensitivity_error" not in result
        sens = result["sensitivity_analysis"]
        masked_std = evalue.outcome_std_from_frame(
            frame, "treatment_initiated", treatment="treatment_arm"
        )
        inputs = evalue.benchmark_inputs_from_frame(
            frame,
            "treatment_arm",
            "treatment_initiated",
            ["disease_severity"],
            naive_effect=0.20,
        )
        expected = evalue.classify(
            0.15,
            (0.08, 0.22),
            randomized=False,
            baseline_risk=inputs.baseline_risk,
            outcome_std=masked_std,
            naive_effect=inputs.naive_effect,
            covariate_factors=inputs.covariate_bias_factors,
            n_rows=inputs.n_rows,
        )
        assert sens["e_value"] == pytest.approx(expected.e_value_point)
        assert sens["reading"] == expected.reading


class TestMeasuredButUnscoreableConfounders:
    """Live re-band 2026-09-10: 6 runs of ``peer_influence_score -> adopted`` declare
    one covariate (``centrality_z``, r = 0.9995 with the continuous treatment) that
    was measured and adjusted for but cannot be scored, and read ``unbenchmarked``.
    The node's state must carry the headline that says so, not "no measured
    confounders"."""

    @staticmethod
    def _collinear_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = rng.normal(0.0, 1.0, n)
        return pd.DataFrame(
            {
                "peer_influence_score": t,
                "adopted": (rng.random(n) < 0.30 + 0.10 * (t > 0)).astype(float),
                "centrality_z": t + 1e-9 * rng.normal(0.0, 1.0, n),
            }
        )

    @pytest.mark.asyncio
    async def test_state_carries_the_measured_unscoreable_headline(self):
        from src.causal_engine import evalue

        state = _state_with_frame(0.15, 0.08, 0.22)
        state["estimation_data"] = self._collinear_frame()
        state["treatment_var"] = "peer_influence_score"
        state["outcome_var"] = "adopted"
        state["estimation_result"]["covariates_adjusted"] = ["centrality_z"]

        result = await SensitivityNode().execute(state)

        assert "sensitivity_error" not in result
        sens = result["sensitivity_analysis"]
        assert sens["reading"] == "unbenchmarked"
        assert sens["benchmark_basis"] == "measured_unscoreable"
        assert sens["headline"] == evalue.HEADLINE_UNBENCHMARKED_MEASURED_UNSCOREABLE
        assert (
            "none of the 1 measured confounder(s) could be scored on this frame"
            in sens["interpretation"]
        )
        assert "no measured confounders exist" not in sens["interpretation"]
