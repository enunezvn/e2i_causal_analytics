"""Spec 2026-09-10 §4.3/§4.6: the refutation node hands the runner FULL-frame
benchmark inputs and records a null finding in ``warnings`` exactly once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes import refutation as node_mod
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)


def _frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = rng.normal(5, 1, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * (sev - 5) - 0.3)))).astype(int)
    y = (rng.random(n) < 0.30 + 0.10 * t + 0.02 * (sev - 5)).astype(int)
    return pd.DataFrame({"treatment_arm": t, "treatment_initiated": y, "disease_severity": sev})


class TestBenchmarkInputs:
    def test_inputs_are_computed_on_the_full_frame_with_the_backdoor_set(self):
        frame = _frame()
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="treatment_arm",
            outcome="treatment_initiated",
            estimation_result={
                "naive_ate": 0.123,
                "covariates_adjusted": ["disease_severity"],
                "baseline_covariates_adjusted": ["noise"],
            },
        )
        p0 = frame.loc[frame.treatment_arm == 0, "treatment_initiated"].mean()
        assert inputs.baseline_risk == pytest.approx(p0)
        assert inputs.naive_effect == 0.123  # the estimation node's contrast wins
        # efficiency controls excluded
        assert set(inputs.covariate_bias_factors) == {"disease_severity"}
        assert inputs.n_rows == len(frame)

    def test_missing_frame_yields_empty_inputs(self):
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=None, treatment="t", outcome="y", estimation_result={}
        )
        assert inputs.baseline_risk is None and inputs.naive_effect is None
        assert inputs.covariate_bias_factors == {}

    def test_missing_columns_yield_empty_inputs_not_an_error(self):
        frame = _frame()
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="not_a_column",
            outcome="treatment_initiated",
            estimation_result={},
        )
        assert inputs.baseline_risk is None and inputs.covariate_bias_factors == {}

    def test_a_raw_categorical_covariate_is_scored_not_dropped(self):
        """#1417/#1351: the live resolver binds string driver columns into the
        adjustment set and the estimator fits their one-hot encoding, so they ARE
        measured confounding. Dropping them would understate the fallback benchmark
        and let a run read 'beyond measured confounding' too easily."""
        frame = _frame()
        frame["trigger_type"] = np.where(
            frame.disease_severity > frame.disease_severity.median(),
            "dosing_gap",
            "adherence_risk",
        )
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="treatment_arm",
            outcome="treatment_initiated",
            estimation_result={
                "naive_ate": 0.123,
                "covariates_adjusted": ["disease_severity", "trigger_type"],
            },
        )
        assert set(inputs.covariate_bias_factors) == {"disease_severity", "trigger_type"}
        assert inputs.covariate_bias_factors["trigger_type"] > 1.0
        assert inputs.naive_effect == 0.123
        assert inputs.baseline_risk is not None

    def test_a_present_non_numeric_treatment_raises_like_the_runner(self):
        """Whole-diff review F1: a PRESENT string treatment is a data error, not a
        missing input. The runner's own fallback (``run_all_tests``) raises
        ``sensitivity_benchmark_failed`` on the same frame; reading it as empty
        inputs here would make the two engines disagree (spec §5)."""
        from src.causal_engine.errors import RefutationError

        frame = _frame()
        frame["arm_label"] = ["control", "treated"] * (len(frame) // 2)
        with pytest.raises(RefutationError) as ei:
            node_mod._sensitivity_benchmark_inputs(
                estimation_data=frame,
                treatment="arm_label",
                outcome="treatment_initiated",
                estimation_result={"covariates_adjusted": ["disease_severity"]},
            )
        assert ei.value.details["reason"] == "sensitivity_benchmark_failed"
        assert "arm_label" in ei.value.message
        assert "object" in ei.value.message

    def test_a_present_non_numeric_outcome_raises_like_the_runner(self):
        from src.causal_engine.errors import RefutationError

        frame = _frame().assign(treatment_initiated="yes")
        with pytest.raises(RefutationError) as ei:
            node_mod._sensitivity_benchmark_inputs(
                estimation_data=frame,
                treatment="treatment_arm",
                outcome="treatment_initiated",
                estimation_result={"covariates_adjusted": ["disease_severity"]},
            )
        assert ei.value.details["reason"] == "sensitivity_benchmark_failed"
        assert "treatment_initiated" in ei.value.message

    def test_a_bool_treatment_is_numeric_and_benchmarked(self):
        """``is_numeric_dtype(bool)`` is True: a bool T/Y must keep being benchmarked."""
        frame = _frame()
        frame["treatment_arm"] = frame["treatment_arm"].astype(bool)
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="treatment_arm",
            outcome="treatment_initiated",
            estimation_result={"covariates_adjusted": ["disease_severity"]},
        )
        assert inputs.baseline_risk is not None
        assert set(inputs.covariate_bias_factors) == {"disease_severity"}

    def test_an_absent_treatment_column_still_yields_empty_inputs(self):
        """The MISSING fallback is kept: no column, nothing to benchmark, no error."""
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=_frame().drop(columns=["treatment_arm"]),
            treatment="treatment_arm",
            outcome="treatment_initiated",
            estimation_result={"covariates_adjusted": ["disease_severity"]},
        )
        assert inputs.baseline_risk is None and inputs.naive_effect is None
        assert inputs.covariate_bias_factors == {}

    def test_n_rows_is_none_when_missing_and_zero_when_computed(self):
        """The node hands ``n_rows`` to the runner verbatim. ``None`` means "no frame
        was looked at" and lets the runner fall back to ``len(data)`` — the refutation
        SUBSAMPLE. A frame that yielded no usable rows must report the computed 0
        instead, or the reading prints the subsample's count for it."""
        missing = node_mod._sensitivity_benchmark_inputs(
            estimation_data=None, treatment="t", outcome="y", estimation_result={}
        )
        assert missing.n_rows is None
        frame = _frame().assign(treatment_arm=np.nan)
        computed = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="treatment_arm",
            outcome="treatment_initiated",
            estimation_result={},
        )
        assert computed.n_rows == 0

    def test_a_computation_failure_raises_refutation_error_with_a_reason(self):
        """A covariate that perfectly separates treatment and outcome makes evalue raise
        ValueError (positivity violation); the node must not fail open to 'unbenchmarked'."""
        from src.causal_engine.errors import RefutationError

        frame = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 0, 0, 0, 0],
                "y": [1, 0, 1, 0, 1, 1, 0, 0],
                "c": [0, 0, 0, 0, 1, 1, 0, 0],
            }
        )
        with pytest.raises(RefutationError) as ei:
            node_mod._sensitivity_benchmark_inputs(
                estimation_data=frame,
                treatment="t",
                outcome="y",
                estimation_result={"covariates_adjusted": ["c"]},
            )
        assert ei.value.details["reason"] == "sensitivity_benchmark_failed"


class TestRefutationNodeSurfacesBenchmarkFailure:
    async def test_a_string_treatment_fails_the_run_closed_with_the_benchmark_reason(self):
        """Whole-diff review F1: the refutation node reaches the shared helper before
        DoWhy reconstruction; a PRESENT string treatment must fail the run closed
        with ``sensitivity_benchmark_failed`` (the node's RefutationError path),
        not read ``unbenchmarked``."""
        frame = _frame()
        frame["treatment_arm"] = np.where(frame["treatment_arm"] == 1, "treated", "control")
        state = {
            "query": "q",
            "query_id": "q-string-t",
            "status": "pending",
            "treatment_var": "treatment_arm",
            "outcome_var": "treatment_initiated",
            "estimation_result": {
                "method": "CausalForestDML",
                "ate": 0.10,
                "ate_ci_lower": 0.05,
                "ate_ci_upper": 0.15,
                "p_value": 0.01,
                "sample_size": len(frame),
                "covariates_adjusted": ["disease_severity"],
            },
            "estimation_data": frame,
        }
        result = await node_mod.RefutationNode().execute(state)  # type: ignore[arg-type]
        assert result["status"] == "failed"
        assert result["refutation_error_details"]["reason"] == "sensitivity_benchmark_failed"
        assert "treatment_arm" in result["refutation_error"]


class TestNullCaveat:
    def _suite(self, reading: str) -> RefutationSuite:
        sens = RefutationResult(
            RefutationTestType.SENSITIVITY_E_VALUE,
            RefutationStatus.WARNING,
            0.01,
            0.01,
            details={
                "reading": reading,
                "message": "The 95 % CI [-0.020, 0.040] includes zero at n = 600. "
                "The estimate is reported as a null finding; no unmeasured confounder "
                "is needed to explain it.",
            },
        )
        ok = [
            RefutationResult(n, RefutationStatus.PASSED, 0.01, 0.01)
            for n in (
                RefutationTestType.PLACEBO_TREATMENT,
                RefutationTestType.RANDOM_COMMON_CAUSE,
                RefutationTestType.DATA_SUBSET,
                RefutationTestType.BOOTSTRAP,
            )
        ]
        return RefutationSuite(
            passed=True,
            confidence_score=0.9,
            tests=[*ok, sens],
            gate_decision=GateDecision.PROCEED,
        )

    def test_null_finding_message_is_returned_once_as_a_new_warning(self):
        warnings = node_mod._sensitivity_caveat_warnings(self._suite("null_finding"))
        assert len(warnings) == 1
        assert warnings[0].startswith("No detectable effect at this sample size: ")
        assert "includes zero" in warnings[0]

    def test_other_readings_add_no_warning(self):
        for reading in (
            "beyond_measured_confounding",
            "within_measured_confounding",
            "not_applicable_randomized",
        ):
            assert node_mod._sensitivity_caveat_warnings(self._suite(reading)) == []


class TestOutcomeStdFull:
    """The node's σ_Y for the FULL frame. NaN treatment/outcome rows are dropped, as
    the estimation node does before it fits — the raw passthrough frame carries them."""

    def test_the_sd_is_measured_on_the_masked_rows(self):
        frame = _frame()
        frame.loc[[0, 5], "treatment_initiated"] = np.nan
        frame.loc[[3], "treatment_arm"] = np.nan
        y = frame["treatment_initiated"].to_numpy(dtype=float)
        t = frame["treatment_arm"].to_numpy(dtype=float)
        ok = ~np.isnan(y) & ~np.isnan(t)
        assert node_mod._outcome_std_full(frame, "treatment_arm", "treatment_initiated") == (
            pytest.approx(float(np.std(y[ok])))
        )

    def test_an_absent_column_is_none_not_a_failure(self):
        frame = _frame()
        assert node_mod._outcome_std_full(frame, "treatment_arm", "not_a_column") is None
        assert node_mod._outcome_std_full(None, "treatment_arm", "treatment_initiated") is None
