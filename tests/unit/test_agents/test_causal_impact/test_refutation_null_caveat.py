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

    def test_a_non_numeric_treatment_yields_empty_inputs(self):
        frame = _frame()
        frame["arm_label"] = ["control", "treated"] * (len(frame) // 2)
        inputs = node_mod._sensitivity_benchmark_inputs(
            estimation_data=frame,
            treatment="arm_label",
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
