"""#2007 T4: the refutation node fits the declared negative-control outcome on the
SAME reconstructed DoWhy model and hands the runner its ``(outcome, effect,
(lo, hi), n)`` tuple -- or one of the runner's closed skip reasons -- never a
fabricated PASSED, and never at the cost of the primary analysis.

Interval source (measured 2026-09-11, ``docs/demos/results/
2026-09-11_negative_control_disproof/ci_availability.md``): DoWhy's
``get_confidence_intervals()`` is NaN for all three econml production methods and
bootstraps for 20-94 s on the two backdoor ones, so the node reads the fitted
econml estimator through DoWhy's OWN encoded effect-modifier frame
(``ate_inference``) and, for ``backdoor.linear_regression``, a delta-method
contrast on DoWhy's own interventional feature difference. Both reproduce
``estimate.value`` exactly; the node refuses any interval whose point does not.

Every fit here is real (DoWhy + econml on a 300-row synthetic frame, ~1 s); the
only patched seams are the ones the existing node tests already patch
(``_reconstruct_dowhy_artifacts`` / ``_fit_negative_control`` at module level and
``node.runner.run_all_tests``).
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes import refutation as _ref_mod
from src.agents.causal_impact.nodes._compute_budget import ComputeBudgetExpired
from src.agents.causal_impact.nodes.refutation import (
    NEGATIVE_CONTROL_MIN_ROWS,
    RefutationNode,
    _build_dowhy_estimate,
    _fit_negative_control,
    _negative_control_interval,
)
from src.causal_engine.errors import RefutationError
from src.causal_engine.refutation_runner import (
    NEGATIVE_CONTROL_SKIP_REASONS,
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from src.repositories.json_utils import to_plain_json

N = 300
TREATMENT = "copay_support"
OUTCOME = "treatment_initiated"
NC = "sample_dropped"
COVS = ["insurance_access_score", "disease_severity"]
N_NC_NULL = 10  # rows whose control value is NULL (the loader keeps them as NaN)


def _frames(n: int = N, seed: int = 2007) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(estimation_data, negative_control_data) sharing one RangeIndex, the shape
    the API task hands the node (T3): binary treatment and outcome, two
    confounders, and a control that shares the confounders but has no treatment
    arrow. The control carries ``N_NC_NULL`` NaNs -- they drop nothing upstream."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(5.0, 1.5, n)
    x2 = rng.uniform(0.0, 1.0, n)
    t = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.4 * (x1 - 5.0) + 1.0 * (x2 - 0.5)))))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.5 + 0.3 * (x1 - 5.0) + 0.8 * x2 + 0.6 * t))))
    nc = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.3 + 0.3 * (x1 - 5.0) + 0.5 * x2)))).astype(float)
    nc[rng.choice(n, size=N_NC_NULL, replace=False)] = np.nan
    frame = pd.DataFrame({TREATMENT: t, OUTCOME: y, COVS[0]: x1, COVS[1]: x2})
    return frame, pd.DataFrame({NC: nc}, index=frame.index)


LINEAR_DML = {"method": "LinearDML", "selected_estimator": "linear_dml"}
OLS = {"method": "linear_regression", "selected_estimator": "ols"}


@pytest.fixture(scope="module")
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _frames()


@pytest.fixture(scope="module")
def linear_dml_fit(frames):
    frame, _ = frames
    return _build_dowhy_estimate(
        data=frame,
        treatment=TREATMENT,
        outcome=OUTCOME,
        common_causes=COVS,
        estimation_result=dict(LINEAR_DML),
    )


# --------------------------------------------------------------- (a) interval
class TestNegativeControlInterval:
    def test_linear_dml_interval_reproduces_the_estimate_value(self, linear_dml_fit, monkeypatch):
        _, _, estimate, method = linear_dml_fit
        assert method == "backdoor.econml.dml.LinearDML"

        def _never(*_a, **_k):
            raise AssertionError("get_confidence_intervals must never be called (NaN + slow)")

        monkeypatch.setattr(estimate, "get_confidence_intervals", _never, raising=False)
        result = _negative_control_interval(estimate, method)
        assert result is not None
        mean, (lo, hi) = result
        assert mean == pytest.approx(float(estimate.value), abs=1e-9)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < mean < hi

    def test_tampered_value_is_refused(self, linear_dml_fit, monkeypatch):
        """The interval is only trusted when its point IS the estimate's value:
        a frame in the wrong column order gave -1.16 vs +0.03 on a live pair."""
        _, _, estimate, method = linear_dml_fit
        monkeypatch.setattr(estimate, "value", float(estimate.value) + 0.5)
        assert _negative_control_interval(estimate, method) is None

    def test_linear_regression_with_modifiers_uses_the_contrast_not_the_treatment_row(self, frames):
        """The reconstruction passes the common causes as effect modifiers, so
        DoWhy's OLS carries T*X interaction terms and ``estimate.value`` is
        b_T + sum(b_TX * mean(X)) -- NOT the treatment coefficient. Measured
        2026-09-11 on the seed-21 frame: params[1] = -0.123 vs value = +0.003."""
        frame, _ = frames
        _, _, estimate, method = _build_dowhy_estimate(
            data=frame,
            treatment=TREATMENT,
            outcome=OUTCOME,
            common_causes=COVS,
            estimation_result=dict(OLS),
        )
        assert method == "backdoor.linear_regression"
        sm_result = estimate.estimator.model
        assert abs(float(sm_result.params.iloc[1]) - float(estimate.value)) > 1e-3
        result = _negative_control_interval(estimate, method)
        assert result is not None
        mean, (lo, hi) = result
        assert mean == pytest.approx(float(estimate.value), abs=1e-9)
        assert lo < mean < hi

    def test_linear_regression_without_modifiers_equals_the_conf_int_row(self, frames):
        frame, _ = frames
        _, _, estimate, method = _build_dowhy_estimate(
            data=frame,
            treatment=TREATMENT,
            outcome=OUTCOME,
            common_causes=[],
            estimation_result=dict(OLS),
        )
        result = _negative_control_interval(estimate, method)
        assert result is not None
        mean, (lo, hi) = result
        row = estimate.estimator.model.conf_int(alpha=0.05).to_numpy()[1]
        assert mean == pytest.approx(float(estimate.value), abs=1e-9)
        assert (lo, hi) == pytest.approx((float(row[0]), float(row[1])), abs=1e-9)

    def test_ipw_and_unknown_methods_give_no_interval(self):
        fake = SimpleNamespace(value=0.1)
        assert _negative_control_interval(fake, "backdoor.propensity_score_weighting") is None
        assert _negative_control_interval(fake, "backdoor.something_else") is None


# --------------------------------------------------------------- (b) the fit
def _fit_kwargs(frames, **overrides):
    frame, nc_df = frames
    kwargs = {
        "refutation_data": frame,
        "negative_control_data": nc_df,
        "treatment": TREATMENT,
        "nc_outcome": NC,
        "common_causes": COVS,
        "estimation_result": dict(LINEAR_DML),
        "deadline": None,
    }
    kwargs.update(overrides)
    return kwargs


class TestFitNegativeControl:
    @pytest.mark.asyncio
    async def test_no_frame_is_column_missing(self, frames):
        result = await _fit_negative_control(**_fit_kwargs(frames, negative_control_data=None))
        assert result == (None, "negative_control_column_missing")

    @pytest.mark.asyncio
    async def test_absent_column_is_column_missing(self, frames):
        _, nc_df = frames
        result = await _fit_negative_control(
            **_fit_kwargs(frames, negative_control_data=nc_df.rename(columns={NC: "other"}))
        )
        assert result == (None, "negative_control_column_missing")

    @pytest.mark.asyncio
    async def test_misaligned_index_is_ci_unavailable_never_a_silent_reindex(self, frames, caplog):
        _, nc_df = frames
        shifted = nc_df.set_index(nc_df.index + 1000)
        with caplog.at_level(logging.WARNING, logger=_ref_mod.logger.name):
            result = await _fit_negative_control(
                **_fit_kwargs(frames, negative_control_data=shifted)
            )
        assert result == (None, "negative_control_ci_unavailable")
        assert any("index" in rec.getMessage().lower() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_too_few_rows_before_any_fit(self, frames, monkeypatch):
        frame, _ = frames
        monkeypatch.setattr(
            _ref_mod, "_build_dowhy_estimate", lambda **_k: pytest.fail("must not fit")
        )
        result = await _fit_negative_control(
            **_fit_kwargs(frames, refutation_data=frame.iloc[: NEGATIVE_CONTROL_MIN_ROWS - 1])
        )
        assert result == (None, "negative_control_too_few_rows")

    @pytest.mark.asyncio
    async def test_null_heavy_control_counts_only_non_null_rows(self, frames, monkeypatch):
        _, nc_df = frames
        sparse = nc_df.copy()
        sparse.iloc[NEGATIVE_CONTROL_MIN_ROWS - 1 :, 0] = np.nan
        monkeypatch.setattr(
            _ref_mod, "_build_dowhy_estimate", lambda **_k: pytest.fail("must not fit")
        )
        result = await _fit_negative_control(**_fit_kwargs(frames, negative_control_data=sparse))
        assert result == (None, "negative_control_too_few_rows")

    @pytest.mark.asyncio
    async def test_constant_control_is_too_few_rows(self, frames, monkeypatch):
        _, nc_df = frames
        constant = nc_df.copy()
        constant[NC] = 1.0
        monkeypatch.setattr(
            _ref_mod, "_build_dowhy_estimate", lambda **_k: pytest.fail("must not fit")
        )
        result = await _fit_negative_control(**_fit_kwargs(frames, negative_control_data=constant))
        assert result == (None, "negative_control_too_few_rows")

    @pytest.mark.asyncio
    async def test_happy_path_returns_the_interval_on_the_non_null_rows(self, frames):
        frame, _ = frames
        tuple_, reason = await _fit_negative_control(**_fit_kwargs(frames))
        assert reason is None
        assert tuple_ is not None
        nc_outcome, effect, (lo, hi), nc_n = tuple_
        assert nc_outcome == NC
        assert nc_n == len(frame) - N_NC_NULL
        assert isinstance(nc_n, int)
        assert np.isfinite(effect) and np.isfinite(lo) and np.isfinite(hi)
        assert lo <= effect <= hi

    @pytest.mark.asyncio
    async def test_positional_subsample_aligns_by_label(self, frames):
        """The #1419 subsample is ``frame.iloc[indices]`` -- the labels survive,
        so the control must be aligned with ``.loc`` on them."""
        frame, _ = frames
        sub = frame.iloc[::2]
        tuple_, reason = await _fit_negative_control(**_fit_kwargs(frames, refutation_data=sub))
        assert reason is None
        assert tuple_ is not None
        expected_n = int(frames[1].loc[sub.index, NC].notna().sum())
        assert tuple_[3] == expected_n

    @pytest.mark.asyncio
    async def test_no_interval_is_ci_unavailable(self, frames, monkeypatch):
        monkeypatch.setattr(_ref_mod, "_negative_control_interval", lambda *_a, **_k: None)
        result = await _fit_negative_control(**_fit_kwargs(frames))
        assert result == (None, "negative_control_ci_unavailable")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("econml exploded on the control"),
            RefutationError("build failed", details={"reason": "dowhy_reconstruction_failed"}),
        ],
    )
    async def test_a_dowhy_failure_never_fails_the_primary_analysis(
        self, frames, monkeypatch, caplog, exc
    ):
        def _boom(**_k):
            raise exc

        monkeypatch.setattr(_ref_mod, "_build_dowhy_estimate", _boom)
        with caplog.at_level(logging.WARNING, logger=_ref_mod.logger.name):
            result = await _fit_negative_control(**_fit_kwargs(frames))
        assert result == (None, "negative_control_ci_unavailable")
        assert any(str(exc) in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_compute_budget_expiry_propagates(self, frames):
        with pytest.raises(ComputeBudgetExpired):
            await _fit_negative_control(**_fit_kwargs(frames, deadline=time.monotonic() - 1.0))

    @pytest.mark.asyncio
    async def test_every_reason_is_a_runner_token(self, frames, monkeypatch):
        _, nc_df = frames
        reasons = set()
        reasons.add(
            (await _fit_negative_control(**_fit_kwargs(frames, negative_control_data=None)))[1]
        )
        reasons.add(
            (
                await _fit_negative_control(
                    **_fit_kwargs(frames, negative_control_data=nc_df.set_index(nc_df.index + 5))
                )
            )[1]
        )
        constant = nc_df.copy()
        constant[NC] = 0.0
        reasons.add(
            (await _fit_negative_control(**_fit_kwargs(frames, negative_control_data=constant)))[1]
        )
        assert reasons <= NEGATIVE_CONTROL_SKIP_REASONS
        assert len(reasons) == 3


# --------------------------------------------------------------- (c) wiring
def _proceed_suite() -> RefutationSuite:
    return RefutationSuite(
        passed=True,
        confidence_score=1.0,
        tests=[
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.05,
                refuted_effect=0.001,
                p_value=0.9,
            )
        ],
        gate_decision=GateDecision.PROCEED,
    )


def _node_state(frame: pd.DataFrame, ate: float = 0.05, **overrides) -> dict:
    state = {
        "query": "negative control plumbing",
        "query_id": "",  # no persistence path
        "treatment_var": TREATMENT,
        "outcome_var": OUTCOME,
        "confounders": list(COVS),
        "data_source": "synthetic",
        "estimation_result": {
            **LINEAR_DML,
            "ate": ate,
            "ate_ci_lower": ate - 0.05,
            "ate_ci_upper": ate + 0.05,
            "effect_size": "small",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": len(frame),
            "covariates_adjusted": list(COVS),
            "heterogeneity_detected": False,
        },
        "estimation_data": frame,
        "status": "pending",
        "errors": [],
        "warnings": [],
    }
    state.update(overrides)
    return state


def _wire(node: RefutationNode, monkeypatch, nc_return):
    seen: dict = {}

    def fake_recon(**kwargs):
        seen["recon"] = kwargs
        return (SimpleNamespace(), object(), object())

    async def fake_fit(**kwargs):
        seen["fit"] = kwargs
        return nc_return

    def spy_run_all_tests(**kwargs):
        seen["runner"] = kwargs
        return _proceed_suite()

    async def _no_signal(outcome):
        return None

    monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
    monkeypatch.setattr(_ref_mod, "_fit_negative_control", fake_fit)
    monkeypatch.setattr(node.runner, "run_all_tests", spy_run_all_tests)
    monkeypatch.setattr(node, "_log_validation_outcome_signal", _no_signal)
    return seen


class TestExecuteWiring:
    @pytest.mark.asyncio
    async def test_nc_fit_uses_the_primary_inputs_and_the_runner_gets_the_tuple(
        self, frames, monkeypatch
    ):
        frame, nc_df = frames
        node = RefutationNode()
        nc_tuple = (NC, 0.01, (-0.02, 0.04), 290)
        seen = _wire(node, monkeypatch, (nc_tuple, None))
        deadline = time.monotonic() + 600.0

        result = await node.execute(
            _node_state(
                frame,
                negative_control_outcome=NC,
                data_cache={"negative_control_data": nc_df},
                compute_deadline=deadline,
            )
        )

        assert result["gate_decision"] == "proceed"
        fit, recon, runner = seen["fit"], seen["recon"], seen["runner"]
        assert fit["nc_outcome"] == NC
        assert fit["negative_control_data"] is nc_df
        # SAME frame, SAME adjustment set, SAME estimator spec as the primary fit.
        assert fit["refutation_data"] is recon["data"]
        assert fit["common_causes"] is recon["common_causes"]
        assert fit["estimation_result"] is recon["estimation_result"]
        assert fit["treatment"] == recon["treatment"] == TREATMENT
        assert fit["deadline"] == deadline
        assert runner["negative_control"] == nc_tuple
        assert runner["negative_control_skip_reason"] is None
        assert runner["data"] is recon["data"]

    @pytest.mark.asyncio
    async def test_a_skip_reason_reaches_the_runner_with_no_tuple(self, frames, monkeypatch):
        frame, nc_df = frames
        node = RefutationNode()
        seen = _wire(node, monkeypatch, (None, "negative_control_too_few_rows"))

        await node.execute(
            _node_state(
                frame,
                negative_control_outcome=NC,
                data_cache={"negative_control_data": nc_df},
            )
        )

        assert seen["runner"]["negative_control"] is None
        assert seen["runner"]["negative_control_skip_reason"] == "negative_control_too_few_rows"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state_extra", [{}, {"negative_control_outcome": None}])
    async def test_absent_key_passes_nothing(self, frames, monkeypatch, state_extra):
        frame, _ = frames
        node = RefutationNode()
        seen = _wire(node, monkeypatch, ((NC, 0.0, (0.0, 0.0), 1), None))

        await node.execute(_node_state(frame, **state_extra))

        assert "fit" not in seen
        assert seen["runner"].get("negative_control") is None
        assert seen["runner"].get("negative_control_skip_reason") is None


# --------------------------------------------------- (d) real runner, weight 0
# The four REFIT refuters are disabled, not shortened: the runner leaves DoWhy's
# placebo / random_common_cause permutations unseeded, so a same-config re-run
# can flip the gate (measured while writing this file: placebo BLOCKed the
# second of three identical runs at num_simulations=2). What stays is the REAL
# runner's analytic sensitivity reading plus the negative-control row -- a
# deterministic verdict the weight-0 assertion below can compare exactly.
DETERMINISTIC_CONFIG = {
    "placebo_treatment": {"enabled": False},
    "random_common_cause": {"enabled": False},
    "data_subset": {"enabled": False},
    "bootstrap": {"enabled": False},
}


def _nc_row(result: dict) -> dict:
    rows = [
        t
        for t in result["refutation_suite"]["tests"]
        if t["test_name"] == RefutationTestType.NEGATIVE_CONTROL_OUTCOME.value
    ]
    assert len(rows) == 1, [t["test_name"] for t in result["refutation_suite"]["tests"]]
    return rows[0]


async def _run_real(node: RefutationNode, state: dict, monkeypatch) -> dict:
    async def _no_signal(outcome):
        return None

    monkeypatch.setattr(node, "_log_validation_outcome_signal", _no_signal)
    result = await node.execute(state)
    assert "refutation_suite" in result, result.get("error_message")
    return result


class TestRealRunnerEndToEnd:
    @pytest.mark.asyncio
    async def test_declared_control_is_read_and_persistable(
        self, frames, linear_dml_fit, monkeypatch
    ):
        frame, nc_df = frames
        ate = float(linear_dml_fit[2].value)
        node = RefutationNode(config=DETERMINISTIC_CONFIG)
        result = await _run_real(
            node,
            _node_state(
                frame,
                ate=ate,
                negative_control_outcome=NC,
                data_cache={"negative_control_data": nc_df},
            ),
            monkeypatch,
        )
        row = _nc_row(result)
        assert row["status"] in {"passed", "warning", "failed"}
        details = row["details"]
        assert details["nc_outcome"] == NC
        assert details["nc_n"] == len(frame) - N_NC_NULL
        assert details["nc_ci"][0] <= details["nc_effect"] <= details["nc_ci"][1]
        assert row["refuted_effect"] == details["nc_effect"]
        # The #1419 provenance stamp lands on the NC row like every other row.
        assert details["refutation_subsampled"] is False
        assert details["refutation_n_rows"] == len(frame)
        # What the evidence writer persists is a JSON object with the reading keys.
        persisted = json.loads(json.dumps(to_plain_json(details)))
        assert isinstance(persisted, dict)
        assert {"nc_outcome", "nc_effect", "nc_ci", "nc_n", "reading", "message"} <= set(persisted)
        assert persisted["reading"] == persisted["message"]
        assert persisted["nc_effect"] is not None

    @pytest.mark.asyncio
    async def test_primary_verdict_is_unchanged_by_the_control(
        self, frames, linear_dml_fit, monkeypatch
    ):
        """Weight 0: the same gate and confidence whether the control was read,
        could not be fitted, or was never declared."""
        frame, nc_df = frames
        ate = float(linear_dml_fit[2].value)
        declared = _node_state(
            frame,
            ate=ate,
            negative_control_outcome=NC,
            data_cache={"negative_control_data": nc_df},
        )

        with_reading = await _run_real(
            RefutationNode(config=DETERMINISTIC_CONFIG), declared, monkeypatch
        )
        assert _nc_row(with_reading)["status"] != "skipped"

        async def _cannot_fit(**_k):
            return None, "negative_control_ci_unavailable"

        monkeypatch.setattr(_ref_mod, "_fit_negative_control", _cannot_fit)
        fit_failed = await _run_real(
            RefutationNode(config=DETERMINISTIC_CONFIG), declared, monkeypatch
        )
        assert _nc_row(fit_failed)["status"] == "skipped"
        assert _nc_row(fit_failed)["details"]["skip_reason"] == "negative_control_ci_unavailable"

        undeclared = await _run_real(
            RefutationNode(config=DETERMINISTIC_CONFIG), _node_state(frame, ate=ate), monkeypatch
        )
        assert _nc_row(undeclared)["status"] == "skipped"
        assert _nc_row(undeclared)["details"]["skip_reason"] == "no_negative_control_declared"

        verdicts = {
            (r["gate_decision"], r["refutation_suite"]["confidence_score"], r["status"])
            for r in (with_reading, fit_failed, undeclared)
        }
        assert len(verdicts) == 1, verdicts
