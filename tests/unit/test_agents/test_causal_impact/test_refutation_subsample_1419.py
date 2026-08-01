"""#1419: refutation runs on the SAME deterministic stratified subsample as
#1392 selection, and the per-refit hint is CALIBRATED with a real 1-sim refute.

Why (measured 2026-07-31, live 37k conversion frame):
* Full-frame per-refit is ~23 s x ~105 configured sims — refutation can never
  fit any enforceable chat budget on the full frame. On the 5,000-row
  stratified subsample (same helper as selection: treatment x outcome-bin
  strata, content-derived seed) point-refit per-sim cost drops to ~1.6-2.6 s
  (bootstrap, which re-runs inference, to ~11.7 s) and the subsample preserves
  the marginals almost exactly (treat share 0.4849 -> 0.4848, outcome mean
  0.1836 -> 0.1836).
* The reconstruction fit on the subsample costs ~11.5 s but a refuter SIM
  costs ~2.1 s — the recon-wall-time ``per_refit_hint`` overestimates ~5x and
  would budget-skip placebo (30 x 11.5 s = 345 s "needed") even though the
  real work fits. When a deadline is set, the node must calibrate the hint
  with one throwaway 1-sim placebo refute (~one sim's true cost) instead.

The reported ATE/CI stays the FULL-frame fit (#1392 contract); refutation
critiques the same estimator refit on the subsample, and the existing
reconstructed-vs-reported tolerance guard (rel 0.20 / abs 0.10) covers the
subsample drift (observed live: 0.0404 vs 0.0352). Every evidence row records
the subsample provenance via test ``details`` -> ``details_json``.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes import refutation as _ref_mod
from src.agents.causal_impact.nodes.refutation import (
    RefutationNode,
    _subsample_for_refutation,
)
from src.causal_engine.energy_score.estimator_selector import (
    SELECTION_MAX_ROWS_DEFAULT,
)
from src.causal_engine.errors import RefutationError


def _frame(n: int, continuous_treatment: bool = False, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if continuous_treatment:
        t = rng.normal(10.0, 3.0, n)
    else:
        t = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "accepted": t,
            "converted": rng.integers(0, 2, n),
            "confidence_score": rng.uniform(0.5, 1.0, n),
            "trigger_type": rng.choice(["adherence_risk", "dosing_gap"], n),
        }
    )


class TestSubsampleForRefutation:
    def test_above_cap_subsamples_with_preserved_marginals(self):
        big = _frame(3 * SELECTION_MAX_ROWS_DEFAULT)
        sub, disclosure = _subsample_for_refutation(big, "accepted", "converted")
        assert len(sub) == SELECTION_MAX_ROWS_DEFAULT
        assert disclosure == {
            "refutation_subsampled": True,
            "refutation_n_rows": SELECTION_MAX_ROWS_DEFAULT,
            "refutation_n_rows_total": len(big),
        }
        assert abs(sub["accepted"].mean() - big["accepted"].mean()) < 0.02
        assert abs(sub["converted"].mean() - big["converted"].mean()) < 0.02

    def test_at_or_below_cap_is_a_noop(self):
        small = _frame(SELECTION_MAX_ROWS_DEFAULT)
        sub, disclosure = _subsample_for_refutation(small, "accepted", "converted")
        assert sub is small
        assert disclosure["refutation_subsampled"] is False
        assert disclosure["refutation_n_rows"] == len(small)
        assert disclosure["refutation_n_rows_total"] == len(small)

    def test_deterministic_across_calls(self):
        big = _frame(2 * SELECTION_MAX_ROWS_DEFAULT)
        sub1, _ = _subsample_for_refutation(big, "accepted", "converted")
        sub2, _ = _subsample_for_refutation(big, "accepted", "converted")
        pd.testing.assert_frame_equal(sub1, sub2)

    def test_continuous_treatment_binarized_at_full_frame_median(self):
        """The estimand is the effect of the treatment binarized at the FULL
        frame's median (estimation.py preprocessing). Subsampling first and
        letting reconstruction binarize at the SUBSAMPLE median would shift the
        split point — so the helper binarizes BEFORE drawing, with the same
        NumPy ops, and reconstruction's integer check then passes it through."""
        big = _frame(3 * SELECTION_MAX_ROWS_DEFAULT, continuous_treatment=True)
        full_median = np.median(big["accepted"].to_numpy())
        expected = (big["accepted"].to_numpy() > full_median).astype(int)
        sub, disclosure = _subsample_for_refutation(big, "accepted", "converted")
        assert disclosure["refutation_subsampled"] is True
        assert set(sub["accepted"].unique()) <= {0, 1}
        # Every selected row carries the FULL-frame-median split value.
        expected_series = pd.Series(expected, index=big.index)
        assert (sub["accepted"] == expected_series.loc[sub.index]).all()

    def test_uncoercible_treatment_dtype_falls_back_to_full_frame(self):
        """Unexpected dtypes (e.g. a string treatment the integer-coercion
        check chokes on) must NOT escape as a raw TypeError/ValueError that
        bypasses the structured fail-closed paths — the helper falls back to
        the full frame and the downstream budget gates own any resulting
        fail-closed skip."""
        big = _frame(3 * SELECTION_MAX_ROWS_DEFAULT)
        big["accepted"] = np.where(big["accepted"] == 1, "yes", "no")
        sub, disclosure = _subsample_for_refutation(big, "accepted", "converted")
        assert sub is big
        assert disclosure == {
            "refutation_subsampled": False,
            "refutation_n_rows": len(big),
            "refutation_n_rows_total": len(big),
        }

    def test_non_dataframe_and_missing_columns_pass_through(self):
        """Reconstruction owns the fail-closed messaging for bad passthroughs —
        the helper must not preempt it with a different crash."""
        sub, disclosure = _subsample_for_refutation(None, "accepted", "converted")
        assert sub is None and disclosure["refutation_subsampled"] is False
        frame = _frame(10)
        sub, disclosure = _subsample_for_refutation(frame, "not_a_column", "converted")
        assert sub is frame and disclosure["refutation_subsampled"] is False


def _node_state(frame: pd.DataFrame, deadline=None) -> dict:
    ate = 0.05
    state = {
        "query": "subsample plumbing",
        "query_id": "sub-ref-1419",
        "treatment_var": "accepted",
        "outcome_var": "converted",
        "confounders": ["confidence_score", "trigger_type"],
        "data_source": "synthetic",
        "estimation_result": {
            "method": "LinearDML",
            "selected_estimator": "linear_dml",
            "ate": ate,
            "ate_ci_lower": ate - 0.01,
            "ate_ci_upper": ate + 0.01,
            "effect_size": "small",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": len(frame),
            "covariates_adjusted": ["confidence_score", "trigger_type"],
            "heterogeneity_detected": False,
        },
        "estimation_data": frame,
        "status": "pending",
        "errors": [],
        "warnings": [],
    }
    if deadline is not None:
        state["compute_deadline"] = deadline
    return state


class TestNodeSubsamplePlumbing:
    @pytest.mark.asyncio
    async def test_reconstruction_and_suite_get_the_same_subsampled_frame(self, monkeypatch):
        big = _frame(3 * SELECTION_MAX_ROWS_DEFAULT)
        node = RefutationNode()
        seen: dict = {}

        def fake_recon(*, data, **kwargs):
            seen["recon_data"] = data
            return (SimpleNamespace(), object(), object())

        def spy_run_all_tests(**kwargs):
            seen["suite_data"] = kwargs.get("data")
            raise RefutationError("stop after capture", details={"reason": "test"})

        monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
        monkeypatch.setattr(node.runner, "run_all_tests", spy_run_all_tests)

        result = await node.execute(_node_state(big))

        assert len(seen["recon_data"]) == SELECTION_MAX_ROWS_DEFAULT
        # The suite must critique the SAME frame the model was rebuilt on.
        assert seen["suite_data"] is seen["recon_data"]
        assert result.get("status") == "failed"  # spy raised -> clean fail-closed


class TestNodeFullFrameOutcomeStd:
    @pytest.mark.asyncio
    async def test_run_all_tests_receives_full_frame_outcome_std(self, monkeypatch):
        """The e-value standardizes the FULL-frame reported effect, but
        ``run_all_tests`` receives the SUBSAMPLE as ``data`` — the node must
        pass the full frame's outcome SD explicitly or the runner would derive
        a different (subsample) scale for a scale-sensitive critical gate.
        Continuous outcome so the two SDs genuinely differ."""
        big = _frame(3 * SELECTION_MAX_ROWS_DEFAULT)
        rng = np.random.default_rng(11)
        big["converted"] = rng.normal(50.0, 12.0, len(big))
        node = RefutationNode()
        captured: dict = {}

        def fake_recon(**kwargs):
            return (SimpleNamespace(), object(), object())

        def spy_run_all_tests(**kwargs):
            captured.update(kwargs)
            raise RefutationError("stop after capture", details={"reason": "test"})

        monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
        monkeypatch.setattr(node.runner, "run_all_tests", spy_run_all_tests)

        await node.execute(_node_state(big))

        expected = float(np.std(big["converted"].to_numpy(dtype=float)))
        assert captured.get("outcome_std") == pytest.approx(expected)
        # And it is NOT the subsample's SD (the frames genuinely differ here).
        sub_std = float(np.std(captured["data"]["converted"].to_numpy(dtype=float)))
        assert sub_std != pytest.approx(expected)


class TestNodeCalibratedPerRefitHint:
    def _fake_clock(self, monkeypatch):
        clock = {"now": 5000.0}
        monkeypatch.setattr(_ref_mod.time, "monotonic", lambda: clock["now"])
        return clock

    @pytest.mark.asyncio
    async def test_deadline_calibrates_hint_with_one_sim_refute(self, monkeypatch):
        """With a deadline, the hint handed to run_all_tests must be the
        measured 1-sim calibration cost (true per-sim), NOT the ~5x larger
        reconstruction wall-time that would budget-skip a fitting suite."""
        clock = self._fake_clock(monkeypatch)
        calibration_calls: list = []

        class _Model:
            def refute_estimate(self, *a, **k):
                calibration_calls.append(k)
                clock["now"] += 2.0  # one sim costs 2 (fake) seconds

        def fake_recon(**kwargs):
            clock["now"] += 10.0  # reconstruction costs 10 (fake) seconds
            return (_Model(), object(), object())

        node = RefutationNode()
        captured: dict = {}

        def spy_run_all_tests(**kwargs):
            captured.update(kwargs)
            raise RefutationError("stop after capture", details={"reason": "test"})

        monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
        monkeypatch.setattr(node.runner, "run_all_tests", spy_run_all_tests)

        frame = _frame(100)
        await node.execute(_node_state(frame, deadline=5000.0 + 10_000.0))

        assert len(calibration_calls) == 1
        assert calibration_calls[0].get("num_simulations") == 1
        assert captured.get("per_refit_hint") == pytest.approx(2.0)
        # The recon wall-time is preserved as the HEAVY hint: bootstrap sims
        # cost ~the recon fit (measured ~11.7 s vs ~2.1 s point refits), so the
        # runner must gate bootstrap on it, not the cheap calibrated hint.
        assert captured.get("per_refit_hint_heavy") == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_no_deadline_keeps_recon_hint_and_skips_calibration(self, monkeypatch):
        """Without a deadline nothing gates on the hint — no throwaway refute
        call is spent and the recon wall-time hint is preserved as before."""
        clock = self._fake_clock(monkeypatch)
        calibration_calls: list = []

        class _Model:
            def refute_estimate(self, *a, **k):
                calibration_calls.append(k)
                clock["now"] += 2.0

        def fake_recon(**kwargs):
            clock["now"] += 10.0
            return (_Model(), object(), object())

        node = RefutationNode()
        captured: dict = {}

        def spy_run_all_tests(**kwargs):
            captured.update(kwargs)
            raise RefutationError("stop after capture", details={"reason": "test"})

        monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
        monkeypatch.setattr(node.runner, "run_all_tests", spy_run_all_tests)

        await node.execute(_node_state(_frame(100)))

        assert calibration_calls == []
        assert captured.get("per_refit_hint") == pytest.approx(10.0)


class TestEvidenceRowsRecordSubsampleProvenance:
    @pytest.mark.asyncio
    async def test_suite_test_details_carry_subsample_disclosure(self, monkeypatch):
        """Every evidence row persists ``details_json`` — the node must annotate
        each suite test's details with the subsample provenance so
        validated-on-subsample is recorded on the row itself (#1419)."""
        from src.causal_engine.refutation_runner import (
            GateDecision,
            RefutationResult,
            RefutationStatus,
            RefutationSuite,
            RefutationTestType,
        )

        big = _frame(3 * SELECTION_MAX_ROWS_DEFAULT)
        node = RefutationNode()

        def fake_recon(**kwargs):
            return (SimpleNamespace(), object(), object())

        def fake_run_all_tests(**kwargs):
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

        monkeypatch.setattr(_ref_mod, "_reconstruct_dowhy_artifacts", fake_recon)
        monkeypatch.setattr(node.runner, "run_all_tests", fake_run_all_tests)

        result = await node.execute(_node_state(big))

        suite_details = result["refutation_suite"]["tests"][0]["details"]
        assert suite_details["refutation_subsampled"] is True
        assert suite_details["refutation_n_rows"] == SELECTION_MAX_ROWS_DEFAULT
        assert suite_details["refutation_n_rows_total"] == len(big)
