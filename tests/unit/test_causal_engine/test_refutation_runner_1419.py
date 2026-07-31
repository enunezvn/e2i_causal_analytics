"""#1419: critical-gates-first ordering + non-critical budget-skip degradation.

Measured context (2026-07-31, live 37k conversion frame, 5k stratified
subsample): point-refit sims (placebo/rcc/subset) cost ~1.6-2.6 s each, so the
critical gates (placebo 30 + random_common_cause 20 + analytic e-value) fit a
240 s chat deadline — but a BOOTSTRAP sim re-runs the full inference machinery
at ~11.7 s (≈ the reconstruction fit), so bootstrap (50 sims ~585 s) never
fits and must be gated on the HEAVY cost hint, not the cheap observed
per-refit (which would start an unfinishable run and orphan the thread). Owner-approved
design (#1419): the suite must complete its CRITICAL gates and degrade the
non-critical tail to honest SKIPPED results — instead of the pre-#1419
behavior where ANY budget-skip raised RefutationError and the whole turn
failed closed with `ran 0 test(s)`.

Ordering rationale:
* ``sensitivity_e_value`` is ANALYTIC (no refits, ~ms) — it runs FIRST, gated
  only on ``now < deadline`` (the per-refit cost model does not apply to it),
  and must NOT feed the observed per-refit average (recording its ~ms elapsed
  would collapse the average and disarm the orphan guard for every refit-based
  test after it).
* Then the two refit-based CRITICAL tests (placebo, random_common_cause),
  then the non-critical data_subset and bootstrap.

Skip policy:
* A budget-skipped CRITICAL test still fails the suite closed (an estimate
  whose placebo test never ran must not be presented as validated).
* Budget-skipped NON-critical tests append SKIPPED results (reason in
  ``details``) and the suite completes on the evidence that ran; the skips
  surface via ``to_legacy_format()['skipped_tests']`` (#1249 channel) and
  persist via ``details_json``.
* Criticality comes from the merged config (custom per-test dicts UPDATE
  ``DEFAULT_CONFIG``, which carries the ``critical`` flags), with an explicit
  ``DEFAULT_CONFIG`` fallback for unknown test keys.
"""

import time as _t

import pandas as pd
import pytest

from src.causal_engine.errors import RefutationError
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationRunner,
    RefutationStatus,
    RefutationSuite,
)
from tests.unit.test_causal_engine.test_refutation_runner import _full_stub_causal_model


def _config(sensitivity_enabled: bool) -> dict:
    # Deliberately WITHOUT "critical" keys: the merge (custom dicts UPDATE
    # DEFAULT_CONFIG) must preserve the default critical flags.
    return {
        "placebo_treatment": {"enabled": True, "num_simulations": 10},
        "random_common_cause": {"enabled": True, "num_simulations": 10},
        "data_subset": {"enabled": True, "num_subsets": 10, "subset_fraction": 0.8},
        "bootstrap": {"enabled": True, "num_bootstraps": 10},
        "sensitivity_e_value": {"enabled": sensitivity_enabled, "e_value_threshold": 2.0},
    }


def _run(runner, **overrides):
    kwargs = {
        "original_effect": 0.15,
        "original_ci": (0.10, 0.20),
        "causal_model": _full_stub_causal_model(),
        "identified_estimand": object(),
        "estimate": object(),
    }
    kwargs.update(overrides)
    return runner.run_all_tests(**kwargs)


def _fake_clock(monkeypatch, runner, cost_s):
    """Deterministic clock: each executed test advances it by ``cost_s``."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])
    real = runner._run_test_with_tracing

    def fake_run(**kwargs):
        clock["now"] += cost_s
        return real(**kwargs)

    monkeypatch.setattr(runner, "_run_test_with_tracing", fake_run)
    return clock


class TestCriticalFirstOrdering:
    def test_execution_order_is_evalue_then_criticals_then_noncriticals(self, monkeypatch):
        """No deadline: all five run, in critical-first order with the analytic
        e-value FIRST (cheapest critical completes before any refit spend)."""
        runner = RefutationRunner(config=_config(sensitivity_enabled=True))
        order: list = []
        real = runner._run_test_with_tracing

        def spy(**kwargs):
            order.append(kwargs["test_name"])
            return real(**kwargs)

        monkeypatch.setattr(runner, "_run_test_with_tracing", spy)
        _run(runner)
        assert order == [
            "sensitivity_e_value",
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "bootstrap",
        ]


class TestNonCriticalBudgetSkipDegrades:
    def test_noncritical_skips_degrade_to_skipped_results_not_raise(self, monkeypatch):
        """Budget dies after the criticals: data_subset + bootstrap are appended
        as SKIPPED results, the suite COMPLETES, and the legacy format carries
        the skip reasons — no RefutationError, no 0-test fail-closed.

        (E-value disabled here so the gate assertion reflects the SKIP policy,
        not the e-value's observational strength semantics.)
        """
        runner = RefutationRunner(config=_config(sensitivity_enabled=False))
        _fake_clock(monkeypatch, runner, cost_s=30.0)
        # t=1000, deadline 1075: placebo free-pass (no hint) runs 30 (t=1030,
        # per-refit 3) -> rcc 10x3=30 fits, runs (t=1060) -> data_subset
        # 10x3=30 > 15 remaining -> SKIP -> bootstrap -> SKIP.
        suite = _run(runner, deadline=1075.0)
        assert isinstance(suite, RefutationSuite)
        by_name = {
            (t.test_name.value if hasattr(t.test_name, "value") else str(t.test_name)): t
            for t in suite.tests
        }
        assert by_name["data_subset"].status == RefutationStatus.SKIPPED
        assert by_name["bootstrap"].status == RefutationStatus.SKIPPED
        assert "budget" in by_name["bootstrap"].details.get("reason", "")
        # The evidence that RAN decides the gate; both refit criticals passed,
        # confidence excludes SKIPPED (0.5/0.5 = 1.0) -> PROCEED.
        assert by_name["placebo_treatment"].status == RefutationStatus.PASSED
        assert by_name["random_common_cause"].status == RefutationStatus.PASSED
        assert suite.gate_decision == GateDecision.PROCEED
        legacy = suite.to_legacy_format()
        assert set(legacy["skipped_tests"]) >= {"data_subset", "bootstrap"}
        # Skipped entries stay OUT of individual_tests/totals (#1219/#1249).
        assert "bootstrap" not in legacy["individual_tests"]

    def test_critical_budget_skip_still_fails_closed(self, monkeypatch):
        """random_common_cause (critical) cannot fit -> the suite must still
        fail closed: an estimate whose critical gates never ran is not
        validated. The custom config omits the ``critical`` key, so the merged
        default flag must be what drives the decision."""
        runner = RefutationRunner(config=_config(sensitivity_enabled=False))
        _fake_clock(monkeypatch, runner, cost_s=30.0)
        # deadline 1050: placebo free-pass runs 30 (t=1030, per-refit 3) ->
        # rcc 10x3=30 -> 1060 > 1050 -> CRITICAL skip -> raise.
        with pytest.raises(RefutationError) as ei:
            _run(runner, deadline=1050.0)
        assert ei.value.details.get("reason") == "time_budget_exceeded"
        assert "random_common_cause" in ei.value.details.get("skipped", [])

    def test_bootstrap_gates_on_heavy_hint_not_cheap_observed(self, monkeypatch):
        """#1419 measured: a bootstrap sim costs ~5x a point-refit sim (~11.7 s
        vs ~2.1 s — it re-runs the inference machinery, like the reconstruction
        fit). The cheap observed per-refit therefore UNDERESTIMATES bootstrap's
        true cost; gating on it would start a run that overshoots the deadline
        ~5x and orphans the worker thread. With ``per_refit_hint_heavy`` the
        runner gates bootstrap on ``max(observed, heavy)`` and skips it
        honestly (non-critical -> SKIPPED result, suite completes)."""
        runner = RefutationRunner(config=_config(sensitivity_enabled=False))
        _fake_clock(monkeypatch, runner, cost_s=30.0)
        # t=1000, deadline 1200: placebo/rcc/subset run (t=1090, observed
        # per-refit 3). Cheap gate would admit bootstrap (1090 + 10x3 = 1120
        # <= 1200) — the heavy hint (50) must veto it (1090 + 10x50 > 1200).
        suite = _run(runner, deadline=1200.0, per_refit_hint_heavy=50.0)
        by_name = {
            (t.test_name.value if hasattr(t.test_name, "value") else str(t.test_name)): t
            for t in suite.tests
        }
        assert by_name["placebo_treatment"].status == RefutationStatus.PASSED
        assert by_name["random_common_cause"].status == RefutationStatus.PASSED
        assert by_name["data_subset"].status == RefutationStatus.PASSED
        assert by_name["bootstrap"].status == RefutationStatus.SKIPPED
        assert "budget" in by_name["bootstrap"].details.get("reason", "")

    def test_evalue_standardizes_on_caller_supplied_full_frame_std(self):
        """#1419: the agent node passes the SUBSAMPLE as ``data`` but the
        e-value standardizes the FULL-frame reported effect — a scale-sensitive
        critical gate must not be standardized by a different frame's SD. The
        caller therefore passes the full-frame outcome SD explicitly and it
        must win over the data-derived one (the decoy frame here has SD 50)."""
        runner = RefutationRunner(config=_config(sensitivity_enabled=True))
        decoy = pd.DataFrame({"y": [0.0, 100.0] * 50})  # SD == 50.0
        suite = _run(runner, data=decoy, outcome="y", outcome_std=0.25)
        by_name = {
            (t.test_name.value if hasattr(t.test_name, "value") else str(t.test_name)): t
            for t in suite.tests
        }
        assert by_name["sensitivity_e_value"].details.get("outcome_std") == pytest.approx(0.25)

    def test_evalue_falls_back_to_data_derived_std_when_no_override(self):
        """Existing callers pass no ``outcome_std`` — the data-derived SD
        (H3 behavior) must be unchanged for them."""
        runner = RefutationRunner(config=_config(sensitivity_enabled=True))
        data = pd.DataFrame({"y": [0.0, 100.0] * 50})
        suite = _run(runner, data=data, outcome="y")
        by_name = {
            (t.test_name.value if hasattr(t.test_name, "value") else str(t.test_name)): t
            for t in suite.tests
        }
        assert by_name["sensitivity_e_value"].details.get("outcome_std") == pytest.approx(50.0)

    def test_evalue_is_not_cost_gated_and_does_not_calibrate_per_refit(self):
        """Two properties in one measured trap: with a HUGE per_refit_hint and a
        modest deadline, (a) the analytic e-value must still RUN (it is gated on
        bare ``now < deadline``, not the refit cost model), and (b) its ~ms
        elapsed must NOT be recorded into the observed per-refit average — if it
        were, placebo's gate (10 sims x ~0s) would pass and the orphan guard
        would be disarmed. The resulting CRITICAL skip of placebo proves both."""
        runner = RefutationRunner(config=_config(sensitivity_enabled=True))
        with pytest.raises(RefutationError) as ei:
            _run(
                runner,
                deadline=_t.monotonic() + 60.0,
                per_refit_hint=1_000.0,  # 10 sims x 1000s never fits 60s
            )
        assert "sensitivity_e_value" in ei.value.details.get("ran", [])
        assert "placebo_treatment" in ei.value.details.get("skipped", [])
