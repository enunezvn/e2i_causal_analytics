"""#2029: placebo_treatment and random_common_cause are seeded from the estimate id.

Two runs with the same estimate id give byte-identical refits; two estimate ids
differ (the positive control against a seed that ignores its input).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.heavy_ml, pytest.mark.xdist_group(name="dowhy_seed_2029")]


def _fitted():
    from dowhy import CausalModel

    rng = np.random.default_rng(2029)
    n = 400
    w = rng.normal(size=n)
    t = (rng.normal(size=n) + 0.8 * w > 0).astype(int)
    y = 0.3 * t + 0.5 * w + rng.normal(size=n)
    frame = pd.DataFrame({"t": t, "y": y, "w": w})
    model = CausalModel(data=frame, treatment="t", outcome="y", common_causes=["w"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    return model, estimand, estimate, float(estimate.value)


def _runner():
    from src.causal_engine.refutation_runner import RefutationRunner

    return RefutationRunner(
        config={
            "placebo_treatment": {"num_simulations": 2},
            "random_common_cause": {"num_simulations": 2},
        }
    )


def _one(test: str, estimate_id: str):
    model, estimand, estimate, ate = _fitted()
    runner = _runner()
    seed = runner._seed_for(estimate_id)
    if test == "placebo_treatment":
        r = runner._run_placebo_test(ate, model, estimand, estimate, True, random_state=seed)
    else:
        r = runner._run_random_common_cause_test(
            ate, (ate - 0.1, ate + 0.1), model, estimand, estimate, True, random_state=seed
        )
    return (r.refuted_effect, r.p_value, r.status.value, r.details.get("random_state"))


@pytest.mark.parametrize("test", ["placebo_treatment", "random_common_cause"])
def test_same_estimate_id_is_byte_identical(test):
    a = _one(test, "estimate-A")
    b = _one(test, "estimate-A")
    assert a == b
    assert a[3] is not None  # the seed is recorded


@pytest.mark.parametrize("test", ["placebo_treatment", "random_common_cause"])
def test_different_estimate_ids_differ(test):
    a = _one(test, "estimate-A")
    b = _one(test, "estimate-B")
    assert a[3] != b[3]
    assert a[0] != b[0]  # at 2 sims the refit means differ for different seeds


def test_no_estimate_id_means_no_seed():
    runner = _runner()
    assert runner._seed_for(None) is None
    assert runner._seed_for("") is None


def test_subset_and_bootstrap_details_carry_resample_seed():
    """Task 2: the resample seed is persisted with the perturbation row.

    ``num_subsets=2`` / ``num_bootstraps=2`` sit BELOW the tests' minimums
    (3 / 10), so both land on the config_below_minimum SKIPPED path after two
    real re-fits -- the seed must be recorded on that post-run path too.
    """
    from src.causal_engine.refutation_runner import (
        RefutationRunner,
        RefutationStatus,
        RefutationTestType,
    )

    model, estimand, estimate, ate = _fitted()
    runner = RefutationRunner(
        config={"data_subset": {"num_subsets": 2}, "bootstrap": {"num_bootstraps": 2}}
    )
    ci = (ate - 0.1, ate + 0.1)
    sub = runner._run_data_subset_test(ate, ci, model, estimand, estimate, True, resample_seed=17)
    boot = runner._run_bootstrap_test(ate, ci, model, estimand, estimate, True, resample_seed=17)
    assert sub.test_name == RefutationTestType.DATA_SUBSET
    # Pin the path this test claims to exercise: the post-run config_below_minimum skip.
    assert sub.status is RefutationStatus.SKIPPED
    assert boot.status is RefutationStatus.SKIPPED
    assert sub.details["reason"].startswith("config_below_minimum")
    assert boot.details["reason"].startswith("config_below_minimum")
    assert sub.details["resamples_completed"] == 2
    assert sub.details["resample_seed"] == 17
    assert boot.details["resamples_completed"] == 2
    assert boot.details["resample_seed"] == 17


def test_scored_subset_and_bootstrap_details_carry_resample_seed():
    """Same invariant on the SCORED path (counts at the minimums, 3 / 10)."""
    from src.causal_engine.refutation_runner import RefutationRunner, RefutationStatus

    model, estimand, estimate, ate = _fitted()
    runner = RefutationRunner(
        config={"data_subset": {"num_subsets": 3}, "bootstrap": {"num_bootstraps": 10}}
    )
    ci = (ate - 0.1, ate + 0.1)
    sub = runner._run_data_subset_test(ate, ci, model, estimand, estimate, True, resample_seed=17)
    boot = runner._run_bootstrap_test(ate, ci, model, estimand, estimate, True, resample_seed=17)
    assert sub.status is not RefutationStatus.SKIPPED
    assert boot.status is not RefutationStatus.SKIPPED
    assert sub.details["resample_seed"] == 17
    assert boot.details["resample_seed"] == 17


# ---------------------------------------------------------------------------
# Codex r1 finding A: EVERY perturbation row carries its seed key, skips included.
# Spec §4: "every persisted test row's details_json carries the seed it used:
# resample_seed for subset and bootstrap, random_state for placebo and rcc,
# absent for the analytic tests". The post-deploy cert counts perturbation rows
# WITH a seed key and expects nulls = 0, so a SKIPPED row without the key fails it.
# ---------------------------------------------------------------------------


def test_zero_width_ci_pre_run_skips_carry_the_seed_key():
    """The three pre-run degenerate-CI skips (rcc / subset / bootstrap) never
    ran a refit, yet the row must still say which seed the run WOULD have used
    (placebo has no pre-run skip -- confirmed: three ``_degenerate_ci_skip_result``
    call sites, none in ``_run_placebo_test``)."""
    from src.causal_engine.refutation_runner import RefutationRunner, RefutationStatus
    from tests.unit.test_causal_engine.test_refutation_runner import (
        _make_stub_causal_model,
        _stub_estimate,
    )

    runner = RefutationRunner()
    ate = 0.15
    ci = (ate, ate)  # zero width -> pre-run SKIPPED before any refit
    model = _make_stub_causal_model({})
    est = _stub_estimate()

    rcc = runner._run_random_common_cause_test(ate, ci, model, object(), est, True, random_state=41)
    sub = runner._run_data_subset_test(ate, ci, model, object(), est, True, resample_seed=41)
    boot = runner._run_bootstrap_test(ate, ci, model, object(), est, True, resample_seed=41)

    for r in (rcc, sub, boot):
        assert r.status is RefutationStatus.SKIPPED
        assert r.details["reason"].startswith("original_ci_degenerate")
        assert r.details["resamples_completed"] == 0
    assert rcc.details["random_state"] == 41
    assert "resample_seed" not in rcc.details
    assert sub.details["resample_seed"] == 41
    assert boot.details["resample_seed"] == 41
    assert "random_state" not in sub.details and "random_state" not in boot.details


def test_zero_width_ci_pre_run_skips_record_none_when_unseeded():
    """An unseeded run (no estimate id) stays unseeded AND says so: the key is
    present with ``None``, never silently absent."""
    from src.causal_engine.refutation_runner import RefutationRunner
    from tests.unit.test_causal_engine.test_refutation_runner import (
        _make_stub_causal_model,
        _stub_estimate,
    )

    runner = RefutationRunner()
    model = _make_stub_causal_model({})
    rcc = runner._run_random_common_cause_test(
        0.15, (0.15, 0.15), model, object(), _stub_estimate(), True, random_state=None
    )
    sub = runner._run_data_subset_test(
        0.15, (0.15, 0.15), model, object(), _stub_estimate(), True, resample_seed=None
    )
    assert "random_state" in rcc.details and rcc.details["random_state"] is None
    assert "resample_seed" in sub.details and sub.details["resample_seed"] is None


def test_budget_skip_rows_carry_the_seed_key_for_perturbation_tests_only(monkeypatch):
    """``run_all_tests`` builds the time-budget SKIPPED rows itself (not via the
    per-test helpers). With random_common_cause demoted to non-critical, a fake
    clock that burns 60 s per refuter and a 100 s budget, placebo runs and the
    three remaining perturbation tests are budget-skipped: their rows must carry
    the run's seed under the right key; the analytic sensitivity row carries none."""
    import time as _t

    from src.causal_engine.refutation_runner import RefutationRunner, RefutationStatus
    from tests.unit.test_causal_engine.test_refutation_runner import (
        _full_stub_causal_model,
        _stub_estimate,
    )

    config = {
        "placebo_treatment": {"enabled": True, "num_simulations": 10},
        "random_common_cause": {"enabled": True, "num_simulations": 10, "critical": False},
        "data_subset": {"enabled": True, "num_subsets": 10},
        "bootstrap": {"enabled": True, "num_bootstraps": 10},
        "sensitivity_e_value": {"enabled": True},
    }
    runner = RefutationRunner(config=config)
    clock = {"now": 1000.0}
    monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])
    real = runner._run_test_with_tracing

    def fake_run(**kwargs):
        clock["now"] += 60.0
        return real(**kwargs)

    monkeypatch.setattr(runner, "_run_test_with_tracing", fake_run)

    suite = runner.run_all_tests(
        original_effect=0.15,
        original_ci=(0.10, 0.20),
        causal_model=_full_stub_causal_model(),
        identified_estimand=object(),
        estimate=_stub_estimate(),
        estimate_id="est-budget",
        deadline=1000.0 + 100.0,
    )
    seed = runner._seed_for("est-budget")
    assert seed is not None
    rows = {t.test_name.value: t for t in suite.tests}
    budget_skipped = {
        n
        for n, t in rows.items()
        if t.status is RefutationStatus.SKIPPED and t.details["reason"].startswith("time_budget")
    }
    # Pin the path this test claims to exercise: three budget skips, placebo ran.
    assert budget_skipped == {"random_common_cause", "data_subset", "bootstrap"}, budget_skipped
    assert rows["placebo_treatment"].status is not RefutationStatus.SKIPPED
    # The negative-control row is a SKIPPED analytic row (none declared): no seed key.
    assert rows["negative_control_outcome"].status is RefutationStatus.SKIPPED
    assert "random_state" not in rows["negative_control_outcome"].details
    assert "resample_seed" not in rows["negative_control_outcome"].details
    assert rows["random_common_cause"].details["random_state"] == seed
    assert "resample_seed" not in rows["random_common_cause"].details
    assert rows["data_subset"].details["resample_seed"] == seed
    assert rows["bootstrap"].details["resample_seed"] == seed
    for n in ("data_subset", "bootstrap"):
        assert "random_state" not in rows[n].details
    assert rows["placebo_treatment"].details["random_state"] == seed
    assert "random_state" not in rows["sensitivity_e_value"].details
    assert "resample_seed" not in rows["sensitivity_e_value"].details
