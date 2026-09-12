"""#2029: placebo_treatment and random_common_cause are seeded from the pair identity.

The seed comes from ``seed_identity_for`` (``brand|treatment|outcome``); the
estimate id is only the fallback when a caller passes no identity. These tests
drive ``_seed_for`` directly, so "estimate id" below stands for whichever identity
string the runner seeds from: two runs with the same identity give byte-identical
refits; two identities differ (the positive control against a seed that ignores
its input).
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
    # Lane 1b: the identity travels with the seed -- PRESENT (None: this run
    # gave no pair identity, so the seed fell back to the estimate id) on every
    # perturbation row, budget skips included; absent on the analytic rows.
    for n in ("placebo_treatment", "random_common_cause", "data_subset", "bootstrap"):
        assert "seed_identity" in rows[n].details, n
        assert rows[n].details["seed_identity"] is None, n
    for n in ("sensitivity_e_value", "negative_control_outcome"):
        assert "seed_identity" not in rows[n].details, n


def test_seed_key_map_covers_every_perturbation_test_type_and_nothing_else():
    """Drift guard for ``_SEED_KEY_BY_TEST`` (#2029, Task 9): the map is THE
    place a test's seed key is spelled -- the budget-skip loop and the
    pre-run degenerate-CI skip helper both derive the key from the test type,
    so a NEW perturbation test type must be added to the map or its skip rows
    will silently lack the key (and the post-deploy cert's nulls = 0 count
    breaks). The two analytic tests (sensitivity, negative control) are the
    only enum members that must NOT carry a key."""
    from src.causal_engine.refutation_runner import _SEED_KEY_BY_TEST, RefutationTestType

    analytic = {"sensitivity_e_value", "negative_control_outcome"}
    assert set(_SEED_KEY_BY_TEST) == {t.value for t in RefutationTestType} - analytic
    assert set(_SEED_KEY_BY_TEST.values()) == {"random_state", "resample_seed"}


# ---------------------------------------------------------------------------
# Lane 1b (#2029): the run is seeded from the content-addressed PAIR identity
# ``brand|treatment|outcome``, the per-run query id only as a fallback. Measured
# on the deployed image 2026-09-12: two consecutive live discovery runs gave
# identical ATEs (11/11) and statuses (22/22) but refit values identical 0/22,
# because the node seeded from ``query_id`` = a uuid4 minted per analysis.
# ---------------------------------------------------------------------------


def test_seed_identity_for_joins_brand_treatment_outcome():
    from src.causal_engine.refutation_runner import seed_identity_for

    assert seed_identity_for(brand="B", treatment="t", outcome="y") == "B|t|y"


def test_seed_identity_for_without_brand_has_a_leading_empty_segment():
    from src.causal_engine.refutation_runner import seed_identity_for

    assert seed_identity_for(brand=None, treatment="t", outcome="y") == "|t|y"
    assert seed_identity_for(brand="", treatment="t", outcome="y") == "|t|y"


@pytest.mark.parametrize(
    ("treatment", "outcome"),
    [(None, "y"), ("", "y"), ("t", None), ("t", ""), (None, None)],
)
def test_seed_identity_for_needs_both_treatment_and_outcome(treatment, outcome):
    from src.causal_engine.refutation_runner import seed_identity_for

    assert seed_identity_for(brand="B", treatment=treatment, outcome=outcome) is None


_PERTURBATION = ("placebo_treatment", "random_common_cause", "data_subset", "bootstrap")


def _suite(estimate_id, seed_identity):
    """One real-DoWhy ``run_all_tests`` on the tiny frame: 2 placebo sims, 2 rcc
    sims, and the two resample loops at their scoring minimums (3 / 10)."""
    from src.causal_engine.refutation_runner import RefutationRunner

    model, estimand, estimate, ate = _fitted()
    runner = RefutationRunner(
        config={
            "placebo_treatment": {"num_simulations": 2},
            "random_common_cause": {"num_simulations": 2},
            "data_subset": {"num_subsets": 3},
            "bootstrap": {"num_bootstraps": 10},
            "sensitivity_e_value": {"enabled": False},
        }
    )
    suite = runner.run_all_tests(
        original_effect=ate,
        original_ci=(ate - 0.1, ate + 0.1),
        causal_model=model,
        identified_estimand=estimand,
        estimate=estimate,
        estimate_id=estimate_id,
        seed_identity=seed_identity,
    )
    return {t.test_name.value: t for t in suite.tests}


def _refit_view(rows):
    return {
        n: (rows[n].refuted_effect, rows[n].p_value, rows[n].status.value) for n in _PERTURBATION
    }


def test_same_pair_identity_different_estimate_ids_is_byte_identical():
    """The pair identity decides the seed; the per-run estimate id does not."""
    from src.causal_engine.refutation_runner import (
        _SEED_KEY_BY_TEST,
        RefutationStatus,
        seed_for_estimate,
    )

    a = _suite("run-A", "B|t|y")
    b = _suite("run-B", "B|t|y")
    # Pin the path: all four perturbation tests were SCORED (a SKIPPED row's
    # refuted_effect is the original effect, which would agree trivially).
    for n in _PERTURBATION:
        assert a[n].status is not RefutationStatus.SKIPPED, (n, a[n].details.get("reason"))
    assert _refit_view(a) == _refit_view(b)
    expected_seed = seed_for_estimate("B|t|y")
    assert expected_seed is not None
    for rows in (a, b):
        for n in _PERTURBATION:
            assert rows[n].details["seed_identity"] == "B|t|y", n
            assert rows[n].details[_SEED_KEY_BY_TEST[n]] == expected_seed, n
    # The estimate id still reaches the suite (persistence / tracing are unchanged).


def test_same_estimate_id_different_identities_differ():
    """Positive control: a seed that ignored the identity would pass the test above."""
    from src.causal_engine.refutation_runner import seed_for_estimate

    a = _suite("run-A", "B|t|y")
    c = _suite("run-A", "B|t|z")
    assert seed_for_estimate("B|t|y") != seed_for_estimate("B|t|z")
    assert (
        a["placebo_treatment"].details["random_state"]
        != c["placebo_treatment"].details["random_state"]
    )
    assert a["placebo_treatment"].refuted_effect != c["placebo_treatment"].refuted_effect
    assert c["placebo_treatment"].details["seed_identity"] == "B|t|z"


def test_no_identity_falls_back_to_the_estimate_id():
    from src.causal_engine.refutation_runner import _SEED_KEY_BY_TEST, seed_for_estimate

    rows = _suite("run-A", None)
    for n in _PERTURBATION:
        assert rows[n].details[_SEED_KEY_BY_TEST[n]] == seed_for_estimate("run-A"), n
        assert "seed_identity" in rows[n].details, n
        assert rows[n].details["seed_identity"] is None, n
    # Analytic rows carry neither the seed key nor the identity.
    assert "seed_identity" not in rows["negative_control_outcome"].details
