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
