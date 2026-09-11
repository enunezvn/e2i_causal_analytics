"""#2005: the scale-free random_common_cause rule is right on planted truth at the live cap.

Calibration pin for ``PASS_THRESHOLDS["common_cause_shift_se"]`` (PASSED ≤ 1 SE,
WARNING ≤ 2 SE). The DGP frame, the 11 planted truths and the 9 structural nulls
are the ones ``test_sensitivity_calibration.py`` pins (imported, not copied),
at n = 1500, the live estimation row cap. Each pair is estimated the way the
route does (``_fit``: LinearDML, RandomForest nuisance) and refuted the way
production does: the DoWhy artifacts come from the refutation node's own
``_reconstruct_dowhy_artifacts`` (same estimator label, seed and nuisance
models, same reconstructed-vs-reported tolerance guard) and the runner's own
``_run_random_common_cause_test`` scores 20 real refits.

A random common cause carries no information about the outcome, so on a
correctly specified model the refit stays inside the estimate's own noise
whatever the effect's size: every truth AND every null must be PASSED or
WARNING. The old ``|Δ| / |ATE|`` rule could not say that — on the first null
pair probed (``copay_support → treatment_initiated``, ATE 0.017, refit 0.004)
the same refit is 74 % of the effect (FAILED at 30 %) and 0.52 SE (PASSED).

Measured 2026-09-11 (the green run of this file as committed, 20 refits per pair,
7 min 39 s): max ``shift_se_units`` over the 11 truths 0.34 (``copay_support ->
adherent_180d`` and ``-> low_gap_180d``), over the 9 nulls 0.65 (``copay_support ->
treatment_initiated``, old-rule 92 %); every row PASSED, none WARNING; 6 of the 9
nulls exceed the retired 30 % cutoff. DoWhy draws the random common cause unseeded,
so the numbers move run to run (an earlier green run of the same file read 0.46 /
0.43; the pre-flight probe read 0.52 on that null pair); the margin to the 2.0
cutoff is what the pin protects. Per-pair table:
``docs/demos/results/2026-09-11_rcc_scale_free/calibration_n1500.md``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from src.agents.causal_impact.nodes.refutation import _reconstruct_dowhy_artifacts
from src.causal_engine.refutation_runner import RefutationRunner, RefutationStatus
from tests.unit.test_causal_engine.test_sensitivity_calibration import (  # noqa: F401
    ARM_REGISTRY,
    N_ROWS,
    NULL_PAIRS,
    _fit,
    _planted_pairs,
    frame,
)

pytestmark = [
    pytest.mark.heavy_ml,
    # 20 pairs x (LinearDML fit + reconstruction + 20 real refits) measured ~21 s
    # per pair (2026-09-11): the module fixture needs ~7 min, far past the 30 s
    # global safety net.
    pytest.mark.timeout(1500),
]

NUM_SIMULATIONS = 20  # the runner's production default for random_common_cause
ACCEPTABLE = (RefutationStatus.PASSED, RefutationStatus.WARNING)
# The cutoffs this file calibrates, as LITERALS (the production constant is pinned
# equal to them below, so a change to either side is caught here).
PASS_SE, WARNING_SE = 1.0, 2.0
# The retired rule's FAILED cutoff (``common_cause_delta`` warning 0.30, as a percent).
OLD_FAILED_PCT = 30.0


def _score(df, treatment: str, outcome: str, covariates: List[str], runner) -> Dict[str, Any]:
    ate, ci = _fit(df, treatment, outcome, covariates)
    model, estimand, estimate = _reconstruct_dowhy_artifacts(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=covariates,
        estimation_result={"ate": ate, "selected_estimator": "linear_dml", "random_state": 42},
    )
    t0 = time.monotonic()
    result = runner._run_random_common_cause_test(
        original_effect=ate,
        original_ci=ci,
        causal_model=model,
        identified_estimand=estimand,
        estimate=estimate,
        use_dowhy=True,
        reference_n=len(df),
        refit_n=len(df),
    )
    return {
        "pair": f"{treatment}->{outcome}",
        "ate": ate,
        "ci": ci,
        "reconstructed": float(estimate.value),
        "refuted": result.refuted_effect,
        "shift_se": result.details["shift_se_units"],
        # the retired rule's statistic, still persisted (DB column), descriptive only
        "delta_percent": result.delta_percent,
        "status": result.status,
        "p_value": result.p_value,
        "seconds": time.monotonic() - t0,
    }


def _table(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{r['pair']}: ate={r['ate']:.4f} ci=({r['ci'][0]:.4f}, {r['ci'][1]:.4f}) "
        f"refit_mean={r['refuted']:.4f} shift_se={r['shift_se']:.2f} {r['status'].value} "
        f"p={r['p_value']:.3f} old_delta_pct={r['delta_percent']:.1f}"
        for r in rows
    )


@pytest.fixture(scope="module")
def scored(frame):  # noqa: F811 - the imported module fixture
    # The population is pinned BEFORE any fit: the live estimation row cap and
    # the sibling file's 11 truths / 9 nulls. A drifted fixture or an empty pair
    # list must fail here, not pass the status tests vacuously.
    assert len(frame) == N_ROWS == 1500
    planted = _planted_pairs(frame)
    assert len(planted) == 11, [p[:2] for p in planted]
    assert len(NULL_PAIRS) == 9
    runner = RefutationRunner(config={"random_common_cause": {"num_simulations": NUM_SIMULATIONS}})
    truths = [_score(frame, a, o, covs, runner) for a, o, covs in planted]
    nulls = [_score(frame, a, o, list(ARM_REGISTRY[a].confounders), runner) for a, o in NULL_PAIRS]
    assert len(truths) == 11 and len(nulls) == 9
    print("\n[rcc calibration] planted truths\n" + _table(truths))
    print("\n[rcc calibration] structural nulls\n" + _table(nulls))
    return {"truths": truths, "nulls": nulls}


def test_the_production_cutoffs_are_the_ones_this_file_calibrates():
    assert RefutationRunner.PASS_THRESHOLDS["common_cause_shift_se"] == {
        "pass": PASS_SE,
        "warning": WARNING_SE,
    }


def test_every_planted_truth_is_passed_or_warning(scored):
    bad = [r for r in scored["truths"] if r["status"] not in ACCEPTABLE]
    assert not bad, "planted truths FAILED under the SE-units rule:\n" + _table(bad)


def test_every_structural_null_is_passed_or_warning(scored):
    bad = [r for r in scored["nulls"] if r["status"] not in ACCEPTABLE]
    assert not bad, "structural nulls FAILED under the SE-units rule:\n" + _table(bad)


def test_no_refit_shift_exceeds_two_reported_se(scored):
    """The two tests above in the rule's own unit, against the LITERAL cutoff
    (not the production constant, so raising it cannot relax this pin) and
    the measured maxima recorded in the module docstring."""
    rows = scored["truths"] + scored["nulls"]
    worst = max(rows, key=lambda r: r["shift_se"])
    assert worst["shift_se"] <= WARNING_SE, _table([worst])


def test_the_rule_discriminates_from_the_retired_percent_rule(scored):
    """A calibration pin that the OLD rule would also pass is not a pin. At
    least one structural null (ATE near zero) must carry an old-rule statistic
    ``|Δ| / |ATE|`` above the retired 30 % FAILED cutoff while the SE-units
    rule reads it PASSED/WARNING: the same refit, FAILED by the denominator
    alone. Any null qualifies (the draws are unseeded); measured 2026-09-11,
    6 of the 9 nulls did."""
    discriminating = [r for r in scored["nulls"] if r["delta_percent"] > OLD_FAILED_PCT]
    assert discriminating, "no null exceeded the retired 30 % cutoff:\n" + _table(scored["nulls"])
    assert all(r["status"] in ACCEPTABLE for r in discriminating), _table(discriminating)
