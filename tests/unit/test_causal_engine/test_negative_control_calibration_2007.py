"""#2007: the negative-control reading is right on planted truth at the live cap.

Calibration pin for ``RefutationRunner._run_negative_control_test`` (rule
``negative_control_ci_vs_zero``: PASSED when the control's CI includes 0,
WARNING when it excludes 0 and |nc| < |original|, FAILED when it excludes 0 and
|nc| >= |original|) and for the registry that decides WHICH arm gets a control
(``src/api/routes/causal.py::_CAUSAL_NEGATIVE_CONTROL_OUTCOMES``, imported, not
retyped). The DGP frame, the 11 planted truths and the 9 structural nulls are
the ones ``test_sensitivity_calibration.py`` pins (imported), at n = 1500, the
live estimation row cap; every fit is the route's ``_fit`` (LinearDML, RF
nuisances, seed 42) and every verdict comes from the real runner.

Two fits per declared control, exactly the pair the registry was measured on
(``docs/demos/results/2026-09-11_negative_control_disproof/disproof.md``):

* ADJUSTED — the control fitted on the truth's own confounders. The control is
  structurally null, so a correct adjustment must read PASSED; a WARNING is a
  5 % chance positive and at most one is tolerated; a FAILED means the reading
  would block a planted truth and must never happen.
* OMITTED — the control fitted with the confounders replaced by a seeded noise
  column (``run_disproof.py::fit_omitted``). The three declared controls are
  the three nulls that MOVED out of their CI under that omission; the pin
  reproduces the recorded movement (seeded fits, |delta| < 0.01) and checks the
  runner reads the leak as WARNING/FAILED, never PASSED.

The two arms the registry leaves out (``sample_dropped``, ``trigger_accepted``)
are pinned ABSENT together with the measurement that keeps them out: none of
their would-be controls (the structural nulls in ``NULL_PAIRS``) leaves its CI
under the omitted fit, so a declared control would PASS under confounding and
read as false assurance. ``treatment_arm`` moves every outcome in the generator
and has no structural null to declare.

Measured 2026-09-11 on the droplet (31 s, ~20 fits): 6 truth-with-control rows
(copay_support x3, psp_enrolled x2, rep_detailing_high x1), all PASSED adjusted,
0 WARNING; under the omitted fit 4 WARNING and 2 FAILED (psp_enrolled ->
persistent_180d and rep_detailing_high -> treatment_initiated, where the leaked
control moves at least as much as the claimed effect), 0 PASSED; 4 undeclared
would-be controls, 0 exclude 0 under omission. Per-row table:
``docs/demos/results/2026-09-11_negative_control_disproof/calibration_n1500.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.api.routes.causal import _CAUSAL_NEGATIVE_CONTROL_OUTCOMES
from src.causal_engine.refutation_runner import (
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_sensitivity_calibration import (  # noqa: F401
    N_ROWS,
    NULL_PAIRS,
    _fit,
    _planted_pairs,
    frame,
)

pytestmark = [
    pytest.mark.heavy_ml,
    # ~20 LinearDML fits at ~2 s each plus the frame build, ~1 min on the
    # droplet; the CI heavy lane's --timeout is 60 s, its stall watchdog 1200 s
    # (backend-tests.yml, #1655 rule: marker <= 600). 300 s is ~5x measured.
    pytest.mark.timeout(300),
]

DATASET = "patient_journeys"  # the only dataset with declared controls
CONTROLS: Dict[str, str] = _CAUSAL_NEGATIVE_CONTROL_OUTCOMES[DATASET]
UNDECLARED_ARMS = ("sample_dropped", "trigger_accepted")
# The recorded omitted-fit movement per declared control (disproof.md, the
# `omitted ATE` column); the fits are seeded, so a drift beyond RECORD_TOL is a
# changed generator or estimator, not noise.
RECORDED_OMITTED_EFFECT = {
    ("copay_support", "treatment_initiated"): 0.0517,
    ("psp_enrolled", "treatment_initiated"): 0.0895,
    ("rep_detailing_high", "persistent_180d"): 0.0577,
}
RECORD_TOL = 0.01
MAX_ADJUSTED_WARNINGS = 1  # alpha = 0.05 chance positive across 3 distinct controls
EXPECTED_ROWS_PER_ARM = {"copay_support": 3, "psp_enrolled": 2, "rep_detailing_high": 1}


def fit_omitted(df, treatment, outcome, seed=42):
    """Copied verbatim from
    docs/demos/results/2026-09-11_negative_control_disproof/run_disproof.py::fit_omitted
    (docs/ is not importable from the test tree): the production estimator with
    the confounders OMITTED — X is a seeded standard-normal noise column, so the
    model has nothing real to condition on."""
    rng = np.random.default_rng(seed)
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = rng.standard_normal((len(df), 1))
    m = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
        discrete_treatment=True,
        random_state=42,
    )
    m.fit(Y, T, X=X, W=None)
    inf = m.ate_inference(X)
    lo, hi = (float(v) for v in inf.conf_int_mean())
    return float(inf.mean_point), (lo, hi)


def excludes_zero(ci: Tuple[float, float]) -> bool:
    return bool(ci[0] > 0 or ci[1] < 0)


def _score(runner: RefutationRunner, original: float, nc_outcome: str, fit, n: int):
    nc_effect, nc_ci = fit
    result = runner._run_negative_control_test(original, (nc_outcome, nc_effect, nc_ci, n))
    assert result.test_name == RefutationTestType.NEGATIVE_CONTROL_OUTCOME
    assert result.details["rule"] == "negative_control_ci_vs_zero"
    return result


def _fmt(fit) -> str:
    eff, (lo, hi) = fit
    return f"{eff:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _table(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{r['arm']}->{r['outcome']} (truth, original {r['original']:+.4f}) | control "
        f"{r['nc_outcome']}: adjusted {_fmt(r['adjusted'])} -> {r['adjusted_status'].value}; "
        f"omitted {_fmt(r['omitted'])} -> {r['omitted_status'].value}"
        for r in rows
    )


def _undeclared_table(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{r['arm']}->{r['outcome']} (would-be control, UNDECLARED): omitted "
        f"{_fmt(r['omitted'])} excludes_zero={r['excludes_zero']}"
        for r in rows
    )


@pytest.fixture(scope="module")
def scored(frame):  # noqa: F811 - the imported module fixture
    # The population is pinned BEFORE any fit: the live row cap and the sibling
    # file's 11 truths / 9 nulls, so a drifted fixture fails here and never
    # passes the status pins vacuously.
    assert len(frame) == N_ROWS == 1500
    planted = _planted_pairs(frame)
    assert len(planted) == 11, [p[:2] for p in planted]
    assert len(NULL_PAIRS) == 9
    runner = RefutationRunner()
    n = len(frame)
    # One adjusted and one omitted fit per DISTINCT (arm, control, covariates);
    # the three copay truths share one control and one confounder set.
    adjusted_cache: Dict[Tuple[str, str, Tuple[str, ...]], Any] = {}
    omitted_cache: Dict[Tuple[str, str], Any] = {}
    rows: List[Dict[str, Any]] = []
    for arm, outcome, covs in planted:
        nc_outcome = CONTROLS.get(arm)
        if nc_outcome is None:
            continue
        # a control cannot be the outcome it is meant to check (the route's
        # _negative_control_outcome returns None for that case)
        assert nc_outcome != outcome, (arm, outcome)
        original, original_ci = _fit(frame, arm, outcome, covs)
        assert excludes_zero(original_ci), ("planted truth not detected", arm, outcome, original_ci)
        akey = (arm, nc_outcome, tuple(covs))
        if akey not in adjusted_cache:
            adjusted_cache[akey] = _fit(frame, arm, nc_outcome, covs)
        okey = (arm, nc_outcome)
        if okey not in omitted_cache:
            omitted_cache[okey] = fit_omitted(frame, arm, nc_outcome)
        adjusted = _score(runner, original, nc_outcome, adjusted_cache[akey], n)
        omitted = _score(runner, original, nc_outcome, omitted_cache[okey], n)
        rows.append(
            {
                "arm": arm,
                "outcome": outcome,
                "confounders": covs,
                "original": original,
                "original_ci": original_ci,
                "nc_outcome": nc_outcome,
                "adjusted": adjusted_cache[akey],
                "adjusted_status": adjusted.status,
                "omitted": omitted_cache[okey],
                "omitted_status": omitted.status,
            }
        )
    undeclared: List[Dict[str, Any]] = []
    for arm, outcome in NULL_PAIRS:
        if arm not in UNDECLARED_ARMS:
            continue
        fit = fit_omitted(frame, arm, outcome)
        undeclared.append(
            {"arm": arm, "outcome": outcome, "omitted": fit, "excludes_zero": excludes_zero(fit[1])}
        )
    print(
        "\n[negative-control calibration] planted truths with a declared control\n" + _table(rows)
    )
    print(
        "\n[negative-control calibration] undeclared arms, would-be controls\n"
        + _undeclared_table(undeclared)
    )
    return {"controls": rows, "undeclared": undeclared}


def test_the_registry_declares_exactly_the_three_measured_responders():
    assert set(CONTROLS) == {"copay_support", "psp_enrolled", "rep_detailing_high"}, CONTROLS
    assert set(_CAUSAL_NEGATIVE_CONTROL_OUTCOMES) == {DATASET}


def test_treatment_arm_has_no_registry_entry():
    assert "treatment_arm" not in CONTROLS, (
        "treatment_arm moves every outcome in the generator; it has no structural "
        "null to declare as a control"
    )


def test_every_planted_truth_with_a_control_scores_passed_on_the_adjusted_fit(scored):
    rows = scored["controls"]
    assert Counter(r["arm"] for r in rows) == EXPECTED_ROWS_PER_ARM, _table(rows)
    failed = [r for r in rows if r["adjusted_status"] == RefutationStatus.FAILED]
    assert not failed, "adjusted control FAILED on a planted truth:\n" + _table(failed)
    warned = [r for r in rows if r["adjusted_status"] == RefutationStatus.WARNING]
    assert len(warned) <= MAX_ADJUSTED_WARNINGS, "adjusted control WARNING:\n" + _table(warned)
    assert all(
        r["adjusted_status"] in (RefutationStatus.PASSED, RefutationStatus.WARNING) for r in rows
    ), _table(rows)


def test_each_declared_control_leaves_its_ci_under_omitted_confounding(scored):
    seen = {}
    for r in scored["controls"]:
        seen[(r["arm"], r["nc_outcome"])] = r["omitted"]
    assert set(seen) == set(RECORDED_OMITTED_EFFECT), seen
    for key, recorded in RECORDED_OMITTED_EFFECT.items():
        eff, ci = seen[key]
        assert excludes_zero(ci), (key, _fmt(seen[key]))
        assert eff > 0 and recorded > 0, (key, eff, recorded)
        assert abs(eff - recorded) < RECORD_TOL, (key, eff, recorded)


def test_the_runner_reads_the_omitted_fit_as_a_leak_never_passed(scored):
    rows = scored["controls"]
    assert len(rows) == 6
    passed = [r for r in rows if r["omitted_status"] == RefutationStatus.PASSED]
    assert not passed, "omitted-confounder control read PASSED:\n" + _table(passed)
    assert all(
        r["omitted_status"] in (RefutationStatus.WARNING, RefutationStatus.FAILED) for r in rows
    ), _table(rows)


def test_undeclared_arms_are_absent_because_their_controls_do_not_respond(scored):
    rows = scored["undeclared"]
    assert {r["arm"] for r in rows} == set(UNDECLARED_ARMS), _undeclared_table(rows)
    assert len(rows) == 4, _undeclared_table(rows)  # sample_dropped x1, trigger_accepted x3
    responders = [r for r in rows if r["excludes_zero"]]
    assert not responders, (
        "a would-be control for an undeclared arm now RESPONDS to omitted confounding; "
        "re-measure before declaring it:\n" + _undeclared_table(responders)
    )
    for arm in UNDECLARED_ARMS:
        assert arm not in CONTROLS, (
            f"{arm} has a declared control but none of its structural-null outcomes leaves "
            f"its CI under omitted confounding (measured 2026-09-11, n = 1500: "
            + _undeclared_table([r for r in rows if r["arm"] == arm])
            + "); a control that cannot move under confounding is false assurance"
        )
