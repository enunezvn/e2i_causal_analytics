"""#2007: the negative-control reading is right on planted truth at the live cap.

Calibration pin for ``RefutationRunner._run_negative_control_test`` (rule
``negative_control_ci_vs_zero``: PASSED when the control's CI includes 0,
WARNING when it excludes 0 and |nc| < |original|, FAILED when it excludes 0 and
|nc| >= |original|) and for the registry that decides WHICH arm gets a control
(``src/api/routes/causal.py::_CAUSAL_NEGATIVE_CONTROL_OUTCOMES``, imported, not
retyped). The DGP frame, the 11 planted truths and the 9 structural nulls are
the ones ``test_sensitivity_calibration.py`` pins (imported), at n = 1500, the
live estimation row cap; every fit is PRODUCTION's LinearDML (RF nuisances from
``src/causal_engine/nuisance_config.py``, X = W, #2031) and every verdict comes
from the real runner.

Two fits per declared control, exactly the pair the registry was measured on
(``docs/demos/results/2026-09-11_negative_control_disproof/disproof.md``):

* ADJUSTED — the control fitted on the truth's own confounders. The control is
  structurally null, so a correct adjustment must read PASSED; a WARNING is a
  5 % chance positive and at most one is tolerated; a FAILED means the reading
  would block a planted truth and must never happen.
* OMITTED — the control fitted with the confounders replaced by a seeded noise
  column (``run_disproof.py::fit_omitted``). The declared controls are the
  nulls that MOVE out of their CI under that omission; the pin reproduces the
  recorded movement (seeded fits, |delta| < 0.01 on the seed mean) and checks
  the runner reads the leak as WARNING/FAILED, never PASSED.

#2031 (codex r1, 2026-09-12): "responds" is a MULTI-SEED rule. The 2026-09-11
registry was measured on ONE seed (42) at RF leaf 5 -- the same single-draw
problem #2031 fixes in production -- and rep_detailing_high's candidate control
``persistent_180d`` turned out to be a high draw: over seeds (42, 7, 123, 2024,
99, 314) it leaves its CI on 0/6 seeds at production's leaf 50 (2/6 at leaf 5;
stable leak ~ +0.041 at SE ~ 0.026, ~ 1.6 SE). It is now UNDECLARED like
``sample_dropped`` / ``trigger_accepted`` -- a control that cannot move under
confounding is false assurance. A declared control must respond on >= 5/6
seeds; an undeclared arm's candidates on <= 1/6; adjusted controls must exclude
0 on <= 1/6.

Measured 2026-09-12 on the droplet (leaf 50, 6 seeds, ~71 fits, ~2 min): 5
truth-with-control rows (copay_support x3, psp_enrolled x2), all PASSED
adjusted at seed 42, adjusted excludes 0 on 0/6 seeds for both controls; under
the omitted fit copay's control responds 6/6 (seed mean +0.0616), psp's 6/6
(+0.0858); the runner at seed 42 reads 5 WARNING / 0 FAILED / 0 PASSED (the
leaked control is smaller than every claimed effect); 7 undeclared would-be
controls (rep_detailing_high x3, sample_dropped x1, trigger_accepted x3), 0/6
exclude 0 on every one. Pre-#2031 single-seed leaf-5 numbers per row:
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
from src.causal_engine.nuisance_config import linear_dml_rf_params
from src.causal_engine.refutation_runner import (
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_sensitivity_calibration import (  # noqa: F401
    N_ROWS,
    NULL_PAIRS,
    _planted_pairs,
    frame,
)

pytestmark = [
    pytest.mark.heavy_ml,
    # ~71 LinearDML fits at ~1.5 s each plus the frame build, ~2 min on the
    # droplet; the CI heavy lane's stall watchdog is 1200 s (backend-tests.yml,
    # #1655 rule: marker <= 600).
    pytest.mark.timeout(600),
]

DATASET = "patient_journeys"  # the only dataset with declared controls
CONTROLS: Dict[str, str] = _CAUSAL_NEGATIVE_CONTROL_OUTCOMES[DATASET]
# #2031: rep_detailing_high joined the undeclared arms (was declared with
# persistent_180d on the single-seed leaf-5 measurement).
UNDECLARED_ARMS = ("rep_detailing_high", "sample_dropped", "trigger_accepted")
PRODUCTION_SEED = 42
SEEDS = (42, 7, 123, 2024, 99, 314)
MIN_RESPONDING_SEEDS = 5  # declared control: CI excludes 0 on >= 5/6 seeds
MAX_CHANCE_SEEDS = 1  # undeclared candidate / adjusted control: <= 1/6 (alpha 0.05)
# The recorded omitted-fit movement per declared control: the 6-SEED MEAN at
# leaf 50 (#2031; the single-seed leaf-5 record was copay +0.0517, psp +0.0895,
# rep +0.0577). The fits are seeded, so a drift beyond RECORD_TOL is a changed
# generator or estimator, not noise.
RECORDED_OMITTED_EFFECT = {
    ("copay_support", "treatment_initiated"): 0.0616,
    ("psp_enrolled", "treatment_initiated"): 0.0858,
}
RECORD_TOL = 0.01
MAX_ADJUSTED_WARNINGS = 1  # alpha = 0.05 chance positive across 2 distinct controls
# #2031: was {"copay_support": 3, "psp_enrolled": 2, "rep_detailing_high": 1}.
EXPECTED_ROWS_PER_ARM = {"copay_support": 3, "psp_enrolled": 2}
# Under the OMITTED fit the runner's FAILED boundary is |control| >= |claimed|.
# Measured 2026-09-12 at leaf 50, seed 42: no row crosses it (copay's control
# +0.0564 vs claimed +0.1097 / +0.0904 / +0.1085; psp's control +0.0863 vs
# claimed +0.0917 / +0.0966), so all five read WARNING. #2031: was 4 WARNING + 2 FAILED
# ({psp -> persistent_180d, rep -> treatment_initiated}) at leaf 5, seed 42.
# Pinned exactly so a runner regression that read every leak as FAILED (or as
# PASSED) cannot pass on "not PASSED" alone; the boundary arithmetic is also
# re-checked row by row below.
EXPECTED_OMITTED_FAILED: set = set()
EXPECTED_OMITTED_WARNING_COUNT = 5


def _rf_params(seed: int) -> dict:
    """Production's RF params (``nuisance_config``) with only random_state swapped."""
    return {**linear_dml_rf_params(), "random_state": seed}


def _fit_production(Y, T, X, seed):
    """Production's LinearDML: RF nuisances from ``nuisance_config`` (#2031), X = W."""
    m = LinearDML(
        model_y=RandomForestRegressor(**_rf_params(seed)),
        model_t=RandomForestClassifier(**_rf_params(seed)),
        discrete_treatment=True,
        random_state=seed,
    )
    m.fit(Y, T, X=X, W=X)
    inf = m.ate_inference(X)
    lo, hi = (float(v) for v in inf.conf_int_mean())
    return float(inf.mean_point), (lo, hi)


def fit_adjusted(df, treatment, outcome, covariates, seed=PRODUCTION_SEED):
    """The route's fit: the arm's declared confounders as X = W."""
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)
    return _fit_production(Y, T, X, seed)


def fit_omitted(df, treatment, outcome, seed=PRODUCTION_SEED, noise_seed=42):
    """From
    docs/demos/results/2026-09-11_negative_control_disproof/run_disproof.py::fit_omitted
    (docs/ is not importable from the test tree), nuisances re-pointed at the
    shared production config (#2031): the production estimator with the
    confounders OMITTED -- X is a seeded standard-normal noise column (noise seed
    fixed; ``seed`` drives the nuisance / estimator random_state), so the model
    has nothing real to condition on."""
    rng = np.random.default_rng(noise_seed)
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = rng.standard_normal((len(df), 1))
    return _fit_production(Y, T, X, seed)


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


def _seeds_fmt(fits: Dict[int, Any]) -> str:
    return " ".join(
        f"s{s}={eff:+.4f}{'*' if excludes_zero(ci) else ''}" for s, (eff, ci) in fits.items()
    )


def _table(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{r['arm']}->{r['outcome']} (truth, original {r['original']:+.4f}) | control "
        f"{r['nc_outcome']}: adjusted {_fmt(r['adjusted'])} -> {r['adjusted_status'].value}; "
        f"omitted {_fmt(r['omitted'])} -> {r['omitted_status'].value}"
        for r in rows
    )


def _seed_table(by_control: Dict[Tuple[str, str], Dict[str, Dict[int, Any]]]) -> str:
    return "\n".join(
        f"{arm}->{nc} adjusted: {_seeds_fmt(d['adjusted'])}\n"
        f"{arm}->{nc} omitted : {_seeds_fmt(d['omitted'])}"
        for (arm, nc), d in by_control.items()
    )


def _undeclared_table(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{r['arm']}->{r['outcome']} (would-be control, UNDECLARED): omitted "
        f"{_seeds_fmt(r['omitted_by_seed'])} responds {r['responding_seeds']}/{len(SEEDS)}"
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
    # One adjusted and one omitted fit per DISTINCT (arm, control) per seed; the
    # three copay truths share one control and one confounder set.
    by_control: Dict[Tuple[str, str], Dict[str, Dict[int, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    for arm, outcome, covs in planted:
        nc_outcome = CONTROLS.get(arm)
        if nc_outcome is None:
            continue
        # a control cannot be the outcome it is meant to check (the route's
        # _negative_control_outcome returns None for that case)
        assert nc_outcome != outcome, (arm, outcome)
        original, original_ci = fit_adjusted(frame, arm, outcome, covs)
        assert excludes_zero(original_ci), ("planted truth not detected", arm, outcome, original_ci)
        key = (arm, nc_outcome)
        if key not in by_control:
            by_control[key] = {
                "adjusted": {s: fit_adjusted(frame, arm, nc_outcome, covs, s) for s in SEEDS},
                "omitted": {s: fit_omitted(frame, arm, nc_outcome, s) for s in SEEDS},
            }
        adjusted_fit = by_control[key]["adjusted"][PRODUCTION_SEED]
        omitted_fit = by_control[key]["omitted"][PRODUCTION_SEED]
        adjusted = _score(runner, original, nc_outcome, adjusted_fit, n)
        omitted = _score(runner, original, nc_outcome, omitted_fit, n)
        rows.append(
            {
                "arm": arm,
                "outcome": outcome,
                "confounders": covs,
                "original": original,
                "original_ci": original_ci,
                "nc_outcome": nc_outcome,
                "adjusted": adjusted_fit,
                "adjusted_status": adjusted.status,
                "omitted": omitted_fit,
                "omitted_status": omitted.status,
            }
        )
    undeclared: List[Dict[str, Any]] = []
    for arm, outcome in NULL_PAIRS:
        if arm not in UNDECLARED_ARMS:
            continue
        fits = {s: fit_omitted(frame, arm, outcome, s) for s in SEEDS}
        undeclared.append(
            {
                "arm": arm,
                "outcome": outcome,
                "omitted_by_seed": fits,
                "responding_seeds": sum(excludes_zero(ci) for _, ci in fits.values()),
            }
        )
    print(
        "\n[negative-control calibration] planted truths with a declared control (seed 42)\n"
        + _table(rows)
    )
    print(
        "\n[negative-control calibration] declared controls over seeds (* = CI excludes 0)\n"
        + _seed_table(by_control)
    )
    print(
        "\n[negative-control calibration] undeclared arms, would-be controls over seeds\n"
        + _undeclared_table(undeclared)
    )
    return {"controls": rows, "by_control": by_control, "undeclared": undeclared}


def test_the_registry_declares_exactly_the_two_measured_responders():
    # #2031: was {copay_support, psp_enrolled, rep_detailing_high}.
    assert set(CONTROLS) == {"copay_support", "psp_enrolled"}, CONTROLS
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


def test_adjusted_controls_stay_inside_their_ci_across_seeds(scored):
    # #2031: the adjusted (correct) fit of a structurally null control excludes
    # 0 on at most 1/6 seeds -- measured 0/6 for both controls at leaf 50.
    by_control = scored["by_control"]
    assert set(by_control) == set(RECORDED_OMITTED_EFFECT), set(by_control)
    for key, d in by_control.items():
        k = sum(excludes_zero(ci) for _, ci in d["adjusted"].values())
        assert k <= MAX_CHANCE_SEEDS, (key, k, _seeds_fmt(d["adjusted"]))


def test_each_declared_control_leaves_its_ci_under_omitted_confounding(scored):
    # #2031: "responds" is >= 5/6 seeds at production's config (was one seed);
    # the recorded movement is the seed MEAN.
    by_control = scored["by_control"]
    assert set(by_control) == set(RECORDED_OMITTED_EFFECT), set(by_control)
    for key, recorded in RECORDED_OMITTED_EFFECT.items():
        fits = by_control[key]["omitted"]
        k = sum(excludes_zero(ci) for _, ci in fits.values())
        assert k >= MIN_RESPONDING_SEEDS, (key, k, _seeds_fmt(fits))
        effects = [eff for eff, _ in fits.values()]
        assert all(e > 0 for e in effects) and recorded > 0, (key, effects, recorded)
        mean_eff = float(np.mean(effects))
        assert abs(mean_eff - recorded) < RECORD_TOL, (key, mean_eff, recorded)


def _boundary(r: Dict[str, Any]) -> str:
    """The FAILED-boundary arithmetic for one row, for assertion messages."""
    nc = abs(r["omitted"][0])
    claimed = abs(r["original"])
    op = ">=" if nc >= claimed else "<"
    return f"{r['arm']}->{r['outcome']}: |control| {nc:.4f} {op} |claimed| {claimed:.4f}"


def test_the_runner_reads_the_omitted_fit_as_a_leak_never_passed(scored):
    rows = scored["controls"]
    assert len(rows) == 5  # #2031: was 6 (rep_detailing_high -> treatment_initiated dropped)
    passed = [r for r in rows if r["omitted_status"] == RefutationStatus.PASSED]
    assert not passed, "omitted-confounder control read PASSED:\n" + _table(passed)
    by_status = Counter(r["omitted_status"] for r in rows)
    assert by_status == {
        RefutationStatus.WARNING: EXPECTED_OMITTED_WARNING_COUNT,
        **(
            {RefutationStatus.FAILED: len(EXPECTED_OMITTED_FAILED)}
            if EXPECTED_OMITTED_FAILED
            else {}
        ),
    }, "omitted-fit verdict split drifted:\n" + "\n".join(
        f"{_boundary(r)} -> {r['omitted_status'].value}" for r in rows
    )
    failed = {
        (r["arm"], r["outcome"]) for r in rows if r["omitted_status"] == RefutationStatus.FAILED
    }
    assert failed == EXPECTED_OMITTED_FAILED, (
        "FAILED rows are not the measured boundary crossers:\n"
        + "\n".join(f"{_boundary(r)} -> {r['omitted_status'].value}" for r in rows)
    )
    # The verdict must agree with the boundary arithmetic row by row.
    for r in rows:
        crosses = abs(r["omitted"][0]) >= abs(r["original"])
        expected = RefutationStatus.FAILED if crosses else RefutationStatus.WARNING
        assert r["omitted_status"] == expected, _boundary(r) + f" -> {r['omitted_status'].value}"


def test_undeclared_arms_are_absent_because_their_controls_do_not_respond(scored):
    rows = scored["undeclared"]
    assert {r["arm"] for r in rows} == set(UNDECLARED_ARMS), _undeclared_table(rows)
    # #2031: was 4 (sample_dropped x1, trigger_accepted x3); rep_detailing_high adds 3.
    assert len(rows) == 7, _undeclared_table(rows)
    responders = [r for r in rows if r["responding_seeds"] > MAX_CHANCE_SEEDS]
    assert not responders, (
        "a would-be control for an undeclared arm now RESPONDS to omitted confounding on "
        f"more than {MAX_CHANCE_SEEDS}/{len(SEEDS)} seeds; re-measure before declaring it:\n"
        + _undeclared_table(responders)
    )
    for arm in UNDECLARED_ARMS:
        assert arm not in CONTROLS, (
            f"{arm} has a declared control but none of its structural-null outcomes leaves "
            f"its CI on more than {MAX_CHANCE_SEEDS}/{len(SEEDS)} seeds under omitted "
            f"confounding (measured 2026-09-12, leaf 50, n = 1500: "
            + _undeclared_table([r for r in rows if r["arm"] == arm])
            + "); a control that cannot move under confounding is false assurance"
        )
