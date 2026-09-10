"""Calibration of the sensitivity READING against the DGP's planted truth (spec §6).

Measured 2026-09-10 (LinearDML, production RF nuisances, seed-21 Remibrutinib frame,
n=1500 = the live row cap): every one of the 11 planted true effects reads
``beyond_measured_confounding``; the null pairs read ``null_finding``; and the
omitted-confounder refits are NOT distinguishable from the correct fits — the
E-value cannot detect confounding, which is why it is a reading and not a gate.
These pins stop anyone re-adding a cutoff believing it would catch a confounder.
~50 s of pytest time (~70 s wall with imports), ~775 MiB peak RSS, slowest test ~17 s (measured on the droplet).
"""

from __future__ import annotations

import pytest
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.causal_engine import evalue
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

pytestmark = pytest.mark.heavy_ml

N_ROWS = 1500  # the live estimation row cap (routes/causal.py ``limit=1500``)
# Arm -> outcome pairs with a ZERO planted effect at n=1500 (spec §2.3): the arm
# does not target the outcome and the outcome is not downstream of it.
NULL_PAIRS = [
    ("treatment_arm", "persistent_180d"),
    ("copay_support", "treatment_initiated"),
    ("psp_enrolled", "treatment_initiated"),
    ("rep_detailing_high", "adherent_180d"),
    ("trigger_accepted", "adherent_180d"),
]


def _fit(df, treatment, outcome, covariates):
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)
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


def _reading(df, treatment, outcome, covariates):
    ate, ci = _fit(df, treatment, outcome, covariates)
    inputs = evalue.benchmark_inputs_from_frame(df, treatment, outcome, covariates)
    return evalue.classify(
        ate,
        ci,
        randomized=False,
        baseline_risk=inputs.baseline_risk,
        outcome_std=float(df[outcome].std()),
        naive_effect=inputs.naive_effect,
        covariate_factors=inputs.covariate_bias_factors,
        n_rows=len(df),
    )


@pytest.fixture(scope="module")
def frame():
    cfg = GeneratorConfig(
        seed=21, n_records=N_ROWS, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    return PatientGenerator(cfg).generate()


def _planted_pairs(frame):
    truth = frame.attrs["true_ate_by_arm"]
    return [
        (arm, outcome, list(ARM_REGISTRY[arm].confounders))
        for arm, outs in truth.items()
        if arm in ARM_REGISTRY and arm in frame.columns
        for outcome in outs
        if outcome in frame.columns
    ]


def test_every_planted_truth_reads_beyond_measured_confounding(frame):
    pairs = _planted_pairs(frame)
    assert len(pairs) == 11, [p[:2] for p in pairs]
    readings = {(a, o): _reading(frame, a, o, covs) for a, o, covs in pairs}
    not_beyond = {k: r.reading for k, r in readings.items() if r.reading != evalue.READING_BEYOND}
    assert not not_beyond, f"planted truths not served as robust: {not_beyond}"
    assert all(r.status == "passed" for r in readings.values())


def test_null_pairs_read_null_finding(frame):
    readings = {
        (a, o): _reading(frame, a, o, list(ARM_REGISTRY[a].confounders)).reading
        for a, o in NULL_PAIRS
    }
    assert all(r == evalue.READING_NULL for r in readings.values()), readings


def test_known_limit_omitted_confounder_is_not_distinguishable_by_e_value(frame):
    """Pinned LIMIT, not a goal: dropping the strongest declared confounder leaves the
    reading unchanged on at least 9 of 11 pairs. A cutoff gate could never catch it."""
    same = 0
    pairs = _planted_pairs(frame)
    for arm, outcome, covs in pairs:
        strongest = max(
            ARM_REGISTRY[arm].confounders, key=lambda c: abs(ARM_REGISTRY[arm].confounders[c])
        )
        reduced = [c for c in covs if c != strongest] or covs
        if (
            _reading(frame, arm, outcome, covs).reading
            == _reading(frame, arm, outcome, reduced).reading
        ):
            same += 1
    assert same >= 9, f"only {same}/11 pairs read the same with a confounder omitted"


def test_sensitivity_never_carries_a_failed_status_on_the_dgp(frame):
    for arm, outcome, covs in _planted_pairs(frame):
        assert _reading(frame, arm, outcome, covs).status in {"passed", "warning"}
