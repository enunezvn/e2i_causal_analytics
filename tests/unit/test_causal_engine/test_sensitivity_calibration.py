"""Calibration of the sensitivity READING against the DGP's planted truth (spec §6).

Measured 2026-09-10 (LinearDML, production RF nuisances, seed-21 Remibrutinib frame,
n=1500 = the live row cap): every one of the 11 planted true effects reads
``beyond_measured_confounding``; the structurally null pairs read ``null_finding``;
the real-but-indirect ``treatment_arm -> persistent_180d`` effect sits below the
detection limit at this n; and the omitted-confounder refits are NOT distinguishable
from the correct fits — the E-value cannot detect confounding, which is why it is a
reading and not a gate. These pins stop anyone re-adding a cutoff believing it would
catch a confounder.
~63 s of pytest time (~80 s wall with imports), ~780 MiB peak RSS, slowest test ~19 s
(measured on the droplet).
"""

from __future__ import annotations

import pytest
from econml.dml import LinearDML

from src.causal_engine import evalue
from src.causal_engine.nuisance_config import linear_dml_model_t, linear_dml_model_y
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

pytestmark = pytest.mark.heavy_ml

N_ROWS = 1500  # the live estimation row cap (routes/causal.py ``limit=1500``)
# Arm -> outcome pairs with a STRUCTURALLY zero effect: the arm is neither a direct
# target of the outcome nor an input to the function that builds it, nor upstream of
# any such input. Verified 2026-09-10 against the generator code (line numbers as of
# this commit), NOT assumed from the truth attrs:
#
#   Outcome builders and their arm inputs
#   - treatment_initiated: dgp/initiation_outcomes.py:57 generate_initiation_outcome
#     takes treatment_arm + rep_detailing_high + sample_dropped + trigger_accepted
#     (lines 66-70); copay_support and psp_enrolled are NOT inputs.
#   - adherent_180d / low_gap_180d: dgp/adherence_outcomes.py:117
#     generate_adherence_outcomes takes treatment_arm + copay_support + psp_enrolled
#     (lines 119-128); rep/sample/trigger are NOT inputs, nor is treatment_initiated.
#   - persistent_180d: generators/cohort_outcomes.py:180
#     generate_discontinuation_outcomes takes treatment_arm (arm_core, line 240) +
#     copay_support (line 257) + psp_enrolled (line 267) + the clinical axis;
#     rep/sample/trigger are NOT inputs, nor is treatment_initiated. The Remibrutinib
#     axis rebuild (generators/patient_generator.py:830) passes the same inputs.
#   Arm assignments (no back path): every arm is a Bernoulli draw on confounders +
#   engagement_score drawn BEFORE any outcome — generators/patient_generator.py:231
#   (treatment_arm), :267 (rep), :279 (sample), :294 (trigger), :361 (copay),
#   :385 (psp); none reads an outcome or another arm.
#   Declared targets (dgp/treatment_arm.py ARM_REGISTRY): copay -> adherent, low_gap,
#   persistent (:191); psp -> adherent, persistent (:203); rep/sample/trigger ->
#   treatment_initiated only (:214, :225, :236).
#
# Two further structural nulls are EXCLUDED because they read
# ``beyond_measured_confounding`` at n=1500 by chance: the spec's named chance positive
# ``sample_dropped -> adherent_180d`` (§2.3; measured ate=0.0627 CI [0.0119, 0.1134])
# and its sibling ``sample_dropped -> low_gap_180d`` (measured ate=0.0595
# CI [0.0109, 0.1080]) — both binaries threshold ONE shared adherence latent
# (adherence_outcomes.py docstring), so the same chance draw lands on both.
NULL_PAIRS = [
    ("copay_support", "treatment_initiated"),
    ("psp_enrolled", "treatment_initiated"),
    ("rep_detailing_high", "adherent_180d"),
    ("trigger_accepted", "adherent_180d"),
    ("rep_detailing_high", "low_gap_180d"),
    ("trigger_accepted", "low_gap_180d"),
    ("rep_detailing_high", "persistent_180d"),
    ("sample_dropped", "persistent_180d"),
    ("trigger_accepted", "persistent_180d"),
]


def _fit(df, treatment, outcome, covariates):
    """The PRODUCTION fit (LinearDMLWrapper): RF nuisances from ``nuisance_config``
    (#2031, codex r1 -- leaf 50; the local leaf-5 mirror no longer matched
    production) and ``X = W = covariates`` like the wrapper."""
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)
    m = LinearDML(
        model_y=linear_dml_model_y(),
        model_t=linear_dml_model_t(),
        discrete_treatment=True,
        random_state=42,
    )
    m.fit(Y, T, X=X, W=X)
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
        covariates_measured=len(covariates),
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


def test_persistence_indirect_effect_is_below_detection_at_the_live_cap(frame):
    """DETECTION-LIMIT observation, not a null truth.

    ``treatment_arm -> persistent_180d`` is a REAL indirect effect: the arm enters the
    discontinuation logit (cohort_outcomes.py:240 ``arm_core``) and spec §2.3 says the
    pair "is not spurious: persistence is built from the same adherence machinery the
    arm moves" — it reads ``beyond`` at n=5000. At the live row cap it reads
    ``null_finding`` because the effect is below what n=1500 can resolve (measured
    2026-09-10: ate=0.0094, 95% CI [-0.0680, 0.0869]). Spec §8.3 uses exactly this
    pair as the live null-finding probe. A null reading here is a PRECISION statement
    about the sample, never a claim that the truth is zero.
    """
    covs = list(ARM_REGISTRY["treatment_arm"].confounders)
    ate, (lo, hi) = _fit(frame, "treatment_arm", "persistent_180d", covs)
    assert lo < 0.0 < hi, (ate, lo, hi)
    assert _reading(frame, "treatment_arm", "persistent_180d", covs).reading == evalue.READING_NULL


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
