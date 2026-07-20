"""Causal DGP primitives for the synthetic causal-validation dataset.

Adds a per-unit BINARY treatment arm confounded with emitted covariates,
a per-(brand) CATE scaling, and a prevalence-banded binary outcome carrying
a known E[tau(X)] = TRUE_ATE. Used by patient_generator (Shard 03) and the
trigger/cohort generators (Shards 05/06).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, overload

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from ..config import DGP_CONFIGS, Brand, DGPType

# Segment labels — MUST match config.cate_by_segment keys and the
# ml_predictions.segment_assignment values the KPI RPC groups by
# (database/migrations/044_kpi_query_allowlist.sql:122).
SEGMENT_HIGH = "high_severity"
SEGMENT_MEDIUM = "medium_severity"
SEGMENT_LOW = "low_severity"


# The covariates the (intentionally confounded) treatment arm is assigned on —
# the SINGLE SOURCE OF TRUTH for "what the DGP confounds on". assign_treatment_arm
# below reads its propensity inputs through this tuple, so it cannot drift from the
# code. EVERY production estimator that estimates an effect of treatment_arm MUST
# adjust for these (carry them in its adjustment set / effect modifiers); otherwise
# it silently reports the confounded naive diff-in-means (biased upward, ~0.28 vs
# the designed ~0.18 true effect) as "the treatment effect". That contract is
# locked by tests/unit/test_synthetic/test_arm_confounder_contract.py and made
# visible to the analyst by the naive-vs-adjusted "confounding bias removed"
# surfacing (Option D).
ARM_CONFOUNDERS: Tuple[str, str] = ("disease_severity", "academic_hcp")


@dataclass(frozen=True)
class ArmSpec:
    """Declarative description of ONE binary treatment arm.

    Adding an arm means adding a spec — the confounder-contract guard then forces
    every declared confounder into the analysis covariate allowlist, so a new arm
    cannot ship with an un-adjustable backdoor (the anti-mocking harm: the
    estimator would silently report the confounded naive diff-in-means).
    """

    name: str
    confounders: Dict[str, float]  # covariate -> propensity logit coefficient
    intercept: float  # sets the base treatment share
    cate_by_segment: Dict[str, float]  # latent CATE per severity segment
    target_outcomes: Tuple[str, ...]  # curated outcomes this arm wires
    # Covariates centered before entering the logit, so the intercept sets the
    # base share rather than being absorbed by the covariate's scale.
    center: Dict[str, float] = field(default_factory=dict)
    propensity_col: str = ""

    def __post_init__(self) -> None:
        if not self.propensity_col:
            object.__setattr__(self, "propensity_col", f"{self.name}_propensity")


# Phase 1 (COMM-ARMS): copay_support. Constants are MEASURED, not guessed — see
# docs/superpowers/plans/2026-07-19-dgp-commercial-arms-phase1-copay-support.md.
# The CATE map needs a WIDE high-medium gap: at n=3000 the forest cannot resolve
# the medium segment at ANY effect size (measured across maps 0.28/0.20/0.12 ..
# 0.40/0.22/0.04), and even at n=8000 the narrower 0.32/0.20/0.08 map leaves
# Remibrutinib/seed21 INVERTED (high 0.1165 < med 0.1264). 0.44/0.18/0.06 orders
# for all 3 brands, |ATE err| 0.003-0.035, planted ATE 0.083-0.141 (design band
# +8-12pp). The wider 0.55/0.16/0.05 also orders but drifts ATE to 0.156 (out of
# band) with worse error -> rejected.
_COPAY_BETA_INS_ACCESS = -0.90  # LOWER access -> MORE support (real-world skew)
_COPAY_BETA_SEVERITY = 0.25  # sicker -> more support
_COPAY_INTERCEPT = -0.40  # base share ~0.34-0.36 (measured)
_COPAY_CATE = {
    "high_severity": 0.44,
    "medium_severity": 0.18,
    "low_severity": 0.06,
}

# Phase 2 (COMM-ARMS): psp_enrolled (patient support program). Constants are MEASURED,
# not guessed (harness: scratchpad psp_disproof/psp_*_sweep, 2026-07-19). psp skews to
# sicker + more-engaged + academic-HCP patients (the three design confounders, all
# already allowlisted numeric covariates). The design band is +5-10pp (a shade weaker
# than copay's +8-12pp). This _PSP_CATE is the ADHERENCE-latent map (adherent_180d); the
# persistence effect lives on the discontinuation logit (_PSP_DISC_LOGIT in
# cohort_outcomes.py). The WIDE high-medium gap is retained because the n=8000
# CausalForestDML medium-segment resolution floor (Phase 1) is an estimator property.
# Map tuning: the first guess {0.34,0.14,0.05} left Fabhalta/seed21 adherent h-m at a
# razor-thin +0.0152; lowering medium to 0.13 and lifting high to 0.38 raised it to
# +0.0327 (m-l stays +0.057) while keeping Remibrutinib(base) ATE 0.098 in-band. Kisqali
# (1.40 scale) lands 0.122 -- above +10pp by brand-scaling design, exactly as copay's
# Kisqali did (0.140); the gate asserts |est-true|<0.15, not an ATE band.
_PSP_BETA_SEVERITY = 0.20  # sicker -> more likely enrolled in a support program
_PSP_BETA_ENGAGEMENT = 0.16  # more engaged -> more likely to enroll (centered at 5)
_PSP_BETA_ACADEMIC = 0.45  # academic HCPs enroll patients in PSPs more often
_PSP_INTERCEPT = -1.20  # base share ~0.376 (MEASURED, brand-invariant; prop AUC ~0.64)
_PSP_CATE = {
    "high_severity": 0.38,
    "medium_severity": 0.13,
    "low_severity": 0.05,
}

# COMM-ARMS Phase 3 (2026-07-20): rep_detailing_high + sample_dropped, TWO arms that
# fold into the INITIATION latent (treatment_initiated) — the SAME latent as
# treatment_arm, unlike copay/psp which target adherence/persistence. Constants are
# MEASURED (harness: scratchpad phase3_disproof v1-v4, 2026-07-20), not guessed. Both
# confound on academic_hcp + engagement_score (already-allowlisted numeric covariates):
# a rep who details heavily and drops samples skews to academic + high-engagement HCPs.
#
# WHY the WIDE rep high-medium gap ({0.36,0.14,0.05}, high/med ratio 2.57 ~ copay's 2.44):
# rep is a WEAK arm (+3-6pp design) and at that magnitude CausalForestDML sits at the
# n=8000 medium-segment resolution floor (the copay/psp phenomenon). The first guess
# {0.30,0.15,0.06} flipped rep's recovered high>med>low ordering at 3/12 seed-brand cells
# (Remi/123, Fabh/7, Fabh/123); the copay-shaped {0.36,0.14,0.05} cut it to 1/12 (only
# Fabhalta/seed123, the flattest 0.70-scale brand) WITHOUT inflating the RD-scale ATE —
# nonlinear RD compression absorbs the wider LATENT gap, so rep's realized ATE stays
# ~+6.5pp (Remi base). ALL cells recover the ATE (|est-true| <= 0.028 << 0.15) and order
# strictly at the gate seed (21); the recovery gate asserts ordering@seed21 + ATE multi-
# seed, mirroring treatment_arm's own seed-21 ordering gate (test_dgp_recovery_probe.py).
# sample {0.20,0.10,0.04} orders 12/12; its intercept -1.05 keeps share ~0.37 (rep -0.80
# => ~0.49) — realistic detailing/sampling penetration, well below the 0.61/0.50 an
# un-lowered intercept gave (share is RD-ATE-neutral: the effect is population-weighted).
_REP_BETA_ACADEMIC = 0.60  # academic HCPs get more high-touch rep detailing
_REP_BETA_ENGAGEMENT = 0.18  # more-engaged HCPs get more detailing (centered at 5)
_REP_INTERCEPT = -0.80  # base share ~0.49 (measured)
_REP_CATE = {
    "high_severity": 0.36,
    "medium_severity": 0.14,
    "low_severity": 0.05,
}
_SAMPLE_BETA_ACADEMIC = 0.40  # academic HCPs more often have samples dropped
_SAMPLE_BETA_ENGAGEMENT = 0.12  # more-engaged HCPs more often sampled (centered at 5)
_SAMPLE_INTERCEPT = -1.05  # base share ~0.37 (measured)
_SAMPLE_CATE = {
    "high_severity": 0.20,
    "medium_severity": 0.10,
    "low_severity": 0.04,
}

ARM_REGISTRY: Dict[str, ArmSpec] = {
    "treatment_arm": ArmSpec(
        name="treatment_arm",
        confounders={"disease_severity": 0.30, "academic_hcp": 0.80},
        intercept=-2.0,
        cate_by_segment={},  # sourced from brand_scaled_cate (brand-dependent)
        target_outcomes=("treatment_initiated", "adherent_180d", "low_gap_180d"),
        center={"disease_severity": 5.0},
        propensity_col="propensity_score",  # legacy column name, NOT <name>_propensity
    ),
    "copay_support": ArmSpec(
        name="copay_support",
        confounders={
            "insurance_access_score": _COPAY_BETA_INS_ACCESS,
            "disease_severity": _COPAY_BETA_SEVERITY,
        },
        intercept=_COPAY_INTERCEPT,
        cate_by_segment=_COPAY_CATE,
        target_outcomes=("adherent_180d", "low_gap_180d", "persistent_180d"),
        center={"disease_severity": 5.0},
    ),
    "psp_enrolled": ArmSpec(
        name="psp_enrolled",
        confounders={
            "disease_severity": _PSP_BETA_SEVERITY,
            "engagement_score": _PSP_BETA_ENGAGEMENT,
            "academic_hcp": _PSP_BETA_ACADEMIC,
        },
        intercept=_PSP_INTERCEPT,
        cate_by_segment=_PSP_CATE,
        target_outcomes=("adherent_180d", "persistent_180d"),
        center={"disease_severity": 5.0, "engagement_score": 5.0},
    ),
    "rep_detailing_high": ArmSpec(
        name="rep_detailing_high",
        confounders={
            "academic_hcp": _REP_BETA_ACADEMIC,
            "engagement_score": _REP_BETA_ENGAGEMENT,
        },
        intercept=_REP_INTERCEPT,
        cate_by_segment=_REP_CATE,
        target_outcomes=("treatment_initiated",),
        center={"engagement_score": 5.0},
    ),
    "sample_dropped": ArmSpec(
        name="sample_dropped",
        confounders={
            "academic_hcp": _SAMPLE_BETA_ACADEMIC,
            "engagement_score": _SAMPLE_BETA_ENGAGEMENT,
        },
        intercept=_SAMPLE_INTERCEPT,
        cate_by_segment=_SAMPLE_CATE,
        target_outcomes=("treatment_initiated",),
        center={"engagement_score": 5.0},
    ),
}


def insurance_access_from_type(insurance_type: np.ndarray) -> np.ndarray:
    """Numeric access gradient (commercial best -> uninsured worst) from the raw
    categorical. Reads the _INIT_INS_ACCESS SSOT so this persisted covariate can
    never drift from the initiation prognostic offset that uses the same map.

    WHY a numeric proxy: the EconML/DoWhy executors cannot adjust on a raw
    categorical, so copay_support's real-world backdoor (insurance coverage) is
    carried by this score; ``insurance_type`` stays a cohort FILTER only.
    Unknown categories map to the neutral 0.0 rather than raising.
    """
    return np.array(
        [_INIT_INS_ACCESS.get(str(i), 0.0) for i in np.asarray(insurance_type)],
        dtype=float,
    )


def assign_arm_from_spec(
    spec: ArmSpec,
    covariates: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confounded binary arm T ~ Bernoulli(e(X)) for ANY ArmSpec.

    e(X) = sigmoid(intercept + sum_k beta_k * (X_k - center_k)), clipped to
    [0.01, 0.99] to GUARANTEE overlap (no near-separation) so the propensity is
    estimable with common support. Generalizes assign_treatment_arm; that
    function now delegates here and is BIT-IDENTICAL (locked by
    tests/unit/test_synthetic/test_arm_spec.py).
    """
    # Derive n from a DECLARED confounder, never from an arbitrary caller key:
    # Phase 2/3 add new callers of this function, and a mismatched dict must fail
    # loud rather than yield a silently mis-broadcast propensity.
    missing = [c for c in spec.confounders if c not in covariates]
    if missing:
        raise KeyError(
            f"arm {spec.name!r} declares confounder(s) {missing} that the caller did not "
            "supply; a propensity omitting a declared backdoor is silently confounded"
        )
    lengths = {c: len(np.asarray(covariates[c])) for c in spec.confounders}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"arm {spec.name!r} got ragged covariate length(s): {lengths}")
    logit = np.full(next(iter(lengths.values())), float(spec.intercept))
    for cov, beta in spec.confounders.items():
        values = np.asarray(covariates[cov], dtype=float)
        logit = logit + beta * (values - spec.center.get(cov, 0.0))
    propensity = np.clip(expit(logit), 0.01, 0.99)
    arm = (rng.random(len(propensity)) < propensity).astype(int)
    return arm, propensity


def assign_treatment_arm(
    covariates: Dict[str, np.ndarray],
    rng: np.random.Generator,
    beta_severity: float = 0.30,
    beta_academic: float = 0.80,
    intercept: float = -2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confounded binary arm T ~ Bernoulli(e(X)).

    e(X) = sigmoid(intercept + beta_severity*(severity-5) + beta_academic*academic),
    clipped to [0.01,0.99] to GUARANTEE overlap (no near-separation) so the
    propensity is estimable with common support. Returns (arm[0/1], propensity).

    Phase 1: delegates to the generalized ``assign_arm_from_spec``. The explicit
    keyword defaults are retained so existing callers/tests that override them
    keep working; when they are at their defaults this is BIT-IDENTICAL to the
    registry spec (locked by test_arm_spec.py::test_delegation_is_byte_identical).
    """
    severity_key, academic_key = ARM_CONFOUNDERS
    spec = ArmSpec(
        name="treatment_arm",
        confounders={severity_key: beta_severity, academic_key: beta_academic},
        intercept=intercept,
        cate_by_segment={},
        target_outcomes=ARM_REGISTRY["treatment_arm"].target_outcomes,
        center={severity_key: 5.0},
        propensity_col="propensity_score",
    )
    return assign_arm_from_spec(spec, covariates, rng)


# Per-brand multiplicative scale on the base CATE map. Distinct per brand so a
# Kisqali probe yields a DIFFERENT structure than Remibrutinib (INDEX gate 6);
# all > 0 so ordering high>medium>low is preserved under scaling.
_BRAND_CATE_SCALE: Dict[Brand, float] = {
    Brand.REMIBRUTINIB: 1.00,  # base map, unscaled
    Brand.KISQALI: 1.40,  # stronger heterogeneity (oncology CDK4/6 responder split)
    Brand.FABHALTA: 0.70,  # flatter (rare PNH, smaller effect spread)
}


# T11 (2026-06-22): the prognostic-driver enrichment of the initiation outcome
# (binary_outcome_with_cate) compresses the binarized RD-scale treatment effect by
# spreading the latent baseline (more prognostic variance → lower threshold-crossing
# density → smaller mean tau_i). Measured: at _INIT_DRIVER_SCALE=0.75 the realized
# true_ate drops to ~0.14, below the designed [0.15,0.50] band. We compensate by planting
# a proportionally larger LATENT CATE so the RECOVERED RD-scale true_ate is restored to
# ~the pre-T11 baseline (~0.177) — verified faithfully (true_ate 0.137→0.177 at boost
# 1.30) with NO AUC change (~0.80), since τ is small vs the prognostic+noise variance and
# treatment_arm is not a model feature. The boost is applied INSIDE binary_outcome_with_cate
# (the initiation-only outcome fn, single SSOT both the generator and the reseed inherit),
# NOT in brand_scaled_cate — so brand_scaled_cate stays the pure base×brand-scale map
# (Remi@1.0 == base config). CATE high>med>low ordering is preserved (uniform factor).
_INIT_LATENT_CATE_BOOST = 1.30

# Initiation-latent baseline coefficients + noise/prevalence, promoted to module SSOT so
# both binary_outcome_with_cate (below) AND the multi-arm initiation folder
# (initiation_outcomes.generate_initiation_outcome, COMM-ARMS Phase 3) build the SAME
# latent. These were TUNED by the Task 03.5 recovery probe — see binary_outcome_with_cate's
# docstring for the full derivation (0.20/0.30/1.0 flipped medium-vs-low; 0.6 is the noise
# sweet spot). A drift here silently de-calibrates BOTH the initiation AUC gate and the
# ATE/CATE recovery gate, so they live in one place.
_INIT_BASE_SEVERITY_COEF = 0.10
_INIT_BASE_ACADEMIC_COEF = 0.15
_INIT_NOISE_STD = 0.6
_INIT_TARGET_PREVALENCE = 0.35


# ---------------------------------------------------------------------------
# Phase 3 (CLIN-SEG-P3, 2026-07-13): biologic-experience differential CATE.
# ---------------------------------------------------------------------------
# Remibrutinib ONLY (biologic_experienced is 100% NULL for Kisqali/Fabhalta by
# design — see BRAND_ELIGIBILITY_FIELDS). A per-unit MULTIPLICATIVE modifier on
# the severity latent CATE: biologic-EXPERIENCED patients (prior omalizumab /
# anti-IgE exposure => refractory CSU) carry an ATTENUATED BTK-inhibitor effect;
# biologic-NAIVE carry a boosted one. The two multipliers are a MEAN-PRESERVING
# spread at the ~40% experienced prevalence (0.60*1.25 + 0.40*0.625 = 1.00), so
# the population-mean Remibrutinib effect — and therefore the existing
# severity-CATE recovery gate — is UNCHANGED, while a large, RECOVERABLE biologic
# gap (~0.10 RD, ~2x ratio) opens up. Both values are a SYNTHETIC DESIGN CHOICE
# (like the 0.70/0.30/0.10 severity map), NOT a number from real Remibrutinib
# trials. Validated recoverable 5/5 across seeds at n>=8000 by the cheapest-
# disproof harness (scratchpad/phase3_biologic_cate_disproof.py, 2026-07-13);
# a subtle spread (<1.5x) sign-flips and is NOT recoverable. IgE stays DESCRIPTIVE
# (Remibrutinib is a BTK inhibitor, not anti-IgE) — no IgE causal axis is planted.
_BIOLOGIC_NAIVE_MULT = 1.25
_BIOLOGIC_EXPERIENCED_MULT = 0.625
# Independent SeedSequence spawn_key for the Remibrutinib biologic-outcome
# recompute, so it never perturbs the generator's main self._rng stream.
_BIOLOGIC_SPAWN_KEY = 0xB10C


def biologic_cate_modifier(biologic_experienced: np.ndarray) -> np.ndarray:
    """Per-unit multiplicative CATE modifier from biologic-experience.

    biologic_experienced is 0 (naive) / 1 (experienced). Returns
    _BIOLOGIC_NAIVE_MULT for naive, _BIOLOGIC_EXPERIENCED_MULT for experienced.
    """
    bio = np.asarray(biologic_experienced, dtype=float)
    return np.where(bio >= 0.5, _BIOLOGIC_EXPERIENCED_MULT, _BIOLOGIC_NAIVE_MULT)


def brand_scaled_cate(brand: Brand) -> Dict[str, float]:
    """Return the per-brand CATE-by-segment map (base map x brand scale).

    Base map is read from config (SSOT: DGP_CONFIGS[HETEROGENEOUS]); never
    re-hardcoded here. Rounded to 4 dp for stable equality / persistence.
    """
    base = DGP_CONFIGS[DGPType.HETEROGENEOUS].cate_by_segment or {}
    scale = _BRAND_CATE_SCALE.get(brand, 1.00)
    return {seg: round(val * scale, 4) for seg, val in base.items()}


def assign_segment(disease_severity: np.ndarray) -> np.ndarray:
    """Map continuous disease_severity (0-10) to the three CATE segments.

    Thresholds match the existing patient_generator logic
    (patient_generator.py:238-242): >7 high, >4 medium, else low.
    """
    sev = np.asarray(disease_severity, dtype=float)
    return np.where(
        sev > 7,
        SEGMENT_HIGH,
        np.where(sev > 4, SEGMENT_MEDIUM, SEGMENT_LOW),
    )


# ---------------------------------------------------------------------------
# T11 (2026-06-22): prognostic drivers on the INITIATION latent baseline.
# ---------------------------------------------------------------------------
# treatment_initiated's outcome eqn (binary_outcome_with_cate) used ONLY
# disease_severity + academic_hcp + arm·τ + N(0,0.6) — so geographic_region (one
# of its 3 "base" covariates) was NOT in the equation and the goldstd initiation
# model's ~0.67 AUC was the Bayes ceiling of that thin eqn (the 2026-06-14 "more
# features HURT" experiment measured that ceiling, NOT a model limit). These 4
# drivers are added to the latent baseline, drawn INDEPENDENTLY of treatment_arm
# in patient_generator, so arm·τ is untouched → the latent ATE and the segment
# CATE ordering the recovery probe recovers are PRESERVED by construction (proven
# by test_dgp_recovery_probe + test_initiation_calibration). Mirrors the T9
# persistence enrichment (cohort_outcomes.py); coefs dialed so the faithful
# FeatureBuilder+train_cohort_model holdout AUC lands ~0.80 (persist/disc parity).
_INIT_INS_ACCESS = {  # insurance access gradient (commercial best → uninsured worst)
    "commercial": 0.45,
    "medicare": 0.10,
    "medicaid": -0.35,
    "uninsured": -0.55,
}
_INIT_AGE_COEF = 0.025  # latent score per year off _INIT_AGE_CENTER
_INIT_AGE_CENTER = 50.0
_INIT_COMORBIDITY_COEF = -0.18  # more burden → lower initiation propensity
_INIT_PRIOR_THERAPY_COEF = -0.15  # more prior lines → lower initiation propensity
_INIT_DRIVER_SCALE = 0.75  # TUNED (faithful FeatureBuilder+train_cohort_model sweep,
# n=20000): lands the initiation holdout AUC ~0.80 (Remi 0.804 / Fab 0.797 / Kis 0.798),
# persist/disc parity, inside the [0.78,0.83] band test_initiation_calibration.py locks.
# NOTE: adding prognostic predictive signal to a fixed-prevalence BINARY outcome
# necessarily COMPRESSES the binarized RD-scale treatment effect (mean tau_i) — at this
# scale it drops to ~0.14, below the designed [0.15,0.50] true_ate band. That compression
# is offset by _INIT_LATENT_CATE_BOOST (below), which plants a proportionally larger
# latent CATE so the RECOVERED RD-scale true_ate is restored to ~the pre-T11 baseline
# (~0.177) WITHOUT lowering AUC (τ is small vs the prognostic+noise variance). This keeps
# both contracts: AUC ~0.80 AND causal fidelity. (User directive 2026-06-22.)


def initiation_prognostic_offset(
    insurance_type: np.ndarray,
    age_at_diagnosis: np.ndarray,
    comorbidity_burden: np.ndarray,
    prior_therapy_lines: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    """Latent-baseline offset from the 4 prognostic drivers (⊥ treatment_arm).

    Returned vector is ADDED to the initiation baseline in binary_outcome_with_cate.
    Drawn independently of the arm so it raises predictive signal WITHOUT changing
    the recoverable treatment effect. ``scale`` overrides _INIT_DRIVER_SCALE (used by
    the tuning sweep); production passes None.
    """
    s = _INIT_DRIVER_SCALE if scale is None else scale
    ins = np.array(
        [_INIT_INS_ACCESS.get(str(i), 0.0) for i in np.asarray(insurance_type)], dtype=float
    )
    age = np.asarray(age_at_diagnosis, dtype=float)
    com = np.asarray(comorbidity_burden, dtype=float)
    prior = np.asarray(prior_therapy_lines, dtype=float)
    return s * (
        ins
        + _INIT_AGE_COEF * (age - _INIT_AGE_CENTER)
        + _INIT_COMORBIDITY_COEF * com
        + _INIT_PRIOR_THERAPY_COEF * prior
    )


def binarize_score(
    score: np.ndarray,
    baseline: np.ndarray,
    tau_latent: np.ndarray,
    segment: np.ndarray,
    *,
    target_prevalence: float = 0.35,
    noise_std: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Threshold an EXISTING latent score into a binary outcome + per-segment
    RECOVERABLE RD CATE. Lets several outcomes be derived from ONE shared score
    (e.g. adherent_180d and low_gap_180d at different prevalences) while keeping a
    SINGLE SSOT for the quantile-threshold + analytic counterfactual-RD math.
    Returns (y, tau_i); tau_i has exactly 3 distinct per-segment RD values.
    """
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))
    score = np.asarray(score, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    tau_latent = np.asarray(tau_latent, dtype=float)
    q = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= q).astype(int)
    rd_unit = _counterfactual_rd(baseline, tau_latent, q, noise_std)
    rd_map = {str(s): float(np.mean(rd_unit[segment == s])) for s in np.unique(segment)}
    tau_i = np.array([rd_map[str(s)] for s in segment], dtype=float)
    return y, tau_i


@overload
def binary_outcome_rd(
    arm: np.ndarray,
    baseline: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    *,
    target_prevalence: float = ...,
    noise_std: float = ...,
    return_score: Literal[False] = ...,
    cate_modifier: np.ndarray | None = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def binary_outcome_rd(
    arm: np.ndarray,
    baseline: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    *,
    target_prevalence: float = ...,
    noise_std: float = ...,
    return_score: Literal[True],
    cate_modifier: np.ndarray | None = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def binary_outcome_rd(
    arm: np.ndarray,
    baseline: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    *,
    target_prevalence: float = 0.35,
    noise_std: float = 0.6,
    return_score: bool = False,
    cate_modifier: np.ndarray | None = None,
) -> Tuple[np.ndarray, ...]:
    """General binary outcome Y + per-unit RECOVERABLE segment RD-scale CATE.

    latent = baseline(X) + arm * tau_latent(segment) + N(0, noise_std);
    Y = 1{latent >= q}, q = (1 - target_prevalence) sample quantile (=> marginal
    prevalence ~= target_prevalence, clamped to [0.20, 0.50]). Returns (y, tau_i)
    where tau_i is the per-segment counterfactual risk difference (exactly 3
    distinct values, de-confounded, RD scale) — the quantity LinearDML/
    CausalForestDML recover. ``baseline`` is the caller-built latent baseline
    (so callers own their own confounding / prognostic structure); ``cate_map``
    is the brand-scaled segment CATE on the latent score scale.

    When ``return_score`` is True, returns (y, tau_i, score) — the continuous
    latent score Y was thresholded from — so a caller can build a noisy continuous
    PROXY of Y from the SAME latent (shared noise), keeping the proxy consistent
    with the authoritative binary; the default 2-tuple (y, tau_i) is unchanged.

    ``cate_modifier`` (Phase 3, optional) is a per-unit MULTIPLIER on the segment
    latent CATE — used to plant a SECOND heterogeneity dimension (biologic-
    experience) orthogonal to the severity segment. When given, tau_latent becomes
    per-unit (``cate_map[segment] * cate_modifier``) and the recoverable RD is
    collapsed over the COMPOSITE (segment x modifier-level) grouping, so tau_i
    keeps the modifier dimension distinct instead of averaging it away. ``None``
    (default) is byte-identical to the pre-Phase-3 behaviour (group == segment).
    """
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))
    baseline = np.asarray(baseline, dtype=float)
    tau_latent = np.array([cate_map[str(s)] for s in segment], dtype=float)
    if cate_modifier is None:
        group = np.asarray(segment)
    else:
        mod = np.asarray(cate_modifier, dtype=float)
        tau_latent = tau_latent * mod
        # composite grouping keeps each (segment, modifier-level) cell a distinct
        # RD value — otherwise binarize_score averages the biologic gap back out.
        group = np.array([f"{s}|{m:.4f}" for s, m in zip(segment, mod, strict=False)])
    noise = rng.normal(0.0, noise_std, len(arm))
    score = baseline + arm.astype(float) * tau_latent + noise
    y, tau_i = binarize_score(
        score,
        baseline,
        tau_latent,
        group,
        target_prevalence=target_prevalence,
        noise_std=noise_std,
    )
    if return_score:
        return y, tau_i, score
    return y, tau_i


def binary_outcome_with_cate(
    arm: np.ndarray,
    covariates: Dict[str, np.ndarray],
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    target_prevalence: float = _INIT_TARGET_PREVALENCE,
    baseline_severity_coef: float = _INIT_BASE_SEVERITY_COEF,
    baseline_academic_coef: float = _INIT_BASE_ACADEMIC_COEF,
    noise_std: float = _INIT_NOISE_STD,
    prognostic_offset: np.ndarray | None = None,
    cate_modifier: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary outcome Y + per-unit RECOVERABLE segment CATE.

    Latent score = baseline(X) + T*tau_latent + N(0,noise_std); Y=1{score>=q}
    with q = the (1-target_prevalence) sample quantile => marginal prevalence ~=
    target_prevalence (in [0.20,0.50] INDEX band) BY CONSTRUCTION, independent of
    effect size. target_prevalence is clamped to [0.20,0.50] so callers cannot
    push the outcome degenerate.

    `cate_map` is the brand-scaled segment CATE on the *latent* score scale. But
    the agents' estimators (LinearDML/CausalForestDML) target the DE-CONFOUNDED
    effect on the *probability/risk-difference* scale — a different scale that
    binarization attenuates. So `tau_i` is the per-unit segment RISK-DIFFERENCE
    CATE: each unit carries its SEGMENT'S average counterfactual risk difference
        E_seg[ P(Y=1|do(T=1),X) - P(Y=1|do(T=0),X) ]
            = E_seg[ Phi((baseline+tau_latent-q)/s) - Phi((baseline-q)/s) ]
    computed analytically (Gaussian noise marginalized). This yields exactly 3
    distinct values (one per segment), monotone high>medium>low>0, de-confounded,
    on the RD scale the estimators recover. `mean(tau_i)` is the RD-scale TRUE_ATE
    the recovery probe (Task 03.5, gate 3) compares LinearDML against; per-unit
    tau_i is persisted to ml_predictions.{treatment_effect_estimate,
    heterogeneous_effect} and grouped by segment for causal_metrics_cate.

    The returned `tau_i` has exactly 3 distinct values; the generator derives the
    {segment: rd} ground-truth map directly from (tau_i, segment) — no re-draw.
    (Phase 3: when ``cate_modifier`` is passed, tau_i carries segment x modifier-
    level cells — e.g. 6 values for the 3 severity x 2 biologic-experience groups —
    and rd_map_from_tau averages over the modifier to recover the severity map.)

    Default coefs (baseline_severity_coef=0.10, baseline_academic_coef=0.15,
    noise_std=0.6) were TUNED by the Task 03.5 recovery probe (cheapest-disproof):
    the plan's initial 0.20/0.30/1.0 left the de-confounded medium-vs-low RD gap
    too small for CausalForestDML to resolve at n=3000, flipping the CATE ordering
    (the forest systematically inflated the low segment toward medium). Lowering
    baseline confounding stops high-severity saturating both arms; lowering noise
    to 0.6 sharpens the segment RD separation. Below 0.6 (0.5/0.4) the high-vs-
    medium ordering INVERTS via high-severity baseline saturation, so 0.6 is the
    sweet spot. With these defaults the probe recovers ATE within 0.15 AND
    high>medium>low CATE ordering across seeds 21/7/99/123 for Remibrutinib +
    Kisqali (the gate's two brands) at n=3000; Fabhalta (flattest 0.70 scale) is
    robust at the gate seed but its med/low gap is the most fragile off-seed.
    Prevalence stays exactly target by the quantile-threshold construction.
    """
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))

    severity = np.asarray(covariates["disease_severity"], dtype=float)
    academic = np.asarray(covariates["academic_hcp"], dtype=float)
    baseline = baseline_severity_coef * (severity - 5.0) + baseline_academic_coef * academic
    if prognostic_offset is not None:
        baseline = baseline + np.asarray(prognostic_offset, dtype=float)
    # initiation keeps its tuned latent-CATE boost (T11) — applied to the map
    # BEFORE delegation so the core stays boost-agnostic.
    boosted_map = {str(s): float(v) * _INIT_LATENT_CATE_BOOST for s, v in cate_map.items()}
    return binary_outcome_rd(
        arm,
        baseline,
        segment,
        boosted_map,
        rng,
        target_prevalence=target_prevalence,
        noise_std=noise_std,
        cate_modifier=cate_modifier,
    )


def _counterfactual_rd(
    baseline: np.ndarray,
    tau_latent: np.ndarray,
    q: float,
    noise_std: float,
) -> np.ndarray:
    """Per-unit counterfactual risk difference P(Y=1|do(T=1)) - P(Y=1|do(T=0)).

    Marginalizes the Gaussian noise analytically against the fixed threshold q.
    """
    p1 = 1.0 - norm.cdf((q - baseline - tau_latent) / noise_std)
    p0 = 1.0 - norm.cdf((q - baseline) / noise_std)
    return np.asarray(p1 - p0)


def rd_map_from_tau(segment: np.ndarray, tau_i: np.ndarray) -> Dict[str, float]:
    """Derive the {segment: RD-scale CATE} map from the per-unit tau_i.

    tau_i carries the per-segment counterfactual risk difference (3 distinct
    values from binary_outcome_rd / binarize_score), so this is a lossless collapse — the
    RD-scale ground-truth CATE map the generator persists to attrs + JSON sidecar.

    Uses the per-segment MEAN so that a Phase-3 frame (where tau_i varies WITHIN a
    severity segment by biologic-experience) collapses to the severity-MARGINAL RD;
    identical to the old ``[0]`` when tau_i is constant within the segment.
    """
    return {str(s): float(np.mean(tau_i[segment == s])) for s in np.unique(segment)}
