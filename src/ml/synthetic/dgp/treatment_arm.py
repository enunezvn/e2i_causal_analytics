"""Causal DGP primitives for the synthetic causal-validation dataset.

Adds a per-unit BINARY treatment arm confounded with emitted covariates,
a per-(brand) CATE scaling, and a prevalence-banded binary outcome carrying
a known E[tau(X)] = TRUE_ATE. Used by patient_generator (Shard 03) and the
trigger/cohort generators (Shards 05/06).
"""

from __future__ import annotations

from typing import Dict, Tuple

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

    The two propensity covariates are read through ``ARM_CONFOUNDERS`` so that
    constant stays load-bearing (the contract guard test perturbs each and
    asserts the propensity moves).
    """
    _severity_key, _academic_key = ARM_CONFOUNDERS
    severity = np.asarray(covariates[_severity_key], dtype=float)
    academic = np.asarray(covariates[_academic_key], dtype=float)
    # center severity at the population mean (5.0) so the intercept sets the
    # base treatment share rather than being absorbed by the severity scale.
    logit = intercept + beta_severity * (severity - 5.0) + beta_academic * academic
    propensity = np.clip(expit(logit), 0.01, 0.99)
    arm = (rng.random(len(propensity)) < propensity).astype(int)
    return arm, propensity


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
    """
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))
    baseline = np.asarray(baseline, dtype=float)
    tau_latent = np.array([cate_map[str(s)] for s in segment], dtype=float)
    noise = rng.normal(0.0, noise_std, len(arm))
    score = baseline + arm.astype(float) * tau_latent + noise
    q = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= q).astype(int)
    rd_unit = _counterfactual_rd(baseline, tau_latent, q, noise_std)
    rd_map = {str(s): float(np.mean(rd_unit[segment == s])) for s in np.unique(segment)}
    tau_i = np.array([rd_map[str(s)] for s in segment], dtype=float)
    if return_score:
        return y, tau_i, score
    return y, tau_i


def binary_outcome_with_cate(
    arm: np.ndarray,
    covariates: Dict[str, np.ndarray],
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    target_prevalence: float = 0.35,
    baseline_severity_coef: float = 0.10,
    baseline_academic_coef: float = 0.15,
    noise_std: float = 0.6,
    prognostic_offset: np.ndarray | None = None,
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
        arm, baseline, segment, boosted_map, rng,
        target_prevalence=target_prevalence, noise_std=noise_std,
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
    values from binary_outcome_with_cate), so this is a lossless collapse — the
    RD-scale ground-truth CATE map the generator persists to attrs + JSON sidecar.
    """
    return {str(s): float(tau_i[segment == s][0]) for s in np.unique(segment)}
