"""Pure DGP for the discontinuation / persistence cohort outcomes.

These mirror the SEMANTICS of convert_optum_rwd.py's _target_discontinued_180d
(scripts/convert_optum_rwd.py:2696) and _target_persistent_at_180d (:2725) —
discontinuation = coverage gap before day 180, persistence = active at day 180 —
but are GENERATED from the Shard-03 treatment arm + confounders so the effect is
KNOWN and recoverable. Persistence is the strict complement of discontinuation so
no row is simultaneously discontinued and persistent.

Anti-degeneracy: prevalence is a DESIGNED parameter. We do NOT reuse the
patient_generator expit>0.5 threshold (which drove the real synthetic init label
to 0.93+ before Shard 03 banded it). The intercept _DISC_INTERCEPT is tuned so the
marginal discontinuation rate lands inside the [0.05,0.60] band across brands.
Confounder coefficients were boosted (severity 0.55, academic -0.80, + per-region
pull ±0.9 logit) so leakage-safe covariates carry real predictive signal for
gold-standard model-eval, while the treatment effect and prevalence band are
preserved. T9 (2026-06-21) adds 4 prognostic drivers — insurance access, comorbidity
burden, age, prior-therapy lines — drawn independently of treatment_arm so they lift
predictive AUC to a realistic ~0.78-0.82 WITHOUT changing the recoverable ATE/CATE.

Sign convention (resource_optimizer-safe): treatment LOWERS discontinuation
(improves retention). retention_benefit is a NON-NEGATIVE per-unit covariate
(higher = more retained), so problem_formulator's _validate_inputs
(src/agents/resource_optimizer/nodes/problem_formulator.py, which appends an error
on expected_response < 0) never rejects this cohort.
"""

from __future__ import annotations

from typing import Dict, TypedDict

import numpy as np
from scipy.special import expit

# Marginal-rate intercept (logit). T9: re-tuned to -2.9 (was -2.4) for the richer
# 7-covariate equation — the 4 net-positive prognostic drivers raise the mean logit,
# so the intercept drops to keep AVG(discontinued_180d) in-band [0.05,0.60] (measured
# ~0.47). Calibration (test_persistence_calibration) verifies prevalence + AUC.
_DISC_INTERCEPT = -2.9

# Designed treatment effect on the discontinuation LOGIT (negative = retention).
# Heterogeneous by segment: high severity benefits MOST from treatment, so its
# discontinuation drops the most (matches Shard 03 cate_by_segment ordering).
_DISC_TREATMENT_LOGIT = {
    "high_severity": -1.20,
    "medium_severity": -0.70,
    "low_severity": -0.35,
}

# Copay effect on the DISCONTINUATION logit (negative = improves persistence).
# Measured 2026-07-19. The LOW arm is deliberately near zero: at low=-0.14 the
# recovered medium-low margin collapses to +0.003..+0.006 at seed 21 (flaky), and
# the planted ATE falls below the design band. At low=-0.04 the worst margin is
# +0.0543 and ATE is 0.0892. Do NOT widen further: copay is invisible to
# FeatureBuilder's 7 covariates, so every logit unit here costs achievable AUC, and
# the faithful post-wiring re-measure (2026-07-19, test_persistence_calibration's own
# _faithful, n=20000/seed=42) is Remi 0.7863 / Fabhalta 0.7900 / Kisqali 0.7805
# against a 0.78 HARD floor -- Kisqali clears it by 0.0005, not the ~0.016 estimated
# pre-wiring. The AUC floor, not the CATE margin, is now the binding constraint.
_COPAY_DISC_LOGIT = {
    "high_severity": -1.20,
    "medium_severity": -0.55,
    "low_severity": -0.04,
}

# Phase 2 (COMM-ARMS): psp_enrolled on the DISCONTINUATION logit (negative = improves
# persistence). MEASURED, not guessed. Unlike the treatment-arm term this is NOT
# brand-scaled, so the planted persistent ATE is ~brand-invariant (~0.079). psp is a
# 9th covariate of the persistence/discontinuation models (_BASE9_COMMERCIAL), so the
# model CAN observe it -- but it is still real outcome signal, so the post-psp AUC falls
# ~0.017 vs Phase 1 (measured; part is irreducible, part is recovered by the +psp
# feature). backlog #43 (folded into this phase) re-shapes that gate to per-brand PINNED
# baselines + tolerance, exactly because each arm erodes the ceiling by the same
# mechanism. GATE tuning: the persistence gate is MULTI-SEED (21/7/99/123, strict
# ordering), and logit->Bernoulli RDs are thin -- the first guess {-0.90,-0.45,-0.05}
# left seed-99 h-m at only +0.012 (gated, a hair's breadth). Widening the HIGH arm to
# -1.10 lifts every seed-99 h-m to >=+0.042 (approaching copay's +0.054 bar) while
# keeping the ATE ~0.079 in the +5-10pp band. Do NOT narrow the gap without re-running
# the multi-seed persistence gate.
_PSP_DISC_LOGIT = {
    "high_severity": -1.10,
    "medium_severity": -0.45,
    "low_severity": -0.05,
}

# Confounder pull on the discontinuation logit (sicker/non-academic -> more gaps).
_DISC_SEVERITY_COEF = 0.55
_DISC_ACADEMIC_COEF = -0.80

# Per-region pull on the discontinuation logit. Previously geographic_region had
# NO effect (a pure-noise feature); this gives the third base covariate real,
# leakage-safe signal. Evenly spaced ±0.9 logit across the 4 regions.
_DISC_REGION_LOGIT = {
    "midwest": -0.9,
    "northeast": -0.3,
    "south": 0.3,
    "west": 0.9,
}

# --- T9 (2026-06-21): NEW prognostic drivers --------------------------------
# Prognostic-only: drawn independently of treatment_arm in patient_generator, so
# they raise predictive AUC WITHOUT changing the true ATE/CATE. Signs are on the
# discontinuation logit (negative = improves persistence).
_INS_DISC_PULL = {  # access gradient: commercial best, medicaid worst
    "commercial": -0.65,
    "medicare": 0.10,
    "medicaid": 0.75,
}
_COMORBIDITY_COEF = 0.28  # per comorbidity: more burden -> more discontinuation
_PRIOR_THERAPY_COEF = 0.32  # per prior line: harder-to-treat -> more discontinuation
_AGE_CENTER = 50.0
_AGE_COEF = 0.018  # per year above center -> slightly more discontinuation
# Gaussian logit-noise SD. T9: reduced (was 0.35 inline) so the added signal lifts
# the achievable AUC toward ~0.80. Calibration locks it.
_DISC_NOISE_SD = 0.25

# Per-unit, non-negative retention covariate scale (read by resource_optimizer as
# expected_response). Strictly positive so the validator never sees a negative.
PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY = 0.05

# --- Fabhalta pilot (#1321): prior-C5-inhibitor differential on persistence -------
# complement_inhibitor_status == "prior" marks the prior-C5 SWITCH population
# (patients inadequately controlled on a C5 inhibitor — eculizumab/ravulizumab —
# before iptacopan). Two FAITHFUL, RECOVERABLE effects, both validated by the
# cheapest-disproof harness at n>=8000 (5/5) BEFORE any substrate write:
#   * MAIN effect — prior-C5 patients are HARDER to retain (higher discontinuation).
#     MEAN-CENTERED on the realized prior-C5 prevalence, so the c5=1-vs-c5=0 contrast
#     is fully recoverable while the marginal discontinuation prevalence
#     (test_persistence_calibration band) is UNCHANGED by construction.
#   * MODIFIER — prior-C5-EXPERIENCED patients carry an ATTENUATED iptacopan effect
#     (multiplicative on the treatment logit term); MEAN-PRESERVING at the prevalence
#     so the population persistence ATE and the existing Fabhalta gate are UNCHANGED
#     (biologic Phase-3 precedent, treatment_arm._BIOLOGIC_*).
# Both values are a SYNTHETIC DESIGN CHOICE (like the biologic 2x spread), NOT a
# number from real iptacopan trials. Fabhalta's CATE scale is deliberately flat
# (0.70), so the mean-preserving spread is WIDER than biologic's (3.5x vs 2x) to
# clear the estimator noise floor — the harness disproved a 1.9x spread (3/5).
_PRIORC5_MAIN_PULL = 0.55  # logit; prior-C5 -> more discontinuation (mean-centered)
_PRIORC5_EXPERIENCED_MULT = 0.40  # experienced -> attenuated treatment effect
# Independent SeedSequence spawn_key for the Fabhalta prior-C5 persistence recompute,
# so the post-hoc rebuild never perturbs the generator's main self._rng stream
# (mirrors treatment_arm._BIOLOGIC_SPAWN_KEY). "C5" in hex.
PRIORC5_SPAWN_KEY = 0xC5


def priorc5_cate_modifier(experienced: np.ndarray) -> np.ndarray:
    """Per-unit MEAN-PRESERVING multiplicative CATE modifier from prior-C5 experience.

    ``experienced`` is 0 (naive) / 1 (prior-C5 switch). The experienced multiplier is
    fixed (_PRIORC5_EXPERIENCED_MULT); the naive multiplier is solved so the
    prevalence-weighted mean is 1.0 — the population treatment effect (and the
    existing Fabhalta persistence gate) are preserved. A degenerate cohort (all one
    level) returns all-ones (no modification). Biologic Phase-3 precedent.
    """
    exp = np.asarray(experienced, dtype=float)
    prev = float(np.mean(exp >= 0.5))
    if prev <= 0.0 or prev >= 1.0:
        return np.ones_like(exp)
    naive_mult = (1.0 - prev * _PRIORC5_EXPERIENCED_MULT) / (1.0 - prev)
    return np.where(exp >= 0.5, _PRIORC5_EXPERIENCED_MULT, naive_mult)


class DiscontinuationOutcomes(TypedDict):
    """Return shape of :func:`generate_discontinuation_outcomes`. The two binaries
    + the retention covariate are per-unit np.ndarrays; ``copay_persistent_rd_by_segment``
    is the per-segment recoverable RD ground truth for the copay arm."""

    discontinued_180d: np.ndarray
    persistent_180d: np.ndarray
    retention_benefit: np.ndarray
    copay_persistent_rd_by_segment: Dict[str, float]
    psp_persistent_rd_by_segment: Dict[str, float]
    # Fabhalta pilot (#1321): the scalar prior-C5 persistence RD ground truth
    # (naive - experienced; positive = prior-C5 patients persist LESS). None off
    # the Fabhalta prior-C5 path. The per-segment maps above stay copay/psp only.
    priorc5_persistent_rd: float | None


def generate_discontinuation_outcomes(
    *,
    rng: np.random.Generator,
    treatment_arm: np.ndarray,
    disease_severity: np.ndarray,
    academic_hcp: np.ndarray,
    geographic_region: np.ndarray,
    insurance_type: np.ndarray,
    age_at_diagnosis: np.ndarray,
    comorbidity_burden: np.ndarray,
    prior_therapy_lines: np.ndarray,
    segment: np.ndarray,
    brand_cate_scale: float,
    copay_support: np.ndarray | None = None,
    psp_enrolled: np.ndarray | None = None,
    priorc5_experienced: np.ndarray | None = None,
) -> DiscontinuationOutcomes:
    """Draw discontinued_180d (and its complement persistent_180d) + a
    non-negative retention_benefit covariate.

    Args:
        treatment_arm: per-unit 0/1 arm from the Shard-03 DGP.
        academic_hcp: per-unit binary academic flag.
        geographic_region: per-unit region string; maps via _DISC_REGION_LOGIT.
        segment: per-unit {high,medium,low}_severity (Shard-03 segmentation).
        brand_cate_scale: brand-distinct multiplier on the treatment logit so a
            Kisqali probe differs from a Remibrutinib probe (INDEX CATE-by-brand).
        copay_support: optional per-unit 0/1 second commercial arm (Phase 1). When
            None the equation is byte-identical to the pre-copay DGP.
        priorc5_experienced: Fabhalta pilot (#1321) optional per-unit 0/1 prior-C5
            switch indicator (1 = complement_inhibitor_status == "prior"). Drives a
            mean-preserving treatment-effect MODIFIER + a mean-centered persistence
            MAIN effect. When None the equation is byte-identical to the pre-pilot DGP.
    """
    n = len(treatment_arm)
    seg_treat = np.array([_DISC_TREATMENT_LOGIT.get(str(s), -0.70) for s in segment], dtype=float)
    region_pull = np.array(
        [_DISC_REGION_LOGIT.get(str(r), 0.0) for r in geographic_region], dtype=float
    )
    ins_pull = np.array([_INS_DISC_PULL.get(str(i), 0.0) for i in insurance_type], dtype=float)
    # Fabhalta pilot (#1321): prior-C5 differential. c5_mod MULTIPLIES the treatment
    # term (attenuated iptacopan effect for prior-C5-experienced, mean-preserving);
    # c5_main is a MEAN-CENTERED additive pull (prior-C5 -> more discontinuation,
    # prevalence-preserving), kept SEPARATE like copay/psp. Both default OFF, and
    # `x * 1.0` / `+ 0.0` are exact for finite floats, so the equation is
    # byte-identical to the pre-pilot DGP when priorc5_experienced is None.
    if priorc5_experienced is not None:
        c5 = np.asarray(priorc5_experienced, dtype=float)
        c5_prev = float(np.mean(c5 >= 0.5))
        c5_mod: np.ndarray | float = priorc5_cate_modifier(c5)
        c5_main: np.ndarray | float = _PRIORC5_MAIN_PULL * (c5 - c5_prev)
    else:
        c5_prev = 0.0
        c5_mod = 1.0
        c5_main = 0.0
    arm_core = brand_cate_scale * seg_treat * treatment_arm  # unmodified causal term
    logit = (
        _DISC_INTERCEPT
        + arm_core * c5_mod  # causal effect (x prior-C5 modifier when pilot active)
        + _DISC_SEVERITY_COEF * disease_severity
        + _DISC_ACADEMIC_COEF * academic_hcp
        + region_pull
        + ins_pull  # T9 prognostic: access gradient
        + _COMORBIDITY_COEF * np.asarray(comorbidity_burden, dtype=float)
        + _PRIOR_THERAPY_COEF * np.asarray(prior_therapy_lines, dtype=float)
        + _AGE_COEF * (np.asarray(age_at_diagnosis, dtype=float) - _AGE_CENTER)
        + rng.normal(0.0, _DISC_NOISE_SD, n)
    )
    # Phase 1 follow-up: copay_support on the DISCONTINUATION logit (negative =
    # improves persistence). Kept SEPARATE from `logit` above so copay's planted
    # RD is a clean counterfactual (expit(base) - expit(base + copay_pull)) rather
    # than a re-derivation from the blended logit.
    if copay_support is not None:
        copay = np.asarray(copay_support, dtype=int)
        copay_pull = np.array([_COPAY_DISC_LOGIT.get(str(s), 0.0) for s in segment], dtype=float)
    else:
        copay = np.zeros(n, dtype=int)
        copay_pull = np.zeros(n, dtype=float)
    # Phase 2: psp_enrolled enters the SAME discontinuation logit additively, kept
    # SEPARATE from `logit` (like copay) so each arm's planted RD is a clean
    # counterfactual holding the OTHER arm at its realized pull, not a re-derivation
    # from the blended logit. When None the equation is byte-identical to pre-psp.
    if psp_enrolled is not None:
        psp = np.asarray(psp_enrolled, dtype=int)
        psp_pull = np.array([_PSP_DISC_LOGIT.get(str(s), 0.0) for s in segment], dtype=float)
    else:
        psp = np.zeros(n, dtype=int)
        psp_pull = np.zeros(n, dtype=float)
    copay_realized = copay_pull * copay
    psp_realized = psp_pull * psp
    # c5_main defaults to 0.0 (scalar) -> `+ 0.0` no-op, byte-identical when off.
    p_disc = expit(logit + copay_realized + psp_realized + c5_main)
    discontinued = (rng.random(n) < p_disc).astype(int)
    persistent = 1 - discontinued
    # Non-negative retention benefit: scales with severity (high-severity persisters
    # are the most valuable to retain) and is ALWAYS >= 0.
    retention_benefit = PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY * disease_severity * persistent
    # Each commercial arm's OWN recoverable RD on PERSISTENCE = -(discontinuation RD),
    # computed per segment as a counterfactual on the logit the row actually got, with
    # the OTHER arm folded into the effective base at its realized pull (additive-
    # independent: no arm's truth is a blend of the other's).
    copay_rd: Dict[str, float] = {}
    psp_rd: Dict[str, float] = {}
    for seg_name in np.unique(segment):
        mask = np.asarray(segment) == seg_name
        if not mask.any():
            continue
        c_pull = float(_COPAY_DISC_LOGIT.get(str(seg_name), 0.0))
        p_pull = float(_PSP_DISC_LOGIT.get(str(seg_name), 0.0))
        base_for_copay = logit[mask] + psp_realized[mask]
        copay_rd[str(seg_name)] = float(
            np.mean(expit(base_for_copay) - expit(base_for_copay + c_pull))
        )
        base_for_psp = logit[mask] + copay_realized[mask]
        psp_rd[str(seg_name)] = float(np.mean(expit(base_for_psp) - expit(base_for_psp + p_pull)))
    # Fabhalta pilot (#1321): scalar prior-C5 persistence RD ground truth (naive -
    # experienced) — the counterfactual the recovery probe compares LinearDML
    # against. Flip BOTH channels (treatment-term modifier + mean-centered main
    # pull) on the logit each row actually got, holding confounders/noise/other-arms
    # fixed. Positive => prior-C5 patients persist LESS. None off the pilot path.
    priorc5_rd: float | None
    if priorc5_experienced is not None and 0.0 < c5_prev < 1.0:
        exp_mult = _PRIORC5_EXPERIENCED_MULT
        naive_mult = (1.0 - c5_prev * exp_mult) / (1.0 - c5_prev)
        base_wo_treat = logit - arm_core * c5_mod  # intercept + confounders + noise
        other_arms = copay_realized + psp_realized
        lg_naive = (
            base_wo_treat + arm_core * naive_mult + other_arms + _PRIORC5_MAIN_PULL * (-c5_prev)
        )
        lg_exp = (
            base_wo_treat + arm_core * exp_mult + other_arms + _PRIORC5_MAIN_PULL * (1.0 - c5_prev)
        )
        # persist_naive - persist_exp = expit(lg_exp) - expit(lg_naive)
        priorc5_rd = float(np.mean(expit(lg_exp) - expit(lg_naive)))
    else:
        priorc5_rd = None
    return {
        "discontinued_180d": discontinued,
        "persistent_180d": persistent,
        "retention_benefit": np.clip(retention_benefit, 0.0, None),
        "copay_persistent_rd_by_segment": copay_rd,
        "psp_persistent_rd_by_segment": psp_rd,
        "priorc5_persistent_rd": priorc5_rd,
    }
