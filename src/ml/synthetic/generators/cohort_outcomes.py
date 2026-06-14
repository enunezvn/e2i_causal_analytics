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
pull ±0.9 logit, noise 0.35) so leakage-safe covariates carry real predictive
signal for gold-standard model-eval, while the treatment effect and prevalence band
are preserved.

Sign convention (resource_optimizer-safe): treatment LOWERS discontinuation
(improves retention). retention_benefit is a NON-NEGATIVE per-unit covariate
(higher = more retained), so problem_formulator's _validate_inputs
(src/agents/resource_optimizer/nodes/problem_formulator.py, which appends an error
on expected_response < 0) never rejects this cohort.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import expit

# Marginal-rate intercept (logit). Tuned so AVG(discontinued_180d) lands in-band
# ~0.47-0.50 with the BOOSTED confounder coefficients below.
_DISC_INTERCEPT = -2.4

# Designed treatment effect on the discontinuation LOGIT (negative = retention).
# Heterogeneous by segment: high severity benefits MOST from treatment, so its
# discontinuation drops the most (matches Shard 03 cate_by_segment ordering).
_DISC_TREATMENT_LOGIT = {
    "high_severity": -1.20,
    "medium_severity": -0.70,
    "low_severity": -0.35,
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

# Per-unit, non-negative retention covariate scale (read by resource_optimizer as
# expected_response). Strictly positive so the validator never sees a negative.
PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY = 0.05


def generate_discontinuation_outcomes(
    *,
    rng: np.random.Generator,
    treatment_arm: np.ndarray,
    disease_severity: np.ndarray,
    academic_hcp: np.ndarray,
    geographic_region: np.ndarray,
    segment: np.ndarray,
    brand_cate_scale: float,
) -> Dict[str, np.ndarray]:
    """Draw discontinued_180d (and its complement persistent_180d) + a
    non-negative retention_benefit covariate.

    Args:
        treatment_arm: per-unit 0/1 arm from the Shard-03 DGP.
        academic_hcp: per-unit binary academic flag.
        geographic_region: per-unit region string; maps via _DISC_REGION_LOGIT.
        segment: per-unit {high,medium,low}_severity (Shard-03 segmentation).
        brand_cate_scale: brand-distinct multiplier on the treatment logit so a
            Kisqali probe differs from a Remibrutinib probe (INDEX CATE-by-brand).
    """
    n = len(treatment_arm)
    seg_treat = np.array([_DISC_TREATMENT_LOGIT.get(str(s), -0.70) for s in segment], dtype=float)
    region_pull = np.array(
        [_DISC_REGION_LOGIT.get(str(r), 0.0) for r in geographic_region], dtype=float
    )
    logit = (
        _DISC_INTERCEPT
        + brand_cate_scale * seg_treat * treatment_arm  # causal effect — UNCHANGED
        + _DISC_SEVERITY_COEF * disease_severity
        + _DISC_ACADEMIC_COEF * academic_hcp
        + region_pull
        + rng.normal(0.0, 0.35, n)
    )
    p_disc = expit(logit)
    discontinued = (rng.random(n) < p_disc).astype(int)
    persistent = 1 - discontinued
    # Non-negative retention benefit: scales with severity (high-severity persisters
    # are the most valuable to retain) and is ALWAYS >= 0.
    retention_benefit = PERSISTENCE_RETENTION_BENEFIT_PER_SEVERITY * disease_severity * persistent
    return {
        "discontinued_180d": discontinued,
        "persistent_180d": persistent,
        "retention_benefit": np.clip(retention_benefit, 0.0, None),
    }
