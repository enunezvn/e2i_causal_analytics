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
    """
    severity = np.asarray(covariates["disease_severity"], dtype=float)
    academic = np.asarray(covariates["academic_hcp"], dtype=float)
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

    # latent per-unit CATE from the brand-scaled segment map (score scale)
    tau_latent = np.array([cate_map[str(s)] for s in segment], dtype=float)

    baseline = baseline_severity_coef * (severity - 5.0) + baseline_academic_coef * academic
    noise = rng.normal(0.0, noise_std, len(arm))
    score = baseline + arm.astype(float) * tau_latent + noise

    # threshold at the (1 - target_prevalence) quantile => P(Y=1)=target_prevalence
    q = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= q).astype(int)

    # per-unit counterfactual risk difference (RECOVERABLE, de-confounded, RD scale)
    rd_unit = _counterfactual_rd(baseline, tau_latent, q, noise_std)
    # collapse to the per-segment mean so tau_i takes exactly 3 distinct values
    rd_map = {str(s): float(np.mean(rd_unit[segment == s])) for s in np.unique(segment)}
    tau_i = np.array([rd_map[str(s)] for s in segment], dtype=float)
    return y, tau_i


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
    return p1 - p0


def rd_map_from_tau(segment: np.ndarray, tau_i: np.ndarray) -> Dict[str, float]:
    """Derive the {segment: RD-scale CATE} map from the per-unit tau_i.

    tau_i carries the per-segment counterfactual risk difference (3 distinct
    values from binary_outcome_with_cate), so this is a lossless collapse — the
    RD-scale ground-truth CATE map the generator persists to attrs + JSON sidecar.
    """
    return {str(s): float(tau_i[segment == s][0]) for s in np.unique(segment)}
