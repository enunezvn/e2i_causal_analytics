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
    Brand.KISQALI: 1.40,       # stronger heterogeneity (oncology CDK4/6 responder split)
    Brand.FABHALTA: 0.70,      # flatter (rare PNH, smaller effect spread)
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
    baseline_severity_coef: float = 0.20,
    baseline_academic_coef: float = 0.30,
    noise_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary outcome Y carrying a known per-segment CATE.

    Latent score = baseline(X) + T*tau_seg + N(0,noise_std); Y=1{score>=q} with
    q = the (1-target_prevalence) sample quantile => marginal prevalence ~=
    target_prevalence (in [0.20,0.50] INDEX band) BY CONSTRUCTION, independent
    of effect size. tau_seg is the brand-scaled segment CATE (high>medium>low>0),
    so the manifest risk-difference is monotone in segment => ordering survives
    binarization. target_prevalence is clamped to [0.20,0.50] so callers cannot
    push the outcome degenerate. Returns (y[0/1], tau_i) where tau_i ==
    cate_map[segment_i] — persisted to ml_predictions.heterogeneous_effect.
    """
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))

    severity = np.asarray(covariates["disease_severity"], dtype=float)
    academic = np.asarray(covariates["academic_hcp"], dtype=float)

    # per-unit latent CATE from the brand-scaled segment map
    tau_i = np.array([cate_map[str(s)] for s in segment], dtype=float)

    baseline = (
        baseline_severity_coef * (severity - 5.0)
        + baseline_academic_coef * academic
    )
    noise = rng.normal(0.0, noise_std, len(arm))
    score = baseline + arm.astype(float) * tau_i + noise

    # threshold at the (1 - target_prevalence) quantile => P(Y=1)=target_prevalence
    q = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= q).astype(int)
    return y, tau_i
