"""Phase 0 of commercial-arms enrichment: binarized adherence outcomes for the
EXISTING treatment_arm, plus clinically-coherent raw proxies drawn from the same
latent score (so adherence_rate>=0.8 agrees with the authoritative binary).

Design note: binary and proxy share the SAME noise draw so that adherence_rate
is a monotone transform of the latent score that also drives adherent_180d.
This guarantees high ordinal agreement after calibration (>0.95 expected),
far above the >=0.80 test floor. The binary stays authoritative; the proxy is
a logistic squash of the same ordering.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import norm

from src.ml.synthetic.dgp.treatment_arm import rd_map_from_tau

# Adherence baseline: sicker patients are modestly LESS adherent; academic-HCP
# patients modestly MORE (kept small so the arm effect is the dominant signal).
_ADH_SEVERITY_COEF = -0.08
_ADH_ACADEMIC_COEF = 0.12
_ADH_NOISE_STD = 0.6
# Map the adherence latent to a PDC in [0,1] via logistic squash.
_PDC_CENTER = 0.0
_PDC_SCALE = 1.1
# Target marginal prevalence of adherent_180d (clamped to [0.20, 0.50]).
_TARGET_PREVALENCE = 0.35
# gap_days: inverse of adherence; ~ (1 - PDC) * 180-day window, floored at 0.
_GAP_WINDOW_DAYS = 180.0


def generate_adherence_outcomes(
    *,
    treatment_arm: np.ndarray,
    disease_severity: np.ndarray,
    academic_hcp: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Return adherent_180d / low_gap_180d (recoverable binaries) + adherence_rate
    / gap_days (raw proxies) + the per-segment RD ground-truth map.

    The binary outcomes are AUTHORITATIVE: adherent_180d is binarized at the
    (1 - target_prevalence) quantile of the shared latent score, with a
    known counterfactual RD per segment (computed analytically). adherence_rate
    is a noisy continuous proxy of the SAME latent score (same noise draw), so
    adherence_rate >= 0.8 agrees with adherent_180d == 1 on >95% of rows after
    calibration. gap_days is the inverse proxy (gap_days <= 30 => low_gap_180d).
    """
    arm = np.asarray(treatment_arm, dtype=int)
    severity = np.asarray(disease_severity, dtype=float)
    academic = np.asarray(academic_hcp, dtype=float)
    baseline = _ADH_SEVERITY_COEF * (severity - 5.0) + _ADH_ACADEMIC_COEF * academic

    # --- Shared latent score (noise drawn ONCE for binary + proxy consistency) ---
    tau_latent = np.array([cate_map[str(s)] for s in segment], dtype=float)
    noise = rng.normal(0.0, _ADH_NOISE_STD, len(arm))
    score = baseline + arm.astype(float) * tau_latent + noise

    # --- Authoritative binary ---
    target_prev = float(np.clip(_TARGET_PREVALENCE, 0.20, 0.50))
    q = float(np.quantile(score, 1.0 - target_prev))
    adherent_180d = (score >= q).astype(int)

    # --- Per-segment RD ground-truth (analytical Gaussian marginalisation) ---
    p1 = 1.0 - norm.cdf((q - baseline - tau_latent) / _ADH_NOISE_STD)
    p0 = 1.0 - norm.cdf((q - baseline) / _ADH_NOISE_STD)
    rd_unit = p1 - p0
    segs = np.asarray(segment)
    rd_map_seg = {str(s): float(np.mean(rd_unit[segs == s])) for s in np.unique(segs)}
    tau_adherent = np.array([rd_map_seg[str(s)] for s in segs], dtype=float)

    # --- Raw PDC proxy from the SAME score (monotone => ordinal agreement preserved) ---
    adherence_rate = np.clip(
        1.0 / (1.0 + np.exp(-(score - _PDC_CENTER) * _PDC_SCALE)), 0.0, 1.0
    )
    # Calibrate: shift so that the (1 - prevalence) quantile of PDC sits at 0.8,
    # aligning the 0.8 PDC cut-point with the binary's threshold q.
    shift = 0.8 - float(np.quantile(adherence_rate, 1.0 - float(adherent_180d.mean())))
    adherence_rate = np.clip(adherence_rate + shift, 0.0, 1.0)

    gap_days = np.clip((1.0 - adherence_rate) * _GAP_WINDOW_DAYS, 0.0, _GAP_WINDOW_DAYS)
    low_gap_180d = (gap_days <= 30.0).astype(int)

    return {
        "adherent_180d": adherent_180d,
        "low_gap_180d": low_gap_180d,
        "adherence_rate": np.round(adherence_rate, 4),
        "gap_days": np.round(gap_days, 1),
        "adherent_rd_by_segment": rd_map_from_tau(segs, tau_adherent),
    }
