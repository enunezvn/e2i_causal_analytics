"""Phase 0 of commercial-arms enrichment: binarized adherence outcomes for the
EXISTING treatment_arm, plus clinically-coherent raw proxies drawn from the same
latent score (so adherence_rate>=0.8 agrees with the authoritative binary).

Design note: the binary (adherent_180d) and the continuous proxy (adherence_rate)
share the SAME latent score — binary_outcome_rd is called with return_score=True
and the PDC proxy is a monotone logistic squash of that returned score. Sharing
the noise draw is what makes adherence_rate>=0.8 agree with adherent_180d on >95%
of rows after calibration, far above the >=0.80 floor. The binary stays
AUTHORITATIVE (its known per-segment counterfactual RD comes from the core); the
quantile/RD math lives ONLY in treatment_arm.py — this module computes neither.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.ml.synthetic.dgp.treatment_arm import binary_outcome_rd, rd_map_from_tau

# Adherence baseline: sicker patients are modestly LESS adherent; academic-HCP
# patients modestly MORE (kept small so the arm effect is the dominant signal).
_ADH_SEVERITY_COEF = -0.08
_ADH_ACADEMIC_COEF = 0.12
_ADH_NOISE_STD = 0.6
# Map the adherence latent to a PDC in [0,1] via logistic squash.
_PDC_CENTER = 0.0
_PDC_SCALE = 1.1
# Target marginal prevalence of adherent_180d (clamped to [0.20, 0.50] by the core).
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

    The binary outcomes are AUTHORITATIVE: adherent_180d and its known per-segment
    counterfactual RD come from binary_outcome_rd (the single RD SSOT). We ask that
    core for the latent score it thresholded (return_score=True) and build the PDC
    proxy as a monotone logistic transform of the SAME score, so adherence_rate>=0.8
    agrees with adherent_180d==1 on >95% of rows after calibration. gap_days is the
    inverse proxy (gap_days <= 30 => low_gap_180d).
    """
    arm = np.asarray(treatment_arm, dtype=int)
    severity = np.asarray(disease_severity, dtype=float)
    academic = np.asarray(academic_hcp, dtype=float)
    baseline = _ADH_SEVERITY_COEF * (severity - 5.0) + _ADH_ACADEMIC_COEF * academic

    # Authoritative binary + its known per-segment RD + the shared latent score.
    adherent_180d, tau_adherent, score = binary_outcome_rd(
        arm, baseline, segment, cate_map, rng,
        target_prevalence=_TARGET_PREVALENCE, noise_std=_ADH_NOISE_STD,
        return_score=True,
    )

    # Raw PDC proxy from the SAME score (monotone => ordinal agreement preserved).
    adherence_rate = np.clip(
        1.0 / (1.0 + np.exp(-(score - _PDC_CENTER) * _PDC_SCALE)), 0.0, 1.0
    )
    # Calibrate: shift so the (1 - prevalence) quantile of PDC sits at 0.8, aligning
    # the 0.8 PDC cut-point with the binary's threshold.
    shift = 0.8 - float(np.quantile(adherence_rate, 1.0 - float(adherent_180d.mean())))
    adherence_rate = np.clip(adherence_rate + shift, 0.0, 1.0)

    gap_days = np.clip((1.0 - adherence_rate) * _GAP_WINDOW_DAYS, 0.0, _GAP_WINDOW_DAYS)
    low_gap_180d = (gap_days <= 30.0).astype(int)

    segs = np.asarray(segment)
    return {
        "adherent_180d": adherent_180d,
        "low_gap_180d": low_gap_180d,
        "adherence_rate": np.round(adherence_rate, 4),
        "gap_days": np.round(gap_days, 1),
        "adherent_rd_by_segment": rd_map_from_tau(segs, tau_adherent),
    }
