"""Phase 0 of commercial-arms enrichment: TWO binarized adherence outcomes for the
EXISTING treatment_arm (adherent_180d, low_gap_180d), plus clinically-coherent raw
proxies (adherence_rate, gap_days) SNAPPED to be EXACTLY consistent with their binary.

Design note: both curated binaries are thresholds on ONE shared latent ``score``
(binary_outcome_rd builds it once, with a single noise draw, and returns it via
return_score=True; binarize_score thresholds the SAME score at a second, rarer
prevalence for low_gap_180d). Because low_gap's threshold is higher, low_gap rows
are a SUBSET of adherent rows by construction. Both binaries therefore carry a known
per-segment counterfactual RD (the quantile/RD math lives ONLY in treatment_arm.py;
this module computes neither).

The continuous proxies are monotone transforms of that same score, then SNAPPED so
the STORED value can NEVER contradict the STORED binary:
  - round(adherence_rate, 4) >= 0.80  <=>  adherent_180d == 1
  - gap_days (integer days)  <= 30    <=>  low_gap_180d == 1
Agreement is 100% by construction (exact), not approximate.
"""

from __future__ import annotations

from typing import Dict, TypedDict

import numpy as np

from src.ml.synthetic.dgp.treatment_arm import (
    binarize_score,
    binary_outcome_rd,
    rd_map_from_tau,
)

# Adherence baseline: sicker patients are modestly LESS adherent; academic-HCP
# patients modestly MORE (kept small so the arm effect is the dominant signal).
_ADH_SEVERITY_COEF = -0.08
_ADH_ACADEMIC_COEF = 0.12
_ADH_NOISE_STD = 0.6
# Map the adherence latent to a PDC in [0,1] via logistic squash.
_PDC_CENTER = 0.0
_PDC_SCALE = 1.1
# gap_days: inverse of adherence; ~ (1 - PDC) * 180-day window, floored at 0.
_GAP_WINDOW_DAYS = 180.0
# Target marginal prevalences (clamped to [0.20, 0.50] by the core). low_gap is the
# rarer outcome (fewer patients hit a clean <=30d gap), so its threshold is higher
# -> low_gap rows are a strict SUBSET of adherent rows on the shared score (clinically
# a <=30d gap implies PDC>=0.8), which keeps the proxy-consistency snap valid.
_TARGET_PREVALENCE = 0.35
_LOW_GAP_PREVALENCE = 0.30

# Latent-CATE boost on the adherence segment CATE map — adopts the initiation outcome's
# proven _INIT_LATENT_CATE_BOOST mechanism (treatment_arm.py; initiation uses 1.30). WHY:
# the flattest brand (Fabhalta, 0.70 CATE scale) sits at CausalForestDML's n=3000 high-vs-
# medium RESOLUTION FLOOR for BOTH adherence outcomes — without a boost, Fabhalta-adherent
# recovers only by a razor-thin est-margin (+0.0135, seed=21) and Fabhalta-low_gap INVERTS;
# tuning the prevalence alone can't lift it off the floor (a seed=21 surface map showed the
# margin is a noise-floor coin-flip peaking at only +0.004). Applied as a UNIFORM positive
# factor on cate_map BEFORE use, so it flows consistently into both the latent score
# (binary_outcome_rd) AND the RD ground truth (binarize_score's tau_latent) — preserving
# high>medium>low ordering and self-consistency (the persisted true ATE/CATE scale WITH the
# planted effect). The PDC/gap proxies are score-based and the snap is boost-independent, so
# HIGH-3's zero-contradiction is untouched.
#
# TUNED for ROBUSTNESS, not a seed-locked green. Worst case is Fabhalta-low_gap; per-seed
# est(high)-est(medium) margins (seeds 21/7/99/123, the recovery_probe set; seed=21 is the
# GATE seed and a hard sample where the forest compresses medium toward high):
#   boost 1.30 -> 21:+0.0112  7:+0.2020  99:+0.1160  123:+0.1821  (min +0.0112; Kisqali ATE 0.324)
#   boost 1.40 -> 21:+0.0255  7:+0.2140  99:+0.1209  123:+0.1771  (min +0.0255; Kisqali ATE 0.347)
#   boost 1.50 -> 21:+0.0156  7:+0.2093  99:+0.1103  123:+0.1488  (min +0.0156; Kisqali ATE 0.371)
# 1.40 is the realistic-band OPTIMUM: it MAXIMISES the worst-case (gate-seed) margin (~2.3x the
# 1.30 value) while keeping effects realistic; 1.30 leaves seed=21 fragile and >=1.50 over-
# inflates Kisqali (>0.37 ATE) WITHOUT improving seed=21. The +0.03 heuristic isn't cleanly
# reachable — seed=21 caps ~+0.025 for this flattest brand at any realistic boost — but EVERY
# seed is positive (min +0.0255), which is the actual proof the recovery is not a coin-flip.
_ADH_LATENT_CATE_BOOST = 1.40


class AdherenceOutcomes(TypedDict):
    """Return shape of :func:`generate_adherence_outcomes`. The two binaries +
    two continuous proxies are per-unit np.ndarrays; the two ``*_rd_by_segment``
    maps are the per-segment recoverable RD ground truth consumed by the probe."""

    adherent_180d: np.ndarray
    low_gap_180d: np.ndarray
    adherence_rate: np.ndarray
    gap_days: np.ndarray
    adherent_rd_by_segment: Dict[str, float]
    low_gap_rd_by_segment: Dict[str, float]
    copay_adherent_rd_by_segment: Dict[str, float]
    copay_low_gap_rd_by_segment: Dict[str, float]


def generate_adherence_outcomes(
    *,
    treatment_arm: np.ndarray,
    disease_severity: np.ndarray,
    academic_hcp: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    copay_support: np.ndarray | None = None,
    copay_cate: Dict[str, float] | None = None,
) -> AdherenceOutcomes:
    """Return adherent_180d / low_gap_180d (recoverable binaries) + adherence_rate
    / gap_days (raw proxies) + the per-segment RD ground-truth map for BOTH binaries.

    Both binaries are AUTHORITATIVE and recovery-gated: they threshold ONE shared
    latent score at two prevalences, so each carries a known per-segment
    counterfactual RD from the single RD SSOT (treatment_arm.py). The proxies are
    monotone transforms of the same score, SNAPPED so the stored continuous value
    can NEVER contradict the stored binary (100% agreement by construction).
    """
    arm = np.asarray(treatment_arm, dtype=int)
    severity = np.asarray(disease_severity, dtype=float)
    academic = np.asarray(academic_hcp, dtype=float)
    segs = np.asarray(segment)
    baseline = _ADH_SEVERITY_COEF * (severity - 5.0) + _ADH_ACADEMIC_COEF * academic

    # Phase 1: the copay arm enters the SAME latent additively. Its contribution
    # folds into the EFFECTIVE BASELINE for treatment_arm's counterfactual RD (and
    # vice versa below) so each arm's ground truth is its OWN effect, not a blend.
    # copay_cate is the brand-scaled latent CATE map; it is NOT boosted by
    # _ADH_LATENT_CATE_BOOST (that boost is calibrated for treatment_arm only).
    if copay_support is not None and copay_cate is not None:
        copay = np.asarray(copay_support, dtype=int)
        tau_copay = np.array([float(copay_cate[str(s)]) for s in segs], dtype=float)
        copay_contribution = copay.astype(float) * tau_copay
    else:
        copay = np.zeros(len(segs), dtype=int)
        tau_copay = np.zeros(len(segs), dtype=float)
        copay_contribution = np.zeros(len(segs), dtype=float)
    baseline = baseline + copay_contribution

    # Boost the latent CATE map BEFORE use (cf. initiation's boosted_map) so the boost
    # flows CONSISTENTLY into both the score (binary_outcome_rd) and the RD ground truth
    # (binarize_score's tau_latent). Uniform positive factor => ordering preserved.
    boosted_map = {
        str(s): float(cate_map[str(s)]) * _ADH_LATENT_CATE_BOOST for s in np.unique(segs)
    }
    tau_latent = np.array([boosted_map[str(s)] for s in segs], dtype=float)

    # One shared recoverable latent score (single noise draw) + the adherent binary.
    adherent_180d, tau_adherent, score = binary_outcome_rd(
        arm,
        baseline,
        segs,
        boosted_map,
        rng,
        target_prevalence=_TARGET_PREVALENCE,
        noise_std=_ADH_NOISE_STD,
        return_score=True,
    )
    # Second recoverable binary on the SAME score, at its own (rarer) prevalence.
    # low_gap is the top _LOW_GAP_PREVALENCE by score -> a subset of adherent.
    low_gap_180d, tau_lowgap = binarize_score(
        score,
        baseline,
        tau_latent,
        segs,
        target_prevalence=_LOW_GAP_PREVALENCE,
        noise_std=_ADH_NOISE_STD,
    )

    # PDC proxy: monotone-INCREASING transform of the same score, calibrated so its
    # 0.8 cut-point rank-matches the adherent prevalence, then SNAPPED so the STORED
    # value can never contradict adherent_180d (round(PDC,4)>=0.8 <=> adherent==1).
    pdc = 1.0 / (1.0 + np.exp(-(score - _PDC_CENTER) * _PDC_SCALE))
    pdc_shift = 0.8 - float(np.quantile(pdc, 1.0 - float(adherent_180d.mean())))
    pdc = np.clip(pdc + pdc_shift, 0.0, 1.0)
    pdc = np.where(adherent_180d == 1, np.maximum(pdc, 0.80), np.minimum(pdc, 0.7999))
    adherence_rate = np.round(np.clip(pdc, 0.0, 1.0), 4)

    # gap_days proxy: monotone-DECREASING in score, SNAPPED so the STORED INTEGER
    # gap can never contradict low_gap_180d. The DB column is INTEGER (migration
    # 033), so gap_days is WHOLE refill-gap days: low_gap==1 -> gap<=30; low_gap==0
    # -> gap>=31 (the non-low_gap floor is 31.0, NOT 30.1, so integer rounding keeps
    # gap>30 and the (gap<=30)<=>low_gap binary stays exact). low_gap rows are a
    # subset of adherent, so pdc>=0.8 there.
    gap = (1.0 - pdc) * _GAP_WINDOW_DAYS
    gap = np.where(low_gap_180d == 1, np.minimum(gap, 30.0), np.maximum(gap, 31.0))
    gap_days = np.clip(np.round(gap), 0.0, _GAP_WINDOW_DAYS).astype(int)

    # copay's OWN recoverable RD: treatment_arm's contribution folds into ITS
    # effective baseline (the mirror of the fold above), thresholded on the SAME
    # shared score so both arms' truths describe the same realized outcome.
    copay_eff_baseline = baseline - copay_contribution + arm.astype(float) * tau_latent
    _, tau_copay_adh = binarize_score(
        score,
        copay_eff_baseline,
        tau_copay,
        segs,
        target_prevalence=_TARGET_PREVALENCE,
        noise_std=_ADH_NOISE_STD,
    )
    _, tau_copay_low = binarize_score(
        score,
        copay_eff_baseline,
        tau_copay,
        segs,
        target_prevalence=_LOW_GAP_PREVALENCE,
        noise_std=_ADH_NOISE_STD,
    )

    return {
        "adherent_180d": adherent_180d,
        "low_gap_180d": low_gap_180d,
        "adherence_rate": adherence_rate,
        "gap_days": gap_days,
        "adherent_rd_by_segment": rd_map_from_tau(segs, tau_adherent),
        "low_gap_rd_by_segment": rd_map_from_tau(segs, tau_lowgap),
        "copay_adherent_rd_by_segment": rd_map_from_tau(segs, tau_copay_adh),
        "copay_low_gap_rd_by_segment": rd_map_from_tau(segs, tau_copay_low),
    }
