"""COMM-ARMS Phase 3: the INITIATION latent (``treatment_initiated``) with the two
Phase-3 commercial arms — ``rep_detailing_high`` + ``sample_dropped`` — folded in
alongside the existing ``treatment_arm``.

This is the initiation-latent analogue of ``adherence_outcomes.generate_adherence_outcomes``
(Phase 0-2). The three arms enter ONE shared latent score additively; each arm's
ground-truth recoverable RD is computed against an effective baseline that HOLDS THE
OTHER TWO ARMS at their realized contributions, thresholded on the SAME shared score —
so no arm's truth is a blend (additive-independent by design, no interactions).

Why a dedicated folder rather than calling ``binary_outcome_with_cate`` directly: that
helper returns only ``(y, tau_i)`` and cannot expose the shared score the per-arm RD
computation needs. This folder mirrors ``binary_outcome_with_cate``'s baseline+boost
math EXACTLY (same coefs, same ``_INIT_LATENT_CATE_BOOST``, same single noise draw via
``binary_outcome_rd``), so with rep/sample absent it is byte-identical to the pre-Phase-3
initiation outcome AND advances the caller's RNG stream identically. ``cate_modifier`` is
threaded to ``binary_outcome_rd`` so the SAME folder serves the Remibrutinib biologic-
experience rebuild (``patient_generator._apply_biologic_differential``), which folds
rep/sample in the SAME slot and must not clobber them.
"""

from __future__ import annotations

from typing import Dict, Optional, TypedDict

import numpy as np

from src.ml.synthetic.dgp.treatment_arm import (
    _INIT_BASE_ACADEMIC_COEF,
    _INIT_BASE_SEVERITY_COEF,
    _INIT_LATENT_CATE_BOOST,
    _INIT_NOISE_STD,
    _INIT_TARGET_PREVALENCE,
    binarize_score,
    binary_outcome_rd,
    rd_map_from_tau,
)


class InitiationOutcome(TypedDict):
    """Return shape of :func:`generate_initiation_outcome`.

    ``treatment_initiated`` is the recoverable binary; ``tau_i`` is treatment_arm's
    per-unit RD-scale CATE (persisted to ``treatment_effect_estimate``); the three
    ``*_rd_by_segment`` maps are the per-segment recoverable RD ground truth for each
    arm (consumed by the recovery probe). rep/sample maps are ``None`` when that arm
    is not supplied (so the folder degrades to the pre-Phase-3 single-arm outcome)."""

    treatment_initiated: np.ndarray
    tau_i: np.ndarray
    arm_rd_by_segment: Dict[str, float]
    rep_rd_by_segment: Optional[Dict[str, float]]
    sample_rd_by_segment: Optional[Dict[str, float]]
    trigger_rd_by_segment: Optional[Dict[str, float]]


def generate_initiation_outcome(
    *,
    treatment_arm: np.ndarray,
    disease_severity: np.ndarray,
    academic_hcp: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    prognostic_offset: np.ndarray | None = None,
    rep_detailing_high: np.ndarray | None = None,
    rep_cate: Dict[str, float] | None = None,
    sample_dropped: np.ndarray | None = None,
    sample_cate: Dict[str, float] | None = None,
    trigger_accepted: np.ndarray | None = None,
    trigger_cate: Dict[str, float] | None = None,
    cate_modifier: np.ndarray | None = None,
) -> InitiationOutcome:
    """Build ``treatment_initiated`` with treatment_arm + rep_detailing_high +
    sample_dropped folded into one shared latent, and return each arm's recoverable
    per-segment RD ground truth.

    Latent baseline = ``severity_coef*(severity-5) + academic_coef*academic +
    prognostic_offset + rep_contribution + sample_contribution``. ``treatment_arm``'s
    boosted segment CATE (``cate_map * _INIT_LATENT_CATE_BOOST``) enters the score via
    ``binary_outcome_rd`` (single noise draw, quantile-thresholded to
    ``_INIT_TARGET_PREVALENCE``). Each new arm's own RD is thresholded on that SAME
    score against a baseline holding the other two arms fixed — the mirror of
    ``adherence_outcomes``' copay/psp fold. ``rep_cate`` / ``sample_cate`` are the
    brand-scaled LATENT CATE maps and are NOT boosted (the boost is treatment_arm's).
    """
    arm = np.asarray(treatment_arm, dtype=int)
    severity = np.asarray(disease_severity, dtype=float)
    academic = np.asarray(academic_hcp, dtype=float)
    segs = np.asarray(segment)
    baseline = _INIT_BASE_SEVERITY_COEF * (severity - 5.0) + _INIT_BASE_ACADEMIC_COEF * academic
    if prognostic_offset is not None:
        baseline = baseline + np.asarray(prognostic_offset, dtype=float)

    # Fold the two Phase-3 arms into the shared baseline (mirror of the copay/psp fold
    # in adherence_outcomes). Each contribution is treated·its brand-scaled latent CATE.
    if rep_detailing_high is not None and rep_cate is not None:
        rep = np.asarray(rep_detailing_high, dtype=int)
        tau_rep = np.array([float(rep_cate[str(s)]) for s in segs], dtype=float)
        rep_contribution = rep.astype(float) * tau_rep
    else:
        tau_rep = np.zeros(len(segs), dtype=float)
        rep_contribution = np.zeros(len(segs), dtype=float)
    if sample_dropped is not None and sample_cate is not None:
        samp = np.asarray(sample_dropped, dtype=int)
        tau_samp = np.array([float(sample_cate[str(s)]) for s in segs], dtype=float)
        samp_contribution = samp.astype(float) * tau_samp
    else:
        tau_samp = np.zeros(len(segs), dtype=float)
        samp_contribution = np.zeros(len(segs), dtype=float)
    # COMM-ARMS Phase 4: trigger_accepted, the fourth arm in the shared latent —
    # identical additive-independent fold, no interactions.
    if trigger_accepted is not None and trigger_cate is not None:
        trig = np.asarray(trigger_accepted, dtype=int)
        tau_trig = np.array([float(trigger_cate[str(s)]) for s in segs], dtype=float)
        trig_contribution = trig.astype(float) * tau_trig
    else:
        tau_trig = np.zeros(len(segs), dtype=float)
        trig_contribution = np.zeros(len(segs), dtype=float)
    baseline = baseline + rep_contribution + samp_contribution + trig_contribution

    # treatment_arm keeps its tuned latent-CATE boost (T11), applied to the map BEFORE
    # delegation so the core stays boost-agnostic (identical to binary_outcome_with_cate).
    boosted_map = {str(s): float(v) * _INIT_LATENT_CATE_BOOST for s, v in cate_map.items()}
    treatment_initiated, tau_arm_i, score = binary_outcome_rd(
        arm,
        baseline,
        segs,
        boosted_map,
        rng,
        target_prevalence=_INIT_TARGET_PREVALENCE,
        noise_std=_INIT_NOISE_STD,
        return_score=True,
        cate_modifier=cate_modifier,
    )

    # Per-unit boosted arm tau (possibly biologic-modified) — the "other arm" term in
    # each new arm's effective baseline, matching what entered the shared score.
    tau_latent_arm = np.array([boosted_map[str(s)] for s in segs], dtype=float)
    if cate_modifier is not None:
        tau_latent_arm = tau_latent_arm * np.asarray(cate_modifier, dtype=float)
    arm_folded = arm.astype(float) * tau_latent_arm

    rep_rd: Optional[Dict[str, float]] = None
    if rep_detailing_high is not None and rep_cate is not None:
        # rep's effective baseline: everything EXCEPT rep's own contribution
        # (baseline already holds rep+sample, so subtract rep, add the arm term).
        rep_eff_baseline = baseline - rep_contribution + arm_folded
        _, tau_rep_i = binarize_score(
            score,
            rep_eff_baseline,
            tau_rep,
            segs,
            target_prevalence=_INIT_TARGET_PREVALENCE,
            noise_std=_INIT_NOISE_STD,
        )
        rep_rd = rd_map_from_tau(segs, tau_rep_i)

    sample_rd: Optional[Dict[str, float]] = None
    if sample_dropped is not None and sample_cate is not None:
        samp_eff_baseline = baseline - samp_contribution + arm_folded
        _, tau_samp_i = binarize_score(
            score,
            samp_eff_baseline,
            tau_samp,
            segs,
            target_prevalence=_INIT_TARGET_PREVALENCE,
            noise_std=_INIT_NOISE_STD,
        )
        sample_rd = rd_map_from_tau(segs, tau_samp_i)

    trigger_rd: Optional[Dict[str, float]] = None
    if trigger_accepted is not None and trigger_cate is not None:
        trig_eff_baseline = baseline - trig_contribution + arm_folded
        _, tau_trig_i = binarize_score(
            score,
            trig_eff_baseline,
            tau_trig,
            segs,
            target_prevalence=_INIT_TARGET_PREVALENCE,
            noise_std=_INIT_NOISE_STD,
        )
        trigger_rd = rd_map_from_tau(segs, tau_trig_i)

    return {
        "treatment_initiated": treatment_initiated,
        "tau_i": tau_arm_i,
        "arm_rd_by_segment": rd_map_from_tau(segs, tau_arm_i),
        "rep_rd_by_segment": rep_rd,
        "sample_rd_by_segment": sample_rd,
        "trigger_rd_by_segment": trigger_rd,
    }
