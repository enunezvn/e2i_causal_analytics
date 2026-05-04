"""Unit tests for ``_compute_anchor_point_metrics`` (Phase 1 W1 day 1).

Covers shard 20 §F anchor-metrics rows of the test plan:
- ``test_anchor_metrics_gate_inactive_at_low_prevalence``
- ``test_anchor_metrics_gate_active_at_high_prevalence``
- ``test_anchor_metrics_disagree_detection``
- ``test_anchor_metrics_no_primary_tau_returns_nan_block``
- ``test_anchor_metrics_threshold_kwarg_override``

Q-W5-2 RESOLVED 2026-05-01: gate ON when ``prevalence > 0.10``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _PREVALENCE_GATE_THRESHOLD,
    _compute_anchor_point_metrics,
)


def _stratified_dgp(
    n: int, prevalence: float, separation: str, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``y`` with target prevalence + ``proba_pos`` with given separation.

    ``separation`` controls how informative the model is:
      * ``"good"`` → bimodal, clear class separation.
      * ``"random"`` → uniform; classifier is ~useless.
      * ``"flipped"`` → labels and probabilities are anti-correlated.
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < prevalence).astype(int)
    if separation == "good":
        p = np.where(
            y == 1,
            rng.beta(8.0, 2.0, size=n),
            rng.beta(2.0, 8.0, size=n),
        )
    elif separation == "random":
        p = rng.uniform(size=n)
    elif separation == "flipped":
        p = np.where(
            y == 1,
            rng.beta(2.0, 8.0, size=n),
            rng.beta(8.0, 2.0, size=n),
        )
    else:
        raise ValueError(separation)
    return y, p


def test_anchor_metrics_gate_inactive_at_low_prevalence() -> None:
    """At prev = 0.05 the gate is silent and ``nb_anchor_passes is None``."""
    y, p = _stratified_dgp(n=4000, prevalence=0.05, separation="good")
    res = _compute_anchor_point_metrics(y, p, primary_tau=0.04, nb_area_relative=0.01)
    assert res["nb_anchor_secondary_gate_active"] is False
    assert res["nb_anchor_passes"] is None
    # Metric values still emit as finite floats (not NaN), only the gate
    # is inactive.
    assert not math.isnan(res["net_benefit_at_primary_tau"])
    assert not math.isnan(res["net_benefit_at_primary_tau_relative_to_treat_all"])


def test_anchor_metrics_gate_active_at_high_prevalence() -> None:
    """At prev = 0.40 the gate is on and ``nb_anchor_passes`` is bool."""
    y, p = _stratified_dgp(n=2000, prevalence=0.40, separation="good")
    res = _compute_anchor_point_metrics(y, p, primary_tau=0.40, nb_area_relative=0.05)
    assert res["nb_anchor_secondary_gate_active"] is True
    assert isinstance(res["nb_anchor_passes"], bool)


def test_anchor_metrics_disagree_detection_deterministic() -> None:
    """Force ``disagree=True`` deterministically (cycle-20 IMPORTANT-2 fix).

    To pin disagree to True we need:
      * gate active (prev > 0.10),
      * ``anchor_passes`` is False (``nb_at_primary_relative <= 0``),
      * ``area_passes`` is True (``nb_area_relative > 0``).

    Construction: prev = 0.30, classifier predicts a constant 0.20 for all
    samples. At ``primary_tau = 0.50``, ``y_pred = (p >= 0.50) = all-zero``,
    so ``TP = FP = 0`` and ``nb_at_primary = 0``. ``nb_treat_all_at_primary
    = 0.30 - 0.70 * 0.50/0.50 = -0.40``. Therefore
    ``nb_at_primary_relative = 0 - (-0.40) = +0.40``... wait that PASSES.

    Better: pass a synthetic ``nb_area_relative`` directly to force the
    area-passes/anchor-fails split. The helper takes ``nb_area_relative``
    as an input — it does NOT recompute it. So we choose a primary_tau
    and probabilities that yield a NEGATIVE anchor-relative, then pass a
    POSITIVE ``nb_area_relative`` separately.

    Construction (forced anchor-fail):
      * prev = 0.30 (gate active).
      * y is a deterministic 30% positive cohort.
      * p is a constant 0.55 for everyone.
      * primary_tau = 0.40. y_pred = (0.55 >= 0.40) = all-positive.
        TP/n = 0.30, FP/n = 0.70, nb = 0.30 - 0.70 * (0.40/0.60) = -0.167.
        nb_treat_all = 0.30 - 0.70 * (0.40/0.60) = -0.167.
        nb_relative = 0 (exact match). NOT a strict pass-or-fail; we need
        a strict negative.
      * primary_tau = 0.45. nb = 0.30 - 0.70 * (0.45/0.55) = -0.273.
        nb_treat_all = 0.30 - 0.70 * (0.45/0.55) = -0.273. Still 0.

    The model-equals-treat-all collision is what blocks a constant
    classifier. Use a class-aware classifier that at primary_tau predicts
    fewer positives than treat-all does (so NB_model - NB_treat_all is
    negative): probabilities below primary_tau for everyone.
      * p = 0.10 for all samples → y_pred = (0.10 >= 0.50) = all-zero.
        TP = FP = 0. nb_at_primary = 0.
        nb_treat_all_at_primary (τ=0.50, prev=0.30) = 0.30 - 0.70*1.0 = -0.40.
        nb_relative = 0 - (-0.40) = +0.40. PASSES (treat-all is so bad
        at τ=0.50 that doing nothing wins).

    The lesson: at high τ, treat-all goes very negative and a do-nothing
    model dominates it. To force anchor-fail we need a τ where treat-all
    is GOOD (low τ) and a model that performs WORSE than treat-all there.
      * τ = 0.05 (low). prev = 0.30. nb_treat_all = 0.30 - 0.70 * (0.05/0.95) = +0.263.
      * Model: y_pred = always 0 (p_const < 0.05 not really useful; use
        p_const = 0.01). y_pred = all-zero. TP=FP=0. nb = 0.
        nb_relative = 0 - 0.263 = -0.263. STRICTLY NEGATIVE → anchor fails.

    With anchor_passes=False, nb_area_relative=+0.01 (positive area) →
    area_passes=True → ``disagree=True``.
    """
    rng = np.random.default_rng(0)
    n = 1000
    y = np.zeros(n, dtype=int)
    n_pos = int(0.30 * n)
    y[:n_pos] = 1
    rng.shuffle(y)
    p = np.full(n, 0.01)  # constant low predictions, no positives at any τ ≥ 0.01
    res = _compute_anchor_point_metrics(y, p, primary_tau=0.05, nb_area_relative=0.01)
    # Pre-conditions for the disagree case.
    assert res["nb_anchor_secondary_gate_active"] is True
    assert res["net_benefit_at_primary_tau_relative_to_treat_all"] < 0.0
    assert res["nb_anchor_passes"] is False
    # Area-passes is computed from the input ``nb_area_relative=0.01 > 0``.
    # → disagree.
    assert res["nb_anchor_vs_area_disagree"] is True


def test_anchor_metrics_agree_when_both_pass() -> None:
    """Sanity: when area + anchor both pass, ``disagree`` is False."""
    y, p = _stratified_dgp(n=2000, prevalence=0.40, separation="good")
    res = _compute_anchor_point_metrics(y, p, primary_tau=0.40, nb_area_relative=0.05)
    if res["nb_anchor_passes"] is True and not math.isnan(
        res["net_benefit_at_primary_tau_relative_to_treat_all"]
    ):
        assert res["nb_anchor_vs_area_disagree"] is False


def test_anchor_metrics_no_primary_tau_returns_nan_block() -> None:
    """``primary_tau=None`` → all NB-at-primary fields NaN, gate inactive."""
    y, p = _stratified_dgp(n=1000, prevalence=0.20, separation="good")
    res = _compute_anchor_point_metrics(y, p, primary_tau=None, nb_area_relative=0.01)
    assert res["primary_tau"] is None
    assert math.isnan(res["net_benefit_at_primary_tau"])
    assert math.isnan(res["net_benefit_at_primary_tau_treat_all"])
    assert math.isnan(res["net_benefit_at_primary_tau_relative_to_treat_all"])
    assert res["nb_anchor_secondary_gate_active"] is False
    assert res["nb_anchor_passes"] is None
    assert res["nb_anchor_vs_area_disagree"] is False


def test_anchor_metrics_no_primary_tau_out_of_range() -> None:
    """An out-of-range ``primary_tau`` is treated like None."""
    y, p = _stratified_dgp(n=500, prevalence=0.20, separation="good")
    res = _compute_anchor_point_metrics(y, p, primary_tau=1.5, nb_area_relative=0.01)
    assert res["primary_tau"] is None
    assert math.isnan(res["net_benefit_at_primary_tau"])


def test_anchor_metrics_threshold_kwarg_override() -> None:
    """``prevalence_threshold=0.05`` flips a 0.07-prev sample to gate-active."""
    y, p = _stratified_dgp(n=4000, prevalence=0.07, separation="good")
    res_default = _compute_anchor_point_metrics(y, p, primary_tau=0.05, nb_area_relative=0.01)
    res_override = _compute_anchor_point_metrics(
        y,
        p,
        primary_tau=0.05,
        nb_area_relative=0.01,
        prevalence_threshold=0.05,
    )
    # Default threshold (0.10): 0.07 prev does NOT exceed → gate OFF.
    assert res_default["nb_anchor_secondary_gate_active"] is False
    assert res_default["nb_anchor_passes"] is None
    # Override threshold (0.05): 0.07 prev DOES exceed → gate ON.
    assert res_override["nb_anchor_secondary_gate_active"] is True
    assert isinstance(res_override["nb_anchor_passes"], bool)


def test_anchor_metrics_prevalence_constant_matches_module_value() -> None:
    """Sanity: the test imports the same constant the helper uses."""
    assert _PREVALENCE_GATE_THRESHOLD == pytest.approx(0.10)
