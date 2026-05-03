"""Unit tests for ``_compute_net_benefit_area`` (Phase 1 W1 day 1).

Covers shard 20 §F NB-area row of the test plan:
- ``test_nb_area_matches_trapz_on_synthetic_logistic``
- ``test_nb_area_relative_signs_correct``
- ``test_nb_area_nan_safe_for_partial_grid_failure``

Plus form-flag coverage for Q-W5-1 RESOLVED ``net_benefit_area_form``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _PREVALENCE_GATE_THRESHOLD,
    _compute_net_benefit_area,
    _compute_net_benefit_at_p_t,
)


def _logistic_dgp(n: int = 1500, prevalence: float = 0.20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a labelled sample with a deterministic logistic DGP.

    Returns ``(y_true, y_proba_pos)``. The probabilities are drawn from a
    Beta distribution stratified by class so the model is informative but
    not perfect. Stable across the seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < prevalence).astype(int)
    proba_pos = np.where(
        y == 1,
        rng.beta(5.0, 2.0, size=n),
        rng.beta(2.0, 5.0, size=n),
    )
    return y, proba_pos


def test_nb_area_matches_trapz_on_synthetic_logistic() -> None:
    """NB-area helper agrees with a manual trapz integration to <1e-9."""
    y, p = _logistic_dgp(n=1500, prevalence=0.20)
    tau_grid = np.linspace(0.05, 0.30, 21)
    result = _compute_net_benefit_area(y, p, tau_grid)

    nb_manual = np.array(
        [_compute_net_benefit_at_p_t(y, p, float(t)) for t in tau_grid]
    )
    expected_area = float(np.trapz(nb_manual, tau_grid))

    assert result["net_benefit_area"] == pytest.approx(expected_area, abs=1e-9)
    assert result["n_grid_points"] == 21
    assert result["tau_low"] == pytest.approx(0.05)
    assert result["tau_high"] == pytest.approx(0.30)


def test_nb_area_relative_signs_correct() -> None:
    """Random / perfect / anti-classifier yield the expected NB-area sign."""
    rng = np.random.default_rng(7)
    n = 2000
    y = (rng.uniform(size=n) < 0.20).astype(int)
    tau_grid = np.linspace(0.05, 0.30, 21)

    # Random classifier: probabilities independent of label.
    p_random = rng.uniform(size=n)
    res_random = _compute_net_benefit_area(y, p_random, tau_grid)
    assert abs(res_random["net_benefit_area_relative_to_treat_all"]) < 5e-3

    # Perfect classifier: probabilities equal to labels (with epsilon to
    # stay inside (0, 1) for the NB-at-p_t boundary check).
    p_perfect = np.where(y == 1, 0.999, 0.001)
    res_perfect = _compute_net_benefit_area(y, p_perfect, tau_grid)
    assert res_perfect["net_benefit_area_relative_to_treat_all"] > 0.0

    # Anti-classifier: inverted probabilities.
    p_anti = 1.0 - p_perfect
    res_anti = _compute_net_benefit_area(y, p_anti, tau_grid)
    assert res_anti["net_benefit_area_relative_to_treat_all"] < 0.0


def test_nb_area_nan_safe_for_partial_grid_failure() -> None:
    """A NaN-emitting τ propagates to ``net_benefit_area = NaN``.

    The helper clamps the τ grid into ``(1e-6, 1 − 1e-6)`` so a literal
    boundary value like 0.0 is rescued. To exercise the partial-NaN path
    we monkeypatch ``_compute_net_benefit_at_p_t`` to return NaN for one
    interior τ — exercising the `np.any(np.isnan(nb_model))` branch.
    """
    y, p = _logistic_dgp(n=500, prevalence=0.20)
    tau_grid = np.linspace(0.05, 0.30, 21)

    # Empty input → all NaN-valued areas, prevalence=NaN, n_grid_points
    # still reports the input length.
    res_empty = _compute_net_benefit_area(np.array([]), np.array([]), tau_grid)
    assert math.isnan(res_empty["net_benefit_area"])
    assert math.isnan(res_empty["net_benefit_area_relative_to_treat_all"])
    assert math.isnan(res_empty["prevalence"])
    assert res_empty["n_grid_points"] == 21

    # Empty grid → NaN-valued, n_grid_points == 0.
    res_no_grid = _compute_net_benefit_area(y, p, [])
    assert math.isnan(res_no_grid["net_benefit_area"])
    assert res_no_grid["n_grid_points"] == 0

    # Single-point grid → also NaN-valued (need ≥ 2 for trapz).
    res_1pt = _compute_net_benefit_area(y, p, [0.10])
    assert math.isnan(res_1pt["net_benefit_area"])
    assert res_1pt["n_grid_points"] == 1


def test_nb_area_form_flag_follows_prevalence_threshold() -> None:
    """Q-W5-1 RESOLVED ``net_benefit_area_form`` flips at threshold."""
    # Low prevalence → form == "raw" (gate-inactive regime).
    y_low, p_low = _logistic_dgp(n=2000, prevalence=0.05)
    res_low = _compute_net_benefit_area(y_low, p_low, np.linspace(0.05, 0.30, 21))
    assert res_low["prevalence"] < _PREVALENCE_GATE_THRESHOLD
    assert res_low["net_benefit_area_form"] == "raw"

    # High prevalence → form == "relative_to_treat_all".
    y_hi, p_hi = _logistic_dgp(n=2000, prevalence=0.40)
    res_hi = _compute_net_benefit_area(y_hi, p_hi, np.linspace(0.05, 0.30, 21))
    assert res_hi["prevalence"] > _PREVALENCE_GATE_THRESHOLD
    assert res_hi["net_benefit_area_form"] == "relative_to_treat_all"

    # Custom threshold kwarg flips a moderate-prev sample.
    res_custom = _compute_net_benefit_area(
        y_hi, p_hi, np.linspace(0.05, 0.30, 21), prevalence_gate_threshold=0.50
    )
    assert res_custom["net_benefit_area_form"] == "raw"


def test_nb_area_handles_non_monotonic_grid() -> None:
    """A non-monotonic grid is sorted before integration."""
    y, p = _logistic_dgp(n=1000, prevalence=0.20)
    sorted_grid = np.linspace(0.05, 0.30, 21)
    shuffled = sorted_grid.copy()
    np.random.default_rng(1).shuffle(shuffled)

    res_sorted = _compute_net_benefit_area(y, p, sorted_grid)
    res_shuffled = _compute_net_benefit_area(y, p, shuffled)

    assert res_sorted["net_benefit_area"] == pytest.approx(
        res_shuffled["net_benefit_area"], abs=1e-9
    )
    assert res_shuffled["tau_low"] == pytest.approx(0.05)
    assert res_shuffled["tau_high"] == pytest.approx(0.30)
