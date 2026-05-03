"""Unit tests for the W1 day-2 NB-area + DCA + anchor emission wiring.

Covers shard 20 §G acceptance #7 + #11 + #12 — emission gating on
``clinical_threshold_range`` presence in ``success_criteria``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _V3_NB_GRID_P_T_VALUES,
    _compute_classification_metrics,
)


def _toy_classification_inputs(seed: int = 42) -> dict:
    """Generate a small but realistic binary classification fixture.

    Returns kwargs for ``_compute_classification_metrics`` — train + val +
    test arrays with deterministic predictions and probabilities.
    """
    rng = np.random.default_rng(seed)

    def _split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = (rng.uniform(size=n) < 0.30).astype(int)
        # Stratified probability draw: positives skew high, negatives low.
        p = np.where(
            y == 1,
            rng.beta(5.0, 2.0, size=n),
            rng.beta(2.0, 5.0, size=n),
        )
        proba = np.column_stack([1.0 - p, p])
        pred = (p >= 0.5).astype(int)
        return y, pred, proba

    y_train, y_train_pred, y_train_proba = _split(2000)
    y_val, y_val_pred, y_val_proba = _split(800)
    y_test, y_test_pred, y_test_proba = _split(800)
    return {
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_train_proba": y_train_proba,
        "y_validation": y_val,
        "y_validation_pred": y_val_pred,
        "y_validation_proba": y_val_proba,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
        "imbalance_detected": False,
        "minority_ratio": 0.30,
    }


def test_emission_gated_off_when_no_clinical_threshold_range() -> None:
    """No ``clinical_threshold_range`` → no NB-area / DCA / anchor fields."""
    kwargs = _toy_classification_inputs()
    out = _compute_classification_metrics(**kwargs, success_criteria=None)
    test_metrics = out["test_metrics"]

    # Legacy net_benefit_grid SHOULD still be present (single-mode parity
    # per shard 20 §G acceptance #7).
    assert "net_benefit_grid" in test_metrics
    # New W1 day-2 fields MUST be absent in legacy mode.
    assert "net_benefit_area" not in test_metrics
    assert "decision_curve_data" not in test_metrics
    assert "primary_tau" not in test_metrics
    assert "nb_anchor_secondary_gate_active" not in test_metrics


def test_emission_gated_on_when_clinical_threshold_range_present() -> None:
    """``clinical_threshold_range`` present → NB-area + DCA + anchor block emit."""
    kwargs = _toy_classification_inputs()
    sc = {
        "dataset_disease": "breast_cancer_recurrence",
        "clinical_threshold_range": {"use_case": "diagnostic"},
    }
    out = _compute_classification_metrics(**kwargs, success_criteria=sc)
    test_metrics = out["test_metrics"]

    # NB-area block fields.
    assert "net_benefit_area" in test_metrics
    assert "net_benefit_area_treat_all" in test_metrics
    assert "net_benefit_area_relative_to_treat_all" in test_metrics
    assert "net_benefit_area_form" in test_metrics
    assert "tau_low" in test_metrics
    assert "tau_high" in test_metrics
    assert "n_grid_points" in test_metrics

    # DCA artifact.
    dca = test_metrics["decision_curve_data"]
    assert isinstance(dca, dict)
    assert len(dca["nb_model"]) == 21
    assert len(dca["nb_treat_all"]) == 21
    assert dca["n_grid_points"] == 21

    # Anchor block (primary_tau resolves from disease default).
    assert test_metrics["primary_tau"] == pytest.approx(0.21)
    assert "nb_anchor_secondary_gate_active" in test_metrics
    # Disease bounds beat use_case=diagnostic → tau_low == 0.10.
    assert test_metrics["tau_low"] == pytest.approx(0.10)
    assert test_metrics["tau_high"] == pytest.approx(0.30)


def test_emission_uses_use_case_when_no_disease() -> None:
    """A bare ``use_case=diagnostic`` resolves to [0.05, 0.30] grid + anchor=None."""
    kwargs = _toy_classification_inputs()
    sc = {"clinical_threshold_range": {"use_case": "diagnostic"}}
    out = _compute_classification_metrics(**kwargs, success_criteria=sc)
    test_metrics = out["test_metrics"]
    assert test_metrics["tau_low"] == pytest.approx(0.05)
    assert test_metrics["tau_high"] == pytest.approx(0.30)
    # Without an explicit disease label, primary_tau is None → anchor NaN.
    # Cycle-21 C-3: also assert the downstream anchor-block state so the
    # `primary_tau=None` semantic propagates correctly.
    assert test_metrics["primary_tau"] is None
    assert test_metrics["nb_anchor_secondary_gate_active"] is False
    assert test_metrics["nb_anchor_passes"] is None
    assert test_metrics["nb_anchor_vs_area_disagree"] is False


def test_emission_warns_when_tau_high_above_080(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cycle-20 IMPORTANT-1: ``tau_high > 0.80`` triggers a WARNING."""
    kwargs = _toy_classification_inputs()
    sc = {
        "clinical_threshold_range": {
            "use_case": "custom",
            "tau_low": 0.20,
            "tau_high": 0.90,
        }
    }
    with caplog.at_level(logging.WARNING):
        _compute_classification_metrics(**kwargs, success_criteria=sc)
    assert any(
        "tau_high=0.9" in rec.getMessage() and "treat-all NB diverges" in rec.getMessage()
        for rec in caplog.records
    )


def test_emission_info_when_legacy_grid_used(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cycle-20 COSMETIC-2: K=6 legacy fallback emits INFO about precision."""
    kwargs = _toy_classification_inputs()
    # Empty CTR sub-dict → resolver returns ``legacy_grid``.
    sc = {"clinical_threshold_range": {}}
    with caplog.at_level(logging.INFO):
        _compute_classification_metrics(**kwargs, success_criteria=sc)
    assert any(
        "K=6 legacy grid" in rec.getMessage()
        for rec in caplog.records
    )
    # Verify the grid that was used matches legacy (length 6).
    # (Inferred via the response: tau_low/tau_high pin the legacy bounds.)


def test_emission_legacy_grid_shape_matches_v3_constants() -> None:
    """Sanity: legacy fallback emits a 6-point grid spanning [0.05, 0.50]."""
    kwargs = _toy_classification_inputs()
    sc = {"clinical_threshold_range": {}}  # empty CTR → legacy
    out = _compute_classification_metrics(**kwargs, success_criteria=sc)
    test_metrics = out["test_metrics"]
    assert test_metrics["n_grid_points"] == len(_V3_NB_GRID_P_T_VALUES)
    assert test_metrics["tau_low"] == pytest.approx(_V3_NB_GRID_P_T_VALUES[0])
    assert test_metrics["tau_high"] == pytest.approx(_V3_NB_GRID_P_T_VALUES[-1])
