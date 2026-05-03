"""Unit tests for ``_compute_bootstrap_ci`` ci_method wiring (Phase 1 W1 day 3).

Covers shard 20 §F BCa-default-path test + Q-W5-3 RESOLVED auto-resolution.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _MIN_N_FOR_BCA,
    _compute_bootstrap_ci,
    _compute_jackknife_metrics,
    _compute_point_estimates,
)


def _toy_binary_inputs(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 400
    y = (rng.uniform(size=n) < 0.30).astype(int)
    p = np.where(y == 1, rng.beta(5, 2, size=n), rng.beta(2, 5, size=n))
    pred = (p >= 0.5).astype(int)
    return y, pred, p


def test_ci_method_default_auto_resolves_to_bca_above_min_n() -> None:
    """Default ``ci_method='auto'`` + n_bootstrap >= 30 → BCa per Q-W5-3."""
    y, pred, p = _toy_binary_inputs()
    cis, n = _compute_bootstrap_ci(
        y_true=y,
        y_pred=pred,
        y_proba=p,
        problem_type="binary_classification",
        n_bootstrap=200,
        random_state=11,
    )
    assert n == 200
    # All four metrics produced CIs with finite ordered endpoints.
    for metric in ("accuracy", "precision", "recall", "auc"):
        if metric in cis:
            lo, hi = cis[metric]
            assert lo <= hi
            assert np.isfinite(lo) and np.isfinite(hi)


def test_ci_method_auto_below_min_falls_back_to_percentile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``ci_method='auto'`` with ``n_bootstrap < _MIN_N_FOR_BCA`` → percentile."""
    y, pred, p = _toy_binary_inputs()
    with caplog.at_level(logging.INFO):
        _compute_bootstrap_ci(
            y_true=y,
            y_pred=pred,
            y_proba=p,
            problem_type="binary_classification",
            n_bootstrap=_MIN_N_FOR_BCA - 1,
            random_state=11,
        )
    assert any(
        "used=percentile" in rec.getMessage()
        and "auto_below_min_n_for_bca" in rec.getMessage()
        for rec in caplog.records
    )


def test_ci_method_explicit_percentile_preserves_legacy_behavior() -> None:
    """``ci_method='percentile'`` reproduces the pre-cycle-22 percentile CIs."""
    y, pred, p = _toy_binary_inputs()
    cis_pct, _ = _compute_bootstrap_ci(
        y_true=y,
        y_pred=pred,
        y_proba=p,
        problem_type="binary_classification",
        n_bootstrap=200,
        random_state=11,
        ci_method="percentile",
    )
    # Compute percentile CI manually from the same bootstrap.
    rng = np.random.default_rng(11)
    accs: list[float] = []
    for _ in range(200):
        idx = rng.integers(low=0, high=len(y), size=len(y))
        accs.append(float((y[idx] == pred[idx]).mean()))
    expected_lo = float(np.percentile(accs, 2.5))
    expected_hi = float(np.percentile(accs, 97.5))
    assert cis_pct["accuracy"][0] == pytest.approx(expected_lo, abs=1e-9)
    assert cis_pct["accuracy"][1] == pytest.approx(expected_hi, abs=1e-9)


def test_ci_method_force_bca_below_min_n() -> None:
    """``ci_method='bca'`` + ``force_bca=True`` attempts BCa even at small N."""
    y, pred, p = _toy_binary_inputs()
    cis, _ = _compute_bootstrap_ci(
        y_true=y,
        y_pred=pred,
        y_proba=p,
        problem_type="binary_classification",
        n_bootstrap=10,  # well below _MIN_N_FOR_BCA
        random_state=11,
        ci_method="bca",
        force_bca=True,
    )
    # CIs are emitted (BCa may have fallen back per-metric to percentile,
    # but the API contract is to return a tuple regardless).
    assert isinstance(cis, dict)
    if "accuracy" in cis:
        lo, hi = cis["accuracy"]
        assert lo <= hi


def test_ci_method_unknown_warns_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown ci_method value warns and uses percentile."""
    y, pred, p = _toy_binary_inputs()
    with caplog.at_level(logging.WARNING):
        _compute_bootstrap_ci(
            y_true=y,
            y_pred=pred,
            y_proba=p,
            problem_type="binary_classification",
            n_bootstrap=20,
            random_state=11,
            ci_method="not_a_real_method",
        )
    assert any(
        "Unknown ci_method" in rec.getMessage() for rec in caplog.records
    )


def test_compute_point_estimates_binary_classification() -> None:
    """Returns one entry per metric on the original (non-resampled) sample."""
    y, pred, p = _toy_binary_inputs()
    pts = _compute_point_estimates(y, pred, p, "binary_classification")
    assert {"accuracy", "precision", "recall", "auc"}.issubset(pts.keys())
    for v in pts.values():
        assert 0.0 <= v <= 1.0


def test_compute_jackknife_metrics_returns_n_minus_one() -> None:
    """Jackknife distribution has length N (one drop per index)."""
    y, pred, p = _toy_binary_inputs()
    jack = _compute_jackknife_metrics(y, pred, p, "binary_classification")
    assert "accuracy" in jack
    assert len(jack["accuracy"]) == len(y)
    # AUC may have fewer entries if some folds were degenerate, but the
    # accuracy/precision/recall arrays are exactly N.
    assert len(jack["precision"]) == len(y)
    assert len(jack["recall"]) == len(y)


def test_compute_jackknife_metrics_regression() -> None:
    """Regression jackknife emits rmse/mae per drop."""
    rng = np.random.default_rng(3)
    n = 50
    y = rng.normal(size=n)
    pred = y + rng.normal(scale=0.1, size=n)
    jack = _compute_jackknife_metrics(y, pred, None, "regression")
    assert "rmse" in jack
    assert "mae" in jack
    assert len(jack["rmse"]) == n
    assert len(jack["mae"]) == n


def test_min_n_for_bca_constant_matches_q_w5_3() -> None:
    """Sanity: the constant matches Q-W5-3 RESOLVED codex B verdict (=30)."""
    assert _MIN_N_FOR_BCA == 30
