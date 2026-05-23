"""Tests for the post-training learning-curve diagnostic node.

PR #463 Phase 2 of the data-sufficiency rollout — the ``learning_curve`` node
is invoked after ``evaluate_model``, conditional on
``state["success_criteria_met"] is False``. It fits a cheap proxy model
(LightGBM at default HPs) on cumulative training-data buckets, scores each on
the validation set, fits a power-law curve, performs a slope-significance
test on the last 3 points, and (if rising significantly) extrapolates the
sample count required to hit ``scope_spec.success_criteria.min_auc``.

The causal-inference branch tracks ATE CI width vs n instead of predictive
score, fitting ``ci_width(n) = k / sqrt(n)`` and solving for the n at which
CI width hits ``scope_spec.sufficiency.target_mde``.

A 180-second walltime cap returns ``verdict="INCONCLUSIVE"`` with a partial
curve when the proxy fit exceeds the budget.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures — kept tiny so the suite runs fast even with k buckets * fit.
# Deterministic via numpy.random.default_rng(42).
# ---------------------------------------------------------------------------

_N_TRAIN = 140
_N_VAL = 60
_N_FEATURES = 4
_SEED = 42


def _make_binary_state(success_met: bool, *, min_auc: float = 0.80) -> dict[str, Any]:
    """Build a minimal ModelTrainerState-shaped dict for binary classification."""
    rng = np.random.default_rng(_SEED)
    X_train = pd.DataFrame(
        rng.normal(size=(_N_TRAIN, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    # Signal-bearing target so larger buckets actually fit a better model.
    logits = X_train.iloc[:, 0] * 1.2 + X_train.iloc[:, 1] * 0.9 - 0.3
    p = 1.0 / (1.0 + np.exp(-logits))
    y_train = pd.Series((rng.uniform(size=_N_TRAIN) < p).astype(int), name="y")

    X_val = pd.DataFrame(
        rng.normal(size=(_N_VAL, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    val_logits = X_val.iloc[:, 0] * 1.2 + X_val.iloc[:, 1] * 0.9 - 0.3
    val_p = 1.0 / (1.0 + np.exp(-val_logits))
    y_val = pd.Series((rng.uniform(size=_N_VAL) < val_p).astype(int), name="y")

    return {
        "success_criteria_met": success_met,
        "problem_type": "binary_classification",
        "train_data": {"X": X_train, "y": y_train, "row_count": len(X_train)},
        "validation_data": {"X": X_val, "y": y_val, "row_count": len(X_val)},
        "scope_spec": {
            "problem_type": "binary_classification",
            "success_criteria": {"min_auc": min_auc},
        },
        "success_criteria": {"minimum_auc": min_auc},
    }


def _make_causal_state(success_met: bool) -> dict[str, Any]:
    """Build a minimal state-shaped dict for the causal-inference branch."""
    rng = np.random.default_rng(_SEED)
    X_train = pd.DataFrame(
        rng.normal(size=(_N_TRAIN, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    # Treatment + continuous outcome with a constant ATE so bootstrap CI
    # width is a function of bucket size only (no confounding noise).
    treatment = rng.integers(0, 2, size=_N_TRAIN)
    outcome = 0.5 * treatment + rng.normal(scale=1.0, size=_N_TRAIN)
    X_train["treatment"] = treatment
    y_train = pd.Series(outcome, name="y")

    X_val = pd.DataFrame(
        rng.normal(size=(_N_VAL, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    X_val["treatment"] = rng.integers(0, 2, size=_N_VAL)
    y_val = pd.Series(rng.normal(size=_N_VAL), name="y")

    return {
        "success_criteria_met": success_met,
        "problem_type": "causal_inference",
        "train_data": {"X": X_train, "y": y_train, "row_count": len(X_train)},
        "validation_data": {"X": X_val, "y": y_val, "row_count": len(X_val)},
        "scope_spec": {
            "problem_type": "causal_inference",
            "sufficiency": {"target_mde": 0.3},
        },
    }


# ---------------------------------------------------------------------------
# Tests — TDD red-first. Importing the not-yet-existing node module fails
# at collection, which is the expected RED state before implementation.
# ---------------------------------------------------------------------------


async def test_short_circuits_when_success_criteria_met() -> None:
    """When evaluator passed, the diagnostic must be a no-op."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=True)
    result = await learning_curve(state)
    assert result == {}, "node must return empty dict when criteria pass"


async def test_runs_when_success_criteria_not_met() -> None:
    """When evaluator failed, the diagnostic must produce a sufficiency_report."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    result = await learning_curve(state)
    assert "sufficiency_report" in result
    report = result["sufficiency_report"]
    assert report["problem_type"] == "binary_classification"
    assert report["n_rows"] == _N_TRAIN
    assert report["learning_curve"] is not None


async def test_curve_has_k_points() -> None:
    """Default k=7 buckets produces a 7-point curve."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    result = await learning_curve(state)
    curve = result["sufficiency_report"]["learning_curve"]
    assert len(curve) == 7
    # Each entry is (n, score_mean, score_std).
    for entry in curve:
        assert len(entry) == 3
        n_i, score_i, std_i = entry
        assert isinstance(n_i, int) and n_i > 0
        assert isinstance(score_i, float)
        assert isinstance(std_i, float)


async def test_proxy_model_recorded() -> None:
    """The proxy model identifier must be lightgbm-default."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    result = await learning_curve(state)
    assert result["sufficiency_report"]["proxy_model"] == "lightgbm-default"


async def test_power_law_fit_quality() -> None:
    """A rising synthetic curve must yield fit_quality_r2 > 0.8."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _fit_power_law,
    )

    # Monotonically rising, concave curve — the canonical power-law shape.
    ns = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)
    scores = 0.90 - 0.50 * np.power(ns, -0.6)

    fit = _fit_power_law(ns, scores)
    assert fit is not None
    assert fit["r2"] > 0.8
    assert {"a", "b", "c", "r2"}.issubset(fit.keys())


async def test_rising_curve_emits_recommendation() -> None:
    """When the slope at max n is significantly positive, emit a recommendation."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _recommend_additional_samples,
    )

    ns = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)
    # Rising curve far below the target — extrapolation should be > current max.
    scores = 0.55 + 0.002 * ns

    recommendation = _recommend_additional_samples(
        ns=ns,
        scores=scores,
        target_score=0.80,
        slope_pvalue=0.001,
        fit_r2=0.99,
    )
    assert recommendation is not None
    assert recommendation > 0


async def test_saturated_curve_no_recommendation() -> None:
    """A flat (saturated) curve must NOT emit a sample-count recommendation."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _recommend_additional_samples,
    )

    ns = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)
    # Flat near target — slope ~0 ⇒ p-value high ⇒ no recommendation.
    scores = np.full_like(ns, 0.78)

    recommendation = _recommend_additional_samples(
        ns=ns,
        scores=scores,
        target_score=0.80,
        slope_pvalue=0.90,
        fit_r2=0.10,
    )
    assert recommendation is None


async def test_walltime_cap_returns_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """If proxy fit blows the 180s cap, return verdict=INCONCLUSIVE with partial curve."""
    # Import the MODULE explicitly via importlib — ``nodes/__init__.py``
    # re-exports the ``learning_curve`` function with the same name as the
    # submodule, so the regular ``import ... as mod`` form would bind ``mod``
    # to the function. importlib bypasses the re-export by going straight to
    # the submodule.
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    # Patch the cap to a tiny value so the test runs fast — semantic check is
    # the same: cap exceeded → INCONCLUSIVE + partial curve + runtime < cap.
    monkeypatch.setattr(mod, "_WALLTIME_CAP_S", 0.05)

    # Patch the proxy-fit helper so each bucket takes longer than the cap.
    def _slow_fit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.10)
        return {"score_mean": 0.70, "score_std": 0.0}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _slow_fit)

    state = _make_binary_state(success_met=False)
    t0 = time.monotonic()
    result = await mod.learning_curve(state)
    elapsed = time.monotonic() - t0

    report = result["sufficiency_report"]
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["learning_curve"] is not None
    assert len(report["learning_curve"]) >= 0  # partial curve allowed
    # Guard against runaway: must respect the patched cap (with small overshoot).
    assert elapsed < 2.0
    # Also assert runtime < 180s (the contractual cap) — the patched run is
    # well below this.
    assert report["diagnostic_runtime_s"] < 180.0


async def test_causal_branch_uses_ci_width() -> None:
    """Causal branch populates ate_ci_width_curve; predictive curve is None."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_causal_state(success_met=False)
    result = await learning_curve(state)
    report = result["sufficiency_report"]
    assert report["problem_type"] == "causal_inference"
    assert report["ate_ci_width_curve"] is not None
    assert len(report["ate_ci_width_curve"]) > 0
    # Predictive curve must be unset on the causal branch.
    assert report["learning_curve"] is None
    # Each entry is (n, ci_width).
    for entry in report["ate_ci_width_curve"]:
        assert len(entry) == 2


async def test_missing_train_df_returns_empty() -> None:
    """Defensive: no train_data → no diagnostic, no crash."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state: dict[str, Any] = {
        "success_criteria_met": False,
        "problem_type": "binary_classification",
        "scope_spec": {"problem_type": "binary_classification"},
    }
    result = await learning_curve(state)
    assert result == {}


async def test_audit_runtime_recorded() -> None:
    """diagnostic_runtime_s must be populated and strictly positive."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    result = await learning_curve(state)
    runtime = result["sufficiency_report"]["diagnostic_runtime_s"]
    assert runtime is not None
    assert runtime > 0.0
