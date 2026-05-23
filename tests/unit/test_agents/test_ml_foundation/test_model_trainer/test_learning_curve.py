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
        _fit_power_law,
        _recommend_additional_samples,
    )

    ns = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)
    # Rising curve far below the target — extrapolation should be > current max.
    scores = 0.55 + 0.002 * ns

    # F13: the recommender takes a pre-fitted dict; the caller fits.
    fit = _fit_power_law(ns, scores)
    assert fit is not None
    # Force a generous R² for the gate (the linear scores will fit
    # acceptably under a power-law with steep concavity; the test cares
    # about the recommendation path, not the precise R²).
    fit["r2"] = 0.99

    recommendation = _recommend_additional_samples(
        fit=fit,
        n_current=int(ns.max()),
        target_score=0.80,
        slope_pvalue=0.001,
        slope_sign=1.0,
    )
    assert recommendation is not None
    assert recommendation > 0


async def test_saturated_curve_no_recommendation() -> None:
    """A flat (saturated) curve must NOT emit a sample-count recommendation."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _fit_power_law,
        _recommend_additional_samples,
    )

    ns = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)
    # Flat near target — slope ~0 ⇒ p-value high ⇒ no recommendation.
    scores = np.full_like(ns, 0.78)

    fit = _fit_power_law(ns, scores) or {"a": 0.78, "b": 0.0, "c": 0.5, "r2": 0.10}

    recommendation = _recommend_additional_samples(
        fit=fit,
        n_current=int(ns.max()),
        target_score=0.80,
        slope_pvalue=0.90,
        slope_sign=0.0,
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


# ---------------------------------------------------------------------------
# Additional tests for the 15 codex findings (F1..F15)
# ---------------------------------------------------------------------------


async def test_f4_curve_fit_with_near_one_scores() -> None:
    """F4: ``p0[0]`` must clamp to a valid bound when scores cluster near 1.0.

    Pre-fix: ``p0[0] = max(0.97, 0.5) + 0.05 = 1.02`` is outside
    ``bounds=[0,1]`` and ``curve_fit`` raises immediately.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _fit_power_law,
    )

    ns = np.array([20, 40, 60, 80, 100, 120, 140], dtype=float)
    # Max score = 0.97 — without the clamp, p0[0] = 1.02 > bound 1.0.
    scores = 0.97 - 0.10 * np.power(ns, -0.6)
    fit = _fit_power_law(ns, scores)
    assert fit is not None
    assert 0.0 <= fit["a"] <= 1.0
    assert fit["r2"] > 0.5  # generous gate — the point is it FITS at all


async def test_f5_unbounded_n_star_capped() -> None:
    """F5: _invert_power_law_for_n must return None when n_star explodes."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _MAX_RECOMMENDED_N,
        _invert_power_law_for_n,
    )

    # Pathological: c tiny + small (a - target) ⇒ huge n_star.
    # n = ((a - target) / b) ** (-1/c)
    # = (0.01 / 1.0) ** (-1/0.01) = 100 ** 100 = 10^200
    fit = {"a": 0.81, "b": 1.0, "c": 0.01, "r2": 0.99}
    result = _invert_power_law_for_n(fit, target_score=0.80)
    assert result is None
    # Sanity: a fit with reasonable c does return a number under the cap.
    fit_ok = {"a": 0.90, "b": 0.5, "c": 0.5, "r2": 0.99}
    result_ok = _invert_power_law_for_n(fit_ok, target_score=0.80)
    assert result_ok is not None
    assert result_ok <= _MAX_RECOMMENDED_N


async def test_f6_decreasing_curve_no_recommendation() -> None:
    """F6: a falling curve with significant p-value yields no recommendation.

    Two-sided p-value alone cannot distinguish "still rising" from
    "trending downward". The recommender must inspect slope sign.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _recommend_additional_samples,
    )

    fit = {"a": 0.75, "b": 0.5, "c": 0.5, "r2": 0.95}
    rec = _recommend_additional_samples(
        fit=fit,
        n_current=140,
        target_score=0.80,
        slope_pvalue=0.001,  # significant
        slope_sign=-0.05,  # but negative
    )
    assert rec is None, "decreasing curve must NOT emit a sample-count recommendation"


async def test_f6_decreasing_curve_emits_hard_fail() -> None:
    """F6: predictive verdict for a falling curve must be HARD_FAIL."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _verdict_predictive,
    )

    verdict, rationale = _verdict_predictive(
        recommended=None,
        slope_pvalue=0.001,
        slope_sign=-0.05,
        fit={"a": 0.75, "b": 0.5, "c": 0.5, "r2": 0.95},
        fit_r2=0.95,
        target_score=0.80,
    )
    assert verdict == "HARD_FAIL"
    assert "downward" in rationale.lower() or "trending" in rationale.lower()


async def test_f8_causal_binary_outcome_detected() -> None:
    """F8: causal branch infers binary outcome from 2-unique y_train.

    The detected outcome_type must be surfaced in the report so downstream
    consumers know which estimand the diagnostic used.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    rng = np.random.default_rng(_SEED)
    X_train = pd.DataFrame(
        rng.normal(size=(_N_TRAIN, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    treatment = rng.integers(0, 2, size=_N_TRAIN)
    # Binary outcome (e.g. survival flag).
    outcome = rng.integers(0, 2, size=_N_TRAIN)
    X_train["treatment"] = treatment
    y_train = pd.Series(outcome, name="y")

    state: dict[str, Any] = {
        "success_criteria_met": False,
        "problem_type": "causal_inference",
        "train_data": {"X": X_train, "y": y_train, "row_count": len(X_train)},
        # No validation_data — F10 covers this.
        "scope_spec": {
            "problem_type": "causal_inference",
            "sufficiency": {"target_mde": 0.3},
        },
    }
    result = await learning_curve(state)
    report = result["sufficiency_report"]
    assert report["outcome_type"] == "binary"


async def test_f9_treatment_column_t_not_auto_resolved() -> None:
    """F9: bare 't' is too ambiguous; must NOT auto-resolve as treatment."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _resolve_treatment_column,
    )

    X = pd.DataFrame({"t": [0, 1, 0, 1], "x1": [1.0, 2.0, 3.0, 4.0]})
    # Scope spec does NOT declare a treatment column.
    state: dict[str, Any] = {"scope_spec": {"problem_type": "causal_inference"}}
    col = _resolve_treatment_column(state, X)
    assert col is None, "'t' must not be auto-picked as the treatment column"


async def test_f9_treatment_column_explicit_t_works() -> None:
    """F9: explicit scope_spec.treatment_column='t' is honored."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _resolve_treatment_column,
    )

    X = pd.DataFrame({"t": [0, 1, 0, 1], "x1": [1.0, 2.0, 3.0, 4.0]})
    state: dict[str, Any] = {
        "scope_spec": {"problem_type": "causal_inference", "treatment_column": "t"}
    }
    assert _resolve_treatment_column(state, X) == "t"


async def test_f9_normalize_treatment_rejects_three_level() -> None:
    """F9: a 3-level treatment must be rejected (returns INCONCLUSIVE)."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _normalize_treatment_series,
    )

    s = pd.Series([0, 1, 2, 0, 1, 2])
    assert _normalize_treatment_series(s) is None


async def test_f9_normalize_treatment_handles_bool() -> None:
    """F9: a boolean treatment is normalized to {0, 1}."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _normalize_treatment_series,
    )

    s = pd.Series([True, False, True, False])
    out = _normalize_treatment_series(s)
    assert out is not None
    assert set(out.unique().tolist()) == {0, 1}


async def test_f11_asymptote_below_target_emits_hard_fail() -> None:
    """F11: trustworthy fit with asymptote < target is HARD_FAIL, not INCONCLUSIVE."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _verdict_predictive,
    )

    # Fit succeeded, R² > 0.8, but a < target_score.
    verdict, rationale = _verdict_predictive(
        recommended=None,
        slope_pvalue=0.20,  # not significant
        slope_sign=0.001,  # near zero
        fit={"a": 0.70, "b": 0.30, "c": 0.5, "r2": 0.95},
        fit_r2=0.95,
        target_score=0.85,
    )
    assert verdict == "HARD_FAIL"
    assert "asymptote" in rationale.lower()
    assert "0.700" in rationale or "0.70" in rationale


async def test_f11_no_target_emits_distinct_inconclusive() -> None:
    """F11: no target_score → INCONCLUSIVE with a target-specific rationale."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _verdict_predictive,
    )

    verdict, rationale = _verdict_predictive(
        recommended=None,
        slope_pvalue=0.001,  # significant rise (but no target to compare)
        slope_sign=0.01,
        fit={"a": 0.90, "b": 0.30, "c": 0.5, "r2": 0.95},
        fit_r2=0.95,
        target_score=None,
    )
    assert verdict == "INCONCLUSIVE"
    assert "target" in rationale.lower()


async def test_f12_pydantic_success_criteria_extracted() -> None:
    """F12: ``_extract_target_score`` must handle ``SuccessCriteriaSchema`` instances.

    Pre-fix the isinstance(legacy, dict) gate dropped pydantic instances on
    the floor — the function silently returned None even though the schema
    carried a usable ``minimum_auc``.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _extract_target_score,
    )
    from src.agents.ml_foundation.scope_definer.schemas import (
        SuccessCriteriaSchema,
    )

    sc = SuccessCriteriaSchema(minimum_auc=0.85)
    state: dict[str, Any] = {"success_criteria": sc}
    assert _extract_target_score(state) == 0.85


async def test_f12_bool_value_not_accepted_as_target() -> None:
    """F12: a boolean in min_auc must NOT be coerced to 1.0."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _extract_target_score,
    )

    state: dict[str, Any] = {
        "scope_spec": {"success_criteria": {"min_auc": True, "minimum_auc": 0.75}}
    }
    # The True value must be rejected; the legitimate float must be returned.
    assert _extract_target_score(state) == 0.75


async def test_f15_bucket_sizes_refuses_to_pad() -> None:
    """F15: ``_bucket_sizes`` must NOT pad to k with duplicates of n_total.

    Pre-fix the loop appended ``n_total`` until len(sizes)==k. Post-fix the
    function returns however many unique buckets it found (capped at k),
    and the caller decides whether to short-circuit on <3 unique.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _bucket_sizes,
    )

    # n=4: linspace(2,4,7) rounds to {2,3,4} = 3 unique. Pre-fix would
    # pad to 7 entries (3 originals + 4 duplicates of 4). Post-fix
    # returns exactly the 3 unique sizes.
    sizes = _bucket_sizes(4, k=7)
    assert len(sizes) == len(set(sizes)), "buckets must be unique (no duplicates)"
    assert sizes == sorted(sizes)


async def test_f15_lt_three_buckets_emits_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F15: when _bucket_sizes yields < 3 unique sizes, verdict is INCONCLUSIVE.

    Realistically the bucket-size logic with n >= 4 always produces >= 3
    unique buckets, so we monkeypatch ``_bucket_sizes`` to simulate the
    pathological case (e.g. when a future caller passes k=1 or asks for a
    degenerate range).
    """
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")
    # Return a single-bucket list — fewer than the minimum 3 unique sizes.
    monkeypatch.setattr(mod, "_bucket_sizes", lambda *_a, **_k: [50])

    state = _make_binary_state(success_met=False)
    result = await mod.learning_curve(state)
    report = result["sufficiency_report"]
    assert report["verdict"] == "INCONCLUSIVE"
    rationale = report["verdict_rationale"].lower()
    # F15 rationale must mention the data-range / bucket issue.
    assert "insufficient" in rationale or "data range" in rationale or "buckets" in rationale


async def test_f1_failed_fits_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """F1: when a bucket's proxy fit fails, the bucket is omitted, not crashed."""
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    # Fail every other bucket, succeed on the rest.
    call_count: dict[str, int] = {"i": 0}

    def _flaky_fit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        call_count["i"] += 1
        if call_count["i"] % 2 == 0:
            return {"score_mean": float("nan"), "score_std": float("nan")}
        return {"score_mean": 0.70 + 0.005 * call_count["i"], "score_std": 0.0}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _flaky_fit)

    state = _make_binary_state(success_met=False)
    result = await mod.learning_curve(state)
    report = result["sufficiency_report"]
    # Some buckets succeeded ⇒ curve is non-empty but shorter than 7.
    assert report["learning_curve"] is not None
    assert 1 <= len(report["learning_curve"]) < 7


async def test_f1_all_fits_failed_emits_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """F1: when every bucket fails, the verdict is INCONCLUSIVE with a fit-error rationale."""
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    def _always_fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"score_mean": float("nan"), "score_std": float("nan")}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _always_fail)

    state = _make_binary_state(success_met=False)
    result = await mod.learning_curve(state)
    report = result["sufficiency_report"]
    assert report["verdict"] == "INCONCLUSIVE"
    rationale = report["verdict_rationale"].lower()
    assert "fit" in rationale or "proxy" in rationale or "fail" in rationale


async def test_f2_error_state_short_circuits() -> None:
    """F2: state['error'] short-circuits the diagnostic entirely."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    state["error"] = "upstream node failed"
    result = await learning_curve(state)
    assert result == {}


async def test_f2_graph_routes_around_learning_curve_on_error() -> None:
    """F2: the graph-level conditional edge skips learning_curve when error is set."""
    from src.agents.ml_foundation.model_trainer.graph import (
        _should_run_learning_curve,
    )

    # No error + criteria not met ⇒ run the diagnostic.
    assert _should_run_learning_curve({"success_criteria_met": False}) == "learning_curve"
    # Error set ⇒ skip to post_evaluation.
    assert (
        _should_run_learning_curve({"success_criteria_met": False, "error": "boom"})
        == "post_evaluation"
    )
    # Criteria met ⇒ skip to post_evaluation.
    assert _should_run_learning_curve({"success_criteria_met": True}) == "post_evaluation"
    # Criteria met + always_run override ⇒ run the diagnostic.
    assert (
        _should_run_learning_curve(
            {"success_criteria_met": True, "always_run_learning_curve": True}
        )
        == "learning_curve"
    )


async def test_f3_async_wait_for_can_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """F3: ``asyncio.wait_for`` cancels a slow learning_curve coroutine.

    The synchronous fit runs in ``asyncio.to_thread`` so the event loop
    stays responsive. ``wait_for`` cancels the awaiting coroutine; the
    underlying thread may keep running but its result is discarded.

    The test uses a single short sleep per bucket and asserts that
    ``wait_for(timeout=0.2)`` raises within ~1s — well below the 30s
    pytest-timeout. The slow thread terminates naturally before the test
    finishes (each sleep is bounded at 0.3s), so worker cleanup is
    deterministic.
    """
    import asyncio
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    # 0.3s per bucket — long enough that wait_for(0.2) cancels before the
    # first bucket completes, short enough that the thread terminates
    # well within the 30s pytest timeout.
    def _slow_fit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.3)
        return {"score_mean": 0.70, "score_std": 0.0}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _slow_fit)
    # Bound the total walltime cap to a small value as a second-line guard
    # in case wait_for doesn't behave as expected — the diagnostic returns
    # INCONCLUSIVE rather than running for 180s.
    monkeypatch.setattr(mod, "_WALLTIME_CAP_S", 0.5)

    state = _make_binary_state(success_met=False)
    t0 = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(mod.learning_curve(state), timeout=0.2)
    elapsed_to_cancel = time.monotonic() - t0
    # The cancellation must fire well before the 0.3 × 7 = 2.1s the full
    # bucket loop would take (proving wait_for is interrupting, not just
    # observing natural completion).
    assert elapsed_to_cancel < 1.5


async def test_f14_extrapolation_gated_on_fit_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    """F14: ``extrapolated_n_for_target`` is None when fit_r2 ≤ gate."""
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    real_fit = mod._fit_power_law

    def _bad_fit_r2(*args: Any, **kwargs: Any) -> dict[str, float] | None:
        result = real_fit(*args, **kwargs)
        if result is None:
            return None
        # Force R² below the gate so the extrapolation MUST be suppressed.
        result["r2"] = 0.10
        return result

    monkeypatch.setattr(mod, "_fit_power_law", _bad_fit_r2)

    state = _make_binary_state(success_met=False)
    result = await mod.learning_curve(state)
    report = result["sufficiency_report"]
    # Gated fields must be unset (or marked untrustworthy).
    assert report["extrapolated_n_for_target"] is None
    assert report["extrapolated_n_ci"] is None
    assert report["fit_trustworthy"] is False


# ---------------------------------------------------------------------------
# Round-2 codex follow-ups (R2.1..R2.7)
# ---------------------------------------------------------------------------


async def test_r2_1_string_dtype_y_does_not_crash() -> None:
    """R2.1: object/string-dtype y must NOT trigger 'binary' path.

    Pre-fix: 2-unique string y returned 'binary', then ``y.to_numpy()[mask].mean()``
    raised ``TypeError`` on strings. Post-fix: non-numeric dtype falls back
    to 'continuous' at the type-detection level; the causal branch ALSO
    gates on numeric y_train at entry and returns INCONCLUSIVE with a
    rationale that points the user at the right fix (encode the outcome).
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _detect_outcome_type,
        learning_curve,
    )

    # _detect_outcome_type must return 'continuous' for string dtype.
    y_str = pd.Series(["a", "b", "a", "b", "a"], dtype=object)
    assert _detect_outcome_type(y_str) == "continuous"

    # End-to-end: causal branch with object-dtype y_train must not raise
    # and must route to INCONCLUSIVE with a numeric-outcome rationale.
    rng = np.random.default_rng(_SEED)
    X_train = pd.DataFrame(
        rng.normal(size=(_N_TRAIN, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    X_train["treatment"] = rng.integers(0, 2, size=_N_TRAIN)
    y_train = pd.Series(
        np.where(rng.uniform(size=_N_TRAIN) < 0.5, "yes", "no"),
        name="y",
        dtype=object,
    )

    state: dict[str, Any] = {
        "success_criteria_met": False,
        "problem_type": "causal_inference",
        "train_data": {"X": X_train, "y": y_train, "row_count": len(X_train)},
        "scope_spec": {
            "problem_type": "causal_inference",
            "sufficiency": {"target_mde": 0.3},
        },
    }
    result = await learning_curve(state)
    report = result["sufficiency_report"]
    # No crash; clear INCONCLUSIVE rationale.
    assert report["verdict"] == "INCONCLUSIVE"
    rationale = report["verdict_rationale"].lower()
    assert "numeric" in rationale or "outcome" in rationale


async def test_r2_2_validation_data_with_none_values_routes_to_inconclusive() -> None:
    """R2.2: validation_data={'X': None, 'y': None} must not crash.

    Pre-fix: ``have_val`` was True (keys present), but ``_coerce_X(None)``
    raised ``ValueError: Must pass 2-d input``. Post-fix: have_val also
    checks values are non-None / non-empty, predictive branch returns
    INCONCLUSIVE.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    # Replace validation_data with None-valued sentinel.
    state["validation_data"] = {"X": None, "y": None}
    result = await learning_curve(state)
    report = result["sufficiency_report"]
    assert report["verdict"] == "INCONCLUSIVE"
    assert "validation_data" in report["verdict_rationale"].lower()


async def test_r2_2_validation_data_with_empty_arrays_routes_to_inconclusive() -> None:
    """R2.2: empty validation arrays are also unusable, not a crash."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state = _make_binary_state(success_met=False)
    state["validation_data"] = {
        "X": pd.DataFrame(columns=[f"f{i}" for i in range(_N_FEATURES)]),
        "y": pd.Series([], dtype=float),
    }
    result = await learning_curve(state)
    report = result["sufficiency_report"]
    assert report["verdict"] == "INCONCLUSIVE"


async def test_r2_3_treatment_with_nan_does_not_crash() -> None:
    """R2.3: NaN treatment values are dropped, not coerced via Int64→int.

    Pre-fix: factorize fallback preserved NaN through ``.map(...).astype('Int64').astype(int)``
    raising ``ValueError: cannot convert NA to integer``. Post-fix: NaN
    rows are dropped before normalization; <2 surviving rows returns None.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _normalize_treatment_series,
    )

    # Mixed treatment with NaN — factorize fallback would have crashed.
    s = pd.Series(["a", np.nan, "b", "a", np.nan, "b"], dtype=object)
    out = _normalize_treatment_series(s)
    assert out is not None
    assert set(out.unique().tolist()) == {0, 1}
    # Only 4 non-NaN rows survive.
    assert len(out) == 4

    # All-NaN: returns None.
    s_all_nan = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    assert _normalize_treatment_series(s_all_nan) is None

    # Single non-NaN row: returns None (need ≥ 2 distinct after dropna).
    s_one_real = pd.Series([np.nan, 1.0, np.nan], dtype=float)
    assert _normalize_treatment_series(s_one_real) is None


async def test_r2_3_causal_branch_drops_nan_treatment_rows() -> None:
    """R2.3 end-to-end: a few NaN treatment rows are silently excluded."""
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    rng = np.random.default_rng(_SEED)
    X_train = pd.DataFrame(
        rng.normal(size=(_N_TRAIN, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    treatment = rng.integers(0, 2, size=_N_TRAIN).astype(float)
    # Sprinkle 5 NaN treatment values.
    treatment[::28] = np.nan
    X_train["treatment"] = treatment
    y_train = pd.Series(rng.normal(size=_N_TRAIN), name="y")

    state: dict[str, Any] = {
        "success_criteria_met": False,
        "problem_type": "causal_inference",
        "train_data": {"X": X_train, "y": y_train, "row_count": len(X_train)},
        "scope_spec": {
            "problem_type": "causal_inference",
            "sufficiency": {"target_mde": 0.3},
        },
    }
    result = await learning_curve(state)
    report = result["sufficiency_report"]
    # No crash; n_rows reflects rows after NaN-treatment exclusion.
    assert report["n_rows"] < _N_TRAIN
    assert report["n_rows"] >= _N_TRAIN - 6


async def test_r2_4_partial_failure_emits_inconclusive_not_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2.4: 5/7 buckets fail → curve has 2 entries → must NOT report 'saturated'.

    Pre-fix: with curve of 2 entries, _fit_power_law returned None,
    _slope_pvalue_last_k returned 1.0 (len<k), all-failures branch
    skipped (curve non-empty), fell through to 'saturated' HARD_FAIL
    with a wrong-cause rationale. Post-fix: emit INCONCLUSIVE with a
    bucket-failure rationale.
    """
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    # Succeed on the first 2 buckets, fail on the remaining 5.
    call_count: dict[str, int] = {"i": 0}

    def _two_pass_five_fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        call_count["i"] += 1
        if call_count["i"] <= 2:
            return {"score_mean": 0.70 + 0.005 * call_count["i"], "score_std": 0.0}
        return {"score_mean": float("nan"), "score_std": float("nan")}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _two_pass_five_fail)

    state = _make_binary_state(success_met=False)
    result = await mod.learning_curve(state)
    report = result["sufficiency_report"]
    assert report["verdict"] == "INCONCLUSIVE"
    rationale = report["verdict_rationale"].lower()
    # Rationale must reference the bucket-fit failures, NOT "saturated".
    assert "saturated" not in rationale
    assert "fail" in rationale or "insufficient" in rationale
    # Curve must reflect the surviving (successful) buckets.
    assert report["learning_curve"] is not None
    assert len(report["learning_curve"]) == 2
    # Fit-trustworthy is False since no fit was attempted.
    assert report["fit_trustworthy"] is False


async def test_r2_5_schema_accepts_new_fields() -> None:
    """R2.5: DataSufficiencyReport(extra='forbid') must accept fit_trustworthy / outcome_type."""
    from src.utils.sufficiency_schemas import DataSufficiencyReport

    # All Phase-2 fields populated — the schema must validate without raising.
    payload = {
        "verdict": "SOFT_FAIL",
        "verdict_rationale": "rising curve; ~50 more samples close gap.",
        "n_rows": 140,
        "n_features": 4,
        "problem_type": "binary_classification",
        "learning_curve": [(20, 0.70, 0.0), (60, 0.75, 0.0)],
        "proxy_model": "lightgbm-default",
        "fit_quality_r2": 0.95,
        "fit_trustworthy": True,
        "outcome_type": "binary",
    }
    rep = DataSufficiencyReport(**payload)
    assert rep.fit_trustworthy is True
    assert rep.outcome_type == "binary"


async def test_r2_6_low_variance_continuous_classified_as_continuous() -> None:
    """R2.6: y = {0.0, 0.5} must be 'continuous', not 'binary'.

    Pre-fix: 2 uniques → 'binary' regardless of values. Post-fix: only
    {0, 1} returns 'binary'; {0.0, 0.5} (or any other 2-value set) is
    treated as continuous.
    """
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        _detect_outcome_type,
    )

    y_zero_half = pd.Series([0.0, 0.5, 0.0, 0.5, 0.5, 0.0])
    assert _detect_outcome_type(y_zero_half) == "continuous"

    # Sanity: legitimate Bernoulli {0,1} is still binary.
    y_bernoulli = pd.Series([0, 1, 0, 1, 1, 0])
    assert _detect_outcome_type(y_bernoulli) == "binary"

    # Float-typed {0.0, 1.0} also recognized as binary.
    y_bernoulli_float = pd.Series([0.0, 1.0, 0.0, 1.0])
    assert _detect_outcome_type(y_bernoulli_float) == "binary"

    # Non-{0,1} 2-value pair (e.g. -1/1) is continuous.
    y_signed = pd.Series([-1.0, 1.0, -1.0, 1.0])
    assert _detect_outcome_type(y_signed) == "continuous"

    # Multi-class continuous: still continuous.
    y_multi = pd.Series([0.0, 0.5, 1.0, 0.2, 0.8])
    assert _detect_outcome_type(y_multi) == "continuous"


async def test_r2_7_fit_trustworthy_set_on_all_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2.7: every report path must include fit_trustworthy explicitly.

    Covers: all-failures branch (line 871-style), walltime cap branch
    (line 888-style), and the _empty_report skeleton used by early
    short-circuits (n<4, missing val_data, treatment-column missing,
    etc.). The schema declares fit_trustworthy as Optional but the
    intent is to surface False on every non-success branch.
    """
    import importlib

    mod = importlib.import_module("src.agents.ml_foundation.model_trainer.nodes.learning_curve")

    # Branch A — all-failures via _fit_proxy_on_bucket returning NaN.
    def _always_fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"score_mean": float("nan"), "score_std": float("nan")}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _always_fail)
    state = _make_binary_state(success_met=False)
    result_a = await mod.learning_curve(state)
    assert result_a["sufficiency_report"]["fit_trustworthy"] is False

    # Branch B — walltime cap hit.
    import time as _time

    monkeypatch.setattr(mod, "_WALLTIME_CAP_S", 0.01)

    def _slow_fit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        _time.sleep(0.05)
        return {"score_mean": 0.70, "score_std": 0.0}

    monkeypatch.setattr(mod, "_fit_proxy_on_bucket", _slow_fit)
    result_b = await mod.learning_curve(state)
    assert result_b["sufficiency_report"]["fit_trustworthy"] is False

    # Branch C — _empty_report early exit (no validation data).
    from src.agents.ml_foundation.model_trainer.nodes.learning_curve import (
        learning_curve,
    )

    state_no_val: dict[str, Any] = {
        "success_criteria_met": False,
        "problem_type": "causal_inference",
        "train_data": {
            "X": pd.DataFrame({"x": list(range(50))}),  # No treatment column.
            "y": pd.Series(range(50)),
            "row_count": 50,
        },
        "scope_spec": {"problem_type": "causal_inference"},
    }
    result_c = await learning_curve(state_no_val)
    # Will route to "no treatment column found" empty_report.
    assert result_c["sufficiency_report"]["fit_trustworthy"] is False
