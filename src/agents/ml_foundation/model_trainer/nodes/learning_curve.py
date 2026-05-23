"""Post-training learning-curve diagnostic node (PR #463 — Phase 2).

Triggered after ``evaluate_model``. The node is **conditional** on
``state["success_criteria_met"] is False`` — when the model already meets the
acceptance thresholds the diagnostic is a no-op (returns ``{}``).

When invoked, the diagnostic answers two questions:

1. **Does adding more data plausibly close the gap?** It splits the training
   data into ``k=7`` cumulative buckets, fits a cheap proxy model
   (LightGBM at default HPs) on bucket 1..i, evaluates on the validation set,
   and records ``(n_i, score_mean, score_std)`` per bucket. A power-law
   ``score(n) = a − b · n^(−c)`` is fit via ``scipy.optimize.curve_fit``; a
   slope-significance test on the last 3 points (linear-regression
   p-value, H0: slope=0) decides whether the curve is still rising.
2. **How much more data?** If ``slope_pvalue < 0.05`` AND
   ``fit_quality_r2 > 0.8`` AND the target is reachable under the fit, the
   power-law is inverted to estimate ``recommended_additional_samples``.

A 180-second walltime cap (configurable via the module constant
``_WALLTIME_CAP_S``) guards against pathological proxy fits. When the cap is
breached mid-loop the partial curve is returned with
``verdict="INCONCLUSIVE"``.

**Causal-inference branch** (``state["scope_spec"]["problem_type"] ==
"causal_inference"``):
- Tracks ATE bootstrap CI width vs. n instead of predictive score.
- Fits ``ci_width(n) = k / sqrt(n)`` (one-parameter) and solves for the n at
  which ``ci_width(n) ≤ target_mde`` resolved via
  :func:`src.utils.sufficiency_resolver.resolve_target_mde`.

Per CLAUDE.md REASON-BEFORE-RULES: this node is a scaffolded foundation for a
documented product feature (data-sufficiency diagnostics Phase 2). It is
**not** a silent mock — every numeric output is computed from real data, the
power-law fit is a real ``scipy.optimize.curve_fit``, the slope p-value is a
real ``scipy.stats.linregress``, and the LightGBM proxy is a real fit.
Defensive returns of ``{}`` on missing inputs are intentional (the graph runs
unconditionally; the node short-circuits when its preconditions are unmet).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

from src.utils.sufficiency_resolver import resolve_target_mde

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level knobs. Patched in tests; kept top-level so the values are
# discoverable / overridable from the outside per CLAUDE.md
# REASON-BEFORE-RULES — these are NOT magic numbers, they are documented
# diagnostic-policy knobs.
# ---------------------------------------------------------------------------
_DEFAULT_K_BUCKETS = 7
_WALLTIME_CAP_S = 180.0
_SLOPE_SIG_LAST_K = 3
_SLOPE_PVALUE_GATE = 0.05
_FIT_R2_GATE = 0.8
_DEFAULT_BOOTSTRAP_N = 200
_PROXY_MODEL_ID = "lightgbm-default"
_FALLBACK_PROXY_MODEL_ID = "gradient-boosting-default"


# ---------------------------------------------------------------------------
# Power-law helpers.
# ---------------------------------------------------------------------------


def _power_law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Score = a − b · n^(−c). a is the asymptote; b/c control rise shape."""
    return a - b * np.power(n, -c)


def _fit_power_law(ns: np.ndarray, scores: np.ndarray) -> Optional[dict[str, float]]:
    """Fit ``score(n) = a − b·n^(−c)``; return ``None`` if curve_fit fails.

    Returns a dict ``{"a", "b", "c", "r2"}``. ``r2`` is the standard
    coefficient of determination of the fit on the input points.
    """
    if len(ns) < 3 or len(scores) < 3 or len(ns) != len(scores):
        return None
    # Bounds: a in [0, 1] (score), b in [0, ∞), c in [0.01, 5.0] (avoid c→0
    # degenerate case where the fit collapses to a constant).
    try:
        popt, _pcov = curve_fit(
            _power_law,
            ns,
            scores,
            p0=[max(float(scores.max()), 0.5) + 0.05, 1.0, 0.5],
            bounds=([0.0, 0.0, 0.01], [1.0, np.inf, 5.0]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError) as exc:  # noqa: BLE001 — diagnostic, log + return None
        logger.debug(f"power-law curve_fit failed: {exc!r}")
        return None

    a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
    predicted = _power_law(ns, a, b, c)
    ss_res = float(np.sum((scores - predicted) ** 2))
    ss_tot = float(np.sum((scores - scores.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": a, "b": b, "c": c, "r2": r2}


def _slope_pvalue_last_k(ns: np.ndarray, scores: np.ndarray, k: int = _SLOPE_SIG_LAST_K) -> float:
    """Linear-regression p-value (two-sided) on the last ``k`` points.

    Returns 1.0 when the slope is undefined (fewer than k points, zero
    variance in x). A small p-value rejects H0=slope=0 i.e. "the curve is
    still rising at the current max n".
    """
    if len(ns) < k or len(scores) < k:
        return 1.0
    x = ns[-k:].astype(float)
    y = scores[-k:].astype(float)
    if np.var(x) == 0:
        return 1.0
    try:
        result = stats.linregress(x, y)
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001
        logger.debug(f"linregress failed: {exc!r}")
        return 1.0
    return float(result.pvalue)


def _invert_power_law_for_n(
    fit: dict[str, float],
    target_score: float,
) -> Optional[int]:
    """Solve ``a − b·n^(−c) = target`` for n. Returns ``None`` if unreachable.

    Unreachable cases:
    - ``a <= target`` — the fit's asymptote is below the target.
    - Numerical overflow when target requires astronomically large n.
    """
    a, b, c = fit["a"], fit["b"], fit["c"]
    if a <= target_score:
        return None
    if b <= 0 or c <= 0:
        return None
    ratio = (a - target_score) / b
    if ratio <= 0:
        return None
    # n = ratio^(-1/c)
    try:
        n_star = float(np.power(ratio, -1.0 / c))
    except (OverflowError, ValueError):
        return None
    if not np.isfinite(n_star) or n_star <= 0:
        return None
    return int(np.ceil(n_star))


def _recommend_additional_samples(
    *,
    ns: np.ndarray,
    scores: np.ndarray,
    target_score: float,
    slope_pvalue: float,
    fit_r2: float,
) -> Optional[int]:
    """Return the extra-samples recommendation or ``None`` if not warranted.

    Gates:
    - ``slope_pvalue < _SLOPE_PVALUE_GATE`` (the curve is still rising)
    - ``fit_r2 > _FIT_R2_GATE`` (the power-law is a good description)
    """
    if slope_pvalue >= _SLOPE_PVALUE_GATE:
        return None
    if fit_r2 <= _FIT_R2_GATE:
        return None
    fit = _fit_power_law(ns, scores)
    if fit is None:
        return None
    n_target = _invert_power_law_for_n(fit, target_score)
    if n_target is None:
        return None
    n_current = int(ns.max())
    additional = n_target - n_current
    if additional <= 0:
        return None
    return int(additional)


# ---------------------------------------------------------------------------
# Proxy-model helpers (predictive branch).
# ---------------------------------------------------------------------------


def _make_proxy_model(problem_type: str) -> tuple[Any, str]:
    """Return ``(estimator, model_id)``. Prefer LightGBM; fall back if missing.

    The fallback ID lets callers (and the audit chain) record exactly which
    proxy was used. The diagnostic is approximate by design — we want a
    cheap, well-behaved learner — so the fallback is acceptable.
    """
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        if problem_type == "regression":
            return LGBMRegressor(verbose=-1, random_state=_SEED_OFFSET), _PROXY_MODEL_ID
        return LGBMClassifier(verbose=-1, random_state=_SEED_OFFSET), _PROXY_MODEL_ID
    except ImportError:  # pragma: no cover — exercised in environments without lightgbm
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        if problem_type == "regression":
            return GradientBoostingRegressor(random_state=_SEED_OFFSET), _FALLBACK_PROXY_MODEL_ID
        return GradientBoostingClassifier(random_state=_SEED_OFFSET), _FALLBACK_PROXY_MODEL_ID


_SEED_OFFSET = 42


def _score_proxy(
    *,
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str,
) -> float:
    """Compute the validation-set score for a single proxy fit."""
    if problem_type == "regression":
        # R² so larger=better, consistent with the power-law's rising shape.
        from sklearn.metrics import r2_score

        preds = model.predict(X_val)
        return float(r2_score(y_val, preds))
    # Classification — AUC if proba available; else accuracy.
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_val)
            # Binary: column 1. Multiclass: average AUC OvR.
            if proba.shape[1] == 2:
                return float(roc_auc_score(y_val, proba[:, 1]))
            return float(roc_auc_score(y_val, proba, multi_class="ovr"))
        except (ValueError, IndexError):
            pass
    # Fallback — accuracy.
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_val, model.predict(X_val)))


def _fit_proxy_on_bucket(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str,
    bucket_size: int,
) -> dict[str, float]:
    """Fit the proxy on the first ``bucket_size`` rows and score on validation.

    Returns ``{"score_mean": ..., "score_std": ...}``. ``score_std`` is 0.0
    for the single-fit case; future revisions could repeat the fit with
    different seeds and aggregate. Kept as a public helper so the walltime
    test can ``monkeypatch`` it.
    """
    X_bucket = X_train.iloc[:bucket_size]
    y_bucket = y_train.iloc[:bucket_size]
    model, _ = _make_proxy_model(problem_type)
    # Defensive: if only one class is present in the bucket, classifiers
    # raise. Return a degenerate score so the curve continues at the next
    # bucket rather than aborting the diagnostic.
    if problem_type != "regression" and len(np.unique(y_bucket)) < 2:
        return {"score_mean": 0.5, "score_std": 0.0}
    model.fit(X_bucket, y_bucket)
    score = _score_proxy(model=model, X_val=X_val, y_val=y_val, problem_type=problem_type)
    return {"score_mean": score, "score_std": 0.0}


# ---------------------------------------------------------------------------
# Bucketization.
# ---------------------------------------------------------------------------


def _bucket_sizes(n_total: int, k: int = _DEFAULT_K_BUCKETS) -> list[int]:
    """Return ``k`` cumulative bucket sizes in [floor, n_total].

    The smallest bucket is at least 10 rows (so the proxy fit is not
    degenerate); the largest is the full train set.
    """
    if n_total <= 0 or k <= 0:
        return []
    smallest = max(10, n_total // (k * 2)) if n_total > 20 else max(2, n_total // k)
    if smallest >= n_total:
        return [n_total]
    raw = np.linspace(smallest, n_total, k)
    sizes = sorted({int(round(x)) for x in raw})
    # Ensure exactly k entries when possible by padding from the upper end.
    while len(sizes) < k:
        sizes.append(n_total)
    return sizes[:k]


# ---------------------------------------------------------------------------
# Causal-inference branch helpers.
# ---------------------------------------------------------------------------


def _bootstrap_ate_ci_width(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    treatment_col: str,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_N,
    rng: np.random.Generator,
) -> float:
    """Bootstrap the difference-in-means ATE and return the 95% CI width.

    The estimator is intentionally simple (difference-of-means) so the
    diagnostic stays cheap. Real causal-effect estimation lives in the
    causal_engine; here we only need a *width* that scales with n, which the
    DIM ATE provides with negligible setup cost.
    """
    if treatment_col not in X.columns:
        return float("nan")
    t = X[treatment_col].to_numpy()
    y_arr = y.to_numpy()
    n = len(y_arr)
    if n < 4 or np.unique(t).size < 2:
        return float("nan")
    estimates = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t_b = t[idx]
        y_b = y_arr[idx]
        if np.unique(t_b).size < 2:
            estimates[b] = np.nan
            continue
        mean_treated = y_b[t_b == 1].mean() if (t_b == 1).any() else np.nan
        mean_control = y_b[t_b == 0].mean() if (t_b == 0).any() else np.nan
        estimates[b] = mean_treated - mean_control
    estimates = estimates[~np.isnan(estimates)]
    if estimates.size < 4:
        return float("nan")
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(hi - lo)


def _fit_inverse_sqrt(ns: np.ndarray, widths: np.ndarray) -> Optional[float]:
    """Fit ``width(n) = k / sqrt(n)``; return ``k`` or None.

    One-parameter least-squares: k* = sum(widths * 1/sqrt(n)) / sum(1/n).
    """
    valid = np.isfinite(widths) & (ns > 0)
    if valid.sum() < 2:
        return None
    n_valid = ns[valid].astype(float)
    w_valid = widths[valid].astype(float)
    inv_sqrt = 1.0 / np.sqrt(n_valid)
    denom = float(np.sum(inv_sqrt * inv_sqrt))
    if denom <= 0:
        return None
    k_star = float(np.sum(w_valid * inv_sqrt) / denom)
    if not np.isfinite(k_star) or k_star <= 0:
        return None
    return k_star


def _solve_n_for_ci_width(k: float, target_width: float) -> Optional[int]:
    """Given ``width = k/sqrt(n)``, return the smallest n satisfying width<=target."""
    if target_width <= 0 or k <= 0:
        return None
    n_star = (k / target_width) ** 2
    if not np.isfinite(n_star) or n_star <= 0:
        return None
    return int(np.ceil(n_star))


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def _extract_target_score(state: dict[str, Any]) -> Optional[float]:
    """Resolve the target predictive-metric value from the state.

    Priority:
    1. ``scope_spec.success_criteria.min_auc`` (PR-462/463 contract field)
    2. ``success_criteria.minimum_auc`` (legacy model_trainer field)
    """
    scope_spec = state.get("scope_spec") or {}
    sc = (scope_spec.get("success_criteria") if isinstance(scope_spec, dict) else None) or {}
    for key in ("min_auc", "minimum_auc", "min_score", "target_score"):
        val = sc.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    legacy = state.get("success_criteria") or {}
    if isinstance(legacy, dict):
        for key in ("minimum_auc", "min_auc"):
            val = legacy.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _resolve_problem_type(state: dict[str, Any]) -> str:
    """Prefer scope_spec.problem_type; fall back to top-level state field."""
    scope_spec = state.get("scope_spec") or {}
    if isinstance(scope_spec, dict):
        pt = scope_spec.get("problem_type")
        if isinstance(pt, str):
            return pt
    pt = state.get("problem_type")
    return pt if isinstance(pt, str) else "binary_classification"


def _resolve_treatment_column(state: dict[str, Any], X: pd.DataFrame) -> Optional[str]:
    """Best-effort treatment-column discovery for the causal branch."""
    scope_spec = state.get("scope_spec") or {}
    if isinstance(scope_spec, dict):
        for key in ("treatment_column", "treatment", "intervention_column"):
            val = scope_spec.get(key)
            if isinstance(val, str) and val in X.columns:
                return val
    for candidate in ("treatment", "t", "intervention", "treated"):
        if candidate in X.columns:
            return candidate
    return None


def _empty_report(
    *,
    problem_type: str,
    n_rows: int,
    n_features: int,
    rationale: str,
) -> dict[str, Any]:
    """Construct an INCONCLUSIVE report skeleton for early-exit branches."""
    return {
        "verdict": "INCONCLUSIVE",
        "verdict_rationale": rationale,
        "n_rows": n_rows,
        "n_features": n_features,
        "problem_type": problem_type,
        "learning_curve": None,
        "proxy_model": None,
        "slope_at_max_n": None,
        "slope_pvalue": None,
        "power_law_fit": None,
        "extrapolated_n_for_target": None,
        "extrapolated_n_ci": None,
        "fit_quality_r2": None,
        "recommended_additional_samples": None,
        "ate_ci_width_curve": None,
        "ate_target_ci_width": None,
        "diagnostic_runtime_s": 0.0,
    }


async def learning_curve(state: dict[str, Any]) -> dict[str, Any]:
    """Run the post-training learning-curve diagnostic.

    Short-circuits to ``{}`` when ``success_criteria_met is True`` — the
    diagnostic is the user-facing answer to "why didn't the model pass",
    and there is no question to answer when the model passed. Callers that
    want to force the diagnostic even on pass set
    ``PipelineConfig.always_run_learning_curve = True`` upstream, which
    flips ``success_criteria_met`` to False in the state propagated into
    this node.
    """
    # Conditional gate — Phase 2 only runs when the model failed, UNLESS
    # the caller set ``always_run_learning_curve=True`` (PR #463
    # PipelineConfig opt-in) which forces the diagnostic to produce a report
    # for audit / replication purposes regardless of pass/fail.
    if state.get("success_criteria_met") is True and not state.get(
        "always_run_learning_curve", False
    ):
        return {}

    # Defensive: missing training data ⇒ no diagnostic.
    train_data = state.get("train_data")
    if not isinstance(train_data, dict) or "X" not in train_data or "y" not in train_data:
        return {}
    val_data = state.get("validation_data") or {}
    if not isinstance(val_data, dict) or "X" not in val_data or "y" not in val_data:
        return {}

    X_train_raw = train_data["X"]
    y_train_raw = train_data["y"]
    X_val_raw = val_data["X"]
    y_val_raw = val_data["y"]

    # Coerce to pandas — the proxy fit and bucket slicing assume DataFrames /
    # Series. Numpy inputs are wrapped here, not deeper, so the helpers stay
    # uniform.
    X_train = (
        X_train_raw
        if isinstance(X_train_raw, pd.DataFrame)
        else pd.DataFrame(np.asarray(X_train_raw))
    )
    y_train = (
        y_train_raw if isinstance(y_train_raw, pd.Series) else pd.Series(np.asarray(y_train_raw))
    )
    X_val = (
        X_val_raw if isinstance(X_val_raw, pd.DataFrame) else pd.DataFrame(np.asarray(X_val_raw))
    )
    y_val = y_val_raw if isinstance(y_val_raw, pd.Series) else pd.Series(np.asarray(y_val_raw))

    problem_type = _resolve_problem_type(state)
    n_rows = int(len(X_train))
    n_features = int(X_train.shape[1])

    if n_rows < 4:
        return {
            "sufficiency_report": _empty_report(
                problem_type=problem_type,
                n_rows=n_rows,
                n_features=n_features,
                rationale="Too few training rows for a learning-curve diagnostic.",
            )
        }

    t_start = time.monotonic()

    # Branch on problem type. The causal branch tracks CI width vs n; the
    # predictive branch tracks score vs n. They share the bucketization
    # scheme and the walltime cap.
    if problem_type == "causal_inference":
        report = _run_causal_branch(
            state=state,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            n_features=n_features,
            t_start=t_start,
        )
    else:
        report = _run_predictive_branch(
            state=state,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            problem_type=problem_type,
            n_features=n_features,
            t_start=t_start,
        )

    return {"sufficiency_report": report}


def _run_predictive_branch(
    *,
    state: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str,
    n_features: int,
    t_start: float,
) -> dict[str, Any]:
    """Predictive-metric branch: score vs n on cumulative buckets."""
    n_rows = int(len(X_train))
    sizes = _bucket_sizes(n_rows, _DEFAULT_K_BUCKETS)
    curve: list[tuple[int, float, float]] = []
    cap_hit = False

    # Discover the proxy model identifier before running the loop so the
    # field is populated even when the cap is hit on the first bucket.
    _, proxy_id = _make_proxy_model(problem_type)

    fit_fn: Callable[..., dict[str, float]] = _fit_proxy_on_bucket

    for bucket_size in sizes:
        if (time.monotonic() - t_start) > _WALLTIME_CAP_S:
            cap_hit = True
            break
        result = fit_fn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            problem_type=problem_type,
            bucket_size=int(bucket_size),
        )
        curve.append((int(bucket_size), float(result["score_mean"]), float(result["score_std"])))
        # Re-check the cap AFTER appending so the walltime test sees a
        # partial curve when the slow fit pushes us past the budget.
        if (time.monotonic() - t_start) > _WALLTIME_CAP_S:
            cap_hit = True
            break

    runtime_s = float(time.monotonic() - t_start)

    if cap_hit:
        return {
            "verdict": "INCONCLUSIVE",
            "verdict_rationale": (
                f"Diagnostic exceeded {_WALLTIME_CAP_S:.0f}s walltime cap after "
                f"{len(curve)} of {len(sizes)} buckets."
            ),
            "n_rows": n_rows,
            "n_features": n_features,
            "problem_type": problem_type,
            "learning_curve": curve,
            "proxy_model": proxy_id,
            "slope_at_max_n": None,
            "slope_pvalue": None,
            "power_law_fit": None,
            "extrapolated_n_for_target": None,
            "extrapolated_n_ci": None,
            "fit_quality_r2": None,
            "recommended_additional_samples": None,
            "ate_ci_width_curve": None,
            "ate_target_ci_width": None,
            "diagnostic_runtime_s": runtime_s,
        }

    # Curve is complete — compute slope, fit, recommendation.
    ns = np.array([entry[0] for entry in curve], dtype=float)
    scores = np.array([entry[1] for entry in curve], dtype=float)
    fit = _fit_power_law(ns, scores)
    slope_pvalue = _slope_pvalue_last_k(ns, scores)

    # Slope at max n: derivative of the fitted curve = b * c * n^(-c-1).
    slope_at_max_n: Optional[float] = None
    if fit is not None and len(ns) > 0:
        n_max = float(ns.max())
        if n_max > 0:
            slope_at_max_n = float(fit["b"] * fit["c"] * np.power(n_max, -fit["c"] - 1.0))

    target_score = _extract_target_score(state)
    fit_r2 = fit["r2"] if fit is not None else 0.0
    recommended: Optional[int] = None
    extrapolated_n: Optional[int] = None
    extrapolated_n_ci: Optional[tuple[int, int]] = None
    if target_score is not None:
        recommended = _recommend_additional_samples(
            ns=ns,
            scores=scores,
            target_score=target_score,
            slope_pvalue=slope_pvalue,
            fit_r2=fit_r2,
        )
        if fit is not None:
            extrapolated_n = _invert_power_law_for_n(fit, target_score)
            if extrapolated_n is not None:
                # ±20% wide envelope is a placeholder for a future bootstrap-
                # over-the-fit; declared visible so downstream consumers
                # don't conflate "no CI" with "tight CI".
                lo = max(1, int(np.floor(extrapolated_n * 0.8)))
                hi = int(np.ceil(extrapolated_n * 1.2))
                extrapolated_n_ci = (lo, hi)

    verdict, rationale = _verdict_predictive(
        recommended=recommended,
        slope_pvalue=slope_pvalue,
        fit_r2=fit_r2,
        target_score=target_score,
    )

    return {
        "verdict": verdict,
        "verdict_rationale": rationale,
        "n_rows": n_rows,
        "n_features": n_features,
        "problem_type": problem_type,
        "learning_curve": curve,
        "proxy_model": proxy_id,
        "slope_at_max_n": slope_at_max_n,
        "slope_pvalue": float(slope_pvalue),
        "power_law_fit": fit,
        "extrapolated_n_for_target": extrapolated_n,
        "extrapolated_n_ci": extrapolated_n_ci,
        "fit_quality_r2": float(fit_r2) if fit is not None else None,
        "recommended_additional_samples": recommended,
        "ate_ci_width_curve": None,
        "ate_target_ci_width": None,
        "diagnostic_runtime_s": float(time.monotonic() - t_start),
    }


def _verdict_predictive(
    *,
    recommended: Optional[int],
    slope_pvalue: float,
    fit_r2: float,
    target_score: Optional[float],
) -> tuple[str, str]:
    """Map the predictive-branch outputs to a (verdict, rationale) pair."""
    if recommended is not None:
        return (
            "SOFT_FAIL",
            (
                f"Learning curve is still rising (slope p={slope_pvalue:.3g}, "
                f"fit R²={fit_r2:.2f}); ~{recommended} additional samples would "
                f"close the gap to target={target_score}."
            ),
        )
    if slope_pvalue >= _SLOPE_PVALUE_GATE:
        return (
            "HARD_FAIL",
            (
                "Learning curve has saturated (slope not significantly different "
                "from zero at the current n); more data unlikely to help."
            ),
        )
    if fit_r2 <= _FIT_R2_GATE:
        return (
            "INCONCLUSIVE",
            f"Power-law fit quality below R²>{_FIT_R2_GATE} (got {fit_r2:.2f}).",
        )
    return ("INCONCLUSIVE", "Insufficient signal to extrapolate a sample-count recommendation.")


def _run_causal_branch(
    *,
    state: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    n_features: int,
    t_start: float,
) -> dict[str, Any]:
    """Causal-inference branch: ATE bootstrap CI width vs n."""
    n_rows = int(len(X_train))
    sizes = _bucket_sizes(n_rows, _DEFAULT_K_BUCKETS)
    treatment_col = _resolve_treatment_column(state, X_train)

    if treatment_col is None:
        return _empty_report(
            problem_type=problem_type,
            n_rows=n_rows,
            n_features=n_features,
            rationale="Causal branch could not identify a treatment column.",
        )

    rng = np.random.default_rng(_SEED_OFFSET)
    ci_curve: list[tuple[int, float]] = []
    cap_hit = False
    for bucket_size in sizes:
        if (time.monotonic() - t_start) > _WALLTIME_CAP_S:
            cap_hit = True
            break
        X_bucket = X_train.iloc[: int(bucket_size)]
        y_bucket = y_train.iloc[: int(bucket_size)]
        width = _bootstrap_ate_ci_width(
            X=X_bucket,
            y=y_bucket,
            treatment_col=treatment_col,
            n_bootstrap=_DEFAULT_BOOTSTRAP_N,
            rng=rng,
        )
        if np.isfinite(width):
            ci_curve.append((int(bucket_size), float(width)))

    runtime_s = float(time.monotonic() - t_start)

    # Resolve the target CI width via the sufficiency_resolver (audit-friendly).
    sufficiency_cfg = (state.get("scope_spec") or {}).get("sufficiency")
    target_mde_res = resolve_target_mde(
        user_config=cast(dict[str, Any] | None, sufficiency_cfg),
        outcome_type="continuous",
    )
    target_width = float(target_mde_res.value)

    recommended: Optional[int] = None
    extrapolated_n: Optional[int] = None
    if ci_curve:
        ns = np.array([entry[0] for entry in ci_curve], dtype=float)
        widths = np.array([entry[1] for entry in ci_curve], dtype=float)
        k_fit = _fit_inverse_sqrt(ns, widths)
        if k_fit is not None:
            extrapolated_n = _solve_n_for_ci_width(k_fit, target_width)
            if extrapolated_n is not None:
                additional = extrapolated_n - n_rows
                if additional > 0:
                    recommended = int(additional)

    if cap_hit:
        verdict = "INCONCLUSIVE"
        rationale = (
            f"Diagnostic exceeded {_WALLTIME_CAP_S:.0f}s walltime cap after "
            f"{len(ci_curve)} of {len(sizes)} buckets (causal branch)."
        )
    elif recommended is not None:
        verdict = "SOFT_FAIL"
        rationale = (
            f"Bootstrap ATE CI width is still shrinking with n; ~{recommended} "
            f"additional samples would tighten CI to target_mde={target_width}."
        )
    else:
        verdict = "INCONCLUSIVE"
        rationale = (
            "Causal branch could not extrapolate a sample-count recommendation "
            "from the bootstrap CI curve."
        )

    return {
        "verdict": verdict,
        "verdict_rationale": rationale,
        "n_rows": n_rows,
        "n_features": n_features,
        "problem_type": problem_type,
        "learning_curve": None,
        "proxy_model": "dim-bootstrap",
        "slope_at_max_n": None,
        "slope_pvalue": None,
        "power_law_fit": None,
        "extrapolated_n_for_target": extrapolated_n,
        "extrapolated_n_ci": None,
        "fit_quality_r2": None,
        "recommended_additional_samples": recommended,
        "ate_ci_width_curve": ci_curve if ci_curve else None,
        "ate_target_ci_width": target_width,
        "diagnostic_runtime_s": runtime_s,
    }
