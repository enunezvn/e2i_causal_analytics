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

import asyncio
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
# F5: Pharma-cohort upper bound; if the inverted power-law demands more
# than this many samples, the asymptote is too flat to give a meaningful
# answer and we should report INCONCLUSIVE rather than a 10^36 fantasy.
_MAX_RECOMMENDED_N = 1_000_000_000  # 1 billion samples — well above realistic cohorts.


# ---------------------------------------------------------------------------
# Power-law helpers.
# ---------------------------------------------------------------------------


def _power_law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Score = a − b · n^(−c). a is the asymptote; b/c control rise shape."""
    return a - b * np.power(n, -c)


def _fit_power_law(
    ns: np.ndarray,
    scores: np.ndarray,
    *,
    problem_type: str = "binary_classification",
) -> Optional[dict[str, float]]:
    """Fit ``score(n) = a − b·n^(−c)``; return ``None`` if curve_fit fails.

    Returns a dict ``{"a", "b", "c", "r2"}``. ``r2`` is the standard
    coefficient of determination of the fit on the input points.

    F4: ``p0[0]`` clamped to <= 0.99 so it never starts outside the bounds
    upper limit on near-perfect curves.

    F7: Bounds are problem-type-aware. AUC/accuracy live in ``[0, 1]``, but
    R² regression scores can be negative (a worse-than-mean predictor) and
    occasionally overshoot 1.0 in noisy fits, so we use ``[-1.0, 1.5]`` for
    ``a`` when ``problem_type == "regression"``. We intentionally do NOT clip
    the raw scores; clipping loses signal for very-bad models and the wider
    bound is the principled fix.
    """
    if len(ns) < 3 or len(scores) < 3 or len(ns) != len(scores):
        return None

    if problem_type == "regression":
        # R² can be negative; allow slight overshoot above 1.0 for noisy fits.
        a_lower, a_upper = -1.0, 1.5
    else:
        # AUC / accuracy ∈ [0, 1].
        a_lower, a_upper = 0.0, 1.0

    # F4: clamp the initial-guess asymptote so it cannot exceed ``a_upper``.
    score_max = float(scores.max())
    p0_a = min(max(score_max, 0.5) + 0.05, a_upper - 0.01)
    p0_a = max(p0_a, a_lower + 0.01)

    # Bounds: a per problem_type, b in [0, ∞), c in [0.01, 5.0] (avoid c→0
    # degenerate case where the fit collapses to a constant).
    try:
        popt, _pcov = curve_fit(
            _power_law,
            ns,
            scores,
            p0=[p0_a, 1.0, 0.5],
            bounds=([a_lower, 0.0, 0.01], [a_upper, np.inf, 5.0]),
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


def _slope_sign_last_k(
    ns: np.ndarray, scores: np.ndarray, k: int = _SLOPE_SIG_LAST_K
) -> Optional[float]:
    """Return the linear-regression slope sign on the last ``k`` points.

    F6 companion to ``_slope_pvalue_last_k``: a two-sided p-value cannot
    distinguish a rising curve from a falling one. Callers must combine
    ``slope > 0 AND pvalue < gate`` to claim "still rising"; ``slope < 0 AND
    pvalue < gate`` is "trending downward, more data unlikely to help".

    Returns ``None`` when the slope is undefined (degenerate inputs).
    """
    if len(ns) < k or len(scores) < k:
        return None
    x = ns[-k:].astype(float)
    y = scores[-k:].astype(float)
    if np.var(x) == 0:
        return None
    try:
        result = stats.linregress(x, y)
    except (ValueError, RuntimeError):
        return None
    return float(result.slope)


def _invert_power_law_for_n(
    fit: dict[str, float],
    target_score: float,
) -> Optional[int]:
    """Solve ``a − b·n^(−c) = target`` for n. Returns ``None`` if unreachable.

    Unreachable cases:
    - ``a <= target`` — the fit's asymptote is below the target.
    - Numerical overflow when target requires astronomically large n.
    - F5: ``n_star > _MAX_RECOMMENDED_N`` — the asymptote is too flat to give
      a meaningful recommendation. A 10^36-sample answer is a fantasy in any
      pharma cohort; we treat this as "unreachable" and let the caller emit
      INCONCLUSIVE.
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
    if n_star > _MAX_RECOMMENDED_N:
        logger.warning(
            "Power-law inversion returned n_star=%.3g exceeding cap %.3g; "
            "asymptote too flat to give a meaningful recommendation.",
            n_star,
            float(_MAX_RECOMMENDED_N),
        )
        return None
    return int(np.ceil(n_star))


def _recommend_additional_samples(
    *,
    fit: Optional[dict[str, float]],
    n_current: int,
    target_score: float,
    slope_pvalue: float,
    slope_sign: Optional[float] = None,
) -> Optional[int]:
    """Return the extra-samples recommendation or ``None`` if not warranted.

    F13: the recommender no longer re-fits the curve. The caller MUST pass a
    pre-fitted ``fit`` dict (or ``None``), so the ``fit_r2`` gate this
    function evaluates is consistent with the same fit object the caller
    inspects elsewhere in the report. This removes the silent decoupling
    where the caller's R² value referred to a different fit than the
    recommender's internal one.

    Gates:
    - ``fit`` is not None and ``fit["r2"] > _FIT_R2_GATE`` (the power-law
      is a good description)
    - ``slope_pvalue < _SLOPE_PVALUE_GATE`` (slope significantly nonzero
      on the last k buckets)
    - F6: ``slope_sign`` is None or positive (rising, not falling). A
      decreasing curve with significant p-value is a HARD_FAIL signal —
      not a "more data" signal — so no recommendation is emitted.
    """
    if fit is None:
        return None
    if fit.get("r2", 0.0) <= _FIT_R2_GATE:
        return None
    if slope_pvalue >= _SLOPE_PVALUE_GATE:
        return None
    # F6: if we know the slope sign and it's negative, refuse to recommend.
    if slope_sign is not None and slope_sign <= 0:
        return None
    n_target = _invert_power_law_for_n(fit, target_score)
    if n_target is None:
        return None
    additional = n_target - int(n_current)
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

    F1: wraps ``model.fit`` in try/except. If the fit raises (e.g. categorical
    X without preprocessing, NaN values, fewer rows than classes), the bucket
    is marked failed via ``score_mean=nan`` so the caller can skip it from
    the curve. We do NOT re-raise — letting the exception propagate would
    crash the LangGraph node and bypass ``log_to_mlflow``.
    """
    X_bucket = X_train.iloc[:bucket_size]
    y_bucket = y_train.iloc[:bucket_size]
    model, _ = _make_proxy_model(problem_type)
    # Defensive: if only one class is present in the bucket, classifiers
    # raise. Return a degenerate score so the curve continues at the next
    # bucket rather than aborting the diagnostic.
    if problem_type != "regression" and len(np.unique(y_bucket)) < 2:
        return {"score_mean": 0.5, "score_std": 0.0}
    try:
        model.fit(X_bucket, y_bucket)
        score = _score_proxy(model=model, X_val=X_val, y_val=y_val, problem_type=problem_type)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
    ) as exc:  # noqa: BLE001 — diagnostic, log + return failed marker
        logger.warning(
            "Proxy fit failed at bucket_size=%d (%s); skipping this bucket.",
            bucket_size,
            exc,
        )
        return {"score_mean": float("nan"), "score_std": float("nan")}
    return {"score_mean": score, "score_std": 0.0}


# ---------------------------------------------------------------------------
# Bucketization.
# ---------------------------------------------------------------------------


_MIN_UNIQUE_BUCKETS = 3


def _bucket_sizes(n_total: int, k: int = _DEFAULT_K_BUCKETS) -> list[int]:
    """Return up to ``k`` cumulative bucket sizes in [floor, n_total].

    F15: if dedup leaves fewer than ``_MIN_UNIQUE_BUCKETS`` unique sizes we
    return an empty list rather than padding with duplicates of ``n_total``.
    Duplicated sizes break the power-law slope estimation (zero x-variance
    on the last k points) and silently produce a meaningless curve. The
    caller short-circuits to INCONCLUSIVE on an empty return.

    The smallest bucket is at least 10 rows (so the proxy fit is not
    degenerate); the largest is the full train set.
    """
    if n_total <= 0 or k <= 0:
        return []
    smallest = max(10, n_total // (k * 2)) if n_total > 20 else max(2, n_total // k)
    if smallest >= n_total:
        # Single bucket — caller will treat as INCONCLUSIVE (<3 unique).
        return [n_total]
    raw = np.linspace(smallest, n_total, k)
    sizes = sorted({int(round(x)) for x in raw})
    if len(sizes) < _MIN_UNIQUE_BUCKETS:
        # F15: refuse to pad with duplicates. Caller emits INCONCLUSIVE.
        return sizes
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


def _coerce_to_mapping(obj: Any) -> dict[str, Any]:
    """Return a dict view of ``obj`` covering pydantic, dict, and dict-like.

    F12 helper: ``state["success_criteria"]`` may be a ``SuccessCriteriaSchema``
    (pydantic BaseModel) or a plain dict, depending on which upstream node
    constructed it. ``isinstance(obj, dict)`` returns False on the pydantic
    instance — meaning the legacy-key fallback silently skipped real values.

    Priority for extracting a dict:
    1. ``obj.model_dump()`` if pydantic v2 BaseModel (preserves alias fields)
    2. ``dict(obj)`` if obj already a Mapping
    3. ``{}`` otherwise
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            dumped = obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except (TypeError, ValueError):
            pass
    if isinstance(obj, dict):
        return obj
    # Last-resort: BaseAgentSchema has a dict-like shim ``__getitem__``; we
    # cannot iterate generically without a known field list, so return {}.
    return {}


def _read_numeric_field(mapping: dict[str, Any], key: str) -> Optional[float]:
    """F12: read a numeric field rejecting bool (which is an ``int`` subclass).

    ``isinstance(True, (int, float))`` is True in Python — without the bool
    guard the helper would coerce a boolean adaptive flag to ``float(True)
    == 1.0`` and report it as a target score.
    """
    val = mapping.get(key)
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _extract_target_score(state: dict[str, Any]) -> Optional[float]:
    """Resolve the target predictive-metric value from the state.

    Priority:
    1. ``scope_spec.success_criteria.min_auc`` (PR-462/463 contract field)
    2. ``success_criteria.minimum_auc`` (legacy model_trainer field)

    F12: handles ``SuccessCriteriaSchema`` pydantic instances (not just
    plain dicts) and rejects bool values that ``isinstance(..., (int,
    float))`` would otherwise accept.
    """
    scope_spec_raw = state.get("scope_spec")
    scope_spec = _coerce_to_mapping(scope_spec_raw)
    sc = _coerce_to_mapping(scope_spec.get("success_criteria"))
    for key in ("min_auc", "minimum_auc", "min_score", "target_score"):
        val = _read_numeric_field(sc, key)
        if val is not None:
            return val
    legacy = _coerce_to_mapping(state.get("success_criteria"))
    for key in ("minimum_auc", "min_auc"):
        val = _read_numeric_field(legacy, key)
        if val is not None:
            return val
    # Also try direct pydantic-attr access when model_dump() omitted the
    # field (e.g. ``exclude_none=True``).
    legacy_raw = state.get("success_criteria")
    for key in ("minimum_auc", "min_auc"):
        if hasattr(legacy_raw, key):
            val = getattr(legacy_raw, key)
            if isinstance(val, bool):
                continue
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
    """Best-effort treatment-column discovery for the causal branch.

    F9: removed single-letter ``'t'`` from the implicit-fallback list. ``'t'``
    collides with timestamp columns, transposed-matrix scratch columns, and
    feature-engineering shorthand; it is too ambiguous to use as a default.
    Callers that genuinely use ``'t'`` MUST declare it explicitly in
    ``scope_spec.treatment_column``.
    """
    scope_spec = state.get("scope_spec") or {}
    if isinstance(scope_spec, dict):
        for key in ("treatment_column", "treatment", "intervention_column"):
            val = scope_spec.get(key)
            if isinstance(val, str) and val in X.columns:
                return val
    # F9: 't' removed from this list — too ambiguous.
    for candidate in ("treatment", "intervention", "treated"):
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


def _coerce_X(obj: Any) -> pd.DataFrame:
    """Coerce ``obj`` to a DataFrame. Numpy → DataFrame; pass-through otherwise."""
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(np.asarray(obj))


def _coerce_y(obj: Any) -> pd.Series:
    """Coerce ``obj`` to a Series. Numpy → Series; pass-through otherwise."""
    return obj if isinstance(obj, pd.Series) else pd.Series(np.asarray(obj))


def _resolve_training_X(state: dict[str, Any], fallback_X: Any) -> pd.DataFrame:
    """F1: prefer pre-fit preprocessed features over raw train_data['X'].

    The raw ``train_data['X']`` can contain string/categorical columns that
    crash sklearn / LightGBM. ``state['X_train_preprocessed']`` is the
    numeric output of ``fit_preprocessing`` and is what the production
    model_trainer actually fits on. When that field is present we use it;
    otherwise we fall back to the raw frame (the node will rely on F1's
    try/except wrapper in ``_fit_proxy_on_bucket`` to handle a failure).
    """
    preprocessed = state.get("X_train_preprocessed")
    if preprocessed is None:
        return _coerce_X(fallback_X)
    # Apply the same coercion path so downstream helpers see a DataFrame.
    df = _coerce_X(preprocessed)
    # Length must match y; if upstream mismatch exists, fall back to raw.
    fallback_df = _coerce_X(fallback_X)
    if len(df) != len(fallback_df):
        logger.warning(
            "X_train_preprocessed length %d != train_data['X'] length %d; "
            "falling back to raw train_data['X'] for learning-curve diagnostic.",
            len(df),
            len(fallback_df),
        )
        return fallback_df
    return df


async def learning_curve(state: dict[str, Any]) -> dict[str, Any]:
    """Run the post-training learning-curve diagnostic.

    Short-circuits to ``{}`` when ``success_criteria_met is True`` — the
    diagnostic is the user-facing answer to "why didn't the model pass",
    and there is no question to answer when the model passed. Callers that
    want to force the diagnostic even on pass set
    ``PipelineConfig.always_run_learning_curve = True`` upstream, which
    flips ``success_criteria_met`` to False in the state propagated into
    this node.

    F2: also short-circuits when an upstream node has set ``state['error']``
    — there is no point running an 180s diagnostic when the pipeline is
    already in an error state and the downstream conditional will route
    straight to END regardless of the diagnostic output.

    F3: sklearn/LightGBM fits are CPU-bound and synchronous; they're
    wrapped in ``asyncio.to_thread(...)`` so the LangGraph event loop is
    not blocked. ``asyncio.wait_for(learning_curve(...), timeout=T)``
    cancels the awaiting coroutine; the underlying thread keeps running
    but its result is discarded — acceptable for a non-critical diagnostic.
    """
    # F2: pipeline already errored — emit nothing, let the downstream
    # conditional route to END without burning 180s on a doomed diagnostic.
    if state.get("error"):
        return {}

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

    # F10: validation_data is only needed by the predictive branch. The
    # causal branch tracks CI width via bootstrap on the train set only —
    # it does NOT touch ``val_data``, so requiring it here would wrongly
    # skip the causal diagnostic when no val split exists.
    val_data = state.get("validation_data") or {}
    have_val = isinstance(val_data, dict) and "X" in val_data and "y" in val_data

    # F1: prefer state['X_train_preprocessed'] when present.
    X_train = _resolve_training_X(state, train_data["X"])
    y_train = _coerce_y(train_data["y"])

    if have_val:
        X_val = _coerce_X(val_data["X"])
        y_val = _coerce_y(val_data["y"])
    else:
        X_val = pd.DataFrame()
        y_val = pd.Series(dtype=float)

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
        # F3: run the sync bootstrap loop off the event loop.
        report = await asyncio.to_thread(
            _run_causal_branch,
            state=state,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            n_features=n_features,
            t_start=t_start,
        )
    else:
        # F10: predictive branch requires val_data; emit INCONCLUSIVE if absent.
        if not have_val:
            return {
                "sufficiency_report": _empty_report(
                    problem_type=problem_type,
                    n_rows=n_rows,
                    n_features=n_features,
                    rationale=(
                        "Predictive learning-curve diagnostic requires "
                        "validation_data; none found in state."
                    ),
                )
            }
        # F3: run the sync proxy-fit loop off the event loop.
        report = await asyncio.to_thread(
            _run_predictive_branch,
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

    # F15: refuse to run with fewer than ``_MIN_UNIQUE_BUCKETS`` distinct
    # bucket sizes — duplicates would silently corrupt slope/p-value math.
    if len(sizes) < _MIN_UNIQUE_BUCKETS:
        return _empty_report(
            problem_type=problem_type,
            n_rows=n_rows,
            n_features=n_features,
            rationale=(
                "Insufficient data range for learning-curve diagnostic "
                f"(n_total={n_rows}, k_unique={len(sizes)}); need n_total "
                "≳ 20 for meaningful buckets."
            ),
        )

    curve: list[tuple[int, float, float]] = []
    cap_hit = False
    fit_failures = 0

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
        score_mean = float(result["score_mean"])
        score_std = float(result["score_std"])
        # F1: skip buckets whose fit failed (marker is NaN). Keep counting
        # so we can emit INCONCLUSIVE if every bucket fails.
        if not np.isfinite(score_mean):
            fit_failures += 1
        else:
            curve.append((int(bucket_size), score_mean, score_std))
        # Re-check the cap AFTER appending so the walltime test sees a
        # partial curve when the slow fit pushes us past the budget.
        if (time.monotonic() - t_start) > _WALLTIME_CAP_S:
            cap_hit = True
            break

    runtime_s = float(time.monotonic() - t_start)

    # F1: if EVERY attempted bucket failed (and we didn't run out of time),
    # the diagnostic cannot proceed — emit INCONCLUSIVE with the failure
    # count rather than a meaningless empty curve.
    if not cap_hit and not curve and fit_failures > 0:
        return {
            **_empty_report(
                problem_type=problem_type,
                n_rows=n_rows,
                n_features=n_features,
                rationale=(
                    f"All {fit_failures} proxy-model fits failed on the "
                    "training buckets (likely categorical X without "
                    "preprocessing or constant target); diagnostic cannot "
                    "produce a learning curve."
                ),
            ),
            "proxy_model": proxy_id,
            "diagnostic_runtime_s": runtime_s,
        }

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
    # F7: fit bounds vary by problem_type.
    fit = _fit_power_law(ns, scores, problem_type=problem_type)
    slope_pvalue = _slope_pvalue_last_k(ns, scores)
    # F6: also capture the slope sign so the verdict can distinguish
    # "rising-and-significant" from "falling-and-significant".
    slope_sign = _slope_sign_last_k(ns, scores)

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
    fit_trustworthy = fit is not None and fit_r2 > _FIT_R2_GATE

    if target_score is not None and fit is not None:
        # F13: pass the fit object instead of letting the recommender re-fit.
        recommended = _recommend_additional_samples(
            fit=fit,
            n_current=int(ns.max()),
            target_score=target_score,
            slope_pvalue=slope_pvalue,
            slope_sign=slope_sign,
        )
        # F14: only expose ``extrapolated_n_for_target`` and its CI when the
        # underlying fit is trustworthy (R² above gate). A flimsy fit's
        # extrapolation is misleading — and consumers would otherwise have
        # no signal to distinguish a confident extrapolation from a coin
        # flip dressed up in an integer.
        if fit_trustworthy:
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
        slope_sign=slope_sign,
        fit=fit,
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
        "fit_trustworthy": fit_trustworthy,
        "recommended_additional_samples": recommended,
        "ate_ci_width_curve": None,
        "ate_target_ci_width": None,
        "diagnostic_runtime_s": float(time.monotonic() - t_start),
    }


def _verdict_predictive(
    *,
    recommended: Optional[int],
    slope_pvalue: float,
    slope_sign: Optional[float],
    fit: Optional[dict[str, float]],
    fit_r2: float,
    target_score: Optional[float],
) -> tuple[str, str]:
    """Map the predictive-branch outputs to a (verdict, rationale) pair.

    F6: a significant slope with negative sign is HARD_FAIL ("trending
    downward"), not "still rising".

    F11: a trustworthy fit whose asymptote ``a`` is below ``target_score`` is
    HARD_FAIL ("more data physically cannot reach target") — distinct from
    INCONCLUSIVE which previously conflated this with "no target was given".
    """
    if recommended is not None:
        return (
            "SOFT_FAIL",
            (
                f"Learning curve is still rising (slope p={slope_pvalue:.3g}, "
                f"fit R²={fit_r2:.2f}); ~{recommended} additional samples would "
                f"close the gap to target={target_score}."
            ),
        )

    # F6: significant downward trend is a hard fail.
    if slope_pvalue < _SLOPE_PVALUE_GATE and slope_sign is not None and slope_sign < 0:
        return (
            "HARD_FAIL",
            (
                f"Curve trending downward (slope p={slope_pvalue:.3g}, slope<0); "
                "more data unlikely to help."
            ),
        )

    # F11: trustworthy fit whose asymptote is below target ⇒ HARD_FAIL.
    if (
        fit is not None
        and fit_r2 > _FIT_R2_GATE
        and target_score is not None
        and fit.get("a", 0.0) < target_score
    ):
        return (
            "HARD_FAIL",
            (
                f"Power-law asymptote (a={fit['a']:.3f}) below target ({target_score:.3f}); "
                "more data physically cannot reach the target."
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
            f"Power-law fit quality below R²>{_FIT_R2_GATE} (got {fit_r2:.2f}); "
            "cannot extrapolate a sample-count recommendation reliably.",
        )
    if target_score is None:
        return (
            "INCONCLUSIVE",
            "No target metric extractable from success_criteria — "
            "cannot determine how many additional samples would close the gap.",
        )
    return (
        "INCONCLUSIVE",
        "Insufficient signal to extrapolate a sample-count recommendation.",
    )


def _detect_outcome_type(y: pd.Series) -> str:
    """F8: classify ``y`` as 'binary' or 'continuous'.

    Two unique numeric values → binary (the causal estimand becomes a
    risk-difference; ``resolve_target_mde`` needs ``outcome_type='binary'``
    plus ``baseline_rate`` to compute the right MDE). Anything else →
    treated as continuous (default DIM-on-means estimator).
    """
    try:
        n_unique = int(y.dropna().nunique())
    except (AttributeError, TypeError):
        return "continuous"
    if n_unique == 2:
        return "binary"
    return "continuous"


def _normalize_treatment_series(t: pd.Series) -> Optional[pd.Series]:
    """F9: coerce a 2-level treatment column to ``{0, 1}``; return None if not.

    Accepts numeric ``{0, 1}``, boolean, or any 2-level categorical. Rejects
    anything with !=2 unique non-null values — the DIM ATE estimator assumes
    a binary intervention indicator and silently mis-coercing a 3-level
    treatment to ``{0, 1}`` would produce a misleading CI width.
    """
    series = t.dropna()
    uniques = series.unique()
    if len(uniques) != 2:
        return None
    # Boolean → int directly preserves ordering (False=0, True=1).
    if series.dtype == bool:
        return t.astype("Int64").astype("Int64").fillna(0).astype(int)
    # Already numeric and exactly {0, 1}: pass through.
    numeric_set = {0, 1}
    try:
        as_int = series.astype(int)
        if set(as_int.unique().tolist()) == numeric_set:
            return t.astype(int)
    except (ValueError, TypeError):
        pass
    # General case: factorize, mapping the alphabetically/numerically lower
    # unique to 0 and the higher to 1. Stable across runs because we sort.
    sorted_levels = sorted(uniques.tolist(), key=lambda v: (str(type(v)), str(v)))
    mapping = {sorted_levels[0]: 0, sorted_levels[1]: 1}
    return t.map(mapping).astype("Int64").astype(int)


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

    # F15: insufficient unique buckets in the causal branch too.
    if len(sizes) < _MIN_UNIQUE_BUCKETS:
        return _empty_report(
            problem_type=problem_type,
            n_rows=n_rows,
            n_features=n_features,
            rationale=(
                "Insufficient data range for causal learning-curve diagnostic "
                f"(n_total={n_rows}, k_unique={len(sizes)}); need n_total "
                "≳ 20 for meaningful buckets."
            ),
        )

    treatment_col = _resolve_treatment_column(state, X_train)

    if treatment_col is None:
        return _empty_report(
            problem_type=problem_type,
            n_rows=n_rows,
            n_features=n_features,
            rationale=(
                "No treatment column found in scope_spec or X_train. Declare "
                "scope_spec.treatment_column explicitly, or rename the column "
                "to one of 'treatment' / 'intervention' / 'treated'."
            ),
        )

    # F9: normalize the treatment series to {0, 1} once. Reject if it
    # doesn't have exactly 2 unique non-null values.
    normalized_t = _normalize_treatment_series(X_train[treatment_col])
    if normalized_t is None:
        return _empty_report(
            problem_type=problem_type,
            n_rows=n_rows,
            n_features=n_features,
            rationale=(
                f"Treatment column '{treatment_col}' does not have exactly 2 "
                "unique values; causal diagnostic requires a binary "
                "intervention indicator."
            ),
        )
    X_train = X_train.copy()
    X_train[treatment_col] = normalized_t.to_numpy()

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

    # F8: detect outcome type from y_train so resolve_target_mde sees the
    # correct estimand. Binary causal outcomes also need ``baseline_rate``.
    outcome_type = _detect_outcome_type(y_train)
    sufficiency_cfg = (state.get("scope_spec") or {}).get("sufficiency")
    extra_kwargs: dict[str, Any] = {}
    if outcome_type == "binary":
        # Pre-treatment baseline rate ≈ control-arm mean. Falls back to
        # overall mean if the control mask is empty (defensive).
        try:
            control_mask = X_train[treatment_col].to_numpy() == 0
            if control_mask.any():
                extra_kwargs["baseline_rate"] = float(y_train.to_numpy()[control_mask].mean())
            else:
                extra_kwargs["baseline_rate"] = float(y_train.mean())
        except (KeyError, TypeError, ValueError):
            extra_kwargs["baseline_rate"] = float(y_train.mean())
    target_mde_res = resolve_target_mde(
        user_config=cast(dict[str, Any] | None, sufficiency_cfg),
        outcome_type=outcome_type,
        **extra_kwargs,
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
        "outcome_type": outcome_type,
        "diagnostic_runtime_s": runtime_s,
    }
