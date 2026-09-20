"""Shared, inference-backed treatment-effect heterogeneity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HeterogeneityResult:
    """Distinguish CATE availability from evidence of non-constant effects."""

    cate_available: bool
    detected: bool
    method: str
    p_value: Optional[float]
    score: float
    segments: list[dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


def _segment_interval(model: Any, X: NDArray[np.float64]) -> Optional[tuple[float, float, float]]:
    try:
        inference = model.ate_inference(X)
        lower, upper = inference.conf_int_mean()
        stderr = float(np.squeeze(inference.stderr_mean))
        values = (float(np.squeeze(lower)), float(np.squeeze(upper)), stderr)
        if all(np.isfinite(v) for v in values) and values[0] < values[1] and values[2] > 0:
            return values
    except Exception:  # absence of segment inference is an honest unavailable state
        return None
    return None


def _build_segments(
    model: Any,
    X: NDArray[np.float64],
    cate: NDArray[np.float64],
    overall_ate: float,
) -> list[dict[str, Any]]:
    """Build exploratory score strata only when each stratum has uncertainty."""

    threshold = float(np.median(cate))
    definitions = (
        ("High CATE", cate >= threshold, "Predicted CATE at or above the median"),
        ("Low CATE", cate < threshold, "Predicted CATE below the median"),
    )
    segments: list[dict[str, Any]] = []
    for name, mask, description in definitions:
        if not np.any(mask):
            continue
        interval = _segment_interval(model, X[mask])
        if interval is None:
            # Do not publish a segment point estimate without uncertainty.
            return []
        lower, upper, stderr = interval
        segment_ate = float(np.mean(cate[mask]))
        segments.append(
            {
                "segment": name,
                "cate": segment_ate,
                "cate_ci_lower": lower,
                "cate_ci_upper": upper,
                "standard_error": stderr,
                "size": int(np.sum(mask)),
                "description": description,
                # This is deliberately descriptive, not a two-sample test: the
                # segment and overall estimates are dependent.
                "interval_excludes_overall_ate": not (lower <= overall_ate <= upper),
                # These are model-score strata, not human-interpretable rules.
                "validation_status": "exploratory_model_score_stratum",
            }
        )
    return segments


def analyze_heterogeneity(
    *,
    model: Any,
    X: NDArray[np.float64],
    cate: Any,
    alpha: float = 0.05,
) -> HeterogeneityResult:
    """Test whether a fitted CATE surface is non-constant.

    Linear/final-stage estimators use a Bonferroni-controlled test of the null
    that every effect-modifier coefficient is zero.  Forest-like estimators use
    their pointwise inference to test deviations from the population ATE, also
    Bonferroni-controlled.  If the estimator exposes no usable inference, CATE
    remains available but heterogeneity is *not* declared.
    """

    arr = np.asarray(cate, dtype=float).ravel() if cate is not None else np.asarray([])
    matrix = np.asarray(X, dtype=float)
    if arr.size < 2 or matrix.shape[0] != arr.size or not np.all(np.isfinite(arr)):
        return HeterogeneityResult(
            cate_available=False,
            detected=False,
            method="unavailable",
            p_value=None,
            score=0.0,
            reason="No finite row-aligned CATE array was produced.",
        )

    spread = float(np.std(arr))
    magnitude = float(np.mean(np.abs(arr)))
    score = spread / magnitude if magnitude > 0 else 0.0
    overall_ate = float(np.mean(arr))

    # DML/LinearDML/linear-final-stage DRLearner: a non-zero modifier
    # coefficient is direct evidence against a constant treatment effect.
    try:
        inference = model.coef__inference()
        pvalues = np.asarray(inference.pvalue(), dtype=float).ravel()
        pvalues = pvalues[np.isfinite(pvalues)]
        if pvalues.size:
            adjusted_p = min(1.0, float(np.min(pvalues)) * int(pvalues.size))
            detected = adjusted_p < alpha
            segments = _build_segments(model, matrix, arr, overall_ate) if detected else []
            return HeterogeneityResult(
                cate_available=True,
                detected=detected,
                method="effect_modifier_coefficients_bonferroni",
                p_value=adjusted_p,
                score=score,
                segments=segments,
            )
    except Exception:
        pass

    # CausalForestDML and other non-parametric estimators: test whether any
    # pointwise effect differs from the estimated population mean, controlling
    # family-wise error over the evaluated rows.
    try:
        inference = model.effect_inference(matrix)
        pvalues = np.asarray(inference.pvalue(value=overall_ate), dtype=float).ravel()
        pvalues = pvalues[np.isfinite(pvalues)]
        if pvalues.size:
            adjusted_p = min(1.0, float(np.min(pvalues)) * int(pvalues.size))
            detected = adjusted_p < alpha
            segments = _build_segments(model, matrix, arr, overall_ate) if detected else []
            return HeterogeneityResult(
                cate_available=True,
                detected=detected,
                method="pointwise_effect_deviation_bonferroni",
                p_value=adjusted_p,
                score=score,
                segments=segments,
            )
    except Exception:
        pass

    return HeterogeneityResult(
        cate_available=True,
        detected=False,
        method="inference_unavailable",
        p_value=None,
        score=score,
        reason="Estimator produced CATEs but no valid heterogeneity inference.",
    )
