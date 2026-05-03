"""Bootstrap confidence-interval helpers.

Phase 1 W3-lite Day-5 minimum surface: ``bca_confidence_interval`` over a small
(k=4..50) sample of fold-level scalars for the cross-fold aggregator
(shard 21 §D). Scope is intentionally narrow — when shard 20 W7 ships its
broader bootstrap utility, this module is the natural home for the percentile
+ basic + studentized variants and the BCa here can be reused unchanged.

BCa rationale (per shard 21 §D codex review verdict B = CHALLENGED 2026-05-01):
The BCa endpoint here is *exploratory* over k split-level fold values, not a
calibrated generalization-error CI. At k=10 the jackknife acceleration term is
on the marginal side of stability per DiCiccio & Tibshirani 1987; we expose
``unstable_warning=True`` whenever the acceleration magnitude exceeds 0.25 OR
the bootstrap endpoint is non-finite OR n_samples < 4. Downstream callers that
hit ``unstable_warning=True`` SHOULD prefer the percentile CI emitted alongside
on ``AggregateStat`` rather than the BCa endpoint. Bengio & Grandvalet 2004
(JMLR) is the load-bearing citation: there is no universal unbiased estimator
of k-fold CV variance — BCa here is an asymmetry-corrected percentile, not a
variance-corrected interval.

Threshold conservatism (cycle-16 C-1): the 0.25 default ``instability_threshold``
is calibrated against DiCiccio & Tibshirani 1987 Table 1 — at n=10 the BCa
coverage breakdown threshold is empirically near |a|·n^{1/2} = 1, i.e. |a| ≈
0.32. The 0.25 default leaves a ~22% safety margin so callers see
``unstable_warning=True`` BEFORE the coverage actually degrades; this matches
shard 21 §D's "exploratory not inferential" framing.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import bootstrap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BcaResult:
    """Bias-corrected accelerated bootstrap CI over a 1-D fold-value sample.

    Returned by :func:`bca_confidence_interval`. ``unstable_warning=True`` when
    the BCa acceleration term is ill-conditioned or n_samples is below the
    minimum threshold; downstream callers SHOULD fall back to a percentile CI
    in that case.
    """

    ci_lo: Optional[float]
    ci_hi: Optional[float]
    unstable_warning: bool
    n_samples: int
    acceleration: Optional[float]


def bca_confidence_interval(
    values: "np.ndarray | list[float] | tuple[float, ...]",
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    rng_seed: int = 42,
    instability_threshold: float = 0.25,
    min_samples: int = 4,
) -> BcaResult:
    """BCa CI over a 1-D sample of fold-level scalars.

    Returns ``ci_lo=ci_hi=None`` and ``unstable_warning=True`` when
    ``len(values) < min_samples``. When :func:`scipy.stats.bootstrap` returns
    NaN/inf endpoints OR the jackknife acceleration magnitude exceeds
    ``instability_threshold``, ``unstable_warning=True`` is set; the
    confidence-interval endpoints are still emitted (so callers can compare
    BCa-vs-percentile drift) unless the bootstrap itself failed.

    The acceleration is computed from the jackknife-leave-one-out distribution
    per the BCa standard formula (DiCiccio & Tibshirani 1987 eq. 6) — scipy's
    ``BootstrapResult`` does not expose acceleration directly, so we recompute
    it here for the diagnostic flag.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"bca_confidence_interval expects 1-D values, got shape {arr.shape}")
    n = arr.size
    if n < min_samples:
        return BcaResult(
            ci_lo=None,
            ci_hi=None,
            unstable_warning=True,
            n_samples=n,
            acceleration=None,
        )

    accel = _jackknife_acceleration(arr)
    accel_unstable = (not np.isfinite(accel)) or (abs(accel) > instability_threshold)

    try:
        rng = np.random.default_rng(rng_seed)
        # Cycle-16 I-2: scipy emits DegenerateDataWarning + RuntimeWarning when
        # the BCa endpoint can't be calculated (e.g., constant fold values).
        # The unstable_warning fallback below is the documented contract; the
        # scipy warning duplicates it noisily. Filter at the call site so
        # downstream loggers see clean output.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The BCa confidence interval cannot be calculated",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered",
            )
            res = bootstrap(
                (arr,),
                np.mean,
                method="BCa",
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
        ci_lo = float(res.confidence_interval.low)
        ci_hi = float(res.confidence_interval.high)
    except Exception as exc:
        logger.warning(
            "bca_confidence_interval scipy.stats.bootstrap failure: %r — emitting None CI",
            exc,
        )
        return BcaResult(
            ci_lo=None,
            ci_hi=None,
            unstable_warning=True,
            n_samples=n,
            acceleration=accel if np.isfinite(accel) else None,
        )

    finite_endpoints = np.isfinite(ci_lo) and np.isfinite(ci_hi)
    if not finite_endpoints:
        return BcaResult(
            ci_lo=None,
            ci_hi=None,
            unstable_warning=True,
            n_samples=n,
            acceleration=accel if np.isfinite(accel) else None,
        )

    return BcaResult(
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        unstable_warning=accel_unstable,
        n_samples=n,
        acceleration=float(accel) if np.isfinite(accel) else None,
    )


def _jackknife_acceleration(values: np.ndarray) -> float:
    """BCa acceleration from leave-one-out jackknife distribution.

    Formula 6 of DiCiccio & Tibshirani 1987:

    .. math::
        \\hat a = \\frac{\\sum_i (\\bar\\theta_{(\\cdot)} - \\theta_{(i)})^3}
                       {6 \\, [\\sum_i (\\bar\\theta_{(\\cdot)} - \\theta_{(i)})^2]^{3/2}}

    Returns NaN when n < 2 or the denominator vanishes (degenerate sample).
    """
    n = values.size
    if n < 2:
        return float("nan")

    total = float(np.sum(values))
    # leave-one-out means: (total - values[i]) / (n - 1)
    loo_means = (total - values) / (n - 1)
    grand = float(np.mean(loo_means))
    diffs = grand - loo_means
    num = float(np.sum(diffs ** 3))
    den_sq_sum = float(np.sum(diffs ** 2))
    if den_sq_sum == 0.0:
        return float("nan")
    den = 6.0 * (den_sq_sum ** 1.5)
    return float(num / den)


__all__ = ["BcaResult", "bca_confidence_interval"]
