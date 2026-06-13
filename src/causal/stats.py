"""Shared statistical helpers for the causal pipeline.

#27: the causal pipeline hardcoded a 95% CI (+/- 1.96 sigma) in several places
(``src/api/routes/causal.py`` consensus CI, the CATE estimator fallback, the
hierarchical segment-CATE normal-approximation CIs) without ever exposing the
confidence level, so the frontend could not honestly label its intervals. This
module centralizes the z-score so the magic number lives in ONE place.

SCOPE: this helper is used by the #27 call sites listed above (consensus CI,
``heterogeneous_optimizer`` CATE fallback, ``causal_engine.hierarchical``
segment-CATE CIs). It does NOT yet cover every CI in the codebase -- the
per-library DoWhy refutation CIs, CausalML uplift CIs, and energy-score
estimator CIs still carry their own (currently 95%) interval math. Migrating
those is out of scope for #27 and tracked separately.

We use ``scipy.stats.norm.ppf`` (scipy is a hard dependency -- ``scipy>=1.13.0``
in ``pyproject.toml``) for the exact two-sided normal quantile rather than a
lookup table of rounded constants. At the 0.95 default this yields 1.9599639...,
which rounds to the legacy 1.96, so existing numeric behavior is unchanged at the
default.
"""

from __future__ import annotations

from scipy.stats import norm

__all__ = ["z_score_for_confidence", "z_score_for_alpha"]


def z_score_for_confidence(confidence_level: float) -> float:
    """Return the two-sided normal z critical value for ``confidence_level``.

    The half-width of a normal-approximation confidence interval is
    ``z * sigma`` where ``z = norm.ppf(1 - (1 - confidence_level) / 2)``.

    Examples (scipy-exact):
        0.90 -> 1.6448536...  (~1.645)
        0.95 -> 1.9599639...  (~1.96, the legacy default)
        0.99 -> 2.5758293...  (~2.576)

    Args:
        confidence_level: Desired confidence level in the open interval (0, 1),
            e.g. 0.95 for a 95% CI.

    Returns:
        The positive z critical value.

    Raises:
        ValueError: if ``confidence_level`` is not strictly inside (0, 1). A
            level of exactly 0 or 1 has no finite z and signals a programming
            error, not a usable interval -- we fail loudly rather than silently
            substitute a default.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in the open interval (0, 1), got {confidence_level!r}"
        )
    return float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))


def z_score_for_alpha(alpha: float) -> float:
    """Return the two-sided normal z critical value for significance ``alpha``.

    Convenience wrapper for callers that carry a significance level (``alpha``,
    e.g. 0.05) rather than a confidence level. Equivalent to
    ``z_score_for_confidence(1 - alpha)``.

    Args:
        alpha: Significance level in the open interval (0, 1), e.g. 0.05.

    Returns:
        The positive z critical value.

    Raises:
        ValueError: if ``alpha`` is not strictly inside (0, 1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    return z_score_for_confidence(1.0 - alpha)
