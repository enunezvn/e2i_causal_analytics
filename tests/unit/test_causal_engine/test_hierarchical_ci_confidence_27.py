"""RED-FIRST test for #27: hierarchical segment-CATE CIs honor the confidence level.

The hierarchical CATE route (POST /api/causal/hierarchical) threads
``request.confidence_level`` into ``SegmentCATEConfig.ci_confidence_level`` and
echoes it back in ``HierarchicalAnalysisResponse.confidence_level``. But the
normal-approximation CI fallbacks in segment_cate.py used a BINARY z-score:

    z = 1.96 if ci_confidence_level == 0.95 else 2.576

so ANY non-0.95 level (including a valid 0.90 request) silently got the 99%
z-score (2.576) while the response claimed 0.90 -- a functional honesty bug of
exactly the #27 class. This test pins the corrected behavior: the half-width
tracks the configured level via the shared z-score helper.

No mocking of results: ``_compute_ci`` is the real method, given a real config
and a model WITHOUT ``effect_inference`` so it deterministically takes the
normal-approximation fallback over a real numpy std.
"""

import numpy as np
import pytest


def _half_width_at_level(level: float) -> tuple[float, float]:
    """Return (half_width, sigma_se) from the real _compute_ci fallback at ``level``."""
    from src.causal_engine.hierarchical.segment_cate import (
        SegmentCATECalculator,
        SegmentCATEConfig,
    )

    calc = SegmentCATECalculator(SegmentCATEConfig(ci_confidence_level=level))

    # A model with no effect_inference -> _compute_ci falls back to the normal
    # approximation (the path that hardcoded the z-score). Real, deterministic
    # CATE values so cate_std and the resulting SE are genuine computations.
    model = object()
    cate_values = np.array([0.10, 0.50] * 10, dtype=float)
    cate_mean = float(np.mean(cate_values))
    cate_std = float(np.std(cate_values))
    n = len(cate_values)
    se = cate_std / np.sqrt(n)

    lower, upper = calc._compute_ci(
        model, X=None, cate_values=cate_values, cate_mean=cate_mean, cate_std=cate_std
    )
    assert lower is not None and upper is not None
    half_width = (upper - lower) / 2.0
    return half_width, se


def test_hierarchical_ci_half_width_tracks_confidence_level():
    # 0.95 -> 1.96 * se (legacy behavior preserved)
    hw_95, se = _half_width_at_level(0.95)
    assert pytest.approx(hw_95, rel=1e-3) == 1.96 * se

    # 0.90 -> 1.645 * se, NOT 2.576 * se (the old binary-else bug)
    hw_90, se90 = _half_width_at_level(0.90)
    assert pytest.approx(hw_90, rel=1e-3) == 1.645 * se90
    assert abs(hw_90 - 2.576 * se90) > 1e-3, "0.90 must NOT use the 99% z (2.576)"
    assert hw_90 < hw_95  # 90% interval strictly narrower than 95%

    # 0.99 -> 2.576 * se (still correct)
    hw_99, se99 = _half_width_at_level(0.99)
    assert pytest.approx(hw_99, rel=1e-3) == 2.576 * se99
