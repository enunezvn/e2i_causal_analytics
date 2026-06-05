"""P7 / H6 — nested CI must use the segment ATE's STANDARD ERROR, not dispersion.

The producers set cate_std = np.std(cate_values) — the spread of per-unit CATE
estimates, which does NOT shrink with n. The bridge fed that in as the segment
ATE's standard error (ate_std) into inverse-variance weights and Q/I²/τ², making
each segment's variance ~n× too large and the aggregate CIs ~√n too wide. The fix
derives a TRUE SE (cate_se) from the segment-MEAN CI (shrinks with n) and bridges
that instead.
"""

from __future__ import annotations

import pytest

from src.causal_engine.hierarchical.analyzer import (
    HierarchicalAnalyzer,
    HierarchicalConfig,
    SegmentResult,
)
from src.causal_engine.hierarchical.segment_cate import (
    SegmentCATECalculator,
    SegmentCATEConfig,
)


class TestSEFromCI:
    def test_se_derived_from_mean_ci_not_dispersion(self):
        calc = SegmentCATECalculator(SegmentCATEConfig(ci_confidence_level=0.95))
        # Segment-MEAN CI half-width ≈ 1.96·0.2 → SE ≈ 0.2, NOT cate_std=99.
        se = calc._se_from_ci(4.608, 5.392, cate_std=99.0, n_samples=50)
        assert se == pytest.approx(0.2, abs=0.01)

    def test_se_fallback_shrinks_with_n(self):
        calc = SegmentCATECalculator(SegmentCATEConfig(ci_confidence_level=0.95))
        se_n = calc._se_from_ci(None, None, cate_std=2.0, n_samples=100)
        se_4n = calc._se_from_ci(None, None, cate_std=2.0, n_samples=400)
        assert se_n == pytest.approx(0.2, abs=1e-6)
        assert se_4n == pytest.approx(0.1, abs=1e-6)  # 4× n → 2× smaller SE


class TestAggregateUsesSE:
    def test_aggregate_ci_uses_se_not_std(self):
        """The aggregate CI must be built from cate_se (narrow), not cate_std.

        cate_se=0.1 (a real SE) vs cate_std=10.0 (the per-unit dispersion). With
        the buggy bridge the inverse-variance weights use 10.0 → a very wide
        aggregate CI; with the fix they use 0.1 → a narrow one.
        """
        analyzer = HierarchicalAnalyzer(HierarchicalConfig(compute_nested_ci=True))
        segs = [
            SegmentResult(
                segment_id=i,
                segment_name=f"s{i}",
                n_samples=100,
                uplift_range=(0.0, 1.0),
                cate_mean=mean,
                cate_std=10.0,  # large dispersion (the wrong quantity)
                cate_se=0.1,  # true SE (shrinks with n)
                cate_ci_lower=mean - 0.2,
                cate_ci_upper=mean + 0.2,
                success=True,
            )
            for i, mean in enumerate([0.4, 0.5, 0.6])
        ]
        _ate, lo, hi = analyzer._aggregate_results(segs, 300)
        assert lo is not None and hi is not None
        width = hi - lo
        # SE-based aggregate CI is narrow (~0.2); dispersion-based would be ~20+.
        assert width < 1.0, f"aggregate CI must use the SE (narrow), got width={width}"
