"""#2142 — the hierarchical overall CI is the CI of the overall ATE it is served beside.

``overall_ate`` is the sample-size-weighted mean of segment CATEs: the segments
partition the analysed rows, so sum(n_i/N * CATE_i) is the population ATE. Before
#2142 the served ``overall_ci_lower/upper`` came from the inverse-variance nested
aggregate (a different estimand); live, the CI (0.0958, 0.1087) excluded the
served ATE 0.1311 (#2027 cert BEFORE run). The CI is now the delta-method CI of
the same weighted sum, SE = sqrt(sum (n_i/N)^2 * se_i^2), over the SAME segment
set as the ATE. A successful segment without a measured SE or either CI bound
withholds the overall CI (None) — never a ``cate_std`` stand-in, never a CI bound
swallowed by ``or``.

The segment results below are planted ``SegmentResult``s (the layer the analyzer
aggregates); the numbers in the first test are the live cert segments.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.hierarchical.analyzer import (
    HierarchicalAnalyzer,
    HierarchicalConfig,
    SegmentResult,
)

Z95 = 1.959963984540054


def _seg(
    i: int,
    n: int,
    mean: float,
    se: float | None,
    lo: float | None,
    hi: float | None,
    cate_std: float | None = 0.07,
) -> SegmentResult:
    return SegmentResult(
        segment_id=i,
        segment_name=f"segment_{i}",
        n_samples=n,
        uplift_range=(0.0, 1.0),
        cate_mean=mean,
        cate_std=cate_std,
        cate_se=se,
        cate_ci_lower=lo,
        cate_ci_upper=hi,
        success=True,
    )


def _cert_segments() -> List[SegmentResult]:
    # #2027 cert BEFORE run, segment_results (se = CI half-width / z).
    lo0, hi0 = 0.001130585253742761, 0.017546870408908004
    lo1, hi1 = 0.2423995079470123, 0.26330101652637766
    return [
        _seg(0, 200, 0.009338727831325383, (hi0 - lo0) / 2 / Z95, lo0, hi0, 0.0592),
        _seg(1, 200, 0.252850262236695, (hi1 - lo1) / 2 / Z95, lo1, hi1, 0.0754),
    ]


def test_overall_ci_is_the_delta_method_ci_of_the_sample_weighted_ate():
    analyzer = HierarchicalAnalyzer(HierarchicalConfig())
    ate, lo, hi = analyzer._aggregate_results(_cert_segments(), 400)

    assert ate == pytest.approx(0.131094, abs=1e-6)
    # Delta method over the planted SEs; the pre-#2142 value was the
    # inverse-variance aggregate's CI (0.095788, 0.108699).
    assert lo == pytest.approx(0.124450, abs=1e-6)
    assert hi == pytest.approx(0.137739, abs=1e-6)
    assert lo < ate < hi


def test_overall_ci_keeps_a_real_zero_bound():
    # One segment, a real 0.0 lower bound: ``0.0 or cate_mean`` used to turn it
    # into 0.1.
    analyzer = HierarchicalAnalyzer(HierarchicalConfig())
    ate, lo, hi = analyzer._aggregate_results([_seg(0, 100, 0.1, 0.1 / Z95, 0.0, 0.2)], 100)

    assert ate == pytest.approx(0.1)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.2, abs=1e-9)


def test_overall_ci_uses_a_zero_segment_mean():
    analyzer = HierarchicalAnalyzer(HierarchicalConfig())
    segs = [_seg(0, 100, 0.0, 0.05, -0.098, 0.098), _seg(1, 300, 0.4, 0.05, 0.302, 0.498)]
    ate, lo, hi = analyzer._aggregate_results(segs, 400)

    assert ate == pytest.approx(0.3)
    se = math.sqrt(0.25**2 * 0.05**2 + 0.75**2 * 0.05**2)
    assert lo == pytest.approx(0.3 - Z95 * se)
    assert hi == pytest.approx(0.3 + Z95 * se)


def test_segment_without_se_withholds_the_overall_ci_not_a_cate_std_stand_in():
    analyzer = HierarchicalAnalyzer(HierarchicalConfig())
    segs = [
        _seg(0, 200, 0.1, 0.02, 0.06, 0.14),
        _seg(1, 200, 0.3, None, 0.26, 0.34, cate_std=0.5),
    ]
    ate, lo, hi = analyzer._aggregate_results(segs, 400)

    assert ate == pytest.approx(0.2)  # the ATE is still served
    assert lo is None and hi is None


@pytest.mark.parametrize("missing", ["lower", "upper"])
def test_segment_missing_a_ci_bound_withholds_the_overall_ci(missing):
    analyzer = HierarchicalAnalyzer(HierarchicalConfig())
    lo_b = None if missing == "lower" else 0.26
    hi_b = None if missing == "upper" else 0.34
    segs = [_seg(0, 200, 0.1, 0.02, 0.06, 0.14), _seg(1, 200, 0.3, 0.02, lo_b, hi_b)]
    ate, lo, hi = analyzer._aggregate_results(segs, 400)

    assert ate == pytest.approx(0.2)
    assert lo is None and hi is None


def test_ci_disabled_yields_no_zero_width_interval():
    # compute_nested_ci=False also switches segment CIs off; the old branch
    # averaged ``ci or cate_mean`` into a zero-width interval at the ATE.
    analyzer = HierarchicalAnalyzer(HierarchicalConfig(compute_nested_ci=False))
    segs = [_seg(0, 200, 0.1, None, None, None), _seg(1, 200, 0.3, None, None, None)]
    ate, lo, hi = analyzer._aggregate_results(segs, 400)

    assert ate == pytest.approx(0.2)
    assert lo is None and hi is None


def test_small_segment_stays_in_the_ci_it_stays_in_the_ate():
    # NestedCIConfig's default min_segment_size=30 dropped this segment from the
    # CI while the ATE kept it.
    analyzer = HierarchicalAnalyzer(HierarchicalConfig(min_segment_size=10))
    segs = [_seg(0, 20, 1.0, 0.2, 0.608, 1.392), _seg(1, 180, 0.0, 0.02, -0.039, 0.039)]
    ate, lo, hi = analyzer._aggregate_results(segs, 200)

    assert ate == pytest.approx(0.1)
    se = math.sqrt(0.1**2 * 0.2**2 + 0.9**2 * 0.02**2)
    assert lo == pytest.approx(0.1 - Z95 * se)
    assert hi == pytest.approx(0.1 + Z95 * se)


def test_overall_ci_honours_the_configured_confidence_level():
    analyzer = HierarchicalAnalyzer(HierarchicalConfig(ci_confidence_level=0.90))
    segs = [_seg(0, 100, 0.1, 0.05, 0.0178, 0.1822), _seg(1, 100, 0.3, 0.05, 0.2178, 0.3822)]
    ate, lo, hi = analyzer._aggregate_results(segs, 200)

    se = math.sqrt(2 * 0.5**2 * 0.05**2)
    assert hi - lo == pytest.approx(2 * 1.6448536269514722 * se)


@pytest.mark.asyncio
async def test_analyze_warns_naming_the_segment_when_the_overall_ci_is_withheld(monkeypatch):
    analyzer = HierarchicalAnalyzer(HierarchicalConfig(n_segments=2, min_segment_size=10))
    planted = [
        _seg(0, 50, 0.1, 0.02, 0.06, 0.14),
        _seg(1, 50, 0.3, None, 0.26, 0.34),
    ]

    async def _planted_segments(*_args, **_kwargs):
        return planted

    monkeypatch.setattr(analyzer, "_compute_segment_cate", _planted_segments)
    rng = np.random.default_rng(0)
    result = await analyzer.analyze(
        X=pd.DataFrame({"x1": rng.normal(size=100)}),
        treatment=rng.integers(0, 2, size=100),
        outcome=rng.normal(size=100),
        uplift_scores=np.linspace(0.0, 1.0, 100),
    )

    assert result.success
    assert result.overall_ate == pytest.approx(0.2)
    assert result.overall_ate_ci_lower is None and result.overall_ate_ci_upper is None
    withheld = [w for w in result.warnings if "overall CI withheld" in w]
    assert len(withheld) == 1
    assert "segment_1" in withheld[0] and "segment_0" not in withheld[0]
