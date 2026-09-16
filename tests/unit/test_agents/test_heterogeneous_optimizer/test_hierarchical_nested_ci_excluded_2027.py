"""#2027 part B (agent path): the nested CI fails closed on unmeasured segments.

``HierarchicalAnalyzerNode._run_hierarchical_analysis`` (wired by default,
``graph.py`` ``enable_hierarchical=True``) bridged every successful segment into
``SegmentEstimate`` with invented uncertainty when the analyzer had none —
``ate_std = cate_std or 0.01``, ``ci = cate_mean ± 0.1`` — and the ``or`` swallowed
a real 0.0 bound. Same contract as the API path: a segment enters the nested
aggregate only with a measured SE AND both CI bounds (``is not None``); the rest are
listed under ``nested_ci_excluded_segments`` with a reason code and named in
``warnings``; no segment left means no nested CI (``compute`` returns ±inf on zero
segments, so the node must decide first).

Real node method + real ``NestedConfidenceInterval.compute``; only the heavy
``HierarchicalAnalyzer.analyze`` fit is stood in for.
"""

from __future__ import annotations

import types
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
    HierarchicalAnalyzerNode,
)
from src.causal_engine.hierarchical import HierarchicalAnalyzer

NO_MEASURED_UNCERTAINTY = "no_measured_uncertainty"
_N = 100


def _segment(
    segment_id: int,
    *,
    cate_mean: float,
    cate_se: Optional[float],
    ci_lower: Optional[float],
    ci_upper: Optional[float],
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        segment_id=segment_id,
        segment_name=f"seg{segment_id}",
        n_samples=_N,
        uplift_range=(0.0, 1.0),
        cate_mean=cate_mean,
        cate_std=0.3,
        cate_se=cate_se,
        cate_ci_lower=ci_lower,
        cate_ci_upper=ci_upper,
        success=True,
        error_message=None,
    )


async def _run(monkeypatch: pytest.MonkeyPatch, segments: list) -> dict[str, Any]:
    async def fake_analyze(self, X, treatment, outcome, uplift_scores=None):  # noqa: ANN001
        return types.SimpleNamespace(
            success=True,
            segment_results=segments,
            overall_ate=0.5,
            overall_ate_ci_lower=0.3,
            overall_ate_ci_upper=0.7,
            segment_heterogeneity=10.0,
            n_segments=len(segments),
            errors=[],
            warnings=[],
        )

    monkeypatch.setattr(HierarchicalAnalyzer, "analyze", fake_analyze)
    node = HierarchicalAnalyzerNode(n_segments=2, estimator_type="ols", min_segment_size=10)
    X = pd.DataFrame({"age": np.arange(40, dtype=float)})
    return await node._run_hierarchical_analysis(
        X, np.arange(40) % 2, 100.0 + np.arange(40, dtype=float), None
    )


@pytest.mark.asyncio
async def test_segment_without_se_is_excluded_and_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await _run(
        monkeypatch,
        [
            _segment(0, cate_mean=0.5, cate_se=None, ci_lower=0.3, ci_upper=0.7),
            _segment(1, cate_mean=0.2, cate_se=0.05, ci_lower=0.1, ci_upper=0.3),
        ],
    )

    assert out["nested_ci"]["n_segments_included"] == 1
    assert out["nested_ci"]["aggregate_ate"] == pytest.approx(0.2)
    (entry,) = out["nested_ci_excluded_segments"]
    assert entry["segment_id"] == 0
    assert entry["segment_name"] == "seg0"
    assert entry["n"] == _N
    assert entry["reason"] == NO_MEASURED_UNCERTAINTY
    assert entry["detail"]
    assert any("seg0" in w for w in out["warnings"]), out.get("warnings")


@pytest.mark.asyncio
async def test_a_real_zero_ci_bound_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await _run(
        monkeypatch,
        [_segment(0, cate_mean=0.2, cate_se=0.1, ci_lower=0.0, ci_upper=0.4)],
    )

    # Single-segment passthrough: the aggregate CI IS the segment's CI.
    assert out["nested_ci"]["aggregate_ci_lower"] == 0.0
    assert out["nested_ci"]["aggregate_ci_upper"] == pytest.approx(0.4)
    assert out["nested_ci"]["aggregate_std"] == pytest.approx(0.1)
    assert out["nested_ci"]["n_segments_included"] == 1
    assert out["nested_ci_excluded_segments"] == []


@pytest.mark.asyncio
async def test_no_measured_segment_means_no_nested_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await _run(
        monkeypatch,
        [
            _segment(0, cate_mean=0.5, cate_se=None, ci_lower=0.3, ci_upper=0.7),
            _segment(1, cate_mean=0.2, cate_se=None, ci_lower=None, ci_upper=None),
        ],
    )

    assert out["nested_ci"] is None
    assert [e["segment_id"] for e in out["nested_ci_excluded_segments"]] == [0, 1]
    assert {e["reason"] for e in out["nested_ci_excluded_segments"]} == {NO_MEASURED_UNCERTAINTY}
    assert out["overall_hierarchical_ate"] == pytest.approx(0.5)
    assert len(out["hierarchical_segment_results"]) == 2


@pytest.mark.asyncio
async def test_segment_with_se_but_no_ci_is_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await _run(
        monkeypatch,
        [
            _segment(0, cate_mean=0.5, cate_se=0.05, ci_lower=None, ci_upper=0.7),
            _segment(1, cate_mean=0.2, cate_se=0.05, ci_lower=0.1, ci_upper=0.3),
        ],
    )

    assert out["nested_ci"]["n_segments_included"] == 1
    assert [e["segment_id"] for e in out["nested_ci_excluded_segments"]] == [0]


@pytest.mark.asyncio
async def test_failure_return_carries_an_empty_exclusion_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_analyze(self, X, treatment, outcome, uplift_scores=None):  # noqa: ANN001
        return types.SimpleNamespace(success=False, errors=["fit failed"])

    monkeypatch.setattr(HierarchicalAnalyzer, "analyze", failed_analyze)
    node = HierarchicalAnalyzerNode(n_segments=2, estimator_type="ols", min_segment_size=10)
    out = await node._run_hierarchical_analysis(
        pd.DataFrame({"age": np.arange(40, dtype=float)}),
        np.arange(40) % 2,
        100.0 + np.arange(40, dtype=float),
        None,
    )

    assert out["nested_ci"] is None
    assert out["nested_ci_excluded_segments"] == []
