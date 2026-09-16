"""#2142 (agent path): the analyzer's "overall CI withheld" warning reaches the node output.

``HierarchicalAnalyzer.analyze`` withholds ``overall_ate_ci_*`` when a successful
segment has no measured uncertainty and says so in ``result.warnings``. The node
emitted only its own nested-CI exclusion lines, so on the agent path
``overall_hierarchical_ci_*`` went None with no word about the overall line.

Real node + real ``analyze`` (segmentation, aggregation, warning); only the
per-segment EconML fit is replaced by planted segment results, and uplift scores
are passed in so no CausalML fit runs.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
    HierarchicalAnalyzerNode,
)
from src.causal_engine.hierarchical import HierarchicalAnalyzer
from src.causal_engine.hierarchical.analyzer import SegmentResult


def _planted(i: int, se: float | None) -> SegmentResult:
    mean = 0.1 + 0.2 * i
    return SegmentResult(
        segment_id=i,
        segment_name=f"segment_{i}",
        n_samples=20,
        uplift_range=(0.0, 1.0),
        cate_mean=mean,
        cate_std=0.3,
        cate_se=se,
        cate_ci_lower=mean - 0.04,
        cate_ci_upper=mean + 0.04,
        success=True,
    )


async def _run(monkeypatch: pytest.MonkeyPatch, planted: List[SegmentResult]) -> dict[str, Any]:
    async def planted_segments(self, *_args, **_kwargs):  # noqa: ANN001
        return planted

    monkeypatch.setattr(HierarchicalAnalyzer, "_compute_segment_cate", planted_segments)
    node = HierarchicalAnalyzerNode(n_segments=2, estimator_type="linear_dml", min_segment_size=10)
    X = pd.DataFrame({"age": np.arange(40, dtype=float)})
    return await node._run_hierarchical_analysis(
        X, np.arange(40) % 2, np.arange(40, dtype=float), np.linspace(0.0, 1.0, 40)
    )


@pytest.mark.asyncio
async def test_withheld_overall_ci_is_named_in_node_warnings(monkeypatch: pytest.MonkeyPatch):
    out = await _run(monkeypatch, [_planted(0, 0.02), _planted(1, None)])

    assert out["overall_hierarchical_ate"] == pytest.approx(0.2)
    assert out["overall_hierarchical_ci_lower"] is None
    assert out["overall_hierarchical_ci_upper"] is None
    withheld = [w for w in out.get("warnings", []) if "overall CI withheld" in w]
    assert len(withheld) == 1 and "segment_1" in withheld[0]


@pytest.mark.asyncio
async def test_measured_segments_serve_the_overall_ci_without_warnings(
    monkeypatch: pytest.MonkeyPatch,
):
    out = await _run(monkeypatch, [_planted(0, 0.02), _planted(1, 0.02)])

    assert out["overall_hierarchical_ci_lower"] < out["overall_hierarchical_ate"]
    assert out["overall_hierarchical_ate"] < out["overall_hierarchical_ci_upper"]
    assert not out.get("warnings")
