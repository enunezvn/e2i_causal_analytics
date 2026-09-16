"""#2027 part B (API path): the nested CI fails closed on unmeasured segments.

``POST /api/causal/hierarchical/analyze`` bridged every successful segment into
``SegmentEstimate`` with invented uncertainty when the analyzer had none —
``ate_std = cate_std or 0.01``, ``ci = cate_mean ± 0.1`` — and the ``or`` also
swallowed a real 0.0 bound. An SE of 0.01 hands an unmeasured segment a large
inverse-variance weight; ±0.1 reads as a real CI on a rate outcome.

Contract: a segment enters the nested aggregate only with a measured SE AND both CI
bounds (``is not None`` — a real 0.0 is kept); the rest are listed under
``nested_ci_excluded_segments`` with a reason code and named in ``warnings``; when
no segment remains there is no nested CI at all (``NestedConfidenceInterval.compute``
would return ±inf on zero segments).

Harness as in ``test_hierarchical_defab.py``: real route + real
``NestedConfidenceInterval.compute``; only ``HierarchicalAnalyzer.analyze`` (the
heavy EconML-within-segments fit) is stood in for, returning segments with the exact
uncertainty fields under test.
"""

from __future__ import annotations

import types
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.causal_engine.hierarchical import HierarchicalAnalyzer

pytestmark = pytest.mark.integration

# The wire-level reason code (a literal on purpose: it is the contract consumers
# branch on, not the name of the constant that emits it).
NO_MEASURED_UNCERTAINTY = "no_measured_uncertainty"

_N = 100  # above the request's min_segment_size so compute's size filter keeps it


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


def _post(monkeypatch: pytest.MonkeyPatch, segments: list) -> dict[str, Any]:
    async def fake_analyze(self, X, treatment, outcome):  # noqa: ANN001
        return types.SimpleNamespace(
            segment_results=segments,
            overall_ate=0.5,
            overall_ate_ci_lower=0.3,
            overall_ate_ci_upper=0.7,
            segment_heterogeneity=10.0,
            n_segments=len(segments),
            errors=[],
            warnings=[],
            success=True,
        )

    monkeypatch.setattr(HierarchicalAnalyzer, "analyze", fake_analyze)
    records = [{"promotion": i % 2, "trx": 100.0 + i, "age": 30 + i} for i in range(40)]
    response = TestClient(app).post(
        "/api/causal/hierarchical/analyze",
        json={
            "treatment_var": "promotion",
            "outcome_var": "trx",
            "effect_modifiers": ["age"],
            "n_segments": 2,
            "min_segment_size": 10,
            "filters": {"estimation_data_records": records},
        },
    )
    assert response.status_code == 200, response.text[:300]
    return response.json()


def test_segment_without_se_is_excluded_and_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _post(
        monkeypatch,
        [
            _segment(0, cate_mean=0.5, cate_se=None, ci_lower=0.3, ci_upper=0.7),
            _segment(1, cate_mean=0.2, cate_se=0.05, ci_lower=0.1, ci_upper=0.3),
        ],
    )

    assert data["nested_ci"]["n_segments_included"] == 1
    assert data["nested_ci"]["aggregate_ate"] == pytest.approx(0.2)
    assert data["nested_ci_excluded_segments"] == [
        {
            "segment_id": 0,
            "segment_name": "seg0",
            "n": _N,
            "reason": NO_MEASURED_UNCERTAINTY,
            "detail": data["nested_ci_excluded_segments"][0]["detail"],
        }
    ]
    assert data["nested_ci_excluded_segments"][0]["detail"]
    assert any("seg0" in w for w in data["warnings"]), data["warnings"]


def test_a_real_zero_ci_bound_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _post(
        monkeypatch,
        [_segment(0, cate_mean=0.2, cate_se=0.1, ci_lower=0.0, ci_upper=0.4)],
    )

    # Single-segment passthrough: the aggregate CI IS the segment's CI, so a
    # lower bound of exactly 0.0 proves the bridge did not treat 0.0 as missing.
    assert data["nested_ci"]["aggregate_ci_lower"] == 0.0
    assert data["nested_ci"]["aggregate_ci_upper"] == pytest.approx(0.4)
    assert data["nested_ci"]["aggregate_std"] == pytest.approx(0.1)
    assert data["nested_ci"]["n_segments_included"] == 1
    assert data["nested_ci_excluded_segments"] == []


def test_no_measured_segment_means_no_nested_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _post(
        monkeypatch,
        [
            _segment(0, cate_mean=0.5, cate_se=None, ci_lower=0.3, ci_upper=0.7),
            _segment(1, cate_mean=0.2, cate_se=None, ci_lower=None, ci_upper=None),
        ],
    )

    assert data["nested_ci"] is None
    assert [e["segment_id"] for e in data["nested_ci_excluded_segments"]] == [0, 1]
    assert {e["reason"] for e in data["nested_ci_excluded_segments"]} == {NO_MEASURED_UNCERTAINTY}
    assert data["overall_ate"] == pytest.approx(0.5)
    assert data["status"] == "completed"


def test_segment_with_se_but_no_ci_is_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _post(
        monkeypatch,
        [
            _segment(0, cate_mean=0.5, cate_se=0.05, ci_lower=None, ci_upper=0.7),
            _segment(1, cate_mean=0.2, cate_se=0.05, ci_lower=0.1, ci_upper=0.3),
        ],
    )

    assert data["nested_ci"]["n_segments_included"] == 1
    assert [e["segment_id"] for e in data["nested_ci_excluded_segments"]] == [0]
