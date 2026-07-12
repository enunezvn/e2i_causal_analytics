"""_resolve_windowed_call routing matrix for the segment/line axes (migration 105).

Mirrors test_windowed_call.py's style (pure-helper tests, no DB). Covers:
- segment -> base `_segment` id, params [brand, segment]
- segment + window -> `_segment_windowed`, params [brand, segment, start, end]
- therapy_line -> base `_line` id, params [brand, therapy_line] (incl. line 0)
- therapy_line + window -> `_line_windowed`, params [brand, therapy_line, start, end]
- precedence: segment wins over region; segment wins over therapy_line
- share never windowed is covered separately in
  test_segment_line_volume_routing.py (it exercises _calc_trx_share directly).
"""

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator


def _calc():
    return BusinessImpactCalculator(db_client=None)  # helper is pure; no DB needed


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    """Pin synthetic flags off so the resolved ids are the bare (non-twin) forms."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")


_WINDOW = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}


def test_segment_base():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx", brand="Kisqali", region=None, window=None, segment="high_severity"
    )
    assert qid == "business_impact_trx_segment"
    assert params == ["Kisqali", "high_severity"]


def test_segment_windowed():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Kisqali",
        region=None,
        window=_WINDOW,
        segment="high_severity",
    )
    assert qid == "business_impact_trx_segment_windowed"
    assert params == ["Kisqali", "high_severity", _WINDOW["start"], _WINDOW["end"]]


def test_therapy_line_base():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region=None, window=None, therapy_line=2
    )
    assert qid == "business_impact_nrx_line"
    assert params == ["Kisqali", 2]


def test_therapy_line_windowed():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx",
        brand="Kisqali",
        region=None,
        window=_WINDOW,
        therapy_line=2,
    )
    assert qid == "business_impact_nrx_line_windowed"
    assert params == ["Kisqali", 2, _WINDOW["start"], _WINDOW["end"]]


def test_therapy_line_zero_is_not_dropped():
    """Line 0 is a real, commonly-populated bucket (462 patients in the
    validated ground truth). `is not None` (not truthiness) must route it,
    not silently fall through to the base/region branch."""
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nbrx", brand="Kisqali", region=None, window=None, therapy_line=0
    )
    assert qid == "business_impact_nbrx_line"
    assert params == ["Kisqali", 0]


def test_segment_takes_precedence_over_region():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Kisqali",
        region="northeast",
        window=None,
        segment="low_severity",
    )
    assert qid == "business_impact_trx_segment"
    assert params == ["Kisqali", "low_severity"]


def test_segment_takes_precedence_over_therapy_line():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Kisqali",
        region=None,
        window=None,
        segment="medium_severity",
        therapy_line=3,
    )
    assert qid == "business_impact_trx_segment"
    assert params == ["Kisqali", "medium_severity"]


def test_therapy_line_takes_precedence_over_region():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx", brand="Kisqali", region="south", window=None, therapy_line=1
    )
    assert qid == "business_impact_trx_line"
    assert params == ["Kisqali", 1]


def test_no_axis_falls_back_to_existing_region_behavior():
    """Byte-identical to the pre-105 behavior when no axis is supplied."""
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx", brand="Kisqali", region="northeast", window=None
    )
    assert qid == "business_impact_trx_region"
    assert params == ["Kisqali", "northeast"]


def test_no_axis_no_region_no_window_unchanged():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx", brand="Kisqali", region=None, window=None
    )
    assert qid == "business_impact_trx"
    assert params == ["Kisqali"]
