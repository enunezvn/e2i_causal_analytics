# tests/unit/test_kpi/test_synthetic_mode_segment_line.py
"""segment_query_id / line_query_id / windowed_axis_query_id (migration 105).

Mirrors test_synthetic_mode_windowed.py's reload pattern: the synthetic flag
is read fresh at call time via os.getenv, but importlib.reload keeps parity
with the sibling test file's idiom.
"""

import importlib

import src.kpi.synthetic_mode as sm


def _reload(monkeypatch, flag):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1" if flag else "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    return importlib.reload(sm)


# --- segment_query_id ------------------------------------------------------


def test_segment_query_id_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.segment_query_id("business_impact_trx") == "business_impact_trx_segment"


def test_segment_query_id_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.segment_query_id("business_impact_trx") == "business_impact_trx_segment_include_synthetic"
    )


# --- line_query_id -----------------------------------------------------


def test_line_query_id_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.line_query_id("business_impact_nrx") == "business_impact_nrx_line"


def test_line_query_id_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert m.line_query_id("business_impact_nrx") == "business_impact_nrx_line_include_synthetic"


# --- windowed_axis_query_id ----------------------------------------------


def test_windowed_axis_segment_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert (
        m.windowed_axis_query_id("business_impact_trx", axis="segment")
        == "business_impact_trx_segment_windowed"
    )


def test_windowed_axis_segment_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.windowed_axis_query_id("business_impact_trx", axis="segment")
        == "business_impact_trx_segment_windowed_include_synthetic"
    )


def test_windowed_axis_line_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert (
        m.windowed_axis_query_id("business_impact_nbrx", axis="line")
        == "business_impact_nbrx_line_windowed"
    )


def test_windowed_axis_line_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.windowed_axis_query_id("business_impact_nbrx", axis="line")
        == "business_impact_nbrx_line_windowed_include_synthetic"
    )


# --- absence from the twin drift-lock set --------------------------------


def test_segment_and_line_ids_not_in_synthetic_twinned_set(monkeypatch):
    """Additive axis variants (migration 105) are absent from
    SYNTHETIC_TWINNED_QUERY_IDS, same as `_region` / `_windowed` -- the
    _include_synthetic suffix is appended by segment_query_id / line_query_id
    / windowed_axis_query_id directly, not by resolve_kpi_query_id."""
    m = _reload(monkeypatch, False)
    assert "business_impact_trx_segment" not in m.SYNTHETIC_TWINNED_QUERY_IDS
    assert "business_impact_nrx_line" not in m.SYNTHETIC_TWINNED_QUERY_IDS
