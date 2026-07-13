# tests/unit/test_kpi/test_synthetic_mode_biologic_ige.py
"""biologic_query_id / ige_tier_query_id / windowed_axis_query_id (migration 108).

Mirrors test_synthetic_mode_segment_line.py's reload pattern: the synthetic flag
is read fresh at call time via os.getenv, but importlib.reload keeps parity with
the sibling test file's idiom.
"""

import importlib

import src.kpi.synthetic_mode as sm


def _reload(monkeypatch, flag):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1" if flag else "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    return importlib.reload(sm)


# --- biologic_query_id -----------------------------------------------------


def test_biologic_query_id_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.biologic_query_id("business_impact_nrx") == "business_impact_nrx_biologic"


def test_biologic_query_id_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.biologic_query_id("business_impact_nrx")
        == "business_impact_nrx_biologic_include_synthetic"
    )


# --- ige_tier_query_id -----------------------------------------------------


def test_ige_tier_query_id_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.ige_tier_query_id("business_impact_trx") == "business_impact_trx_ige_tier"


def test_ige_tier_query_id_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.ige_tier_query_id("business_impact_trx")
        == "business_impact_trx_ige_tier_include_synthetic"
    )


# --- windowed_axis_query_id (biologic / ige_tier) --------------------------


def test_windowed_axis_biologic_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert (
        m.windowed_axis_query_id("business_impact_nrx", axis="biologic")
        == "business_impact_nrx_biologic_windowed"
    )


def test_windowed_axis_biologic_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.windowed_axis_query_id("business_impact_nrx", axis="biologic")
        == "business_impact_nrx_biologic_windowed_include_synthetic"
    )


def test_windowed_axis_ige_tier_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert (
        m.windowed_axis_query_id("business_impact_nbrx", axis="ige_tier")
        == "business_impact_nbrx_ige_tier_windowed"
    )


def test_windowed_axis_ige_tier_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert (
        m.windowed_axis_query_id("business_impact_nbrx", axis="ige_tier")
        == "business_impact_nbrx_ige_tier_windowed_include_synthetic"
    )


# --- absence from the twin drift-lock set --------------------------------


def test_biologic_and_ige_ids_not_in_synthetic_twinned_set(monkeypatch):
    """Additive axis variants (migration 108) are absent from
    SYNTHETIC_TWINNED_QUERY_IDS, same as `_segment` / `_line` -- the
    _include_synthetic suffix is appended by the axis helpers directly, not by
    resolve_kpi_query_id."""
    m = _reload(monkeypatch, False)
    assert "business_impact_nrx_biologic" not in m.SYNTHETIC_TWINNED_QUERY_IDS
    assert "business_impact_trx_ige_tier" not in m.SYNTHETIC_TWINNED_QUERY_IDS
