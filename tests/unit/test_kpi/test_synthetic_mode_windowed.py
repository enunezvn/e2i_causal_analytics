# tests/unit/test_kpi/test_synthetic_mode_windowed.py
import importlib, src.kpi.synthetic_mode as sm

def _reload(monkeypatch, flag):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1" if flag else "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    return importlib.reload(sm)

def test_windowed_base(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.windowed_query_id("business_impact_nrx", region=False) == "business_impact_nrx_windowed"

def test_windowed_region(monkeypatch):
    m = _reload(monkeypatch, False)
    assert m.windowed_query_id("business_impact_nrx", region=True) == "business_impact_nrx_windowed_region"

def test_windowed_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert m.windowed_query_id("business_impact_nrx", region=False) == "business_impact_nrx_windowed_include_synthetic"

def test_windowed_region_synthetic(monkeypatch):
    m = _reload(monkeypatch, True)
    assert m.windowed_query_id("business_impact_nrx", region=True) == "business_impact_nrx_windowed_region_include_synthetic"
