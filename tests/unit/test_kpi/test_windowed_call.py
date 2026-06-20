import pytest
from src.kpi.calculators.business_impact import BusinessImpactCalculator


def _calc():
    return BusinessImpactCalculator(db_client=None)  # helper is pure; no DB needed


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    """Pin synthetic flags off so windowed_query_id returns bare ids."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")


def test_no_window_brand_only():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region=None, window=None)
    assert qid == "business_impact_nrx"
    assert params == ["Kisqali"]


def test_window_brand():
    w = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region=None, window=w)
    assert qid == "business_impact_nrx_windowed"
    assert params == ["Kisqali", w["start"], w["end"]]


def test_window_region():
    w = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="Kisqali", region="northeast", window=w)
    assert qid == "business_impact_nrx_windowed_region"
    assert params == ["Kisqali", "northeast", w["start"], w["end"]]
