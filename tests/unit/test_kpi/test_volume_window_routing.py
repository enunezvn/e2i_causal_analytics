"""Part 1: the business_impact VOLUME calculators route through the window helper.

Each of the 4 volume methods (_calc_trx / _calc_nrx / _calc_nbrx / _calc_roi),
when given a window in context, must send a `*_windowed` query_id with the
positional params [brand, start, end]. Synthetic flags are pinned OFF so the
windowed_query_id helper returns the bare (non-_include_synthetic) ids.
"""

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator


class _Resp:
    def __init__(self, data):
        self.data = data


class _Exec:
    def __init__(self, data):
        self._d = data

    def execute(self):
        return _Resp(self._d)


class _StubClient:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def rpc(self, name, payload):
        self.calls.append(payload)
        return _Exec([self.row])


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    """Pin synthetic flags off so windowed_query_id returns bare ids."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


_WINDOW = {"start": "S", "end": "E"}
_CONTEXT = {"brand": "Kisqali", "window": _WINDOW}


def test_trx_routes_windowed():
    client = _StubClient({"trx": 100})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_trx(_CONTEXT)
    assert value == 100.0
    assert len(client.calls) == 1
    assert client.calls[0]["query_id"] == "business_impact_trx_windowed"
    assert client.calls[0]["params"] == ["Kisqali", "S", "E"]


def test_nrx_routes_windowed():
    client = _StubClient({"nrx": 3394})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nrx(_CONTEXT)
    assert value == 3394.0
    assert len(client.calls) == 1
    assert client.calls[0]["query_id"] == "business_impact_nrx_windowed"
    assert client.calls[0]["params"] == ["Kisqali", "S", "E"]


def test_nbrx_routes_windowed():
    client = _StubClient({"nbrx": 50})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nbrx(_CONTEXT)
    assert value == 50.0
    assert len(client.calls) == 1
    assert client.calls[0]["query_id"] == "business_impact_nbrx_windowed"
    assert client.calls[0]["params"] == ["Kisqali", "S", "E"]


def test_roi_routes_windowed():
    client = _StubClient({"avg_roi": 1.5})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_roi(_CONTEXT)
    assert value == 1.5
    assert len(client.calls) == 1
    assert client.calls[0]["query_id"] == "business_impact_roi_windowed"
    assert client.calls[0]["params"] == ["Kisqali", "S", "E"]
