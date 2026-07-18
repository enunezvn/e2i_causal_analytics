"""The business_impact VOLUME + share calculators thread segment/therapy_line
into the resolved query (migration 105).

Mirrors test_volume_window_routing.py's `_StubClient` pattern exactly, one
axis level up: NRx/TRx/NBRx/TRx-Share (_calc_nrx/_calc_trx/_calc_nbrx/
_calc_trx_share), when given `segment` or `therapy_line` in context, must
send the `*_segment` / `*_line` query_id with params [brand, axis_value].
Synthetic flags are pinned OFF so the resolved ids are the bare forms.
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
    """Pin synthetic flags off so the resolved query_ids are the bare (non-twin) forms."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


_SEGMENT_CONTEXT = {"brand": "Remibrutinib", "segment": "high_severity"}
_LINE_CONTEXT = {"brand": "Remibrutinib", "therapy_line": 2}
_LINE_ZERO_CONTEXT = {"brand": "Remibrutinib", "therapy_line": 0}


def test_trx_threads_segment():
    client = _StubClient({"trx": 3271})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_trx(_SEGMENT_CONTEXT)
    assert value == 3271.0
    assert client.calls[0]["query_id"] == "business_impact_trx_segment"
    assert client.calls[0]["params"] == ["Remibrutinib", "high_severity"]


def test_nrx_threads_segment():
    client = _StubClient({"nrx": 490})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nrx(_SEGMENT_CONTEXT)
    assert value == 490.0
    assert client.calls[0]["query_id"] == "business_impact_nrx_segment"
    assert client.calls[0]["params"] == ["Remibrutinib", "high_severity"]


def test_nbrx_threads_segment():
    client = _StubClient({"nbrx": 12})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nbrx(_SEGMENT_CONTEXT)
    assert value == 12.0
    assert client.calls[0]["query_id"] == "business_impact_nbrx_segment"
    assert client.calls[0]["params"] == ["Remibrutinib", "high_severity"]


def test_trx_threads_therapy_line():
    client = _StubClient({"trx": 500})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_trx(_LINE_CONTEXT)
    assert value == 500.0
    assert client.calls[0]["query_id"] == "business_impact_trx_line"
    assert client.calls[0]["params"] == ["Remibrutinib", 2]


def test_nrx_threads_therapy_line():
    client = _StubClient({"nrx": 467})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nrx(_LINE_CONTEXT)
    assert value == 467.0
    assert client.calls[0]["query_id"] == "business_impact_nrx_line"
    assert client.calls[0]["params"] == ["Remibrutinib", 2]


def test_nbrx_threads_therapy_line():
    client = _StubClient({"nbrx": 9})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nbrx(_LINE_CONTEXT)
    assert value == 9.0
    assert client.calls[0]["query_id"] == "business_impact_nbrx_line"
    assert client.calls[0]["params"] == ["Remibrutinib", 2]


def test_nrx_threads_therapy_line_zero():
    """Line 0 (462 patients in the validated ground truth) must not be
    dropped by a truthiness check."""
    client = _StubClient({"nrx": 462})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nrx(_LINE_ZERO_CONTEXT)
    assert value == 462.0
    assert client.calls[0]["query_id"] == "business_impact_nrx_line"
    assert client.calls[0]["params"] == ["Remibrutinib", 0]


def test_trx_share_threads_segment_windowed():
    """Migration 111 registered windowed share variants: segment + window now
    routes to `_segment_windowed` with [brand, segment, start, end] (before
    111 the window was pinned off and silently dropped)."""
    client = _StubClient({"share": 0.42})
    calc = BusinessImpactCalculator(db_client=client)
    context = {
        "brand": "Remibrutinib",
        "segment": "high_severity",
        "window": {"start": "S", "end": "E"},
    }
    value = calc._calc_trx_share(context)
    assert value == 0.42
    assert client.calls[0]["query_id"] == "business_impact_trx_share_segment_windowed"
    assert client.calls[0]["params"] == ["Remibrutinib", "high_severity", "S", "E"]


def test_trx_share_threads_therapy_line_windowed():
    client = _StubClient({"share": 0.31})
    calc = BusinessImpactCalculator(db_client=client)
    context = {
        "brand": "Remibrutinib",
        "therapy_line": 3,
        "window": {"start": "S", "end": "E"},
    }
    value = calc._calc_trx_share(context)
    assert value == 0.31
    assert client.calls[0]["query_id"] == "business_impact_trx_share_line_windowed"
    assert client.calls[0]["params"] == ["Remibrutinib", 3, "S", "E"]


def test_trx_share_segment_without_window_stays_base_axis():
    """No window in context -> the frontier-anchored `_segment` id (105) as before."""
    client = _StubClient({"share": 0.42})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_trx_share({"brand": "Remibrutinib", "segment": "high_severity"})
    assert value == 0.42
    assert client.calls[0]["query_id"] == "business_impact_trx_share_segment"
    assert client.calls[0]["params"] == ["Remibrutinib", "high_severity"]


def test_trx_share_windowed_plain():
    client = _StubClient({"share": 0.3338})
    calc = BusinessImpactCalculator(db_client=client)
    context = {"brand": "Remibrutinib", "window": {"start": "S", "end": "E"}}
    value = calc._calc_trx_share(context)
    assert value == 0.3338
    assert client.calls[0]["query_id"] == "business_impact_trx_share_windowed"
    assert client.calls[0]["params"] == ["Remibrutinib", "S", "E"]


def test_trx_share_window_plus_region_fails_loud():
    """No windowed-region share variant is registered; dropping either filter
    silently would misrepresent the figure, so the combination must raise."""
    calc = BusinessImpactCalculator(db_client=_StubClient({"share": 0.5}))
    with pytest.raises(RuntimeError, match="segment.*or line-of-therapy"):
        calc._calc_trx_share(
            {"brand": "Remibrutinib", "region": "northeast", "window": {"start": "S", "end": "E"}}
        )


def test_trx_share_still_requires_brand():
    calc = BusinessImpactCalculator(db_client=_StubClient({"share": 0.5}))
    with pytest.raises(RuntimeError, match="no brand specified"):
        calc._calc_trx_share({"segment": "high_severity"})
