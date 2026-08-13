"""Conversion Rate (WS3-BI-009) brand/axis/window routing (migration 111).

Before 111, `_calc_conversion_rate` honored ONLY `region` and silently dropped
brand/segment/therapy_line/window — a "Remibrutinib high-severity over the
last year" ask returned the overall portfolio figure while the tool response
echoed the brand, presenting a brand-agnostic number as a brand figure
(session_1784387374342). These tests pin the full routing table plus the
fail-loud paths for combinations with no registered variant.

Mirrors test_segment_line_volume_routing.py's `_StubClient` pattern; synthetic
flags pinned off so resolved ids are the bare forms.
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


_WINDOW = {"start": "2025-07-14", "end": "2026-07-14"}


def _calc(row):
    client = _StubClient(row)
    return BusinessImpactCalculator(db_client=client), client


def test_no_filters_uses_certified_base():
    calc, client = _calc({"conversion_rate": 0.639})
    assert calc._calc_conversion_rate({}) == 0.639
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate"
    assert client.calls[0]["params"] == []


def test_brand_only_routes_to_brand_variant():
    calc, client = _calc({"conversion_rate": 0.6289})
    assert calc._calc_conversion_rate({"brand": "Remibrutinib"}) == 0.6289
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_brand"
    assert client.calls[0]["params"] == ["Remibrutinib"]


def test_segment_routes_to_segment_variant():
    calc, client = _calc({"conversion_rate": 0.728})
    value = calc._calc_conversion_rate({"brand": "Remibrutinib", "segment": "high_severity"})
    assert value == 0.728
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_segment"
    assert client.calls[0]["params"] == ["Remibrutinib", "high_severity"]


def test_segment_windowed_routes_with_four_params():
    calc, client = _calc({"conversion_rate": 0.6779})
    value = calc._calc_conversion_rate(
        {"brand": "Remibrutinib", "segment": "high_severity", "window": _WINDOW}
    )
    assert value == 0.6779
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_segment_windowed"
    assert client.calls[0]["params"] == [
        "Remibrutinib",
        "high_severity",
        _WINDOW["start"],
        _WINDOW["end"],
    ]


def test_therapy_line_zero_not_dropped():
    """Line 0 is a real bucket — must survive an `is not None` check, not truthiness."""
    calc, client = _calc({"conversion_rate": 0.55})
    value = calc._calc_conversion_rate({"brand": "Remibrutinib", "therapy_line": 0})
    assert value == 0.55
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_line"
    assert client.calls[0]["params"] == ["Remibrutinib", 0]


def test_line_windowed_routes_with_four_params():
    calc, client = _calc({"conversion_rate": 0.5996})
    value = calc._calc_conversion_rate(
        {"brand": "Remibrutinib", "therapy_line": 2, "window": _WINDOW}
    )
    assert value == 0.5996
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_line_windowed"
    assert client.calls[0]["params"] == ["Remibrutinib", 2, _WINDOW["start"], _WINDOW["end"]]


def test_window_only_routes_to_windowed_with_null_brand():
    calc, client = _calc({"conversion_rate": 0.61})
    assert calc._calc_conversion_rate({"window": _WINDOW}) == 0.61
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_windowed"
    assert client.calls[0]["params"] == [None, _WINDOW["start"], _WINDOW["end"]]


def test_region_alone_keeps_legacy_region_variant():
    calc, client = _calc({"conversion_rate": 0.64})
    assert calc._calc_conversion_rate({"region": "northeast"}) == 0.64
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_region"
    assert client.calls[0]["params"] == ["northeast"]


def test_segment_takes_precedence_over_region():
    """Axis precedence mirrors _resolve_windowed_call: segment > region."""
    calc, client = _calc({"conversion_rate": 0.7})
    calc._calc_conversion_rate(
        {"brand": "Remibrutinib", "segment": "low_severity", "region": "northeast"}
    )
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_segment"


def test_biologic_axis_fails_loud():
    """Triggers carry no biologic/IgE dimension — never silently drop the filter."""
    calc, _ = _calc({"conversion_rate": 0.6})
    with pytest.raises(RuntimeError, match="biologic"):
        calc._calc_conversion_rate({"brand": "Remibrutinib", "biologic": "naive"})


def test_ige_axis_fails_loud():
    calc, _ = _calc({"conversion_rate": 0.6})
    with pytest.raises(RuntimeError, match="IgE"):
        calc._calc_conversion_rate({"brand": "Remibrutinib", "ige_tier": "high"})


def test_region_plus_window_fails_loud():
    calc, _ = _calc({"conversion_rate": 0.6})
    with pytest.raises(RuntimeError, match="region"):
        calc._calc_conversion_rate({"region": "northeast", "window": _WINDOW})


def test_brand_plus_region_routes_to_brand_region_variant():
    """#1575: brand+region routes to the migration-128 joint leg — before it,
    this combination raised ('brand and region cannot be combined') and the
    chat layer answered 'KPI unavailable' for e.g. Kisqali-in-the-west."""
    calc, client = _calc({"conversion_rate": 0.612})
    context = {"brand": "Kisqali", "region": "west"}
    assert calc._calc_conversion_rate(context) == 0.612
    assert client.calls[0]["query_id"] == "business_impact_conversion_rate_brand_region"
    assert client.calls[0]["params"] == ["Kisqali", "west"]


def test_brand_plus_region_sets_region_provenance_marker():
    """#1538: the region echo comes FROM the routing marker, never the raw arg —
    the new joint path must attest region application like the region-only path."""
    calc, _ = _calc({"conversion_rate": 0.612})
    context = {"brand": "Kisqali", "region": "west"}
    calc._calc_conversion_rate(context)
    assert context.get("_region_routed") is True


def test_brand_plus_region_plus_window_still_fails_loud():
    """#1575 keeps the honest-failure note for combos still genuinely unserved:
    no windowed brand+region conversion variant is registered."""
    calc, _ = _calc({"conversion_rate": 0.6})
    with pytest.raises(RuntimeError, match="region"):
        calc._calc_conversion_rate({"brand": "Kisqali", "region": "west", "window": _WINDOW})
