"""e2i_data_query_tool(kpi) answers TRx/NRx/NBRx/TRx Share through kpi_calculate's own call.

Task 15A (codex r1 HIGH, revised r2/r3). ``_query_kpis`` returned raw
business_metrics ROWS -- newest-first, limit-capped, no region sum, in-progress
month included -- while ``kpi_calculate_tool`` returned the canonical aggregate.
One KPI, two shapes.

That divergence used to be FENCED: pre-lane the canonical KPIs rested on
treatment_events, so ``cross_substrate_conflict`` fired on every stored-row answer.
Once this lane repoints them at business_metrics the two agree on SUBSTRATE and the
fence goes silent -- while the SHAPES still differ. Measured at both shas:
origin/main 4bcd37e76 FIRES, lane 0fe8e3f95 SILENT. So routing is what keeps the
answer honest; there is no longer a warning behind it.
"""

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from src.kpi import canonical_volume_stored as cvs


class _Calculator:
    """Records calculate() calls and answers like KPICalculator.calculate."""

    def __init__(self, result):
        self.result, self.calls = result, []

    def calculate(self, kpi_id, use_cache=True, force_refresh=False, context=None):
        self.calls.append((kpi_id, use_cache, force_refresh, dict(context or {})))
        return self.result


def _result(value=800349.18, error=None, include_synthetic=False):
    context = {} if error else {"data_month": "2026-08-01", "data_through": "2026-08-31"}
    return SimpleNamespace(
        value=None if error else value,
        error=error,
        metadata={"context": context, "include_synthetic": include_synthetic},
    )


def _run(kpi_name, filters, calculator, window_start="2026-06-01", limit=24):
    return asyncio.run(
        cvs.canonical_volume_stored_rows(
            kpi_name, filters, window_start, limit, calculator=calculator
        )
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TRx", "WS3-BI-005"),
        ("trx", "WS3-BI-005"),
        ("total prescriptions", "WS3-BI-005"),
        ("NRx", "WS3-BI-006"),
        ("NBRx", "WS3-BI-007"),
        ("TRx share", "WS3-BI-008"),
        ("market share", None),
        ("market_share", None),
        ("conversion rate", None),
        ("TRx panel", None),
        (None, None),
    ],
)
def test_only_the_canonical_volume_family_is_routed(name, expected):
    assert cvs.canonical_volume_kpi_id(name) == expected


def test_the_stored_only_guard_is_load_bearing_not_decorative():
    """``market share`` RESOLVES to WS3-BI-008, so only the explicit guard stops it.

    Measured 2026-09-17: ``recognize_kpi("market share") -> WS3-BI-008``. Without
    ``_STORED_ONLY_KEYS`` the vocabulary would route a request for the stored
    modeled brand market share to the canonical TRx-Share aggregate -- a different
    quantity. Pinning the premise here means this test cannot pass by describing a
    resolver that no longer behaves that way.
    """
    from src.services.kpi_resolution import recognize_kpi

    assert getattr(recognize_kpi("market share"), "id", None) == "WS3-BI-008", (
        "premise changed: 'market share' no longer resolves to the TRx Share KPI, so "
        "re-derive whether _STORED_ONLY_KEYS is still needed"
    )
    assert cvs.canonical_volume_kpi_id("market share") is None


def test_the_calculation_is_kpi_calculates_own_call_with_the_same_context():
    calc = _Calculator(_result())
    out = _run("TRx", {"brand": "Kisqali", "metric_name": "trx"}, calc)
    # kpi_calculate_tool: calculator.calculate(kpi.id, context=context)
    assert calc.calls == [("WS3-BI-005", True, False, {"brand": "Kisqali"})]
    assert out["success"] is True
    assert out["canonical_aggregate"] is True
    assert out["count"] == 1
    assert out["data"] == [
        {
            "metric_date": "2026-08-01",
            "metric_name": "trx",
            "kpi_id": "WS3-BI-005",
            "brand": "Kisqali",
            "region": "",
            "value": 800349.18,
            "data_through": "2026-08-31",
        }
    ]
    assert out["data_month"] == "2026-08-01"
    assert out["data_through"] == "2026-08-31"
    assert out["measure_basis"]["substrate"] == ["business_metrics"]
    assert out["cross_substrate_conflict"] is None


def test_a_named_region_rides_the_same_context():
    calc = _Calculator(_result(value=1234.5))
    out = _run("NRx", {"brand": "Kisqali", "region": "midwest", "metric_name": "nrx"}, calc)
    assert calc.calls[0][0] == "WS3-BI-006"
    assert calc.calls[0][3] == {"brand": "Kisqali", "region": "midwest"}
    assert out["data"][0]["region"] == "midwest"
    assert out["data"][0]["value"] == 1234.5


@pytest.mark.parametrize(
    "window_start,served",
    # the August headline became complete on 2026-09-01
    [
        ("2013-01-01", True),
        ("2026-08-16", True),
        ("2026-09-01", True),
        ("2026-09-02", False),
        ("2026-09-08", False),
    ],
)
def test_the_lookback_is_honoured_and_never_becomes_a_window(window_start, served):
    """time_range gates eligibility; it never selects the month."""
    calc = _Calculator(_result())
    out = asyncio.run(
        cvs.canonical_volume_stored_rows(
            "TRx", {"brand": "Kisqali"}, window_start, 24, calculator=calc, today=date(2026, 9, 15)
        )
    )
    assert out["success"] is served
    if served:
        assert out["count"] == 1
        assert out["data"][0]["metric_date"] == "2026-08-01"
        assert "out_of_range" not in out
    else:
        assert out["count"] == 0
        assert out["data"] == []
        # A refusal must name the frontier and a destination that can serve it.
        assert out["out_of_range"]["frontier_month"] == "2026-08-01"
        assert out["out_of_range"]["frontier_complete_on"] == "2026-09-01"
        assert out["out_of_range"]["requested_since"] == window_start
        assert out["out_of_range"]["use_instead"] == "kpi_calculate_tool(window=...)"
        assert "kpi_calculate_tool" in out["error"]


def test_the_refusal_never_carries_a_value():
    """An out-of-range answer with a value would be the two-shapes bug again."""
    calc = _Calculator(_result())
    out = asyncio.run(
        cvs.canonical_volume_stored_rows(
            "TRx", {"brand": "Kisqali"}, "2026-09-08", 24, calculator=calc, today=date(2026, 9, 15)
        )
    )
    assert out["data"] == []
    assert out["data_month"] is None
    assert out["data_through"] is None


def test_a_calculator_error_is_reported_not_swallowed():
    calc = _Calculator(_result(error="no complete month of business_metrics trx rows"))
    out = _run("TRx", {"brand": "Nonesuch"}, calc)
    assert out["success"] is False
    assert out["count"] == 0
    assert out["data"] == []
    assert "no complete month" in out["error"]
    assert "out_of_range" not in out


def test_synthetic_provenance_follows_the_calculation():
    calc = _Calculator(_result(include_synthetic=True))
    out = _run("TRx", {"brand": "Kisqali"}, calc)
    assert out["data_source"] == "synthetic"
    calc = _Calculator(_result(include_synthetic=False))
    assert _run("TRx", {"brand": "Kisqali"}, calc)["data_source"] == "database"


def test_a_non_family_kpi_name_is_not_routed_at_all():
    """None means 'not mine' -- the caller must fall through to the stored-row path."""
    calc = _Calculator(_result())
    assert _run("conversion rate", {"brand": "Kisqali"}, calc) is None
    assert calc.calls == [], "the calculator must not be called for a non-family KPI"
