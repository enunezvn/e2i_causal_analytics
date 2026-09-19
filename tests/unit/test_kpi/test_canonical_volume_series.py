"""The read API a forecaster calls: brand (optionally region) monthly TRx/NRx/NBRx,
complete months only, with its data-through date (canonical TRx lane)."""

import asyncio
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.kpi import canonical_volume_series as cvs

ROWS = [
    {"metric_date": "2026-07-01", "value": 801104.0, "n_rows": 4},
    {"metric_date": "2026-06-01", "value": 784777.0, "n_rows": 4},
    {"metric_date": "2026-08-01", "value": 800349.18, "n_rows": 4},
    {"metric_date": "2026-05-01", "value": 500000.0, "n_rows": 3},  # a region missing
    {"metric_date": "2026-09-01", "value": 939465.61, "n_rows": 4},  # in progress
]


def _shape(rows=ROWS, **kw):
    params = {
        "metric": "trx",
        "brand": "Kisqali",
        "region": None,
        "as_of": date(2026, 9, 15),
        "query_id": "q",
    }
    params.update(kw)
    return cvs.shape_monthly_series(rows, **params)


def test_points_are_sorted_complete_months_before_the_in_progress_month():
    s = _shape()
    assert [p.month for p in s.points] == [date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)]
    assert s.points[-1].value == 800349.18 and s.points[-1].n_rows == 4
    assert s.data_through == date(2026, 8, 31)
    assert s.dropped_in_progress == (date(2026, 9, 1),)
    assert s.dropped_incomplete == (date(2026, 5, 1),)
    assert (s.metric, s.brand, s.region, s.query_id) == ("trx", "Kisqali", None, "q")


def test_the_first_of_a_month_is_still_in_progress():
    s = _shape(as_of=date(2026, 9, 1))
    assert s.dropped_in_progress == (date(2026, 9, 1),)


def test_an_empty_series_has_no_data_through():
    s = _shape(rows=[])
    assert s.points == () and s.data_through is None


def test_february_month_end_is_calendar_correct():
    s = _shape(
        rows=[{"metric_date": "2028-02-01", "value": 1.0, "n_rows": 1}], as_of=date(2028, 3, 5)
    )
    assert s.data_through == date(2028, 2, 29)


def test_unsupported_metric_is_refused():
    with pytest.raises(cvs.CanonicalSeriesError):
        _shape(metric="market_share")


def test_day_three_with_the_previous_month_absent_serves_the_last_loaded_month():
    """A stalled cron (September never loaded) on 3 October: the series ends at the
    last loaded complete month and says so through data_through."""
    rows = [r for r in ROWS if r["metric_date"] != "2026-09-01"]
    s = _shape(rows=rows, as_of=date(2026, 10, 3))
    assert s.points[-1].month == date(2026, 8, 1)
    assert s.data_through == date(2026, 8, 31)
    assert s.dropped_in_progress == ()


def test_an_empty_scope_is_empty_not_zero():
    s = _shape(rows=[], brand="NoSuchBrand")
    assert s.points == () and s.data_through is None and s.dropped_incomplete == ()


class _Client:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def rpc(self, name, payload):
        assert name == "kpi_query"
        self.calls.append(payload)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=self.rows))


def test_fetch_binds_metric_brand_region(monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    client = _Client(ROWS)
    s = cvs.fetch_canonical_volume_series(
        "trx", "Kisqali", "west", client=client, as_of=date(2026, 9, 15)
    )
    assert client.calls == [
        {"query_id": "canonical_volume_monthly_series", "params": ["trx", "Kisqali", "west"]}
    ]
    assert s.data_through == date(2026, 8, 31)


def test_fetch_selects_the_twin_under_the_showcase_flag(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    client = _Client([])
    s = cvs.fetch_canonical_volume_series("nbrx", client=client, as_of=date(2026, 9, 15))
    assert client.calls[0]["query_id"] == "canonical_volume_monthly_series_include_synthetic"
    assert client.calls[0]["params"] == ["nbrx", None, None]
    assert s.query_id == "canonical_volume_monthly_series_include_synthetic"


def test_fetch_without_a_client_fails_loud(monkeypatch):
    import src.api.dependencies.supabase_client as sc

    monkeypatch.setattr(sc, "get_supabase", lambda: None)
    with pytest.raises(cvs.CanonicalSeriesError):
        cvs.fetch_canonical_volume_series("trx", "Kisqali", as_of=date(2026, 9, 15))


def test_async_wrapper_matches_the_sync_read(monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    sync = cvs.fetch_canonical_volume_series(
        "trx", "Kisqali", client=_Client(ROWS), as_of=date(2026, 9, 15)
    )
    async_ = asyncio.run(
        cvs.afetch_canonical_volume_series(
            "trx", "Kisqali", client=_Client(ROWS), as_of=date(2026, 9, 15)
        )
    )
    assert async_ == sync


def test_supported_metrics_are_the_ones_the_statement_accepts():
    """The allowlist is CHECKED against the deployed statement, not declared twice.

    ``canonical_volume_monthly_series`` carries its own guard —
    ``WHERE $1::text IN ('trx','nrx','nbrx')`` — so the SQL is the capability and
    this tuple is a mirror of it. If they drift, ``SUPPORTED_METRICS`` either
    refuses a metric the statement would serve (a silent gap in the forecaster's
    reach) or admits one the statement filters to nothing, which reads as "no data"
    rather than "not supported".
    """
    sql = (
        Path(__file__).resolve().parents[3]
        / "database"
        / "migrations"
        / "143_canonical_volume_kpis.sql"
    ).read_text()
    body = re.search(r"\('canonical_volume_monthly_series', \$kpi\$(.*?)\$kpi\$", sql, re.S)
    assert body, "canonical_volume_monthly_series not found in migration 143"
    allowed = re.search(r"\$1::text IN \(([^)]*)\)", body.group(1))
    assert allowed, "the statement no longer guards $1 with an IN list"
    declared = tuple(sorted(re.findall(r"'(\w+)'", allowed.group(1))))
    assert tuple(sorted(cvs.SUPPORTED_METRICS)) == declared
