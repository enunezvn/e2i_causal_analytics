"""Canonical Rx-volume history rests on business_metrics.value and follows the
headline's synthetic policy (canonical TRx lane; codex r1 HIGH)."""

import asyncio
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.kpi.history_backfill as hb
from src.kpi import history_canonical_volume as hcv


def _r(month, brand, region, metric, value, synthetic=True):
    return {
        "metric_id": f"{metric}-{brand}-{region}-{month}-{synthetic}",
        "metric_date": month,
        "brand": brand,
        "region": region,
        "metric_type": metric,
        "value": value,
        "is_synthetic": synthetic,
    }


ROWS = [
    _r("2026-07-01", "Kisqali", "midwest", "trx", 100.0),
    _r("2026-07-01", "Kisqali", "west", "trx", 200.0),
    _r("2026-07-01", "Fabhalta", "west", "trx", 50.0),
    _r("2026-08-01", "Kisqali", "midwest", "trx", 110.0),
    _r("2026-08-01", "Kisqali", "west", "trx", 190.0),
    _r("2026-08-01", "Fabhalta", "west", "trx", 60.0),
    _r("2026-09-01", "Kisqali", "west", "trx", 999.0),  # in progress on 2026-09-15
    _r("2026-08-01", "Kisqali", "west", "nrx", 40.0),
]
AS_OF = date(2026, 9, 15)


def _meta(kpi_id):
    return SimpleNamespace(id=kpi_id, threshold=None)


def _by(points):
    return {(p["brand"], p["region"], p["metric_date"]): p["value"] for p in points}


def _flags(points):
    return {(p["brand"], p["region"], p["metric_date"]): p["is_synthetic"] for p in points}


@pytest.fixture(autouse=True)
def _no_ambient_flags(monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def test_trx_national_brand_region_and_brand_region_series_in_showcase_mode():
    points = hcv.aggregate_canonical_points(
        ROWS, _meta("WS3-BI-005"), AS_OF, include_synthetic=True
    )
    got = _by(points)
    assert got[("", "", "2026-07-01")] == 350.0
    assert got[("", "", "2026-08-01")] == 360.0
    assert got[("Kisqali", "", "2026-08-01")] == 300.0
    assert got[("", "west", "2026-07-01")] == 250.0
    assert got[("Kisqali", "west", "2026-08-01")] == 190.0
    assert not any(k[2] == "2026-09-01" for k in got), "in-progress month emitted"
    assert {p["source"] for p in points} == {"business_metrics.value"}
    assert all(p["is_synthetic"] is True and p["status"] == "informational" for p in points)


def test_real_mode_excludes_synthetic_inputs_exactly_like_the_base_statement():
    assert (
        hcv.aggregate_canonical_points(ROWS, _meta("WS3-BI-005"), AS_OF, include_synthetic=False)
        == []
    )
    mixed = ROWS + [_r("2026-08-01", "Kisqali", "west", "trx", 5.0, synthetic=False)]
    points = hcv.aggregate_canonical_points(
        mixed, _meta("WS3-BI-005"), AS_OF, include_synthetic=False
    )
    assert _by(points)[("", "", "2026-08-01")] == 5.0
    assert all(p["is_synthetic"] is False for p in points), (
        "real-only inputs must not be marked synthetic"
    )


def test_showcase_mode_taints_a_point_only_when_an_input_was_synthetic():
    rows = [
        _r("2026-08-01", "Kisqali", "west", "trx", 5.0, synthetic=False),
        _r("2026-08-01", "Kisqali", "midwest", "trx", 7.0, synthetic=True),
        _r("2026-07-01", "Kisqali", "west", "trx", 3.0, synthetic=False),
    ]
    flags = _flags(
        hcv.aggregate_canonical_points(rows, _meta("WS3-BI-005"), AS_OF, include_synthetic=True)
    )
    assert flags[("", "", "2026-08-01")] is True  # mixed month -> tainted
    assert flags[("Kisqali", "west", "2026-08-01")] is False
    assert flags[("", "", "2026-07-01")] is False  # real-only month stays real


def test_stalled_cron_history_ends_at_the_last_loaded_month():
    rows = [r for r in ROWS if r["metric_date"] != "2026-09-01"]
    points = hcv.aggregate_canonical_points(
        rows, _meta("WS3-BI-005"), date(2026, 10, 3), include_synthetic=True
    )
    assert max(p["metric_date"] for p in points) == "2026-08-01"


def test_nrx_reads_only_nrx_rows():
    got = _by(
        hcv.aggregate_canonical_points(ROWS, _meta("WS3-BI-006"), AS_OF, include_synthetic=True)
    )
    assert got == {
        ("", "", "2026-08-01"): 40.0,
        ("Kisqali", "", "2026-08-01"): 40.0,
        ("", "west", "2026-08-01"): 40.0,
        ("Kisqali", "west", "2026-08-01"): 40.0,
    }


def test_share_is_per_brand_and_brand_region_only():
    got = _by(
        hcv.aggregate_canonical_points(ROWS, _meta("WS3-BI-008"), AS_OF, include_synthetic=True)
    )
    assert got[("Kisqali", "", "2026-07-01")] == pytest.approx(300.0 / 350.0)
    assert got[("Kisqali", "west", "2026-07-01")] == pytest.approx(200.0 / 250.0)
    assert got[("Kisqali", "midwest", "2026-08-01")] == pytest.approx(1.0)
    assert not any(k[0] == "" for k in got), "a portfolio share of the portfolio is undefined"


class _Query:
    def __init__(self, rows):
        self.rows, self.metric_types, self.before, self.bounds, self.real_only = (
            rows,
            None,
            None,
            None,
            False,
        )

    def select(self, _cols):
        return self

    def in_(self, col, values):
        assert col == "metric_type"
        self.metric_types = set(values)
        return self

    def lt(self, col, value):
        assert col == "metric_date"
        self.before = value
        return self

    def eq(self, col, value):
        assert col == "is_synthetic" and value is False
        self.real_only = True
        return self

    def order(self, _col):
        return self

    def range(self, lo, hi):
        self.bounds = (lo, hi)
        return self

    async def execute(self):
        data = [
            r
            for r in self.rows
            if r["metric_type"] in self.metric_types
            and r["metric_date"] < self.before
            and (not self.real_only or r["is_synthetic"] is False)
        ]
        lo, hi = self.bounds
        return SimpleNamespace(data=data[lo : hi + 1])


class _Client:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def table(self, name):
        assert name == "business_metrics"
        query = _Query(self.rows)
        self.queries.append(query)
        return query


def test_the_handler_follows_the_deployment_flag_and_caches_per_mode(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    client, cache = _Client(ROWS), {}
    points = asyncio.run(
        hcv.backfill_canonical_volume(client, _meta("WS3-BI-005"), cache, as_of=AS_OF)
    )
    assert client.queries[0].real_only is False and client.queries[0].before == "2026-09-01"
    assert client.queries[0].metric_types == {"trx", "nrx", "nbrx"}
    assert _by(points)[("", "", "2026-08-01")] == 360.0
    asyncio.run(hcv.backfill_canonical_volume(client, _meta("WS3-BI-006"), cache, as_of=AS_OF))
    assert len(client.queries) == 1, "the four canonical KPIs share one fetch per run and mode"
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    points = asyncio.run(
        hcv.backfill_canonical_volume(client, _meta("WS3-BI-005"), cache, as_of=AS_OF)
    )
    assert len(client.queries) == 2 and client.queries[1].real_only is True
    assert points == []


def test_backfill_registry_wiring():
    for kpi_id in ("WS3-BI-005", "WS3-BI-006", "WS3-BI-007", "WS3-BI-008"):
        assert hb.HANDLERS[kpi_id] is hcv.backfill_canonical_volume
        assert hb.HANDLER_SOURCES[kpi_id] == "business_metrics.value"
    for kpi_id, handler in (
        ("WS3-BI-011", hb._backfill_trx),
        ("WS3-BI-012", hb._backfill_nrx),
        ("WS3-BI-013", hb._backfill_nbrx),
        ("WS3-BI-014", hb._backfill_trx_share),
    ):
        assert hb.HANDLERS[kpi_id] is handler
        assert hb.HANDLER_SOURCES[kpi_id] == "treatment_events.event_date"
        assert kpi_id in hb.REGION_AXIS_KPI_IDS and kpi_id in hb.BRAND_AXIS_KPI_IDS


def test_canonical_history_is_comparable_with_stored_business_metrics():
    from src.kpi.measure_basis import (
        BUSINESS_METRICS_BASIS,
        materialized_history_basis,
        substrates_agree,
    )
    from src.kpi.registry import get_registry

    basis = materialized_history_basis(
        get_registry().get("WS3-BI-005"), rows=[{"source": "business_metrics.value"}]
    )
    assert basis["comparison_key"] == ["business_metrics"]
    assert substrates_agree(basis, BUSINESS_METRICS_BASIS)


def test_the_metric_each_kpi_sums_is_the_one_its_live_statement_sums():
    """The map is not trusted against a second hand-maintained map — it is checked
    against the SQL that actually runs (migration 143).

    ``CANONICAL_METRIC_FOR_KPI`` decides which ``business_metrics`` rows a history
    point sums, so a drift from the deployed statement would make the history
    disagree with its own headline while every unit test stayed green. The value
    predicate is the one on the ``bm`` alias: the ``f`` alias carries the GLOBAL TRx
    frontier subquery, which is 'trx' for nrx and nbrx too and would mask a drift if
    it were matched instead.
    """
    sql = (
        Path(__file__).resolve().parents[3]
        / "database"
        / "migrations"
        / "143_canonical_volume_kpis.sql"
    ).read_text()
    from src.kpi.calculators.canonical_volume import CANONICAL_VOLUME_STATEMENTS

    for kpi_id, metric in hcv.CANONICAL_METRIC_FOR_KPI.items():
        base = CANONICAL_VOLUME_STATEMENTS[kpi_id][0]
        body = re.search(r"\('" + base + r"', \$kpi\$(.*?)\$kpi\$", sql, re.S)
        assert body, f"{base} not found in migration 143"
        summed = sorted(set(re.findall(r"bm\.metric_type = '(\w+)'", body.group(1))))
        assert summed == [metric], (
            f"{kpi_id}: history sums {metric!r} but {base} sums {summed} — the "
            f"history would disagree with its own headline"
        )
