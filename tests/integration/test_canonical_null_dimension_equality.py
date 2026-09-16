"""Headline, history and series apply ONE NULL-dimension rule (canonical TRx lane, codex r2 HIGH).

Planted business_metrics rows (canonical_sql_harness: a TEMP shadow table, rolled back)
include a NULL-region and a NULL-brand row in the frontier month. For all four KPIs,
national and region scopes, and both synthetic modes, the generated headline statement
(chosen by canonical_query_call, twin resolved like the calculator), the history
aggregation and the monthly series agree, and none counts the NULL rows. Skips without docker.
"""

from datetime import date
from types import SimpleNamespace

import pytest

from src.kpi.calculators.canonical_volume import canonical_query_call
from src.kpi.canonical_volume_series import fetch_canonical_volume_series
from src.kpi.history_canonical_volume import aggregate_canonical_points
from src.kpi.synthetic_mode import resolve_kpi_query_id
from tests.integration.canonical_sql_harness import PlantedBusinessMetrics, bm_row

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 15)
AUGUST = "2026-08-01"
SCALE = {"trx": 1.0, "nrx": 0.5, "nbrx": 0.25}
#: (month, brand, region, trx value, is_synthetic); nrx = 0.5x, nbrx = 0.25x
BASE = [
    ("2026-07-01", "Kisqali", "northeast", 100, False),
    ("2026-07-01", "Kisqali", "south", 110, False),
    ("2026-07-01", "Fabhalta", "northeast", 50, False),
    ("2026-07-01", "Fabhalta", "south", 50, False),
    ("2026-08-01", "Kisqali", "northeast", 200, False),
    ("2026-08-01", "Kisqali", "south", 210, True),
    ("2026-08-01", "Fabhalta", "northeast", 60, False),
    ("2026-08-01", "Fabhalta", "south", 60, False),
    ("2026-08-01", "Kisqali", None, 7777, False),  # planted NULL region
    ("2026-08-01", None, "northeast", 8888, False),  # planted NULL brand
    ("2026-09-01", "Kisqali", "northeast", 999, False),  # in-progress month
]
#: (kpi_id, metric, result key, brand, region, {include_synthetic: August value or None})
CASES = [
    ("WS3-BI-005", "trx", "trx", "Kisqali", None, {False: 200.0, True: 410.0}),
    ("WS3-BI-005", "trx", "trx", None, "northeast", {False: 260.0, True: 260.0}),
    ("WS3-BI-005", "trx", "trx", "Kisqali", "south", {False: None, True: 210.0}),
    ("WS3-BI-006", "nrx", "nrx", "Kisqali", None, {False: 100.0, True: 205.0}),
    ("WS3-BI-006", "nrx", "nrx", None, None, {False: 160.0, True: 265.0}),
    ("WS3-BI-007", "nbrx", "nbrx", "Kisqali", None, {False: 50.0, True: 102.5}),
    ("WS3-BI-007", "nbrx", "nbrx", None, "northeast", {False: 65.0, True: 65.0}),
    ("WS3-BI-008", "trx", "share", "Kisqali", None, {False: 200 / 320, True: 410 / 530}),
    ("WS3-BI-008", "trx", "share", "Kisqali", "northeast", {False: 200 / 260, True: 200 / 260}),
]


@pytest.fixture(scope="module")
def db():
    rows = [
        bm_row(metric, month, brand, region, value * SCALE[metric], synthetic=synthetic)
        for metric in SCALE
        for month, brand, region, value, synthetic in BASE
    ]
    return PlantedBusinessMetrics(rows, TODAY)


@pytest.mark.parametrize("include_synthetic", [False, True])
@pytest.mark.parametrize("kpi_id,metric,key,brand,region,expected", CASES)
def test_headline_history_and_series_agree_and_ignore_null_dimensions(
    db, monkeypatch, include_synthetic, kpi_id, metric, key, brand, region, expected
):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true" if include_synthetic else "0")
    want = expected[include_synthetic]

    context = {k: v for k, v in (("brand", brand), ("region", region)) if v}
    query_id, params, result_key = canonical_query_call(kpi_id, context)
    assert result_key == key
    [head] = db.query(resolve_kpi_query_id(query_id), params)
    assert head["data_month"] == AUGUST
    if want is None:
        assert head[key] is None
    else:
        assert float(head[key]) == pytest.approx(want, rel=1e-12)

    points = aggregate_canonical_points(
        db.eligible(include_synthetic),
        SimpleNamespace(id=kpi_id, threshold=None),
        TODAY,
        include_synthetic=include_synthetic,
    )
    history = {
        p["metric_date"]: p["value"]
        for p in points
        if p["brand"] == (brand or "") and p["region"] == (region or "")
    }
    if want is None:
        assert AUGUST not in history
    else:
        assert history[AUGUST] == pytest.approx(want, rel=1e-12)

    if kpi_id != "WS3-BI-008":
        series = fetch_canonical_volume_series(metric, brand, region, client=db, as_of=TODAY)
        served = {p.month.isoformat(): p.value for p in series.points}
        for month, value in served.items():
            assert history[month] == pytest.approx(value, rel=1e-12), month
        if AUGUST in served:
            assert served[AUGUST] == pytest.approx(want, rel=1e-12)
        else:
            # the series' documented coverage filter, never a NULL-dimension row, decides
            assert want is None or date(2026, 8, 1) in series.dropped_incomplete
