"""e2i_data_query_tool(kpi) and kpi_calculate return the SAME answer on planted rows
(canonical TRx lane, Task 15A, codex r2 HIGH): all four KPIs, both synthetic modes,
incomplete regional coverage, a synthetic-only cell, a brand with no row in the global
frontier month, NULL-dimension rows, and an explicit window. The real calculator,
assembled by get_kpi_calculator, runs over canonical_sql_harness (TEMP shadow table,
rolled back). Expected values are computed by hand from the planted rows.
Skips without docker.

Why planted rather than live-only: the cases that break a "rebuild the aggregate from
the monthly series" implementation are exactly the ones live data does not currently
contain -- incomplete regional coverage in the frontier month, a synthetic-only cell,
and a brand missing from the frontier entirely. Asserting the two surfaces agree on
rows the database happens to hold proves much less than asserting it on the rows that
discriminate.
"""

from datetime import date

import pytest

from src.kpi import canonical_volume_stored as cvs
from src.services.time_window import parse_window
from tests.integration.canonical_sql_harness import PlantedBusinessMetrics, bm_row

pytestmark = pytest.mark.integration

TODAY = date(2026, 9, 15)
SCALE = {"trx": 1.0, "nrx": 0.5, "nbrx": 0.25}
#: (month, brand, region, trx value, is_synthetic); nrx = 0.5x, nbrx = 0.25x
BASE = [
    ("2026-07-01", "Kisqali", "northeast", 100, False),
    ("2026-07-01", "Kisqali", "south", 110, False),
    ("2026-07-01", "Kisqali", "midwest", 120, False),
    ("2026-07-01", "Kisqali", "west", 130, False),
    ("2026-07-01", "Fabhalta", "northeast", 50, False),
    ("2026-07-01", "Fabhalta", "south", 50, False),
    ("2026-07-01", "Fabhalta", "midwest", 50, False),
    ("2026-07-01", "Fabhalta", "west", 50, False),
    ("2026-07-01", "Remibrutinib", "northeast", 30, False),
    ("2026-07-01", "Remibrutinib", "south", 40, False),
    # August = the global frontier. Kisqali covers 2 of 4 regions plus one synthetic-only
    # cell; Remibrutinib has no August row at all.
    ("2026-08-01", "Kisqali", "northeast", 200, False),
    ("2026-08-01", "Kisqali", "south", 210, False),
    ("2026-08-01", "Kisqali", "west", 500, True),
    ("2026-08-01", "Fabhalta", "northeast", 60, False),
    ("2026-08-01", "Fabhalta", "south", 60, False),
    ("2026-08-01", "Fabhalta", "midwest", 60, False),
    ("2026-08-01", "Fabhalta", "west", 60, False),
    ("2026-08-01", "Kisqali", None, 7777, False),  # planted NULL region
    ("2026-08-01", None, "northeast", 8888, False),  # planted NULL brand
    ("2026-09-01", "Kisqali", "northeast", 999, False),  # in-progress month
]
#: (kpi_id, data-query kpi_name, brand, region, {include_synthetic: August value, None = unavailable})
CASES = [
    ("WS3-BI-005", "TRx", "Kisqali", None, {False: 410.0, True: 910.0}),
    ("WS3-BI-005", "TRx", "Kisqali", "northeast", {False: 200.0, True: 200.0}),
    ("WS3-BI-005", "TRx", None, None, {False: 650.0, True: 1150.0}),
    ("WS3-BI-005", "TRx", "Kisqali", "west", {False: None, True: 500.0}),
    ("WS3-BI-005", "TRx", "Remibrutinib", None, {False: None, True: None}),
    ("WS3-BI-006", "NRx", "Kisqali", None, {False: 205.0, True: 455.0}),
    ("WS3-BI-006", "NRx", "Fabhalta", "west", {False: 30.0, True: 30.0}),
    ("WS3-BI-007", "NBRx", "Kisqali", None, {False: 102.5, True: 227.5}),
    ("WS3-BI-007", "NBRx", None, "south", {False: 67.5, True: 67.5}),
    ("WS3-BI-008", "TRx share", "Kisqali", None, {False: 410 / 650, True: 910 / 1150}),
    ("WS3-BI-008", "TRx share", "Kisqali", "northeast", {False: 200 / 260, True: 200 / 260}),
]


@pytest.fixture(scope="module")
def db():
    rows = [
        bm_row(metric, month, brand, region, value * SCALE[metric], synthetic=synthetic)
        for metric in SCALE
        for month, brand, region, value, synthetic in BASE
    ]
    return PlantedBusinessMetrics(rows, TODAY)


def _calculator(monkeypatch, db, include_synthetic):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true" if include_synthetic else "0")
    # A closed port: KPICache pings, fails and disables itself, so every scenario is computed.
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    import src.api.routes.kpi as kpi_routes

    monkeypatch.setattr(kpi_routes, "get_supabase", lambda: db)
    return kpi_routes.get_kpi_calculator()


@pytest.mark.parametrize("include_synthetic", [False, True])
@pytest.mark.parametrize("kpi_id,name,brand,region,expected", CASES)
async def test_data_query_equals_kpi_calculate_and_the_hand_computed_headline(
    db, monkeypatch, include_synthetic, kpi_id, name, brand, region, expected
):
    calculator = _calculator(monkeypatch, db, include_synthetic)
    context = {k: v for k, v in (("brand", brand), ("region", region)) if v}
    db.calls.clear()
    reference = calculator.calculate(kpi_id, use_cache=False, context=dict(context))
    reference_calls = list(db.calls)
    db.calls.clear()
    out = await cvs.canonical_volume_stored_rows(
        name,
        {"brand": brand, "region": region, "metric_name": name},
        "2026-06-01",
        24,
        calculator=calculator,
    )
    # the same statements with the same params, in the same order
    assert db.calls == reference_calls
    assert any(c["query_id"].startswith("canonical_volume_") for c in reference_calls)
    want = expected[include_synthetic]
    if want is None:
        assert reference.error is not None
        assert out["success"] is False
        assert out["data"] == []
        assert out["error"] == str(reference.error)
    else:
        assert reference.error is None, reference.error
        assert reference.value == pytest.approx(want, rel=1e-12)
        [row] = out["data"]
        assert row["value"] == pytest.approx(reference.value, rel=1e-12)
        assert row["metric_date"] == out["data_month"] == "2026-08-01"
        assert out["data_through"] == "2026-08-31"


async def test_an_explicit_window_is_served_by_kpi_calculate_and_the_lookback_is_never_one(
    db, monkeypatch
):
    calculator = _calculator(monkeypatch, db, include_synthetic=False)
    window = parse_window({"start": "2026-07-01", "end": "2026-08-31T23:59:59"}).as_dict()
    windowed = calculator.calculate(
        "WS3-BI-005", use_cache=False, context={"brand": "Kisqali", "window": window}
    )
    # July (all four regions) + August (the two real cells); NULL-region row excluded
    assert windowed.error is None
    assert windowed.value == pytest.approx(460.0 + 410.0)
    db.calls.clear()
    out = await cvs.canonical_volume_stored_rows(
        "TRx", {"brand": "Kisqali"}, "2026-07-01", 24, calculator=calculator
    )
    assert [c["query_id"] for c in db.calls if c["query_id"].startswith("canonical_volume_")] == [
        "canonical_volume_trx"
    ]
    assert out["data"][0]["value"] == pytest.approx(410.0)
    assert "window=" in out["note"]
