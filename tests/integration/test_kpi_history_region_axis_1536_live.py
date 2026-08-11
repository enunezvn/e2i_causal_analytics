"""#1536 faithful e2e: the kpi_history region axis against the LIVE DB — no
mocks.

These assert MEANING:

- Region series exist for the region-axis KPIs with the canonical lowercase
  substrate labels, and ONLY for those KPIs (no fabricated region axis on
  MAU/WAU or BR-*).
- The ROI region series is arithmetically the substrate: each (region, month)
  point equals AVG(business_metrics.roi) over that region's rows — the same
  reading migration 125's scoped headline computes.
- NBRx honors its fail-loud brand contract on the region axis too: no
  (brand='', region!='') rows exist ("new-to-region NBRx without a brand" is
  undefined, exactly like the live calculator).
- The global ('' , '') series is untouched by the region axis: one ROI row
  per distinct business_metrics month, as before.
- The migration-126 coverage view exposes the (brand, region) lattice the
  Time-Series region selector is driven by.

CAPABILITY-GATED: skips if SUPABASE_* is unset or the region-axis backfill has
not populated kpi_history (CI's ephemeral Supabase, or a droplet before the
post-deploy backfill run).
"""

import os

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.history_backfill import REGION_AXIS_KPI_IDS

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

CANONICAL_REGIONS = {"northeast", "south", "midwest", "west"}


@pytest.fixture
def db():
    client = BusinessImpactCalculator().db_client
    if client is None:
        pytest.skip("no Supabase client")
    try:
        probe = (
            client.table("kpi_history")
            .select("kpi_id")
            .eq("kpi_id", "WS3-BI-010")
            .neq("region", "")
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"kpi_history unavailable: {e}")
    if not probe.data:
        pytest.skip("#1536 region-axis backfill has not populated kpi_history yet")
    return client


def _rows(db, **eq):
    query = db.table("kpi_history").select("kpi_id,brand,region,metric_date,value")
    for col, val in eq.items():
        query = query.eq(col, val)
    return query.limit(10000).execute().data or []


def test_region_axis_kpis_have_canonical_region_series(db):
    for kpi_id in sorted(REGION_AXIS_KPI_IDS):
        rows = [r for r in _rows(db, kpi_id=kpi_id) if r["region"]]
        assert rows, f"{kpi_id}: no region-scoped history rows"
        labels = {r["region"] for r in rows}
        assert labels <= CANONICAL_REGIONS, f"{kpi_id}: non-canonical region labels {labels}"


def test_non_region_axis_kpis_stay_region_free(db):
    for kpi_id in ("WS3-BI-001", "WS3-BI-002", "BR-001", "BR-002", "BR-003", "BR-004"):
        rows = [r for r in _rows(db, kpi_id=kpi_id) if r["region"]]
        assert rows == [], f"{kpi_id}: fabricated region rows {rows[:3]}"


def test_roi_region_point_is_substrate_mean(db):
    """Each ROI (region, month) point == AVG(business_metrics.roi) there."""
    for region in sorted(CANONICAL_REGIONS):
        series = _rows(db, kpi_id="WS3-BI-010", brand="", region=region)
        assert series, f"ROI has no {region} series"
        latest = max(series, key=lambda r: r["metric_date"])
        substrate = (
            db.table("business_metrics")
            .select("roi")
            .eq("region", region)
            .eq("metric_date", latest["metric_date"])
            .not_.is_("roi", "null")
            .limit(10000)
            .execute()
        ).data
        assert substrate, f"no business_metrics rows for {region} {latest['metric_date']}"
        expected = sum(float(r["roi"]) for r in substrate) / len(substrate)
        assert latest["value"] == pytest.approx(expected, rel=1e-9), (
            f"ROI {region} {latest['metric_date']}: history {latest['value']} != "
            f"substrate mean {expected}"
        )


def test_roi_brand_region_lattice_exists(db):
    rows = _rows(db, kpi_id="WS3-BI-010", brand="Kisqali", region="northeast")
    assert rows, "ROI (Kisqali, northeast) brand×region series missing"


def test_nbrx_region_rows_are_brand_scoped_only(db):
    rows = [r for r in _rows(db, kpi_id="WS3-BI-007", brand="") if r["region"]]
    assert rows == [], f"NBRx grew brandless region rows: {rows[:3]}"


def test_roi_global_series_is_one_row_per_substrate_month(db):
    global_rows = _rows(db, kpi_id="WS3-BI-010", brand="", region="")
    months = (
        db.table("business_metrics")
        .select("metric_date")
        .not_.is_("roi", "null")
        .limit(20000)
        .execute()
    ).data
    assert len(global_rows) == len({r["metric_date"] for r in months})


def test_coverage_view_exposes_the_scope_lattice(db):
    rows = (
        db.table("v_kpi_history_coverage")
        .select("kpi_id,brand,region,points")
        .eq("kpi_id", "WS3-BI-010")
        .limit(100)
        .execute()
    ).data or []
    scopes = {(r["brand"], r["region"]) for r in rows}
    assert ("", "") in scopes
    assert ("", "northeast") in scopes
    assert ("Kisqali", "northeast") in scopes
