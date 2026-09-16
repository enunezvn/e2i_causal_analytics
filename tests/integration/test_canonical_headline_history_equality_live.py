"""The canonical headline and its history agree for the same scope, in BOTH synthetic
modes (canonical TRx lane, codex r1 HIGH). Read-only; skips without a client."""

from datetime import date
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def _client():
    from src.api.dependencies.supabase_client import get_supabase, init_supabase

    if get_supabase() is None:
        init_supabase()
    client = get_supabase()
    if client is None:
        pytest.skip("no Supabase client available")
    return client


def _rows(client, include_synthetic):
    out, offset = [], 0
    while True:
        query = (
            client.table("business_metrics")
            .select("metric_id,metric_date,metric_type,brand,region,value,is_synthetic")
            .in_("metric_type", ["trx", "nrx", "nbrx"])
            .lt("metric_date", date.today().replace(day=1).isoformat())
        )
        if not include_synthetic:
            query = query.eq("is_synthetic", False)
        page = query.order("metric_id").range(offset, offset + 4999).execute().data or []
        out.extend(page)
        if len(page) < 5000:
            return out
        offset += 5000


@pytest.mark.parametrize("include_synthetic", [True, False])
@pytest.mark.parametrize(
    "kpi_id,base,key",
    [
        ("WS3-BI-005", "canonical_volume_trx", "trx"),
        ("WS3-BI-006", "canonical_volume_nrx", "nrx"),
        ("WS3-BI-007", "canonical_volume_nbrx", "nbrx"),
    ],
)
def test_headline_equals_the_history_point_for_its_data_month(include_synthetic, kpi_id, base, key):
    from src.kpi.history_canonical_volume import aggregate_canonical_points

    client = _client()
    query_id = base + ("_include_synthetic" if include_synthetic else "")
    head = (
        client.rpc("kpi_query", {"query_id": query_id, "params": ["Kisqali"]}).execute().data
        or [{}]
    )[0]
    points = aggregate_canonical_points(
        _rows(client, include_synthetic),
        SimpleNamespace(id=kpi_id, threshold=None),
        date.today(),
        include_synthetic=include_synthetic,
    )
    series = {
        p["metric_date"]: p["value"]
        for p in points
        if p["brand"] == "Kisqali" and p["region"] == ""
    }
    if head.get(key) is None:
        assert not series, "history must be empty when the headline has no eligible rows"
    else:
        month = str(head["data_month"])[:10]
        assert series[month] == pytest.approx(float(head[key]), rel=1e-9)
        assert max(series) == month
