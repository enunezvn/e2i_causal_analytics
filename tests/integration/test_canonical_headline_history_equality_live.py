"""The canonical headline and its history agree for the same scope, in BOTH synthetic
modes (canonical TRx lane, codex r1 HIGH). Read-only.

⚠ THIS FILE RUNS IN CI — it is NOT skipped there, and it is not ignored. Measured:
the ``integration-tests`` job runs ``pytest tests/integration/`` with an 18-path
``--ignore`` list that does not name this file; the conftest's automatic
service-skip fires only for tests marked ``requires_supabase`` (this one is marked
``integration``); and ``_check_supabase_service`` deliberately checks CREDENTIALS,
not connectivity, which CI supplies (``SUPABASE_URL=http://localhost:54321``,
``SUPABASE_KEY=test-key``). A ``client is None`` guard therefore protects nothing
in CI: ``create_client`` does not connect eagerly, so ``get_supabase()`` returns a
live Client object against a dead address. Without a real precondition check this
file failed 6/6 (``kpi_query: unknown query_id canonical_volume_trx``) and turned
Integration Tests red.

So the precondition is checked as a CAPABILITY — is the statement deployed? — not
as a proxy for it. ``_headline`` skips on exactly two conditions this gate cannot
create for itself (statements not deployed; no reachable database) and re-raises
everything else, so a genuine numeric DISAGREEMENT can never be skipped away.

⚠ The skip self-extinguishes when migration 143 deploys, and a permanent skip is
indistinguishable from a pass in a CI summary. Task 31 must verify these 6 rows
actually RAN, not merely that Integration Tests is green.
"""

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


def _undeployed(exc: Exception) -> bool:
    """The migration-143 statements are not in the registry yet."""
    return "unknown query_id" in str(exc)


def _unreachable(exc: Exception) -> bool:
    """No database answered at all (CI's dead localhost, a stopped container)."""
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("connection refused", "connect error", "failed to establish", "timed out")
    )


def _headline(client, query_id, params):
    """The headline row, or a SKIP on a missing precondition — never on a mismatch.

    Only the two conditions this gate cannot create for itself are skipped. Anything
    else propagates: if the statement runs and the numbers disagree, that is the
    finding this file exists for and it must fail.
    """
    try:
        rows = client.rpc("kpi_query", {"query_id": query_id, "params": params}).execute().data
    except Exception as exc:  # noqa: BLE001 - narrowed immediately, then re-raised
        if _undeployed(exc):
            pytest.skip(f"{query_id} not deployed (migration 143) — Task 31 runs this")
        if _unreachable(exc):
            pytest.skip(f"no reachable database for {query_id} — Task 31 runs this")
        raise
    return (rows or [{}])[0]


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
    head = _headline(client, query_id, ["Kisqali"])
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
