"""For identical scope, e2i_data_query_tool(kpi) and kpi_calculate_tool return the SAME
number for TRx / NRx / NBRx / TRx Share (canonical TRx lane, Task 15A, codex r1 HIGH).

Read-only; skips without a reachable, deployed, seeded client. Runs for real in the
Task 31 certification, once migration 143 and its canonical series are live -- before
that the lane gate only proves it COLLECTS.

This is the end-to-end statement of the property the unit tests approximate: the two
chat surfaces that can be asked for the same volume figure must not disagree. Pre-lane
that divergence was at least FENCED (the canonical KPIs rested on treatment_events, so
``cross_substrate_conflict`` fired on a stored-row answer); once both rest on
business_metrics the fence goes silent, so equality has to be asserted directly.
"""

import datetime

import pytest

pytestmark = pytest.mark.integration


def _require_client():
    from src.api.dependencies.supabase_client import get_supabase, init_supabase

    if get_supabase() is None:
        init_supabase()
    if get_supabase() is None:
        pytest.skip("no Supabase client available")


def _skip_if_unavailable(error):
    """Skip only when this read-only live test lacks its external precondition."""
    text = str(error).lower()
    if "unknown query_id" in text:
        pytest.skip("migration 143 canonical-volume statements are not deployed")
    if "no complete month of business_metrics" in text and "rows for this scope" in text:
        pytest.skip("canonical volume series has not been seeded for this scope")
    if any(
        marker in text
        for marker in ("connection refused", "connect error", "failed to establish", "timed out")
    ):
        pytest.skip("no reachable Supabase database")


@pytest.mark.parametrize("region", [None, "midwest"])
@pytest.mark.parametrize(
    "kpi_id,name",
    [
        ("WS3-BI-005", "TRx"),
        ("WS3-BI-006", "NRx"),
        ("WS3-BI-007", "NBRx"),
        ("WS3-BI-008", "TRx share"),
    ],
)
async def test_data_query_latest_row_equals_kpi_calculate(kpi_id, name, region):
    _require_client()
    from src.api.routes.chatbot_tools import _query_kpis
    from src.api.routes.kpi import get_kpi_calculator

    context = {"brand": "Kisqali", **({"region": region} if region else {})}
    result = get_kpi_calculator().calculate(kpi_id, use_cache=False, context=context)
    if result.error is not None:
        _skip_if_unavailable(result.error)
    assert result.error is None, result.error
    out = await _query_kpis(
        brand="Kisqali",
        region=region,
        kpi_name=name,
        since=datetime.datetime.now() - datetime.timedelta(days=45),
        limit=24,
    )
    assert out.get("canonical_aggregate") is True, out
    assert out["count"] >= 1, out
    latest = out["data"][0]
    assert latest["metric_date"] == str(result.metadata["context"]["data_month"])[:10]
    assert latest["value"] == pytest.approx(result.value, rel=1e-9, abs=1e-9)
