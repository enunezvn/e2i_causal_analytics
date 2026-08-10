"""#1534: the WS3-BI-010 ROI headline honors brand/region request context.

Intent (issue #1534, investigated 2026-08-10): the dashboard grid mirrors the
selected brand (8ebca82b, product-owner-directed), every WS3-BI sibling honors
``context["brand"]``, and the chatbot echoes brand/region as "the brand/region
the figure was computed for" (d72ac745) — so the ROI headline ignoring context
while surfaces label it brand-scoped was a latent bug, not a decision. The fix:

* ``_calc_roi`` resolves scope ``(brand, region)`` from context and queries the
  2-nullable-param ``business_impact_roi_business_metrics_scoped`` registry id
  (migration 125; additive — the 0-param id stays registered for existing
  callers). ``[None, None]`` is value-identical to the 0-param query.
* The #1532 temporal-variability band receives the SAME resolved scope the
  headline used — band population == headline population by construction
  (the #1532 codex iter-2 invariant, generalized).
* ``agent_activities`` has NO brand/region columns (measured 2026-08-10), so a
  scoped request that business_metrics cannot answer FAILS LOUD instead of
  silently serving a portfolio number under a brand label (WS3-BI-009
  fail-loud precedent). The unscoped fallback is unchanged.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator

SCOPED_ID = "business_impact_roi_business_metrics_scoped"
FALLBACK_ID = "business_impact_roi_agent_activities"


@pytest.fixture()
def base_ids_env(monkeypatch):
    """Resolve base registry ids (not _include_synthetic twins) regardless of
    this box's showcase .env — resolve_kpi_query_id reads env per call, and
    BOTH the KPI-specific flag and the deployment-wide showcase switch flip
    it (test_unified_deployment_flag_flips_kpi_reads)."""
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def _client_with(responses: dict[str, list[dict[str, Any]]]) -> MagicMock:
    """db_client stub that dispatches kpi_query calls by query_id and records
    every (query_id, params) pair — the same db-client seam the #1532 unit
    tests mock, never our own code."""
    client = MagicMock()
    calls: list[tuple[str, list[Any]]] = []

    def rpc(_fn: str, payload: dict[str, Any]) -> MagicMock:
        calls.append((payload["query_id"], payload["params"]))
        holder = MagicMock()
        holder.execute.return_value = MagicMock(data=responses.get(payload["query_id"], []))
        return holder

    client.rpc.side_effect = rpc
    client.recorded_calls = calls
    return client


@pytest.mark.unit
def test_headline_query_passes_brand_and_region_scope(base_ids_env):
    client = _client_with({SCOPED_ID: [{"avg_roi": 1.9, "data_through": "2026-08-01"}]})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_roi({"brand": "Kisqali", "region": "northeast"})
    assert value == pytest.approx(1.9)
    query_id, params = client.recorded_calls[0]
    assert query_id == SCOPED_ID
    assert params == ["Kisqali", "northeast"]


@pytest.mark.unit
def test_unscoped_context_passes_null_scope(base_ids_env):
    client = _client_with({SCOPED_ID: [{"avg_roi": 1.8, "data_through": "2026-08-01"}]})
    calc = BusinessImpactCalculator(db_client=client)
    assert calc._calc_roi({}) == pytest.approx(1.8)
    query_id, params = client.recorded_calls[0]
    assert query_id == SCOPED_ID
    assert params == [None, None]


@pytest.mark.unit
def test_band_receives_the_same_scope_the_headline_used(base_ids_env):
    """Population consistency by construction: the band query is called with
    the exact scope params the headline resolved — not re-derived from context,
    not hardwired unscoped (that was the pre-#1534 world)."""
    client = _client_with({SCOPED_ID: [{"avg_roi": 1.9, "data_through": "2026-08-01"}]})
    calc = BusinessImpactCalculator(db_client=client)
    calc._calc_roi({"brand": "Kisqali", "region": "northeast"})
    band_calls = [
        params
        for query_id, params in client.recorded_calls
        if query_id == "business_impact_roi_temporal_band"
    ]
    assert band_calls == [["Kisqali", "northeast"]]


@pytest.mark.unit
def test_scoped_request_never_falls_back_to_portfolio_agent_activities(base_ids_env):
    """agent_activities has no brand/region dimension: serving it under a
    brand/region filter would relabel a portfolio number as brand-scoped —
    the exact defect #1534 removes. Fail loud instead."""
    client = _client_with({})
    calc = BusinessImpactCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="Kisqali"):
        calc._calc_roi({"brand": "Kisqali"})
    assert all(query_id != FALLBACK_ID for query_id, _ in client.recorded_calls)


@pytest.mark.unit
def test_region_only_request_also_fails_loud_without_business_metrics(base_ids_env):
    client = _client_with({})
    calc = BusinessImpactCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="northeast"):
        calc._calc_roi({"region": "northeast"})
    assert all(query_id != FALLBACK_ID for query_id, _ in client.recorded_calls)


@pytest.mark.unit
def test_unscoped_request_keeps_agent_activities_fallback(base_ids_env):
    """Pin the pre-existing unscoped fallback: business_metrics empty and no
    scope requested -> agent_activities still answers (no band: different
    substrate, unchanged #1532 contract)."""
    client = _client_with({FALLBACK_ID: [{"avg_roi": 2.4, "data_through": "2026-08-01"}]})
    calc = BusinessImpactCalculator(db_client=client)
    context: dict[str, Any] = {}
    assert calc._calc_roi(context) == pytest.approx(2.4)
    query_ids = [query_id for query_id, _ in client.recorded_calls]
    assert query_ids[0] == SCOPED_ID
    assert FALLBACK_ID in query_ids
    assert "temporal_variability_band" not in context
