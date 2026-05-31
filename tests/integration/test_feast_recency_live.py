"""#559 faithful premise test: real MAX(<timestamp col>) recency against the LIVE,
prod-equivalent Supabase.

This codifies the premise the unit tests (which mock the Supabase client) cannot prove:
that the PostgREST ``.order().limit()`` recency mechanism actually returns a real,
tz-aware datetime from the target DB — and that the ``execute_sql`` RPC is NOT required
(it does not exist in the target Supabase). It SKIPS when Supabase is not configured
(e.g. CI without a DB), so it is a local/staging repeatability guard, not a CI gate.

Run against the local self-contained Supabase with SUPABASE_URL + SUPABASE_ANON_KEY set.
"""

import os
from datetime import datetime, timezone

import pytest

from src.feature_store.feast_client import FeastClient

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
requires_supabase = pytest.mark.skipif(
    not HAS_SUPABASE,
    reason="SUPABASE_URL and SUPABASE_ANON_KEY not set — live recency premise test skipped",
)

pytestmark = [pytest.mark.integration, requires_supabase]

# The canonical mapped source tables (see FeastClient._TABLE_TIMESTAMP_COLUMNS).
MAPPED_TABLES = [
    "hcp_profiles",
    "triggers",
    "business_metrics",
    "patient_journeys",
    "territory_metrics",
]


@pytest.fixture
def live_client():
    """The shared sync Supabase client the production stats path self-resolves."""
    from src.api.dependencies.supabase_client import get_supabase

    client = get_supabase()
    if client is None:
        pytest.skip("get_supabase() returned None — Supabase unavailable")
    return client


@pytest.mark.parametrize("table", MAPPED_TABLES)
async def test_query_max_recency_invariants(live_client, table):
    """For every mapped table, recency is either None (genuinely empty) or a real,
    tz-aware UTC datetime that is not in the future — never a fabricated now(), and never
    dependent on the (non-existent) execute_sql RPC."""
    recency = await FeastClient()._query_max_recency(live_client, table)
    if recency is not None:
        assert recency.tzinfo is not None, f"{table}: recency must be tz-aware UTC"
        assert recency <= datetime.now(timezone.utc), f"{table}: recency {recency} is in the future"


async def test_hcp_profiles_has_real_recency(live_client):
    """The premise the mock unit tests cannot prove: a REAL recency signal flows through
    the PostgREST mechanism for the canonically-populated hcp_profiles table."""
    recency = await FeastClient()._query_max_recency(live_client, "hcp_profiles")
    assert recency is not None, "hcp_profiles must yield a real MAX(updated_at), not None"
    assert recency.tzinfo is not None


async def test_unmapped_table_recency_is_none(live_client):
    """An unmapped table issues no query and yields None (recency genuinely unknown)."""
    recency = await FeastClient()._query_max_recency(live_client, "table_that_does_not_exist")
    assert recency is None
