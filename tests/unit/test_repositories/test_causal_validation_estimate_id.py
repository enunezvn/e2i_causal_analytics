"""Canonical causal_paths -> causal_validations linkage (#1352, migration 119).

``causal_validations.estimate_id`` is a UUID while ``causal_paths.path_id`` is
a varchar(20). The semantics pin (validation_status='validated' means
"RefutationSuite evidence exists and passed") needs ONE deterministic mapping
shared by:

* migration 119's SQL (``public.causal_path_estimate_id`` =
  ``extensions.uuid_generate_v5(extensions.uuid_ns_url(),
  'e2i:causal_paths:' || path_id)``) — evidence seeding + trigger enforcement;
* the chat surfacing path (``CausalValidationRepository.get_rows_for_paths``);
* the future RefutationNode promoter (#1352 item 3, separate resolver lane).

The known-answer vectors below were measured READ-ONLY against the live
Postgres (supabase-db) with::

    SELECT uuid_generate_v5(uuid_ns_url(), 'e2i:causal_paths:CP-0001');
    -- 8a0c3f5f-247f-59f7-bf51-983e12274243
    SELECT uuid_generate_v5(uuid_ns_url(), 'e2i:causal_paths:scp_001cdc6864504');
    -- d28b51a1-775c-531d-9998-e7240741ab10

so these tests lock Python's uuid5 to Postgres' uuid_generate_v5 byte-for-byte.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.causal_validation import (
    CAUSAL_PATH_ESTIMATE_NAMESPACE,
    CausalValidationRepository,
    derive_causal_path_estimate_id,
)

# Measured against live pg (see module docstring) — do NOT recompute in Python;
# the whole point is the cross-language pin.
PG_VECTOR_CP0001 = "8a0c3f5f-247f-59f7-bf51-983e12274243"
PG_VECTOR_SCP = "d28b51a1-775c-531d-9998-e7240741ab10"


@pytest.mark.unit
def test_namespace_prefix_is_pinned():
    """The prefix is load-bearing: migration 119 hardcodes the same literal."""
    assert CAUSAL_PATH_ESTIMATE_NAMESPACE == "e2i:causal_paths:"


@pytest.mark.unit
def test_derive_estimate_id_matches_postgres_known_vectors():
    assert derive_causal_path_estimate_id("CP-0001") == PG_VECTOR_CP0001
    assert derive_causal_path_estimate_id("scp_001cdc6864504") == PG_VECTOR_SCP


@pytest.mark.unit
def test_derive_estimate_id_deterministic_and_distinct():
    a1 = derive_causal_path_estimate_id("scp_aaaa")
    a2 = derive_causal_path_estimate_id("scp_aaaa")
    b = derive_causal_path_estimate_id("scp_bbbb")
    assert a1 == a2
    assert a1 != b


def _client_returning(rows):
    """Mock the supabase async chain table().select().in_().eq().execute()."""
    client = MagicMock()
    execute = AsyncMock(return_value=MagicMock(data=rows))
    chain = MagicMock()
    chain.execute = execute
    chain.in_ = MagicMock(return_value=chain)
    chain.eq = MagicMock(return_value=chain)
    chain.select = MagicMock(return_value=chain)
    client.table = MagicMock(return_value=chain)
    return client


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_rows_for_paths_maps_rows_back_to_path_ids():
    rows = [
        {"estimate_id": PG_VECTOR_CP0001, "test_type": "placebo_treatment", "status": "passed"},
        {"estimate_id": PG_VECTOR_CP0001, "test_type": "bootstrap", "status": "passed"},
        {"estimate_id": PG_VECTOR_SCP, "test_type": "placebo_treatment", "status": "passed"},
    ]
    repo = CausalValidationRepository(_client_returning(rows))
    out = await repo.get_rows_for_paths(["CP-0001", "scp_001cdc6864504", "scp_no_evidence"])
    assert set(out.keys()) == {"CP-0001", "scp_001cdc6864504"}
    assert len(out["CP-0001"]) == 2
    assert len(out["scp_001cdc6864504"]) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_rows_for_paths_empty_input_returns_empty():
    repo = CausalValidationRepository(_client_returning([]))
    assert await repo.get_rows_for_paths([]) == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_rows_for_paths_propagates_query_errors():
    """Deliberately NOT the log-and-return-[] sibling convention: the chat
    caller must distinguish 'no evidence on record' (honest None) from
    'lookup failed' (must not be presented as absence of evidence)."""
    client = _client_returning([])
    client.table.return_value.execute = AsyncMock(side_effect=RuntimeError("boom"))
    repo = CausalValidationRepository(client)
    with pytest.raises(RuntimeError):
        await repo.get_rows_for_paths(["CP-0001"])
