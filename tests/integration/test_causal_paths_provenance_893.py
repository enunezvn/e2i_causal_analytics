"""#893 faithful integration: causal_paths provenance (the chat leak), live DB.

``causal_paths`` is an ``is_synthetic``-tagged table (migration 063:18) whose
live substrate is 250/250 synthetic (post the 2026-06-11 synthetic-gold-only
cleanup), but ``CausalPathRepository`` never set ``HAS_PROVENANCE = True`` —
so the user-visible chat tool ``_query_causal_chains`` returned planted
ground-truth test paths as real insight on every call.

These tests drive the REAL repository and the REAL chat helper against the
live DB, red-first:

  * RED   — default (real-mode) reads return the seeded synthetic row and the
            chat helper surfaces synthetic paths; ``include_synthetic`` does
            not even exist on the helper.
  * GREEN — ``HAS_PROVENANCE = True`` makes every real-mode read
            default-exclude synthetic rows (on today's all-synthetic substrate
            the chat tool honestly returns EMPTY — the #872 fail-closed
            semantics, not a regression) while ``include_synthetic=True``
            opt-in still reaches the synthetic substrate.

Self-cleaning: each test brackets the causal_paths row count and deletes its
seeded rows; post-count must equal pre-count.

Run with the shared-DB lock::

    flock -w 2400 /tmp/e2i_dbtest.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_causal_paths_provenance_893.py'
"""

import os
import uuid
from datetime import datetime, timezone

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB causal_paths provenance test; "
        "set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]

# The live table holds 250 synthetic rows; any read that should see the whole
# table must use a limit comfortably above that.
_WHOLE_TABLE_LIMIT = 400


def _seed_rows(marker: str) -> tuple[dict, dict]:
    """One synthetic + one real causal path, isolated by a unique brand marker."""
    token = uuid.uuid4().hex[:10]
    common = {
        "discovery_date": "2026-06-12",
        "start_node": "hcp_engagement",
        "end_node": "trx_growth",
        "causal_effect_size": 0.123,
        "confidence_level": 0.9,
        "brand": marker,
        "data_split": "unassigned",
    }
    synthetic = {
        **common,
        "path_id": f"T893{token}S",  # varchar(20) PK
        "causal_chain": {"nodes": ["hcp_engagement", "trx_growth"], "provenance": "synthetic"},
        "is_synthetic": True,
    }
    real = {
        **common,
        "path_id": f"T893{token}R",
        "causal_chain": {"nodes": ["hcp_engagement", "trx_growth"], "provenance": "real"},
        "is_synthetic": False,
    }
    return synthetic, real


async def _count_rows(client) -> int:
    res = await client.table("causal_paths").select("path_id", count="exact").limit(1).execute()
    return int(res.count)


@pytest.fixture
async def seeded():
    """Seed 1 synthetic + 1 real causal path; bracket + restore the row count.

    Resets the factory's cached async client around each test (the established
    realdb-suite idiom) so the per-function event loop never reuses a pooled
    connection bound to a closed loop.
    """
    import src.memory.services.factories as factories
    from src.memory.services.factories import get_async_supabase_client

    factories._async_supabase_client = None
    client = await get_async_supabase_client()
    assert client is not None, "async supabase client unavailable (creds?)"

    pre_count = await _count_rows(client)
    marker = f"e2i893-{uuid.uuid4().hex[:8]}"
    synthetic, real = _seed_rows(marker)
    await client.table("causal_paths").insert([synthetic, real]).execute()
    try:
        yield {
            "client": client,
            "marker": marker,
            "synthetic_id": synthetic["path_id"],
            "real_id": real["path_id"],
        }
    finally:
        await (
            client.table("causal_paths")
            .delete()
            .in_("path_id", [synthetic["path_id"], real["path_id"]])
            .execute()
        )
        post_count = await _count_rows(client)
        factories._async_supabase_client = None
        assert post_count == pre_count, (
            f"causal_paths row count not restored: pre={pre_count} post={post_count}"
        )


# =============================================================================
# Repository read paths
# =============================================================================


async def test_get_many_default_excludes_synthetic(seeded):
    """Real-mode get_many must not return the seeded synthetic row (RED pre-fix)."""
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=seeded["client"])
    rows = await repo.get_many(filters={"brand": seeded["marker"]})

    returned_ids = {row["path_id"] for row in rows}
    assert seeded["synthetic_id"] not in returned_ids, (
        "real-mode get_many leaked a synthetic causal path"
    )
    assert returned_ids == {seeded["real_id"]}, (
        f"real-mode get_many should return exactly the real row, got {returned_ids}"
    )


async def test_get_many_opt_in_returns_synthetic(seeded):
    """include_synthetic=True must still reach the synthetic substrate."""
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=seeded["client"])
    rows = await repo.get_many(filters={"brand": seeded["marker"]}, include_synthetic=True)

    returned_ids = {row["path_id"] for row in rows}
    assert returned_ids == {seeded["synthetic_id"], seeded["real_id"]}


async def test_get_by_brand_default_excludes_synthetic(seeded):
    """The bespoke get_by_brand read inherits the same real-mode exclusion."""
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=seeded["client"])
    rows = await repo.get_by_brand(brand=seeded["marker"])

    returned_ids = {row["path_id"] for row in rows}
    assert returned_ids == {seeded["real_id"]}


async def test_get_by_brand_opt_in_returns_synthetic(seeded):
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=seeded["client"])
    rows = await repo.get_by_brand(brand=seeded["marker"], include_synthetic=True)

    returned_ids = {row["path_id"] for row in rows}
    assert returned_ids == {seeded["synthetic_id"], seeded["real_id"]}


# =============================================================================
# The user-visible chat tool (no mocks — the real helper, the real DB)
# =============================================================================


async def test_chat_tool_real_mode_returns_no_synthetic_rows(seeded):
    """_query_causal_chains must never surface synthetic paths to chat users.

    Pre-fix RED: the live 250/250-synthetic substrate flows straight into the
    chat answer. Post-fix: only non-synthetic rows survive (on an all-synthetic
    substrate that means honest-empty — the seeded real row is the only hit).
    """
    from src.api.routes.chatbot_tools import _query_causal_chains

    result = await _query_causal_chains(
        brand=None,
        kpi_name=None,
        since=datetime.now(timezone.utc),
        limit=_WHOLE_TABLE_LIMIT,
    )

    assert result["success"] is True, f"chat helper errored: {result.get('error')}"
    synthetic_returned = [row["path_id"] for row in result["data"] if row.get("is_synthetic")]
    assert synthetic_returned == [], (
        f"chat tool leaked {len(synthetic_returned)} synthetic causal paths "
        f"(e.g. {synthetic_returned[:3]})"
    )
    returned_ids = {row["path_id"] for row in result["data"]}
    assert seeded["real_id"] in returned_ids, "chat tool dropped a genuinely real row"
    assert seeded["synthetic_id"] not in returned_ids


async def test_chat_tool_opt_in_reaches_synthetic_substrate(seeded):
    """Explicit include_synthetic=True (agent-context/validation) still reads it."""
    from src.api.routes.chatbot_tools import _query_causal_chains

    result = await _query_causal_chains(
        brand=None,
        kpi_name=None,
        since=datetime.now(timezone.utc),
        limit=_WHOLE_TABLE_LIMIT,
        include_synthetic=True,
    )

    assert result["success"] is True, f"chat helper errored: {result.get('error')}"
    returned_ids = {row["path_id"] for row in result["data"]}
    assert seeded["synthetic_id"] in returned_ids
    assert seeded["real_id"] in returned_ids
