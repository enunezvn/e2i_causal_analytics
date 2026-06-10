"""Faithful real-DB integration tests for the feedback_learner knowledge stores (#837).

Proves, against the REAL Supabase, that wiring real ``knowledge_stores`` makes
``update_effectiveness`` a genuine measured ratio:

* ``SupabaseKnowledgeStore.update`` durably persists the recorded value, bumps
  version on re-update, READS BACK to confirm, and exposes it via ``get`` — and
  refuses (returns False, writes nothing) for an empty value;
* the REAL ``KnowledgeUpdaterNode`` with real stores applies the proposed updates
  (``update_backend_wired=True``), and the REAL ``_finalize_training_signal``
  then computes ``update_effectiveness > 0`` (not None, not a fabricated 0.0).

Run gate: ``E2I_DB_INTEGRATION=1`` + a reachable async Supabase client
(``SUPABASE_URL`` in env). NO mocks; every row is isolated by a unique tag and
torn down deterministically in a ``finally`` block.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason=(
        "E2I_DB_INTEGRATION!=1; integration test requires a real Supabase client "
        "and explicit opt-in. Run with E2I_DB_INTEGRATION=1 and SUPABASE_URL set."
    ),
)

from src.agents.feedback_learner.graph import _finalize_training_signal  # noqa: E402
from src.agents.feedback_learner.knowledge_stores import (  # noqa: E402
    build_knowledge_stores,
)
from src.agents.feedback_learner.nodes.knowledge_updater import (  # noqa: E402
    KnowledgeUpdaterNode,
)
from src.memory.services.factories import get_async_supabase_client  # noqa: E402

_TABLE = "agent_knowledge_store"


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the cached async client so each test builds a fresh one on its own
    event loop (mirrors the other tests/integration suites)."""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


async def _cleanup(client, keys) -> None:
    for kt, key in keys:
        try:
            await client.table(_TABLE).delete().eq("knowledge_type", kt).eq("key", key).execute()
        except Exception:
            pass


async def test_store_update_persists_reads_back_and_bumps_version():
    client = await get_async_supabase_client()
    stores = build_knowledge_stores(client)
    assert set(stores) == {"baseline", "agent_config", "prompt", "threshold"}

    tag = uuid.uuid4().hex[:10]
    key = f"f837_{tag}"
    none_key = f"f837none_{tag}"
    keys = [("baseline", key), ("baseline", none_key)]
    try:
        store = stores["baseline"]

        # First update: persisted + read-back True, version 1.
        assert await store.update(key=key, value="raise baseline X", justification="why") is True
        assert await store.get(key) == "raise baseline X"
        rows = (
            await client.table(_TABLE)
            .select("*")
            .eq("knowledge_type", "baseline")
            .eq("key", key)
            .execute()
        ).data
        assert len(rows) == 1
        assert rows[0]["value"] == "raise baseline X"
        assert rows[0]["version"] == 1
        assert rows[0]["justification"] == "why"

        # Re-update: same (type,key) reused, version bumps to 2, new value read back.
        assert (
            await store.update(key=key, value="raise baseline X v2", justification="why2") is True
        )
        assert await store.get(key) == "raise baseline X v2"
        rows2 = (
            await client.table(_TABLE)
            .select("version,value")
            .eq("knowledge_type", "baseline")
            .eq("key", key)
            .execute()
        ).data
        assert len(rows2) == 1
        assert rows2[0]["version"] == 2
        assert rows2[0]["value"] == "raise baseline X v2"

        # Empty value: not applied, nothing written.
        assert await store.update(key=none_key, value=None, justification="x") is False
        assert (
            await client.table(_TABLE)
            .select("key")
            .eq("knowledge_type", "baseline")
            .eq("key", none_key)
            .execute()
        ).data == []
    finally:
        await _cleanup(client, keys)


async def test_node_with_real_stores_yields_positive_effectiveness():
    """The acceptance criterion: real stores → a proposed update is applied +
    persisted → the real finalize emits update_effectiveness > 0."""
    client = await get_async_supabase_client()
    stores = build_knowledge_stores(client)

    tag = uuid.uuid4().hex[:10]
    agent = f"f837agent_{tag}"
    recommendations = [
        {
            "recommendation_id": f"R1_{tag}",
            "category": "data_update",  # -> baseline
            "affected_agents": [agent],
            "proposed_change": "update baseline knowledge",
            "description": "d1",
        },
        {
            "recommendation_id": f"R2_{tag}",
            "category": "config_change",  # -> agent_config
            "affected_agents": [agent],
            "proposed_change": "tune retry config",
            "description": "d2",
        },
        {
            "recommendation_id": f"R3_{tag}",
            "category": "prompt_update",  # -> prompt
            "affected_agents": [agent],
            "proposed_change": "add formatting guidance",
            "description": "d3",
        },
        {
            "recommendation_id": f"R4_{tag}",
            "category": "threshold",  # -> threshold
            "affected_agents": [agent],
            "proposed_change": "0.7",
            "description": "d4",
        },
    ]
    keys = [(kt, agent) for kt in ("baseline", "agent_config", "prompt", "threshold")]
    try:
        node = KnowledgeUpdaterNode(stores)
        out = await node.execute(
            {"status": "running", "learning_recommendations": recommendations}  # type: ignore[arg-type]
        )
        assert out["status"] == "completed"
        assert out["update_backend_wired"] is True
        assert len(out["proposed_updates"]) == 4
        assert len(out["applied_updates"]) == 4  # all four persisted + read-back confirmed

        # The REAL finalize computes a genuine, positive update_effectiveness.
        final = await _finalize_training_signal(out)  # type: ignore[arg-type]
        signal = final["training_signal"]
        assert signal.update_effectiveness is not None
        assert signal.update_effectiveness > 0
        assert signal.update_effectiveness == pytest.approx(1.0)

        # The reward now includes a real update_effectiveness term (> 0).
        assert signal.compute_reward() > 0

        # Each store's value is durably persisted + readable.
        for kt in ("baseline", "agent_config", "prompt", "threshold"):
            assert await stores[kt].get(agent) is not None
    finally:
        await _cleanup(client, keys)


async def test_build_production_feedback_stores_wires_real_stores():
    """The shared builder used by the Celery task, the /feedback/learn route, and
    process_feedback_batch returns a real feedback_store + the 4 knowledge stores
    when the async Supabase client is available — so those entry points run a
    fully-wired cycle (update_effectiveness measurable)."""
    from src.agents.feedback_learner.agent import build_production_feedback_stores

    feedback_store, knowledge_stores = await build_production_feedback_stores()
    assert feedback_store is not None
    assert knowledge_stores is not None
    assert set(knowledge_stores) == {"baseline", "agent_config", "prompt", "threshold"}
