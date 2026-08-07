"""Real-DB drain-cycle tests for #1515 — the REAL 035 lifecycle functions.

Runs the enqueue -> peek -> claim -> status-transition -> close-out cycle
against the live local Supabase, exercising the actual
``insert_optimization_request`` / ``get_next_optimization_request`` /
``update_optimization_request_status`` / ``cancel_stale_optimization_requests``
SQL functions plus the drainer's compare-and-set claim, so the unit suite's
FakeQueueDB cannot drift from the schema.

The ONE substitution is at the LLM boundary: ``ChatbotOptimizer.
optimize_module`` is monkeypatched in the full-cycle test, because the real
call fires an unbounded GEPA optimization (dspy 3.1.0 auto="light" is ~390
metric calls). That substitution is deliberate and visible here; the seam
itself is pinned to the real optimizer by
tests/unit/test_tasks/test_chatbot_optimization_drainer_1515.py::
TestExecutorSeam. Everything else in this file is real: real client, real
table, real SQL functions, real RLS/SECURITY DEFINER paths.

Opt-in (real docker supabase-db required), skipped in CI by default:

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_chatbot_optimization_queue_db.py -n 0 -p no:cacheprovider

Every row this file inserts has a ``test1515_`` request_id prefix and is
deleted in fixture teardown. Tests refuse to run (skip) if the live queue
holds foreign pending/processing rows, rather than draining real work.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

QUEUE_TABLE = "chatbot_optimization_requests"
PREFIX = "test1515_"


async def _client():
    from src.memory.services.factories import get_async_supabase_service_client

    return await get_async_supabase_service_client()


async def _cleanup(client) -> None:
    await client.table(QUEUE_TABLE).delete().like("request_id", f"{PREFIX}%").execute()


async def _fetch(client, request_id: str) -> dict:
    result = await client.table(QUEUE_TABLE).select("*").eq("request_id", request_id).execute()
    assert result.data, f"row {request_id} not found"
    return result.data[0]


async def _foreign_queue_rows(client) -> int:
    result = (
        await client.table(QUEUE_TABLE)
        .select("request_id,status")
        .in_("status", ["pending", "processing"])
        .execute()
    )
    return sum(1 for r in (result.data or []) if not r["request_id"].startswith(PREFIX))


@pytest.fixture(autouse=True)
def _fresh_service_client():
    """Reset the cached async service client so each test builds one on its
    OWN event loop (the global cache binds httpx.AsyncClient to the creating
    loop; pytest's per-test loops would otherwise reuse a client from a closed
    loop). Same discipline as test_async_supabase_client_realdb.py."""
    import src.memory.services.factories as factories

    factories._async_supabase_service_client = None
    yield
    factories._async_supabase_service_client = None


@pytest.fixture
async def db():
    client = await _client()
    await _cleanup(client)
    if await _foreign_queue_rows(client):
        pytest.skip("live optimization queue holds foreign pending work; refusing to interfere")
    yield client
    await _cleanup(client)


async def _enqueue(client, request_id: str, module_name: str = "intent_classifier", **kw) -> None:
    """Enqueue via the REAL 035 insert function."""
    params = {
        "p_request_id": request_id,
        "p_module_name": module_name,
        "p_signal_count": kw.get("signal_count", 60),
        "p_min_reward": kw.get("min_reward", 0.5),
        "p_budget": kw.get("budget", "light"),
        "p_priority": kw.get("priority", 3),
    }
    result = await client.rpc("insert_optimization_request", params).execute()
    assert result.data, "insert_optimization_request returned no id"


@pytest.mark.asyncio
async def test_lifecycle_enqueue_peek_claim_closeout(db):
    """The full row lifecycle over the real SQL functions + the CAS claim."""
    import src.tasks.chatbot_optimization_tasks as drain_mod

    rid = f"{PREFIX}lifecycle_{uuid.uuid4().hex[:8]}"
    await _enqueue(db, rid, "agent_router", budget="medium", min_reward=0.6)

    # Peek via the REAL get_next_optimization_request (priority 3 sorts first).
    row = await drain_mod._peek_next(db)
    assert row is not None and row["request_id"] == rid
    assert row["module_name"] == "agent_router"
    assert row["budget"] == "medium"
    assert float(row["min_reward"]) == 0.6

    # Claim once: CAS pending -> processing.
    assert await drain_mod._claim(db, rid) is True
    stored = await _fetch(db, rid)
    assert stored["status"] == "processing"
    assert stored["started_at"] is not None

    # Claim twice: the compare-and-set must LOSE on a non-pending row.
    assert await drain_mod._claim(db, rid) is False

    # A claimed row is invisible to the peek.
    assert await drain_mod._peek_next(db) is None

    # Close out via the REAL update_optimization_request_status.
    assert (
        await drain_mod._close_out(db, rid, "completed", baseline_score=0.5, optimized_score=0.6)
        is True
    )
    stored = await _fetch(db, rid)
    assert stored["status"] == "completed"
    assert stored["completed_at"] is not None
    assert float(stored["optimized_score"]) == 0.6
    # 035 computes improvement server-side: (0.6-0.5)/0.5 * 100 = 20%.
    assert abs(float(stored["improvement_percent"]) - 20.0) < 1e-6


@pytest.mark.asyncio
async def test_enqueued_request_survives_process_restart(db):
    """#1515 core defect: a request enqueued by ANOTHER process (no shared
    memory) must be visible to a fresh ChatbotOptimizer — table-backed reads."""
    from src.api.routes.chatbot_dspy import ChatbotOptimizer

    rid = f"{PREFIX}durable_{uuid.uuid4().hex[:8]}"
    await _enqueue(db, rid, "query_rewriter")

    optimizer = ChatbotOptimizer()  # fresh instance == empty process memory
    assert optimizer._pending_requests == []

    pending = await optimizer.get_pending_requests()
    assert rid in {r.request_id for r in pending}

    filtered = await optimizer.get_pending_requests("query_rewriter")
    assert rid in {r.request_id for r in filtered}
    assert await optimizer.get_pending_requests("synthesizer") == []


@pytest.mark.asyncio
async def test_full_drain_cycle_completes_request_against_real_db(db, monkeypatch):
    """enqueue -> _drain_cycle() -> completed, with ONLY the GEPA execution
    substituted (explicit LLM-boundary stub; see module docstring)."""
    import src.tasks.chatbot_optimization_tasks as drain_mod
    from src.api.routes.chatbot_dspy import ChatbotOptimizer

    rid = f"{PREFIX}drain_{uuid.uuid4().hex[:8]}"
    await _enqueue(db, rid, "synthesizer", budget="light", min_reward=0.55)

    # EXPLICIT substitution at the expensive-LLM boundary: a real call here
    # would launch a real GEPA run (#1513 cost-gate rationale).
    fake_optimize = AsyncMock(return_value={"success": True, "best_score": 0.77})
    monkeypatch.setattr(ChatbotOptimizer, "optimize_module", fake_optimize)

    # The producer must hold off while our request is in flight (dedup guard);
    # patching it to a mock ALSO keeps the run off chatbot_training_signals.
    producer = AsyncMock()
    monkeypatch.setattr("src.api.routes.chatbot_dspy.submit_signals_for_optimization", producer)

    monkeypatch.setenv(drain_mod.DRAIN_ENABLED_ENV, "1")
    result = await drain_mod._drain_cycle()

    assert result["status"] == "completed"
    assert {
        "request_id": rid,
        "module_name": "synthesizer",
        "status": "completed",
    } in result["executed"]
    fake_optimize.assert_awaited_once_with("synthesizer", budget="light", min_reward=0.55)
    producer.assert_not_awaited()

    stored = await _fetch(db, rid)
    assert stored["status"] == "completed"
    assert stored["started_at"] is not None
    assert stored["completed_at"] is not None
    assert float(stored["optimized_score"]) == 0.77

    # And the durable pending view no longer serves it.
    optimizer = ChatbotOptimizer()
    assert rid not in {r.request_id for r in await optimizer.get_pending_requests()}


@pytest.mark.asyncio
async def test_orphaned_processing_row_is_requeued(db):
    """Worker-death recovery over the real table: a 'processing' row whose
    started_at predates the zombie cutoff returns to 'pending'."""
    import src.tasks.chatbot_optimization_tasks as drain_mod

    rid = f"{PREFIX}zombie_{uuid.uuid4().hex[:8]}"
    await _enqueue(db, rid)
    assert await drain_mod._claim(db, rid) is True

    # Backdate the claim far past the default 12h cutoff.
    stale_start = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await db.table(QUEUE_TABLE).update({"started_at": stale_start}).eq("request_id", rid).execute()

    recovered = await drain_mod._recover_zombies(db)
    assert recovered == 1
    stored = await _fetch(db, rid)
    assert stored["status"] == "pending"
    assert stored["started_at"] is None

    # And it is claimable again — the drain loop can retry it.
    row = await drain_mod._peek_next(db)
    assert row is not None and row["request_id"] == rid
