"""#1515: ``ChatbotOptimizer.get_pending_requests`` must read the 035 table.

The defect: the method read only the in-process ``_pending_requests`` list, so
a worker restart lost the queue and the drainer had nothing durable to show.
These tests pin the new contract:

- with a DB client available, the method returns what the
  ``chatbot_optimization_requests`` TABLE says — process memory is ignored;
- the module filter is pushed into the query;
- the in-memory list remains ONLY as a fallback when no DB client can be
  built (mirrors ``get_training_signals``' fallback discipline).

The DB is substituted with the explicit FakeQueueDB stand-in (see
_fake_supabase_queue.py); the real table read runs in
tests/integration/test_chatbot_optimization_queue_db.py under
E2I_DB_INTEGRATION=1.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.api.routes.chatbot_dspy import (
    ChatbotOptimizationRequest,
    ChatbotOptimizer,
)
from tests.unit.test_tasks._fake_supabase_queue import FakeQueueDB, make_request_row

FACTORY = "src.memory.services.factories.get_async_supabase_service_client"


def _memory_only_request(request_id: str = "mem_only") -> ChatbotOptimizationRequest:
    return ChatbotOptimizationRequest(
        request_id=request_id,
        module_name="synthesizer",
        signal_count=10,
        min_reward=0.5,
        budget="light",
    )


@pytest.mark.asyncio
async def test_reads_table_not_process_memory():
    """DB rows win; a process-local entry absent from the table is NOT returned."""
    db = FakeQueueDB(
        rows=[
            make_request_row("db_req_1", "intent_classifier", priority=2),
            make_request_row("db_req_2", "agent_router"),
            make_request_row("db_done", "agent_router", status="completed"),
        ]
    )
    optimizer = ChatbotOptimizer()
    optimizer._pending_requests = [_memory_only_request()]

    with patch(FACTORY, new=AsyncMock(return_value=db)):
        pending = await optimizer.get_pending_requests()

    ids = {r.request_id for r in pending}
    assert ids == {"db_req_1", "db_req_2"}, (
        "get_pending_requests must return the TABLE's pending rows "
        f"(worker-restart durable), got {ids}"
    )
    assert all(isinstance(r, ChatbotOptimizationRequest) for r in pending)
    # Priority-ordered: db_req_1 has priority 2 and must come first.
    assert pending[0].request_id == "db_req_1"


@pytest.mark.asyncio
async def test_module_filter_hits_the_query():
    db = FakeQueueDB(
        rows=[
            make_request_row("db_intent", "intent_classifier"),
            make_request_row("db_router", "agent_router"),
        ]
    )
    optimizer = ChatbotOptimizer()

    with patch(FACTORY, new=AsyncMock(return_value=db)):
        pending = await optimizer.get_pending_requests("agent_router")

    assert [r.request_id for r in pending] == ["db_router"]
    # The filter must be part of the DB query, not client-side post-filtering
    # of an unfiltered read.
    select_ops = [op for op in db.table_ops if op["mode"] == "select"]
    assert select_ops, "expected a table select"
    assert any(("eq", "module_name", "agent_router") in op["filters"] for op in select_ops), (
        f"module filter not pushed into the query: {select_ops}"
    )


@pytest.mark.asyncio
async def test_row_fields_map_onto_dataclass():
    db = FakeQueueDB(
        rows=[
            make_request_row(
                "db_full",
                "query_rewriter",
                priority=3,
                budget="medium",
                min_reward=0.7,
                signal_count=123,
            )
        ]
    )
    optimizer = ChatbotOptimizer()

    with patch(FACTORY, new=AsyncMock(return_value=db)):
        (req,) = await optimizer.get_pending_requests()

    assert req.module_name == "query_rewriter"
    assert req.priority == 3
    assert req.budget == "medium"
    assert req.min_reward == 0.7
    assert req.signal_count == 123
    assert req.status == "pending"


@pytest.mark.asyncio
async def test_fallback_to_memory_when_client_unavailable():
    """No DB client -> pre-#1515 in-memory behavior (explicit, degraded path)."""
    optimizer = ChatbotOptimizer()
    optimizer._pending_requests = [_memory_only_request("mem_1")]

    with patch(FACTORY, new=AsyncMock(side_effect=RuntimeError("no supabase env"))):
        pending = await optimizer.get_pending_requests()

    assert [r.request_id for r in pending] == ["mem_1"]

    with patch(FACTORY, new=AsyncMock(side_effect=RuntimeError("no supabase env"))):
        filtered = await optimizer.get_pending_requests("intent_classifier")
    assert filtered == []
