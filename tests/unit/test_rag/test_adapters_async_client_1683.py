"""#1683: the remaining ``execute()``-in-executor sites must work with an ASYNC client.

#1682 fixed ``get_signals_for_optimization`` (#1681). The same sync/async
mismatch remained at two more sites in the same module, one of which fails
WORSE than the fixed one:

- ``SignalCollectorAdapter.flush()``: with an async client the INSERT coroutine
  is never awaited, and because a write never reads ``.data`` on the result,
  **no exception fires** — the write silently never happens while the method
  logs "Flushed N training signals" and returns N. #1681 raised; this lies.

- ``ProceduralMemoryAdapter._execute_procedure_search``: both the RPC path and
  the table fallback raise on ``.data``, both raises are caught, and the
  method returns ``[]`` — a swallowed failure.

Test design follows ``test_signals_async_client_1681.py``:

- assert VALUES, not keys — and for the write path assert the insert actually
  EXECUTED and which rows it carried, because "no exception" is precisely this
  defect's failure mode: a test that only checks the return value passes on
  the broken code;
- each async case is paired with a sync case that passes BEFORE the fix — an
  in-file positive control, so the pair cannot be satisfied by a test that is
  simply broken;
- the event-loop offload guard is retained: the obvious fix (bare
  ``await execute()``) repairs the async client while silently moving the sync
  client's blocking call onto the loop thread.
"""

import asyncio
import threading
from typing import Any, Dict, List

import pytest

from src.rag.memory_adapters import ProceduralMemoryAdapter, SignalCollectorAdapter


class _Response:
    def __init__(self, data):
        self.data = data


class _ExecRecorder:
    """Log of execute() calls that actually RAN (not merely built a coroutine)."""

    def __init__(self):
        self.executed: List[Any] = []


# ---------------------------------------------------------------------------
# ProceduralMemoryAdapter._execute_procedure_search
# ---------------------------------------------------------------------------

PROC_ROWS: List[Dict[str, Any]] = [
    {"procedure_id": "p1", "content": "if churn risk high then prioritize call", "score": 0.9},
    {"procedure_id": "p2", "content": "if TRx dips then check access events", "score": 0.7},
]


class _ProcQuery:
    """Mimics the postgrest builder; ``execute`` is sync or async, or fails."""

    def __init__(self, is_async: bool, recorder: _ExecRecorder, tag: str, fail: bool = False):
        self._is_async = is_async
        self._recorder = recorder
        self._tag = tag
        self._fail = fail

    def select(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self._fail:
            raise RuntimeError("rpc function missing")
        if self._is_async:

            async def _run():
                self._recorder.executed.append(self._tag)
                return _Response(PROC_ROWS)

            return _run()  # a coroutine — the shape that broke #1681
        self._recorder.executed.append(self._tag)
        return _Response(PROC_ROWS)


class _ProcClient:
    def __init__(self, is_async: bool, rpc_fails: bool = False):
        self._is_async = is_async
        self._rpc_fails = rpc_fails
        self.recorder = _ExecRecorder()

    def rpc(self, _name, _params):
        return _ProcQuery(self._is_async, self.recorder, "rpc", fail=self._rpc_fails)

    def table(self, _name):
        return _ProcQuery(self._is_async, self.recorder, "table")


# ``_execute_procedure_search`` is called directly: the public search path
# goes through the semantic/global-embedding service first, and this method
# is the self-contained unit that held the defect.


@pytest.mark.parametrize("is_async", [True, False], ids=["async_client", "sync_client"])
def test_procedure_search_rpc_returns_rows_for_either_client(is_async):
    """RED before the fix on the async client: both paths raise on ``.data``, return []."""
    adapter = ProceduralMemoryAdapter(supabase_client=_ProcClient(is_async))

    rows = asyncio.run(adapter._execute_procedure_search("kisqali adoption", 5))

    assert rows == PROC_ROWS, f"expected the real rows back, got {rows!r}"


@pytest.mark.parametrize("is_async", [True, False], ids=["async_client", "sync_client"])
def test_procedure_search_table_fallback_returns_rows_for_either_client(is_async):
    """The RPC-missing fallback must also survive an async client."""
    client = _ProcClient(is_async, rpc_fails=True)
    adapter = ProceduralMemoryAdapter(supabase_client=client)

    rows = asyncio.run(adapter._execute_procedure_search("kisqali adoption", 5))

    assert rows == PROC_ROWS, f"expected the real rows back, got {rows!r}"
    assert "table" in client.recorder.executed, "fallback table query never actually ran"


# ---------------------------------------------------------------------------
# SignalCollectorAdapter.flush
# ---------------------------------------------------------------------------


class _InsertBuilder:
    def __init__(self, is_async: bool, recorder: _ExecRecorder, records):
        self._is_async = is_async
        self._recorder = recorder
        self._records = records

    def execute(self):
        if self._is_async:

            async def _run():
                self._recorder.executed.append(("insert", self._records))
                return _Response(self._records)

            return _run()
        self._recorder.executed.append(("insert", self._records))
        return _Response(self._records)


class _SignalTable:
    def __init__(self, is_async: bool, recorder: _ExecRecorder):
        self._is_async = is_async
        self._recorder = recorder

    def insert(self, records):
        return _InsertBuilder(self._is_async, self._recorder, records)


class _SignalClient:
    def __init__(self, is_async: bool):
        self._is_async = is_async
        self.recorder = _ExecRecorder()

    def table(self, name):
        assert name == "dspy_agent_training_signals"
        return _SignalTable(self._is_async, self.recorder)


SIGNALS: List[Dict[str, Any]] = [
    {"type": "feedback_learner", "query": "q1", "response": "r1", "reward": 0.8},
    {"type": "feedback_learner", "query": "q2", "response": "r2", "reward": 0.2},
]


@pytest.mark.parametrize("is_async", [True, False], ids=["async_client", "sync_client"])
def test_flush_actually_persists_for_either_client(is_async):
    """RED before the fix on the async client: flush returns 2 but the INSERT never ran.

    The count alone cannot catch this — the broken path returns the correct
    count while writing nothing. The assertion that matters is that the insert
    EXECUTED and carried the collected rows.
    """
    client = _SignalClient(is_async)
    adapter = SignalCollectorAdapter(supabase_client=client)

    async def _run():
        await adapter.collect(SIGNALS)
        return await adapter.flush()

    count = asyncio.run(_run())

    assert count == 2
    assert client.recorder.executed, "flush() reported success but the INSERT never ran"
    tag, records = client.recorder.executed[0]
    assert tag == "insert"
    assert [r["input_context"]["query"] for r in records] == ["q1", "q2"]
    assert [r["source_agent"] for r in records] == ["feedback_learner", "feedback_learner"]


def test_sync_flush_still_runs_off_the_event_loop():
    """Guards the regression the obvious fix would introduce.

    The executor offload exists so a blocking sync ``execute()`` does not stall
    the loop. "Just await execute()" fixes the async client and quietly moves
    the sync client's blocking call onto the loop thread. This asserts the sync
    insert is still made from a worker thread.
    """
    seen: Dict[str, str] = {}
    loop_thread: Dict[str, str] = {}

    class _RecordingBuilder(_InsertBuilder):
        def execute(self):
            seen["thread"] = threading.current_thread().name
            return super().execute()

    class _RecordingTable(_SignalTable):
        def insert(self, records):
            return _RecordingBuilder(False, self._recorder, records)

    class _RecordingClient(_SignalClient):
        def table(self, name):
            return _RecordingTable(False, self.recorder)

    client = _RecordingClient(is_async=False)
    adapter = SignalCollectorAdapter(supabase_client=client)

    async def _run():
        loop_thread["name"] = threading.current_thread().name
        await adapter.collect(SIGNALS)
        return await adapter.flush()

    count = asyncio.run(_run())

    assert count == 2
    assert client.recorder.executed
    assert seen["thread"] != loop_thread["name"], (
        "sync execute() ran on the event-loop thread — the run_in_executor offload was lost"
    )
