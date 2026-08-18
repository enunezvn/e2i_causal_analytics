"""#1681: ``get_signals_for_optimization`` must work with an ASYNC Supabase client.

The live defect this pins:

    /api/feedback/health -> "Optimizer gate status unavailable
                             ('coroutine' object has no attribute 'data')"
    every count null, operator card renders "— / 40"

``src/api/routes/feedback.py`` builds its client with ``get_async_supabase_client()``
and hands it down to ``get_optimizer_gate_status`` -> ``read_optimizer_signal_pool``
-> this adapter. The adapter ran ``query.execute()`` inside ``run_in_executor`` —
correct for a SYNC client, which would otherwise block the event loop — but an
ASYNC client's ``execute()`` returns a **coroutine**, and ``run_in_executor``
hands that coroutine back unawaited. ``response.data`` then raises.

These tests assert VALUES, not keys. The probe that missed this in production
checked that the response had the right field NAMES, which stayed true while
every value went null — a rename is visible in the keys, a dead read only in the
values. So each test below asserts the actual rows come back.

Both client shapes are covered in one parametrized pair on purpose: fixing the
async path by simply awaiting ``execute()`` on the loop would silently
re-introduce event-loop blocking for the sync client, and only the sync case
catches that regression.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from src.rag.memory_adapters import SignalCollectorAdapter

ROWS: List[Dict[str, Any]] = [
    {"signal_id": "s1", "source_agent": "feedback_learner", "reward": 0.9},
    {"signal_id": "s2", "source_agent": "feedback_learner", "reward": 0.4},
]


class _Response:
    def __init__(self, data):
        self.data = data


class _Query:
    """Mimics the postgrest builder chain; ``execute`` is sync or async."""

    def __init__(self, is_async: bool, calls: list):
        self._is_async = is_async
        self._calls = calls

    def _chain(self, name):
        self._calls.append(name)
        return self

    def select(self, *a, **k):
        return self._chain("select")

    def eq(self, *a, **k):
        return self._chain("eq")

    def gte(self, *a, **k):
        return self._chain("gte")

    def order(self, *a, **k):
        return self._chain("order")

    def limit(self, *a, **k):
        return self._chain("limit")

    def execute(self):
        if self._is_async:

            async def _run():
                return _Response(ROWS)

            return _run()  # a coroutine — the shape that broke production
        return _Response(ROWS)


class _Client:
    def __init__(self, is_async: bool):
        self.is_async = is_async
        self.calls: list = []

    def table(self, _name):
        return _Query(self.is_async, self.calls)


@pytest.mark.parametrize("is_async", [True, False], ids=["async_client", "sync_client"])
def test_get_signals_for_optimization_returns_rows_for_either_client(is_async):
    """RED before the fix on the async client: raises 'coroutine' has no attribute 'data'."""
    adapter = SignalCollectorAdapter(supabase_client=_Client(is_async))

    rows = asyncio.run(
        adapter.get_signals_for_optimization(
            source_agent="feedback_learner", min_reward=0.0, limit=10, strict=True
        )
    )

    # VALUES, not keys — the assertion the production probe failed to make.
    assert rows == ROWS, f"expected the real rows back, got {rows!r}"


def test_async_client_read_does_not_report_an_empty_pool():
    """A swallowed failure here publishes a fabricated 0 on a health surface.

    With ``strict=False`` the adapter returns ``[]`` on error. That is right for
    best-effort readers, but these rows are COUNTED and the count is published,
    so "0 examples" would be indistinguishable from a genuinely empty corpus —
    the fabricated zero #1661/#1668 forbid. Before the fix this returned ``[]``.
    """
    adapter = SignalCollectorAdapter(supabase_client=_Client(is_async=True))

    rows = asyncio.run(
        adapter.get_signals_for_optimization(
            source_agent="feedback_learner", min_reward=0.0, limit=10, strict=False
        )
    )

    assert rows != [], "async client read silently produced an empty pool (fabricated zero)"
    assert len(rows) == len(ROWS)


def test_sync_client_execute_still_runs_off_the_event_loop():
    """Guards the regression the obvious fix would introduce.

    ``run_in_executor`` exists here so a blocking sync ``execute()`` does not
    stall the loop. "Just await execute()" fixes the async client and quietly
    moves the sync client's blocking call onto the loop thread. This asserts the
    sync call is still made from a worker thread, not the loop thread.
    """
    import threading

    loop_thread_name = {}
    seen = {}

    class _RecordingQuery(_Query):
        def execute(self):
            seen["thread"] = threading.current_thread().name
            return _Response(ROWS)

    class _RecordingClient(_Client):
        def table(self, _name):
            return _RecordingQuery(False, self.calls)

    async def _run():
        loop_thread_name["name"] = threading.current_thread().name
        adapter = SignalCollectorAdapter(supabase_client=_RecordingClient(False))
        return await adapter.get_signals_for_optimization(
            source_agent="feedback_learner", min_reward=0.0, limit=10, strict=True
        )

    rows = asyncio.run(_run())

    assert rows == ROWS
    assert seen["thread"] != loop_thread_name["name"], (
        "sync execute() ran on the event-loop thread — the run_in_executor offload was lost"
    )
