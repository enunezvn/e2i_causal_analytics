"""#1659 — the chat SSE stream must never go silent longer than nginx tolerates.

MEASURED on production (2026-08-16, request_id ``probe1659-1786891760``, sent
through the live host nginx at https://eznomics.site/api/copilotkit/chat/stream
on a turn whose ``dispatch_info.agents_dispatched`` was
``["heterogeneous_optimizer", "gap_analyzer"]``):

    t=    860.9 ms   115 B   {"type": "session_id", ...}
    t=  35256.6 ms   712 B   {"type": "text", ...}        <- gap 34395.7 ms
    t=  35446.9 ms   103 B   {"type": "conversation_title", ...}
    t=  35449.7 ms  1490 B   {"type": "dispatch_info", ...}
    t=  35450.5 ms    36 B   {"type": "done", ...}

The server-side span for the same request:

    node_wall_ms={"init": 741.4, "load_context": 577.4, "classify_intent": 2628.8,
                  "retrieve_rag": 23463.9, "orchestrator": 6977.8,
                  "generate": 0.1, "finalize": 184.6}

Those node times sum to 34389.4 ms against a measured 34395.7 ms client-side
silent window — 6 ms apart. In other words the stream emits ``session_id`` when
the connection opens and then **nothing until the whole graph finishes**: the
silent window is the entire turn, not just the agent dispatch.

``proxy_read_timeout`` bounds exactly that silent window (it resets on every
byte nginx reads from upstream), so with no keepalive the binding constraint is
``total turn wall time < 300 s``, which the ``heterogeneous_optimizer`` dispatch
budget (420 s, ``router.py``) blows through on its own.

These tests pin the fix: the SSE body must interleave keepalive frames while the
upstream generator is quiet.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List

import pytest

from src.api.utils.sse_keepalive import (
    PROXY_READ_TIMEOUT_SECONDS,
    SSE_KEEPALIVE_FRAME,
    SSE_KEEPALIVE_INTERVAL_SECONDS,
    with_sse_keepalive,
)


async def _silent_then_answer(quiet_seconds: float) -> AsyncGenerator[str, None]:
    """Reproduce the measured shape: one frame, a long silence, then the answer."""
    yield 'data: {"type": "session_id", "data": "x"}\n\n'
    await asyncio.sleep(quiet_seconds)
    yield 'data: {"type": "text", "data": "answer"}\n\n'
    yield 'data: {"type": "done", "data": ""}\n\n'


@pytest.mark.asyncio
async def test_keepalive_breaks_the_silent_window() -> None:
    """A quiet upstream must not produce a quiet socket."""
    interval = 0.05
    quiet = 0.35

    frames: List[str] = []
    async for frame in with_sse_keepalive(_silent_then_answer(quiet), interval_seconds=interval):
        frames.append(frame)

    keepalives = [f for f in frames if f == SSE_KEEPALIVE_FRAME]
    payloads = [f for f in frames if f != SSE_KEEPALIVE_FRAME]

    # The real events survive, in order, unmodified.
    assert payloads == [
        'data: {"type": "session_id", "data": "x"}\n\n',
        'data: {"type": "text", "data": "answer"}\n\n',
        'data: {"type": "done", "data": ""}\n\n',
    ]

    # And the silence was broken repeatedly rather than once.
    assert len(keepalives) >= 3, f"expected >=3 keepalives across {quiet}s, got {len(keepalives)}"


@pytest.mark.asyncio
async def test_no_gap_exceeds_the_keepalive_interval() -> None:
    """The invariant that actually matters: max inter-frame gap <= interval (+slack).

    This is the property ``proxy_read_timeout`` measures. Asserting the *gap*
    rather than the keepalive count is what makes this test a regression guard
    for the measured 34 395.7 ms window.
    """
    interval = 0.05
    quiet = 0.4

    gaps: List[float] = []
    last = asyncio.get_running_loop().time()
    async for _frame in with_sse_keepalive(_silent_then_answer(quiet), interval_seconds=interval):
        now = asyncio.get_running_loop().time()
        gaps.append(now - last)
        last = now

    assert gaps, "generator produced no frames"
    # Generous slack: this asserts the ORDER OF MAGNITUDE (interval-bounded, not
    # turn-bounded), not scheduler precision.
    assert max(gaps) < interval * 4, f"largest gap {max(gaps):.3f}s exceeded {interval * 4:.3f}s"


@pytest.mark.asyncio
async def test_keepalive_frame_is_an_sse_comment() -> None:
    """The frame must be inert to every consumer.

    An SSE comment (a line starting with ``:``) is ignored by spec-compliant
    parsers and is skipped by ``scripts/demos/copilot_chat_perf_runner.py``,
    which only decodes lines beginning with ``data: ``. Emitting a synthetic
    ``data:`` event instead would surface as a bogus chat event.
    """
    assert SSE_KEEPALIVE_FRAME.startswith(":")
    assert SSE_KEEPALIVE_FRAME.endswith("\n\n")
    assert not SSE_KEEPALIVE_FRAME.lstrip(":").lstrip().startswith("data:")


@pytest.mark.asyncio
async def test_fast_upstream_emits_no_keepalives() -> None:
    """No keepalive when the upstream is already chatty — this adds no noise."""

    async def chatty() -> AsyncGenerator[str, None]:
        for i in range(5):
            yield f'data: {{"type": "text", "data": "{i}"}}\n\n'

    frames = [f async for f in with_sse_keepalive(chatty(), interval_seconds=5.0)]
    assert SSE_KEEPALIVE_FRAME not in frames
    assert len(frames) == 5


@pytest.mark.asyncio
async def test_upstream_exception_propagates() -> None:
    """The wrapper must not swallow upstream failures into an infinite keepalive."""

    async def boom() -> AsyncGenerator[str, None]:
        yield "data: {}\n\n"
        raise RuntimeError("upstream exploded")

    with pytest.raises(RuntimeError, match="upstream exploded"):
        async for _ in with_sse_keepalive(boom(), interval_seconds=0.01):
            pass


@pytest.mark.asyncio
async def test_consumer_disconnect_closes_upstream() -> None:
    """Breaking out of the loop must tear the upstream generator down.

    Without this the orchestrator run would keep burning a heavy-compute slot
    after the client is gone.
    """
    closed = asyncio.Event()

    async def tracked() -> AsyncGenerator[str, None]:
        try:
            yield "data: {}\n\n"
            await asyncio.sleep(10)
            yield "data: {}\n\n"
        finally:
            closed.set()

    agen = with_sse_keepalive(tracked(), interval_seconds=0.01)
    async for _frame in agen:
        break
    await agen.aclose()

    await asyncio.wait_for(closed.wait(), timeout=2)
    assert closed.is_set()


def test_keepalive_interval_leaves_room_under_the_proxy_ceiling() -> None:
    """The whole point: the keepalive cadence, not the turn length, faces nginx."""
    assert SSE_KEEPALIVE_INTERVAL_SECONDS > 0
    assert SSE_KEEPALIVE_INTERVAL_SECONDS <= PROXY_READ_TIMEOUT_SECONDS / 10
