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
from typing import AsyncGenerator, List, get_args

import pytest

from src.api.utils.sse_keepalive import (
    PROXY_READ_TIMEOUT_SECONDS,
    SSE_KEEPALIVE_FRAME,
    SSE_KEEPALIVE_INTERVAL_SECONDS,
    SSEFrame,
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


# ---------------------------------------------------------------------------
# #1669 widened the accepted frame type. These pin that it was widened HONESTLY
# — i.e. the wrapper really does handle what it now claims to handle.
# ---------------------------------------------------------------------------
#
# #1672 CI caught the original bound: ``T = TypeVar("T", bound=str)`` could not
# absorb ``StreamingResponse.body_iterator``, whose declared element type is
# ``str | bytes | memoryview``. The two possible responses were to suppress the
# ``type-var`` error or to widen the bound. Widening is correct because the
# wrapper never decodes, concatenates or inspects a frame — it only re-yields
# it — so the frame's type was never load-bearing. Suppressing would have left a
# real mismatch behind a comment.
#
# MEASURED: the body that motivated the widening actually yields ``str`` today
# (43/43 chunks on both AG-UI entry points), so this is guarding a contract, not
# an observed bytes producer. That is exactly why it needs a test — nothing else
# in the repo exercises the bytes half of the signature.


async def _bytes_then_quiet_then_bytes(quiet_seconds: float) -> AsyncGenerator[bytes, None]:
    """A ``bytes`` body with a silent window — the half the ``str`` bound excluded."""
    yield b'data: {"type": "session_id"}\n\n'
    await asyncio.sleep(quiet_seconds)
    yield b'data: {"type": "text", "data": "answer"}\n\n'
    yield b'data: {"type": "done"}\n\n'


@pytest.mark.asyncio
async def test_bytes_frames_pass_through_untouched_with_keepalives_interleaved() -> None:
    """A ``bytes`` source keeps its type, its payload and its ORDER.

    The keepalive is INTERLEAVED, never substituted, and never coerces the
    upstream frames — which is what "frames pass through untouched" has to mean
    now that the frame type is no longer fixed to ``str``.
    """
    interval = 0.05
    quiet = 0.35

    frames = [
        f
        async for f in with_sse_keepalive(
            _bytes_then_quiet_then_bytes(quiet), interval_seconds=interval
        )
    ]

    payloads = [f for f in frames if f != SSE_KEEPALIVE_FRAME]
    keepalives = [f for f in frames if f == SSE_KEEPALIVE_FRAME]

    assert payloads == [
        b'data: {"type": "session_id"}\n\n',
        b'data: {"type": "text", "data": "answer"}\n\n',
        b'data: {"type": "done"}\n\n',
    ]
    assert all(isinstance(p, bytes) for p in payloads), (
        f"upstream bytes frames were coerced: {[type(p).__name__ for p in payloads]}"
    )
    assert len(keepalives) >= 3, f"expected >=3 keepalives across {quiet}s, got {len(keepalives)}"
    assert all(isinstance(k, str) for k in keepalives)

    # The keepalives landed INSIDE the silent window, not bunched at either end.
    first_payload = frames.index(payloads[0])
    second_payload = frames.index(payloads[1])
    assert second_payload - first_payload > 1, (
        f"no keepalive between the two payload frames: {[type(f).__name__ for f in frames]}"
    )


@pytest.mark.asyncio
async def test_bytes_body_survives_a_real_asgi_round_trip() -> None:
    """The decisive one: mixed ``str`` keepalive + ``bytes`` payload, over HTTP.

    Reading starlette's ``stream_response`` says it encodes any non-bytes chunk
    before ``send``. This proves it, because "the sink encodes it" is the whole
    justification for letting the wrapper emit a ``str`` keepalive into a
    ``bytes`` stream. If that were wrong the response would raise or arrive
    corrupted rather than merely mistyped.
    """
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.get("/stream")
    async def _stream() -> StreamingResponse:  # pragma: no cover - driven via client
        return StreamingResponse(
            with_sse_keepalive(_bytes_then_quiet_then_bytes(0.3), interval_seconds=0.05),
            media_type="text/event-stream",
        )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://keepalive") as client:
        async with client.stream("GET", "/stream", timeout=30) as response:
            assert response.status_code == 200
            body = "".join([chunk async for chunk in response.aiter_text()])

    records = [r for r in body.split("\n\n") if r.strip()]
    data_records = [r for r in records if r.startswith("data: ")]
    comment_records = [r for r in records if r.startswith(":")]

    assert data_records == [
        'data: {"type": "session_id"}',
        'data: {"type": "text", "data": "answer"}',
        'data: {"type": "done"}',
    ], f"bytes payload did not survive the round trip: {data_records}"
    assert comment_records, "no keepalive reached the wire"
    assert len(records) == len(data_records) + len(comment_records), (
        f"unclassifiable records on the wire: {records}"
    )


def test_sse_frame_mirrors_starlettes_content_alias() -> None:
    """``SSEFrame`` is a MIRROR of starlette's ``Content`` — keep them pinned.

    ``sse_keepalive`` deliberately does not import from starlette (it is
    transport-agnostic), so the union is duplicated. Without this test a
    starlette release that widens ``Content`` would silently reintroduce the
    #1672 mismatch at the ``body_iterator`` assignment in ``copilotkit.py``.
    """
    import starlette.responses

    starlette_content = getattr(starlette.responses, "Content", None)
    if starlette_content is None:  # pragma: no cover - alias renamed upstream
        pytest.skip("starlette.responses.Content is gone; re-derive SSEFrame against the new name")

    def _members(union: object) -> set:
        return set(get_args(union)) or {union}

    assert _members(SSEFrame) == _members(starlette_content), (
        f"SSEFrame {SSEFrame} has drifted from starlette's Content {starlette_content}; "
        "with_sse_keepalive must accept whatever a StreamingResponse body can carry (#1669)"
    )
