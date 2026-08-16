"""Bound the SILENT window of an SSE stream so nginx never severs a live turn.

#1659. ``proxy_read_timeout`` does not bound how long a request may take — it
bounds how long nginx will wait between two successive reads from upstream. It
resets on every byte. So the number a long-running streaming endpoint has to
respect is the largest gap between frames, not the total duration.

Measured on production 2026-08-16 (request_id ``probe1659-1786891760``, sent
through the live host nginx to ``POST /api/copilotkit/chat/stream``, on a turn
whose ``agents_dispatched`` was ``["heterogeneous_optimizer", "gap_analyzer"]``)::

    t=    860.9 ms   115 B   {"type": "session_id", ...}
    t=  35256.6 ms   712 B   {"type": "text", ...}          <- 34 395.7 ms of silence
    t=  35446.9 ms   103 B   {"type": "conversation_title", ...}
    t=  35449.7 ms  1490 B   {"type": "dispatch_info", ...}
    t=  35450.5 ms    36 B   {"type": "done", ...}

Server-side span for the same request::

    node_wall_ms={"init": 741.4, "load_context": 577.4, "classify_intent": 2628.8,
                  "retrieve_rag": 23463.9, "orchestrator": 6977.8,
                  "generate": 0.1, "finalize": 184.6}

Those sum to 34 389.4 ms against the 34 395.7 ms measured client-side gap — 6 ms
apart. The stream emits ``session_id`` at connection open and then nothing until
the graph finishes, because every frame originates from a LangGraph
node-completion update (``chatbot_graph.py`` ``astream``) and the orchestrator
is one node that internally ``ainvoke``s a nested graph. So *the silent window
is the whole turn*, and the effective pre-fix constraint was

    total turn wall time < PROXY_READ_TIMEOUT_SECONDS

which no single dispatch budget can honour on its own: ``heterogeneous_optimizer``
alone is budgeted at 420 s (``src/agents/orchestrator/nodes/router.py``) on a
MEASURED 269.7 s complete run, and the same turn also pays for
``retrieve_rag`` (23.5 s in the trace above; up to ~41 s on a novel query in a
fresh session, #1484) plus classification and finalisation.

Wrapping the response body in :func:`with_sse_keepalive` replaces that constraint
with ``SSE_KEEPALIVE_INTERVAL_SECONDS < PROXY_READ_TIMEOUT_SECONDS``, a relation
between two constants in this module — so dispatch budgets and the proxy ceiling
can no longer drift into contradiction.

What this does NOT fix: a keepalive is a coroutine, so it can only fire while the
event loop is free. Compute that blocks the loop (the #1548 / #1592 class — e.g.
``cate_estimator._calculate_cate_by_segment``'s fallback branch calls
``cf.effect_interval`` synchronously on the loop) still starves it. That is a
separate defect class, tracked separately; the keepalive strictly dominates
having none, because without it *every* long turn is severed, blocked loop or not.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, AsyncIterable, TypeVar

logger = logging.getLogger(__name__)

#: ``proxy_read_timeout`` for the locations that front the chat SSE surfaces.
#:
#: Mirrors ``docker/nginx/host-nginx.conf`` locations ``/api/`` and
#: ``/copilotkit/``. That file is the config actually deployed — verified
#: 2026-08-16 by diffing it against ``/etc/nginx/sites-enabled/e2i-analytics``
#: on the production droplet (identical apart from one comment), with
#: ``eznomics.site`` resolving straight to the droplet (no CDN in front) and
#: sslh splicing 443 -> 127.0.0.1:4443 without an idle timeout of its own.
#: ``docker/nginx/nginx.conf`` (60 s) is referenced by no compose file and
#: fronts nothing.
#:
#: ``tests/unit/test_tests_meta/test_proxy_ceiling_coherence_1659.py`` parses the
#: nginx file and fails if this constant drifts from it.
PROXY_READ_TIMEOUT_SECONDS = 300

#: How often to emit a keepalive while the upstream generator is quiet.
#:
#: 15 s is 20x under the ceiling, so nineteen consecutive missed keepalives are
#: needed before nginx severs. Small enough to be safe, large enough that a
#: normal chatty turn never emits one.
SSE_KEEPALIVE_INTERVAL_SECONDS = 15

#: An SSE *comment*: per the EventSource spec a line beginning with ``:`` is
#: ignored by conforming parsers, so this resets nginx's read timer without
#: appearing as an event to any consumer. Deliberately NOT a ``data:`` frame —
#: a synthetic event would need a type every client's schema already knows, and
#: ``scripts/demos/copilot_chat_perf_runner.py`` decodes every ``data: `` line
#: it sees.
SSE_KEEPALIVE_FRAME = ": keepalive\n\n"

#: What a streaming HTTP body may carry. Mirrors starlette's own ``Content``
#: alias (``starlette/responses.py``: ``Content = str | bytes | memoryview``),
#: because a ``StreamingResponse.body_iterator`` is exactly an
#: ``AsyncIterable[Content]`` and #1669 wraps one that a third-party package
#: built — so this wrapper must accept whatever starlette accepts.
#:
#: Deliberately a MIRROR rather than an import: this module is transport-
#: agnostic on purpose and pulling starlette into it would invert that. The two
#: are pinned together by ``test_sse_frame_mirrors_starlettes_content_alias``,
#: so a starlette widening fails loudly here instead of drifting.
SSEFrame = str | bytes | memoryview

#: Bounded by :data:`SSEFrame`, NOT by ``str``.
#:
#: The bound describes what this function actually accepts, and it accepts
#: anything: frames are re-yielded untouched — never decoded, concatenated or
#: inspected — so their type is irrelevant to the logic. The original ``str``
#: bound was an over-restriction that only held because the first call site
#: (``stream_chat``) happened to yield ``str``. #1669's second call site wraps a
#: ``body_iterator``, whose declared element type is the full union.
T = TypeVar("T", bound=SSEFrame)


async def with_sse_keepalive(
    source: AsyncIterable[T],
    interval_seconds: float = SSE_KEEPALIVE_INTERVAL_SECONDS,
) -> AsyncGenerator[T | str, None]:
    """Yield everything ``source`` yields, interleaving keepalives when it is quiet.

    The yield type is ``T | str`` because the ONLY value this function creates is
    :data:`SSE_KEEPALIVE_FRAME`, which is a ``str``. A ``str`` interleaved into a
    ``bytes`` body is safe for the sink this exists to serve: starlette's
    ``StreamingResponse.stream_response`` does ``if not isinstance(chunk, bytes |
    memoryview): chunk = chunk.encode(self.charset)`` before every ``send``
    (``starlette/responses.py``), so mixed chunk types are encoded per chunk
    rather than concatenated. ``test_bytes_body_survives_a_real_asgi_round_trip``
    proves that end to end rather than trusting the reading.

    A consumer that instead joins frames without encoding would need to
    normalise; none exists in this repo, and both call sites are
    ``StreamingResponse``.

    Args:
        source: the real SSE body generator. Its frames pass through untouched
            and in order — whatever their type.
        interval_seconds: emit a keepalive after this long without an upstream
            frame. Must be well under :data:`PROXY_READ_TIMEOUT_SECONDS`.

    Notes:
        ``asyncio.wait`` is used rather than ``asyncio.wait_for`` precisely
        because it does NOT cancel the pending task on timeout — the upstream
        ``__anext__`` keeps running across as many keepalives as it needs.
        Cancellation happens only when the consumer goes away, where tearing the
        graph run down is the desired behaviour (it frees the heavy-compute slot).
    """
    iterator = source.__aiter__()
    pending: asyncio.Task[T] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())

            done, _ = await asyncio.wait({pending}, timeout=interval_seconds)

            if not done:
                # Upstream still working. Reset nginx's read timer and wait again
                # on the SAME task — no cancellation, no lost work.
                yield SSE_KEEPALIVE_FRAME
                continue

            try:
                frame = pending.result()
            except StopAsyncIteration:
                pending = None
                return
            finally:
                if pending is not None and pending.done():
                    pending = None

            yield frame
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            except Exception:  # noqa: BLE001 — teardown must not mask the real error
                logger.debug("SSE keepalive: upstream raised during teardown", exc_info=True)
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — same rationale
                logger.debug("SSE keepalive: upstream aclose failed", exc_info=True)


__all__ = [
    "PROXY_READ_TIMEOUT_SECONDS",
    "SSE_KEEPALIVE_FRAME",
    "SSE_KEEPALIVE_INTERVAL_SECONDS",
    "SSEFrame",
    "with_sse_keepalive",
]
