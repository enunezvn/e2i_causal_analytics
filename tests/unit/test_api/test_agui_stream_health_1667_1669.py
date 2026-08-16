"""#1667 / #1669 — the AG-UI stream must be graded on what reaches the user.

Two defects with one root cause: **nothing in the suite ever drove the AG-UI
streaming path end to end**, so both "the stream is dead" and "the stream is
silent for longer than nginx tolerates" were invisible to every existing guard.

WHY THE EXISTING GUARDS COULD NOT SEE EITHER
--------------------------------------------
Every stream guard we had drives the *per-event helper*::

    test_copilotkit_tool_stream_leak_1547.py:81       agent._handle_single_event(...)
    test_copilotkit_classifier_stream_leak_1636.py:74 agent._handle_single_event(...)
    test_copilotkit_nested_graph_stream_leak_1641.py:147 agent._handle_single_event(...)

All three build their agent with ``object.__new__(LangGraphAgent)`` — no
``__init__``, no graph — which is precisely why they cannot reach
``execute()``. #1662's raise was on ``execute``'s FIRST statement, before any
helper ran, and 291 copilotkit tests passed against a 100% dead surface for a
full day.

WHY HTTP STATUS CANNOT SEE IT EITHER
------------------------------------
``StreamingResponse`` commits the status line before the body generator is
iterated, so a raise inside the body yields a **well-formed HTTP 200 with an
empty body**. Measured live during #1662::

    client:      HTTP 200  frames=0  elapsed=0.6s
    e2i_api log: UnboundLocalError at copilotkit.py:846

``test_status_200_is_not_evidence_of_a_live_stream`` below pins that trap as a
executable fact rather than a war story.

TWO AG-UI ENTRY POINTS, NOT ONE
-------------------------------
``copilotkit_custom_handler`` reaches ``LangGraphAgent.execute`` by two
different routes, and a guard that covered only one would miss half the surface:

* **root POST** ``/api/copilotkit`` with body ``{"method": "agent/run"}`` →
  this module's own ``stream_agent_events`` →
  ``StreamingResponse(..., media_type="text/event-stream")``.
* **``POST /api/copilotkit/agent/{name}``** → ``path`` is neither ``""`` nor
  ``"info"``, so it falls through to the third-party SDK handler
  (``copilotkit/integrations/fastapi.py`` regex ``agent/([a-zA-Z0-9_-]+)`` →
  ``handle_execute_agent``) → ``StreamingResponse(events,
  media_type="application/json")``, where ``events`` is
  ``sdk.execute_agent(...)``, which returns **our** ``execute`` unchanged
  (``copilotkit/sdk.py`` ~347).

The wire bytes are IDENTICAL on both — ``data: {json}\\n\\n``, produced by our
``execute`` — so the SDK's ``application/json`` is a mislabel rather than a
different protocol. Only the response header differs, and only the root branch
sets ``X-Accel-Buffering: no``.

The sub-path is the one with known consumers: ``scripts/demos/copilot_agui_runner.py``
posts to ``{api_base}/copilotkit/agent/default``, and the #1669 verification
probe used the same URL. (A 71-minute production log window — the container's
full uptime at the time of writing — showed 6 requests to the sub-path and 0 to
the bare root, but that window contained essentially only probe traffic, so it
is evidence about the runner, NOT a measurement of organic user routing. Both
branches are covered below rather than betting on that.)

``frames > 0`` IS NOT SUFFICIENT ON ITS OWN
-------------------------------------------
#1667 proposed ``frames > 0``. That is right for the SDK path — the exception
escapes the body iterator and starlette aborts with zero frames — but WRONG for
the root branch, whose ``stream_agent_events`` catches the exception and emits a
single ``RUN_ERROR`` frame. A dead stream there has ``frames == 1``. Every
assertion here therefore keys on **delivered answer content**, not frame count.

WHAT IS NOT MOCKED
------------------
The graph is a stub (a real ``StateGraph`` over a real ``GenericFakeChatModel``)
because the *event sequence* has to be controllable and the real graph would
make paid LLM calls. Everything downstream of it is production code: the real
route handler, the real ``LangGraphAgent``, the real ``ag_ui_langgraph``
translation, the real ``astream_events``, the real ``StreamingResponse``. The
stub controls the INPUT; nothing on the path under test is replaced.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any, AsyncIterable, Dict, List, Tuple, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from starlette.requests import Request
from starlette.responses import StreamingResponse

from src.api.utils.sse_keepalive import (
    SSE_KEEPALIVE_FRAME,
    with_sse_keepalive,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

#: The answer the stub graph streams. Asserted verbatim — "the user got their
#: answer" is the property both issues are really about.
STUB_ANSWER = "Kisqali TRx grew 12% quarter over quarter."

#: Keepalive cadence used by the route-level tests. The production default is
#: 15 s (``SSE_KEEPALIVE_INTERVAL_SECONDS``); a test that waited for that would
#: add 15 s to CI. ``test_route_uses_the_production_keepalive_interval`` pins the
#: production call site to the constant so shrinking it here cannot hide a drift.
TEST_KEEPALIVE_INTERVAL = 0.05

#: How long the stub graph stays quiet in the keepalive tests. Comfortably
#: several intervals, still fast.
TEST_QUIET_SECONDS = 0.6


# ---------------------------------------------------------------------------
# Stub graph — controls the EVENT SEQUENCE, nothing else
# ---------------------------------------------------------------------------


class _StubState(TypedDict):
    messages: Annotated[list, add_messages]
    tools: list


def _build_stub_graph(quiet_seconds: float = 0.0, answer: str = STUB_ANSWER):
    """A real compiled ``StateGraph`` whose single node streams ``answer``.

    ``GenericFakeChatModel`` is a real ``BaseChatModel``, so invoking it raises
    genuine ``on_chat_model_start`` / ``on_chat_model_stream`` /
    ``on_chat_model_end`` callbacks into ``astream_events`` — which is what the
    AG-UI translation layer consumes. No part of the streaming machinery is
    replaced.

    ``quiet_seconds`` models a long tool call: an ``await`` during which the
    graph raises NO events at all. That is the shape of the #1669 exposure, and
    it is an ``await`` rather than a busy loop deliberately — a keepalive is a
    coroutine and can only fire while the event loop is free (see the "What this
    does NOT fix" note in ``src/api/utils/sse_keepalive``).
    """
    llm = GenericFakeChatModel(messages=iter([answer] * 64))

    async def chat(state: _StubState) -> dict:
        if quiet_seconds:
            await asyncio.sleep(quiet_seconds)
        return {"messages": [await llm.ainvoke(state["messages"])]}

    workflow = StateGraph(_StubState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=MemorySaver())


def _agui_agent(graph, name: str = "default"):
    """A REAL ``LangGraphAgent`` — full ``__init__``, not ``object.__new__``.

    The distinction is the whole point of #1667: the existing guards bypass
    ``__init__`` and so can never reach ``execute()``.
    """
    from src.api.routes.copilotkit import LangGraphAgent

    return LangGraphAgent(name=name, description="stub agent for stream health", graph=graph)


def _registry(*agents):
    """A REAL ``CopilotKitRemoteEndpoint`` holding the stub agent.

    The production endpoint is built by ``create_copilotkit_sdk()``, which wires
    the real e2i chat graph and would make paid LLM calls; only the AGENT is
    substituted here. The endpoint itself is the genuine third-party object, so
    the SDK's own dispatch (``copilotkit/sdk.py::execute_agent`` ->
    ``copilotkit/integrations/fastapi.py::handle_execute_agent`` ->
    ``StreamingResponse``) runs exactly as it does in production. Stubbing this
    out was how the first draft of these tests silently skipped the SDK path
    altogether.
    """
    from copilotkit import CopilotKitRemoteEndpoint

    return CopilotKitRemoteEndpoint(agents=list(agents), actions=[])


# ---------------------------------------------------------------------------
# Driving the real route handler
# ---------------------------------------------------------------------------

_BEARER = "Bearer stream-health-token"
_USER = {"id": "33333333-3333-3333-3333-333333333333", "email": "stream@e2i.local"}


def _make_request(path: str, body: bytes) -> Request:
    """A real Starlette ``Request`` over a COMPLETE ASGI scope.

    Mirrors ``test_copilotkit_agent_name_auth_gate_1432.py``: the handler reads
    ``.method``, ``.headers``, ``.scope``, ``.path_params`` and awaits
    ``.body()``, and partial hand-rolled scopes are the ones that crash xdist
    workers.
    """
    url = f"/api/copilotkit/{path}".rstrip("/")
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": url,
            "raw_path": url.encode(),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", _BEARER.encode()),
            ],
            "server": ("testserver", 80),
            "client": ("testclient", 12345),
            "root_path": "",
            "path_params": {"path": path},
            "state": {},
        },
        lambda: _receive(body),
    )


async def _receive(body: bytes) -> dict:
    return {"type": "http.request", "body": body, "more_body": False}


def _root_agent_run_body(question: str = "hello") -> bytes:
    """The body shape the custom root branch expects (``method: agent/run``)."""
    return json.dumps(
        {
            "method": "agent/run",
            "params": {"agentId": "default"},
            "body": {
                "threadId": "thread-stream-health",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": question}],
            },
        }
    ).encode()


def _sdk_subpath_body(question: str = "hello") -> bytes:
    """The body shape the SDK ``agent/{name}`` handler expects.

    Note the ABSENCE of a ``method`` key — that is exactly why this request
    misses the root branch and reaches the SDK handler.
    """
    return json.dumps(
        {
            "threadId": "thread-stream-health",
            "state": {},
            "messages": [{"id": "m1", "role": "user", "content": question}],
            "actions": [],
        }
    ).encode()


@pytest.fixture
def authed(monkeypatch):
    """Satisfy the real auth gate without bypassing it.

    ``_require_auth_for_copilotkit_execution`` is left fully in place; only the
    Supabase round trip is stubbed, matching
    ``test_copilotkit_agent_name_auth_gate_1432.py``. Patching ``TESTING_MODE``
    instead would skip the branch under test.
    """
    import src.api.routes.copilotkit as ck

    async def _verify(_token):
        return _USER

    monkeypatch.setattr(ck, "TESTING_MODE", False)
    monkeypatch.setattr(ck, "verify_supabase_token", _verify)
    monkeypatch.setattr(ck, "set_authenticated_user", lambda _uid: None)
    return ck


@pytest.fixture
def fast_keepalive(monkeypatch):
    """Run the REAL wrapper at a CI-sized interval.

    ``with_sse_keepalive``'s default is bound at definition time, so patching
    the constant would not reach it. This replaces the module-level NAME the
    route calls with a thin partial over the real implementation: if the route
    stops calling it, no keepalive appears and the mechanism assertions fail —
    which is the regression these tests exist to catch.
    """
    import src.api.routes.copilotkit as ck

    def _fast(source, interval_seconds: float = TEST_KEEPALIVE_INTERVAL):
        return with_sse_keepalive(source, interval_seconds=interval_seconds)

    monkeypatch.setattr(ck, "with_sse_keepalive", _fast)
    return ck


class Frames:
    """What actually reached the wire, split the way a consumer must split it."""

    def __init__(self, chunks: List[str], gaps: List[float]):
        self.raw = chunks
        self.gaps = gaps
        self.data: List[Dict[str, Any]] = []
        self.unparseable: List[str] = []
        self.keepalives: List[str] = []
        self.other: List[str] = []
        for chunk in chunks:
            if chunk.startswith(":"):
                self.keepalives.append(chunk)
            elif chunk.startswith("data: "):
                payload = chunk[len("data: ") :].strip()
                try:
                    self.data.append(json.loads(payload))
                except json.JSONDecodeError:
                    self.unparseable.append(chunk)
            else:
                self.other.append(chunk)

    @property
    def types(self) -> List[str]:
        return [str(d.get("type", "")) for d in self.data]

    @property
    def answer(self) -> str:
        return "".join(
            d.get("delta") or "" for d in self.data if str(d.get("type")) == "TEXT_MESSAGE_CONTENT"
        )

    @property
    def max_gap(self) -> float:
        return max(self.gaps) if self.gaps else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Frames(data={len(self.data)}, keepalives={len(self.keepalives)}, "
            f"other={len(self.other)}, types={self.types})"
        )


async def _drain(body_iterator: AsyncIterable[Any]) -> Frames:
    """Consume a response body, timing the gap before each chunk."""
    chunks: List[str] = []
    gaps: List[float] = []
    previous = time.monotonic()
    async for chunk in body_iterator:
        now = time.monotonic()
        gaps.append(now - previous)
        previous = now
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk))
    return Frames(chunks, gaps)


def _execute(agent, question: str = "hello"):
    """Call ``execute()`` the way BOTH production callers call it.

    ``actions`` is passed explicitly because both real call sites do
    (``copilotkit.py``'s root branch and ``copilotkit/sdk.py::execute_agent``
    both default it to ``[]``) — and because ``execute`` forwards it straight
    into ``RunAgentInput(tools=...)``, which rejects ``None``. Its own signature
    defaults it to ``None``, so the defaults are not a runnable call; matching
    the real callers keeps this harness faithful rather than papering over that.
    """
    return agent.execute(
        thread_id="thread-stream-health",
        state={},
        messages=[{"id": "m1", "role": "user", "content": question}],
        config=None,
        actions=[],
        node_name=None,
    )


async def _drive_route(ck, registry, path: str, body: bytes) -> Tuple[Any, Frames]:
    """Call the REAL handler and drain whatever streaming body it returns."""
    response = await ck.copilotkit_custom_handler(_make_request(path, body), registry, path=path)
    if not isinstance(response, StreamingResponse):
        return response, Frames([], [])
    return response, await _drain(response.body_iterator)


# ---------------------------------------------------------------------------
# #1667 — a dead stream must not read as success
# ---------------------------------------------------------------------------


class TestExecuteIsActuallyDriven:
    """The missing rung: drive ``execute()`` itself, not the per-event helper."""

    async def test_execute_yields_frames_and_delivers_the_answer(self):
        """RED under #1662: ``execute()`` raised on its first statement, yielding nothing.

        This is the assertion no helper-level test can make, because the helper
        is never reached when the function that calls it raises.
        """
        agent = _agui_agent(_build_stub_graph())

        frames = await _drain(_execute(agent))

        assert frames.data, f"execute() produced no frames at all — the stream is dead ({frames!r})"
        assert frames.answer == STUB_ANSWER, (
            "execute() streamed frames but no answer text reached the user; "
            f"got {frames.answer!r} from types {frames.types}"
        )

    async def test_execute_frames_are_well_formed_sse(self):
        """Every frame must be a parseable ``data: <json>`` record.

        A frame the consumer cannot parse is indistinguishable from no frame at
        all, so well-formedness is part of "the stream is alive".
        """
        agent = _agui_agent(_build_stub_graph())

        frames = await _drain(_execute(agent))

        assert not frames.unparseable, f"unparseable frames on the wire: {frames.unparseable[:3]}"
        assert not frames.other, f"frames that are neither data nor comment: {frames.other[:3]}"
        assert all(chunk.endswith("\n\n") for chunk in frames.raw), (
            "SSE records must be terminated by a blank line, or consumers "
            "concatenate two events into one"
        )

    async def test_execute_completes_the_agui_run_lifecycle(self):
        """A truncated run leaves the client waiting forever, which also looks like 200/OK."""
        agent = _agui_agent(_build_stub_graph())

        frames = await _drain(_execute(agent))

        assert "RUN_STARTED" in frames.types
        assert "RUN_FINISHED" in frames.types, (
            f"run never reached RUN_FINISHED — types were {frames.types}"
        )
        assert "RUN_ERROR" not in frames.types


class TestRouteLevelStreamHealth:
    """Both AG-UI entry points, driven through the real handler."""

    async def test_sdk_subpath_streams_the_answer(self, authed):
        """``POST agent/default`` — the sub-path, delegated to the SDK handler.

        The URL ``scripts/demos/copilot_agui_runner.py`` drives, and the one
        both the #1662 and #1669 investigations measured. Here the exception
        escapes the body iterator with NO frames at all, which is the shape
        #1662 presented as ``HTTP 200, frames=0``.
        """
        registry = _registry(_agui_agent(_build_stub_graph()))

        response, frames = await _drive_route(
            authed, registry, "agent/default", _sdk_subpath_body()
        )

        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200
        assert frames.answer == STUB_ANSWER, f"HTTP 200 but the answer never arrived: {frames!r}"

    async def test_root_agent_run_branch_streams_the_answer(self, authed):
        """The custom ``{"method": "agent/run"}`` branch (``copilotkit.py:4174``).

        Zero production traffic today, but it is live code and #1662 would have
        broken it identically.
        """
        registry = _registry(_agui_agent(_build_stub_graph()))

        response, frames = await _drive_route(authed, registry, "", _root_agent_run_body())

        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200
        assert response.media_type == "text/event-stream"
        assert frames.answer == STUB_ANSWER, f"HTTP 200 but no answer: {frames!r}"

    async def test_status_200_is_not_evidence_of_a_live_stream(self, authed, monkeypatch):
        """Pin the trap that made #1662 invisible for a full day.

        ``StreamingResponse`` commits the status line before the body generator
        runs, so a raise inside ``execute`` still produces a well-formed 200.
        This test asserts the trap EXISTS (so nobody re-adds a status-code smoke
        test and calls it monitoring) and that the content assertion catches
        what the status assertion cannot.

        It also shows why #1667's proposed ``frames > 0`` is not sufficient on
        its own: on the root branch ``stream_agent_events`` catches the
        exception and emits a single ``RUN_ERROR`` frame, so a fully dead stream
        has ``frames == 1``.
        """
        from src.api.routes.copilotkit import LangGraphAgent

        def _exploding_execute(self, **_kwargs):
            async def _gen():
                raise UnboundLocalError(
                    "cannot access local variable 'time' where it is not associated with a value"
                )
                yield  # pragma: no cover - unreachable, makes this an async generator

            return _gen()

        monkeypatch.setattr(LangGraphAgent, "execute", _exploding_execute)
        registry = _registry(_agui_agent(_build_stub_graph()))

        response, frames = await _drive_route(authed, registry, "", _root_agent_run_body())

        # The trap: status says everything is fine.
        assert response.status_code == 200
        # The reality: no answer reached the user.
        assert frames.answer == "", "the exploding stub somehow produced answer text"
        # And frame count alone would NOT have caught it on this surface.
        assert "RUN_ERROR" in frames.types, (
            "the root branch is expected to convert the raise into a RUN_ERROR frame; "
            "if this changed, the 'frames > 0' caveat in this module's docstring needs revisiting"
        )


# ---------------------------------------------------------------------------
# #1669 — the silent window must be bounded, on the surface that carries traffic
# ---------------------------------------------------------------------------


class TestSilentWindowIsBounded:
    """Assert on the MECHANISM (a keepalive fired), never on a threshold.

    #1669's own verification probe asserted ``max_gap <= 25s`` against the AG-UI
    surface and PASSED on 23.1 s while zero keepalives ever fired. A threshold
    can be satisfied by luck; ``keepalives > 0`` can only be satisfied by the
    wrapper actually running.
    """

    async def test_unwrapped_silence_tracks_the_graph_one_for_one(self):
        """Baseline: with no wrapper the gap IS the quiet time, unbounded.

        Nothing in ``ag_ui_langgraph`` heartbeats a quiet graph — it emits a
        frame per ``astream_events`` event and nothing else — so the silent
        window has no ceiling of its own. Driving ``execute()`` raw (the
        pre-#1669 shape) demonstrates that directly.
        """
        agent = _agui_agent(_build_stub_graph(quiet_seconds=TEST_QUIET_SECONDS))

        frames = await _drain(_execute(agent))

        assert not frames.keepalives, "execute() itself must stay transport-agnostic"
        assert frames.max_gap >= TEST_QUIET_SECONDS * 0.8, (
            f"expected the quiet node to show up as a ~{TEST_QUIET_SECONDS}s silent "
            f"window, saw {frames.max_gap:.3f}s — has something started heartbeating?"
        )

    async def test_sdk_subpath_emits_keepalives_while_the_graph_is_quiet(
        self, authed, fast_keepalive
    ):
        """RED before #1669: zero keepalives on the SDK-delegated sub-path.

        This is the branch #1669's suggested fix would have MISSED — it names
        ``copilotkit.py:4280``, which is the root branch's ``StreamingResponse``,
        while the response here is constructed inside the third-party package.
        """
        registry = _registry(_agui_agent(_build_stub_graph(quiet_seconds=TEST_QUIET_SECONDS)))

        _response, frames = await _drive_route(
            fast_keepalive, registry, "agent/default", _sdk_subpath_body()
        )

        assert frames.keepalives, (
            "no keepalive ever fired on POST agent/{name} — the production AG-UI "
            "surface is still exposed to nginx's proxy_read_timeout (#1669). "
            f"Saw {len(frames.data)} data frames, max_gap={frames.max_gap:.3f}s."
        )
        assert all(k == SSE_KEEPALIVE_FRAME for k in frames.keepalives)

    async def test_root_agent_run_branch_emits_keepalives_while_the_graph_is_quiet(
        self, authed, fast_keepalive
    ):
        """The other entry point must be bounded too, or the fix depends on routing."""
        registry = _registry(_agui_agent(_build_stub_graph(quiet_seconds=TEST_QUIET_SECONDS)))

        _response, frames = await _drive_route(fast_keepalive, registry, "", _root_agent_run_body())

        assert frames.keepalives, (
            "no keepalive fired on the root agent/run branch (#1669); "
            f"saw {len(frames.data)} data frames, max_gap={frames.max_gap:.3f}s"
        )

    async def test_keepalives_do_not_disturb_the_delivered_answer(self, authed, fast_keepalive):
        """A heartbeat that corrupts the payload is worse than no heartbeat.

        The keepalive must be INTERLEAVED, never substituted: same AG-UI event
        sequence, same delivered answer, and no frame the consumer cannot parse.

        Deliberately NOT a byte comparison. Two runs differ in ``runId`` /
        ``messageId`` / ``timestamp`` by design, so byte equality would be false
        for reasons that have nothing to do with the keepalive. Event-type order
        plus assembled answer text is the strongest property that is actually
        invariant between the two runs.
        """
        quiet = TEST_QUIET_SECONDS
        registry = _registry(_agui_agent(_build_stub_graph(quiet_seconds=quiet)))
        _response, wrapped = await _drive_route(
            fast_keepalive, registry, "agent/default", _sdk_subpath_body()
        )

        bare_agent = _agui_agent(_build_stub_graph(quiet_seconds=quiet))
        bare = await _drain(_execute(bare_agent))

        assert wrapped.keepalives, "precondition: the wrapped run must have emitted keepalives"
        assert wrapped.answer == bare.answer == STUB_ANSWER
        assert wrapped.types == bare.types, (
            "interleaving keepalives changed the AG-UI event sequence:\n"
            f"  wrapped: {wrapped.types}\n  bare:    {bare.types}"
        )
        assert not wrapped.unparseable and not wrapped.other

    async def test_keepalive_frame_is_an_ignorable_sse_comment(self, authed, fast_keepalive):
        """The heartbeat must be invisible to the CLIENT, not merely spec-legal.

        "SSE comments are ignored by conforming parsers" is a claim about the
        spec, not about the third-party consumer, so it was VERIFIED against the
        real one rather than assumed.

        ``@copilotkit/react-core@1.51.2`` (``frontend/package.json``) depends on
        ``@ag-ui/client@^0.0.42``. In that package's ``dist/index.mjs``:

        * ``transformHttpEventStream`` branches on the RESPONSE content-type —
          ``headers.get("content-type") === AGUI_MEDIA_TYPE`` (which is
          ``"application/vnd.ag-ui.event+proto"``) selects the protobuf reader,
          and **everything else falls through to the SSE reader**. Our
          ``application/json`` therefore lands on the SSE reader, mislabel and
          all.
        * ``parseSSEStream`` buffers bytes, splits records on ``/\\n\\n/``,
          retains the partial tail, and per record does::

              for (let i of lines) if (i.startsWith("data: ")) o.push(i.slice(6));
              if (o.length > 0) { ... JSON.parse(...) }

          A record with no ``data: `` line leaves ``o`` empty, so ``JSON.parse``
          is never reached — the record is dropped silently, with no error.

        Confirmed empirically by driving that real ``parseSSEStream`` with the
        keepalive interleaved between frames, before the first frame, after
        ``RUN_FINISHED``, and split across chunk boundaries mid-record: every
        case produced an event sequence identical to baseline and zero errors.

        ``scripts/demos/copilot_agui_runner.py`` applies the same
        ``line.startswith("data:")`` filter, so both known consumers agree.

        What this test can assert in Python is the frame's side of that
        contract: the bytes we emit must satisfy the properties the parser
        relies on.
        """
        registry = _registry(_agui_agent(_build_stub_graph(quiet_seconds=TEST_QUIET_SECONDS)))

        _response, frames = await _drive_route(
            fast_keepalive, registry, "agent/default", _sdk_subpath_body()
        )

        assert frames.keepalives
        for keepalive in frames.keepalives:
            assert keepalive.startswith(":"), "not an SSE comment — a consumer would parse it"
            assert not keepalive.startswith("data:")
            assert keepalive.endswith("\n\n"), "an unterminated comment swallows the next record"
            # The decisive property: a data-prefix reader sees nothing.
            assert [ln for ln in keepalive.split("\n") if ln.startswith("data:")] == []


class TestKeepaliveWiringIsNotAccidental:
    """Guard the two ways this fix could silently become inert."""

    async def test_route_uses_the_production_keepalive_interval(self, authed, monkeypatch):
        """The route must not pin its own cadence — the constant is the SSOT.

        Captures the ``interval_seconds`` the route actually passes. If a future
        edit hardcodes a number here, the coherence guard in
        ``test_proxy_ceiling_coherence_1659.py`` (which checks the CONSTANT
        against nginx) would keep passing while the wire cadence drifted.
        """
        import src.api.routes.copilotkit as ck
        from src.api.utils import sse_keepalive as ka

        seen: List[Any] = []

        def _record(source, interval_seconds: Any = "<default>"):
            seen.append(interval_seconds)
            return with_sse_keepalive(source, interval_seconds=TEST_KEEPALIVE_INTERVAL)

        monkeypatch.setattr(ck, "with_sse_keepalive", _record)
        registry = _registry(_agui_agent(_build_stub_graph()))
        await _drive_route(ck, registry, "agent/default", _sdk_subpath_body())

        assert seen, "the AG-UI route never called with_sse_keepalive at all"
        assert all(value in ("<default>", ka.SSE_KEEPALIVE_INTERVAL_SECONDS) for value in seen), (
            f"the AG-UI route passed a hardcoded keepalive interval {seen!r}; it must "
            "inherit SSE_KEEPALIVE_INTERVAL_SECONDS so the nginx coherence guard binds it"
        )

    async def test_non_streaming_responses_are_left_alone(self, authed):
        """The wrap must key on the response TYPE, not on the route.

        ``agent/connect`` and the info endpoints return ``JSONResponse``; a wrap
        applied indiscriminately would corrupt them.
        """
        import src.api.routes.copilotkit as ck

        registry = _registry(_agui_agent(_build_stub_graph()))
        body = json.dumps({"method": "agent/connect"}).encode()

        response = await ck.copilotkit_custom_handler(_make_request("", body), registry, path="")

        assert not isinstance(response, StreamingResponse)
        assert response.status_code == 200


class TestOverRealHttp:
    """The same properties, but through a real ASGI round-trip.

    Everything above drains ``response.body_iterator`` directly. That is what
    starlette itself iterates, but it skips the ASGI send path — chunk encoding,
    header commit, and the exact byte framing an HTTP client sees. Since BOTH
    defects are about "what actually reached the user", at least one guard has to
    read the answer off the wire rather than out of the response object.

    ``httpx.ASGITransport`` speaks to the app in-process: no socket, no nginx, no
    LLM spend, but a genuine HTTP request/response.
    """

    @staticmethod
    def _app(ck, registry):
        from fastapi import FastAPI

        app = FastAPI()

        @app.post("/api/copilotkit/{path:path}")
        async def _handler(request: Request, path: str = ""):  # pragma: no cover - via client
            return await ck.copilotkit_custom_handler(request, registry, path=path)

        return app

    @staticmethod
    async def _stream(app, body: dict):
        import httpx

        transport = httpx.ASGITransport(app=app)
        data: List[Dict[str, Any]] = []
        keepalives = 0
        other: List[str] = []
        async with httpx.AsyncClient(transport=transport, base_url="http://stream-health") as c:
            async with c.stream(
                "POST",
                "/api/copilotkit/agent/default",
                headers={"Authorization": _BEARER},
                json=body,
                timeout=60,
            ) as response:
                status = response.status_code
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith(":"):
                        keepalives += 1
                    elif line.startswith("data:"):
                        data.append(json.loads(line[len("data:") :].strip()))
                    else:
                        other.append(line)
        answer = "".join(
            d.get("delta") or "" for d in data if str(d.get("type")) == "TEXT_MESSAGE_CONTENT"
        )
        return status, data, keepalives, other, answer

    async def test_answer_reaches_an_http_client_alongside_keepalives(self, authed, fast_keepalive):
        """#1667 + #1669 in one assertion set, measured off the wire."""
        registry = _registry(_agui_agent(_build_stub_graph(quiet_seconds=TEST_QUIET_SECONDS)))
        app = self._app(fast_keepalive, registry)

        status, data, keepalives, other, answer = await self._stream(
            app,
            {
                "threadId": "thread-stream-health",
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "hello"}],
                "actions": [],
            },
        )

        assert status == 200
        assert data, "HTTP 200 with zero frames on the wire — the #1662 signature (#1667)"
        assert answer == STUB_ANSWER, f"the answer never reached the client: {answer!r}"
        assert keepalives > 0, "no keepalive reached the wire (#1669)"
        assert not other, (
            "lines that are neither an SSE data record nor an SSE comment reached "
            f"the client — a consumer would have to guess what they are: {other[:3]}"
        )
