"""#1673 — every AG-UI streaming response must tell nginx not to buffer it.

THIS FILE IS THE CHEAP GUARD, NOT THE PROOF.

The property #1673 is about is ``TTFB << total`` on the wire, and no in-process
test can observe it: buffering happens in a proxy, and an in-process ASGI call
has no proxy and no wire. That measurement lives in
``tests/integration/api/test_agui_edge_buffering_1673.py``, which runs a real
nginx carrying the production gzip directives in front of a real uvicorn and
asserts on elapsed time.

What this file adds is coverage that survives an environment with no nginx
binary: it pins the *mechanism* the edge test proved sufficient — the
``X-Accel-Buffering: no`` response header — on every streaming entry point of
the AG-UI surface. Losing the header is the realistic regression; losing nginx's
respect for it is not.

WHY THE HEADER, WHEN THE CAUSE IS GZIP
--------------------------------------
Measured on production, same URL, same nginx, same ``location /api/``, differing
only in ``Accept-Encoding``::

    gzip       TTFB 8.754s / total 8.756s   ttfb/total 0.9998   2 chunks
    identity   TTFB 0.683s / total 9.218s   ttfb/total 0.0741  99 chunks

So ``proxy_buffering on`` — which #1673 named as the cause — is not it. nginx
forwards proxy buffers as they fill; it is the **gzip filter** that holds the
whole turn, because it only emits when its deflate buffer fills or a flush
marker arrives, and the third-party SDK labels this SSE byte stream
``media_type="application/json"``, which is in the live ``gzip_types``.

``X-Accel-Buffering: no`` disables proxy buffering, not gzip — so that it fixes a
gzip problem is exactly the kind of claim that has to be measured rather than
reasoned. It was. On a replica nginx with the production directives, one variable
per cell::

    json + gzip + no header          TTFB 9.234 / total 9.235   ratio 0.9999   1 chunk
    json + gzip + X-Accel: no        TTFB 0.065 / total 9.095   ratio 0.0071  61 chunks

Still ``Content-Encoding: gzip``, and streaming. Turning proxy buffering off makes
nginx flag each upstream read with ``flush``, and the gzip filter honours flush
with a ``Z_SYNC_FLUSH``.

WHY NOT FIX IT IN THE NGINX CONFIG
----------------------------------
``location /copilotkit/`` — the block #1673 proposes editing — is dead: it
proxies to ``127.0.0.1:8000/copilotkit/``, a prefix the app does not serve, and
authenticated probes return ``404 EndpointNotFoundError`` on every path. The live
surface is ``location /api/`` (the shipped bundle bakes ``apiUrl:"/api"``; the
runner defaults to ``--api-base https://eznomics.site/api``), and that block
serves the entire REST API — disabling gzip there to fix one endpoint would
de-optimise every JSON response the platform sends. The nginx config also reaches
production only through a manual root ``cp`` + reload, never through CI.

TWO ENTRY POINTS, BOTH COVERED
------------------------------
* ``POST /api/copilotkit`` body ``{"method": "agent/run"}`` -> our own
  ``stream_agent_events`` -> ``StreamingResponse(media_type="text/event-stream")``.
  This one ALREADY sent the header. It is pinned here anyway, because it is the
  branch a reader is most likely to "tidy" once the SDK branch grows its own.
* ``POST /api/copilotkit/agent/{name}`` -> third-party ``sdk_handler``. This is
  the one the frontend, the eval runner and #1662 all actually drive, and the one
  that was missing the header.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated, Any, AsyncIterator, List, Optional, TypedDict

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

ANSWER = "Kisqali TRx grew 12% quarter over quarter."


class _OneShotChatModel(BaseChatModel):
    """A real ``BaseChatModel`` that streams ``ANSWER`` in a single chunk.

    Cadence does not matter here — this file asserts on headers, and the header
    is set before the body iterator is ever pulled. The paced variant that the
    edge test needs would only add seconds to the unit lane.
    """

    @property
    def _llm_type(self) -> str:
        return "one-shot-stub"

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=ANSWER))
        if run_manager:
            await run_manager.on_llm_new_token(ANSWER, chunk=chunk)
        yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=ANSWER))])


class _StubState(TypedDict):
    messages: Annotated[list, add_messages]
    tools: list


def _build_stub_graph():
    llm = _OneShotChatModel()

    async def chat(state: _StubState) -> dict:
        return {"messages": [await llm.ainvoke(state["messages"])]}

    workflow = StateGraph(_StubState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=MemorySaver())


#: How long the quiet-window graph stays silent. Several keepalive intervals at
#: ``TEST_KEEPALIVE_INTERVAL``, still a fraction of a second.
QUIET_SECONDS = 0.4

#: Keepalive cadence for the quiet-window test. Production's default is 15 s
#: (``SSE_KEEPALIVE_INTERVAL_SECONDS``); waiting for that would add 15 s to the
#: unit lane. ``test_agui_stream_health_1667_1669.py`` already pins the
#: production call site to the real constant, so shrinking it here cannot mask a
#: drift in what production actually uses.
TEST_KEEPALIVE_INTERVAL = 0.05


def _build_quiet_graph():
    """A real graph with a genuine silent window before it answers.

    The pause is an ``await``, not a busy loop, deliberately: a keepalive is a
    coroutine and can only fire while the event loop is free. A blocking sleep
    would produce a silent window that the wrapper legitimately cannot cover, and
    the test would fail for a reason that has nothing to do with either fix.
    """
    llm = _OneShotChatModel()

    async def chat(state: _StubState) -> dict:
        await asyncio.sleep(QUIET_SECONDS)
        return {"messages": [await llm.ainvoke(state["messages"])]}

    workflow = StateGraph(_StubState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=MemorySaver())


@pytest.fixture
def quiet_graph_registry():
    """A registry whose agent goes quiet long enough for a keepalive to fire."""
    from copilotkit import CopilotKitRemoteEndpoint

    from src.api.routes.copilotkit import LangGraphAgent

    agent = LangGraphAgent(
        name="default",
        description="stub agent with a quiet window",
        graph=_build_quiet_graph(),
    )
    return CopilotKitRemoteEndpoint(agents=[agent], actions=[])


@pytest.fixture
def fast_keepalive(monkeypatch):
    """Run the REAL wrapper at a unit-lane interval.

    ``with_sse_keepalive``'s default interval is bound at definition time, so
    patching the constant would not reach it. This replaces the module-level NAME
    the route calls with a thin partial over the real implementation — so if the
    route ever stops calling it, no keepalive appears and the assertion fails,
    which is the regression this exists to catch.
    """
    import src.api.routes.copilotkit as ck
    from src.api.utils.sse_keepalive import with_sse_keepalive

    def _fast(source, interval_seconds: float = TEST_KEEPALIVE_INTERVAL):
        return with_sse_keepalive(source, interval_seconds=interval_seconds)

    monkeypatch.setattr(ck, "with_sse_keepalive", _fast)
    return ck


def _registry():
    """A REAL ``CopilotKitRemoteEndpoint`` over a REAL ``LangGraphAgent``.

    Only the graph is substituted. The third-party dispatch
    (``copilotkit/sdk.py::execute_agent`` -> ``handle_execute_agent`` ->
    ``StreamingResponse``) must run for real, because the response under test is
    the one that package constructs — a stubbed registry would test our stub's
    headers instead of the SDK's.
    """
    from copilotkit import CopilotKitRemoteEndpoint

    from src.api.routes.copilotkit import LangGraphAgent

    agent = LangGraphAgent(
        name="default",
        description="stub agent for accel-buffering guard",
        graph=_build_stub_graph(),
    )
    return CopilotKitRemoteEndpoint(agents=[agent], actions=[])


def _make_request(path: str, body: bytes, method: str = "POST") -> Request:
    """A real Starlette ``Request`` over a COMPLETE ASGI scope."""
    url = f"/api/copilotkit/{path}".rstrip("/")
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": url,
            "raw_path": url.encode(),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer accel-buffering-guard"),
            ],
            "server": ("testserver", 80),
            "client": ("testclient", 12345),
            "root_path": "",
            "path_params": {"path": path},
        },
        receive,
    )


def _turn_body() -> bytes:
    return json.dumps(
        {
            "threadId": str(uuid.uuid4()),
            "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "What is Kisqali TRx?"}],
            "actions": [],
        }
    ).encode()


def _root_agent_run_body() -> bytes:
    return json.dumps(
        {
            "method": "agent/run",
            "params": {"agentId": "default"},
            "body": {
                "threadId": str(uuid.uuid4()),
                "state": {},
                "messages": [{"id": "m1", "role": "user", "content": "What is Kisqali TRx?"}],
                "tools": [],
            },
        }
    ).encode()


async def _drive(path: str, body: bytes, method: str = "POST"):
    from src.api.routes.copilotkit import copilotkit_custom_handler

    return await copilotkit_custom_handler(
        _make_request(path, body, method), _registry(), path=path
    )


def _accel(response) -> Optional[str]:
    return response.headers.get("x-accel-buffering")


class TestBothStreamingEntryPointsOptOutOfBuffering:
    """The AG-UI surface reaches ``execute()`` two ways. Both are user-facing."""

    async def test_sdk_subpath_response_disables_proxy_buffering(self):
        """RED before the fix — this is the surface #1673 measured as buffered.

        ``POST /api/copilotkit/agent/{name}`` is delegated to the third-party
        ``sdk_handler``, which constructs its ``StreamingResponse`` inside the
        installed package with no such header. It is the URL the CopilotKit
        frontend drives, the URL ``scripts/demos/copilot_agui_runner.py`` drives,
        and the URL the #1662 outage was measured on.
        """
        response = await _drive("agent/default", _turn_body())

        assert isinstance(response, StreamingResponse), (
            f"expected the SDK's streaming response, got {type(response).__name__} — "
            f"the surface under test was not reached"
        )
        assert _accel(response) == "no", (
            "the SDK-built AG-UI response does not opt out of proxy buffering, so "
            "nginx delivers the whole turn in one flush at the end: measured "
            "TTFB 8.754s of an 8.756s turn (#1673). Headers present: "
            f"{dict(response.headers)}"
        )

    async def test_root_agent_run_branch_disables_proxy_buffering(self):
        """Already GREEN. Pinned so the two branches cannot drift apart again.

        #1673 existed precisely because one of these two branches had the header
        and the other did not, and the one that did was not the one users are on.
        """
        response = await _drive("", _root_agent_run_body())

        assert isinstance(response, StreamingResponse), (
            f"expected the custom streaming branch, got {type(response).__name__}"
        )
        assert _accel(response) == "no", (
            f"the root agent/run branch lost its X-Accel-Buffering header: {dict(response.headers)}"
        )


class TestNonStreamingResponsesAreUntouched:
    """The hook is keyed on response TYPE, and must stay that way."""

    async def test_info_response_does_not_get_a_streaming_header(self):
        """``sdk_handler`` also returns ``JSONResponse`` for info / state / errors.

        Stamping a streaming-only header onto an ordinary JSON body would
        needlessly disable proxy buffering for responses that genuinely benefit
        from it, and would be a silent scope creep past what #1673 asked for.
        """
        response = await _drive("info", b"{}", method="GET")

        assert isinstance(response, JSONResponse), (
            f"expected the info branch's JSONResponse, got {type(response).__name__}"
        )
        assert _accel(response) is None, (
            f"a non-streaming JSON response was given a streaming header: {dict(response.headers)}"
        )


class TestTheHeaderIsNotTheOnlyThingThatMustSurvive:
    """#1672's keepalive and #1673's header ride the same seam; neither may
    displace the other."""

    async def test_sdk_subpath_keeps_both_the_keepalive_wrap_and_the_header(
        self, quiet_graph_registry, fast_keepalive
    ):
        """Two fixes now mutate the same SDK-built response on the way out.

        #1669's keepalive bounds the SILENT window (what nginx's
        ``proxy_read_timeout`` kills); #1673's header bounds the BUFFERED window
        (what the user perceives as no streaming at all). They are independent
        defects with independent failure modes, and a future edit to
        ``_bound_silent_window`` that keeps one while dropping the other would
        look correct in isolation.

        Both are asserted on OBSERVED OUTPUT, not on the presence of a symbol: a
        real quiet window in the graph must produce a real ``: keepalive`` record
        on the wire, and the response must carry the header. Asserting that the
        wrapper's frame constant merely exists would pass with the wrap deleted.
        """
        from src.api.routes.copilotkit import copilotkit_custom_handler
        from src.api.utils.sse_keepalive import SSE_KEEPALIVE_FRAME

        response = await copilotkit_custom_handler(
            _make_request("agent/default", _turn_body()),
            quiet_graph_registry,
            path="agent/default",
        )

        assert _accel(response) == "no", f"the #1673 header was dropped: {dict(response.headers)}"

        chunks = [
            c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
            async for c in response.body_iterator
        ]
        keepalives = [c for c in chunks if c.strip() == SSE_KEEPALIVE_FRAME.strip()]
        payloads = [c for c in chunks if c.startswith("data: ")]

        assert keepalives, (
            f"no keepalive ever fired across a {QUIET_SECONDS}s quiet window — the "
            f"#1669 wrap was lost while the #1673 header survived. Chunks seen: "
            f"{[c[:40] for c in chunks[:6]]}"
        )
        assert payloads, f"the stream delivered no data frames at all: {chunks[:3]}"
        types = {json.loads(c[len("data: ") :]).get("type") for c in payloads}
        assert "RUN_FINISHED" in types, f"the turn never completed; saw {sorted(types)}"
