"""#1673 — the AG-UI stream must actually STREAM through the production edge.

THE ASSERTION IS ``TTFB << total``. Nothing weaker catches this defect.

A fully buffered turn still delivers every frame, still returns HTTP 200, still
produces the correct answer, and still passes ``frames > 0`` — which is #1667's
guard — and it passes a ``max_gap`` bound too, because once nginx flushes, the
frames land microseconds apart. Measured on production before the fix::

    through nginx   TTFB = 8.754s   total = 8.756s   ttfb/total = 0.9998   chunks = 2
    direct to app   TTFB = 0.263s   total = 7.709s   ttfb/total = 0.0341   chunks = 81

The user waits ~9 s at a blank screen and then receives the whole turn at once.
Token-level streaming is produced by the application and destroyed at the proxy.

WHAT ACTUALLY CAUSES IT — MEASURED, AND NOT WHAT #1673 ASSUMED
--------------------------------------------------------------
#1673 named ``proxy_buffering on``. That is NOT the cause. The same production
URL, same nginx, same ``location /api/``, differing only in ``Accept-Encoding``::

    Accept-Encoding: gzip       TTFB 8.754s / total 8.756s   -> buffered
    Accept-Encoding: identity   TTFB 0.683s / total 9.218s   -> streams, 99 chunks

The cause is nginx's **gzip filter**. The third-party CopilotKit SDK builds its
``StreamingResponse`` with ``media_type="application/json"``
(``copilotkit/integrations/fastapi.py::handle_execute_agent``) — a mislabel for
what is really an SSE byte stream — and ``application/json`` is in the live
``gzip_types``. The gzip filter accumulates deflate output until its buffer fills
or a **flush marker** arrives, and with ``proxy_buffering on`` no flush markers
are produced, so it holds the entire turn.

Our other two SSE responses escape by accident, not by design: both declare
``media_type="text/event-stream"``, which is absent from ``gzip_types``.

WHY THE FIX IS ``X-Accel-Buffering: no`` DESPITE THE CAUSE BEING GZIP
--------------------------------------------------------------------
The header turns off *proxy buffering*, not gzip. That it nevertheless defeats
gzip buffering was VERIFIED, not assumed — it is the single load-bearing
assumption of the fix, so it got the cheapest experiment that could falsify it.
A replica nginx (same 1.24.0 binary, ``gzip``/``gzip_types``/``gzip_proxied``/
``proxy_buffer*`` directives copied verbatim from the live config) in front of a
stub upstream, one variable per cell::

    cell                                       TTFB   total  ttfb/tot  chunks   enc   verdict
    A  json, no header, gzip        (= prod)  9.234   9.235    0.9999       1  gzip   buffered
    B  json, no header, identity                0.06   9.096    0.0066      60     -   streams
    C  json, X-Accel-Buffering:no, gzip        0.065   9.095    0.0071      61  gzip   STREAMS
    D  event-stream, no header, gzip           0.071   9.098    0.0078      60     -   streams
    F  nginx `gzip off`, json, gzip            0.074   9.110    0.0081      60     -   streams
    G  nginx `proxy_buffering off`, json       0.175   9.211    0.0190      61  gzip   streams

Cell C is the disproof that did not happen: the response is **still gzipped**
(``enc=gzip``) and still streams in 61 chunks. Turning proxy buffering off makes
nginx mark each upstream read with ``flush``, and the gzip filter honours flush
with a ``Z_SYNC_FLUSH``. So the header is sufficient even though gzip stays on.

WHY NOT FIX IT IN NGINX
-----------------------
Two measured reasons, either one decisive:

1. **The location #1673 proposes editing is dead.** ``location /copilotkit/``
   proxies to ``127.0.0.1:8000/copilotkit/``, a prefix the app does not serve —
   authenticated probes return ``404 EndpointNotFoundError`` on every path,
   through nginx and direct alike. The live surface is ``location /api/``: the
   shipped frontend bundle bakes ``apiUrl:"/api"`` and builds its runtime URL as
   ``${apiUrl}/copilotkit/``, and ``scripts/demos/copilot_agui_runner.py``
   defaults to ``--api-base https://eznomics.site/api``. Editing
   ``docker/nginx/host-nginx.conf`` line 127 would ship an inert fix.
2. **``location /api/`` is the whole REST API.** Disabling gzip or proxy
   buffering there to fix one streaming endpoint de-optimises every JSON
   response the platform serves.

And the nginx config reaches production only by a manual root ``cp`` +
``nginx -t`` + ``systemctl reload`` (``docs/runbooks/frontend-serving-flip.md``);
no workflow ships it. ``/etc/nginx/sites-enabled/e2i-analytics`` is a regular
file, not a symlink, and ``sites-available/e2i-analytics`` currently holds a
*different, older* file — the tracked-to-live path has already drifted once. The
application header ships with the ordinary image deploy.

WHAT THIS TEST DOES NOT MOCK
----------------------------
The transport is real end to end: a real nginx process running the production
gzip and proxy-buffer directives, over a real TCP socket, in front of a real
uvicorn serving the real ``copilotkit_custom_handler`` and the real third-party
``CopilotKitRemoteEndpoint`` -> ``handle_execute_agent`` -> ``StreamingResponse``.
A mocked transport cannot observe buffering at all — buffering is a property of
the bytes on the wire, and an in-process ASGI call has no wire.

Only the GRAPH is a stub — a real ``StateGraph`` over a real ``BaseChatModel``
subclass whose ``_astream`` paces its chunks — so the event cadence is
controllable and no paid LLM call is made. That is the same boundary
``test_agui_stream_health_1667_1669.py`` draws, with one difference that matters
here: that file's ``GenericFakeChatModel`` emits everything as fast as the loop
will run, and a turn that finishes in milliseconds cannot distinguish a buffered
response from a streamed one.

Two fidelity gaps, stated rather than hidden: this runs uvicorn directly where
production runs it under gunicorn's ``UvicornWorker``, and it speaks cleartext
where production terminates TLS at nginx (behind ``sslh``). Both were checked
against production instead of being reasoned about — the direct-to-app control
through the real gunicorn returned TTFB 0.263 s over 81 chunks, and the
``Accept-Encoding: identity`` run through the real TLS edge streamed in 99
chunks. Neither layer buffers.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional, TypedDict

import httpx
import pytest

# Module scope, not function scope, and deliberately so: this module uses
# ``from __future__ import annotations``, which turns every annotation into a
# string. FastAPI resolves a route handler's annotations against its MODULE
# namespace, so a function-local ``from fastapi import Request`` leaves
# ``"Request"`` unresolvable and FastAPI silently downgrades the parameter to a
# required query param — every POST then 422s before the handler ever runs.
from fastapi import Request
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

pytestmark = [pytest.mark.integration]

#: nginx must emit its first byte inside this fraction of the total turn.
#: Production before the fix measured 0.9998; after, the replica cells that
#: stream measured 0.0066-0.0190. Anything under a third is unambiguously
#: streaming and leaves an order of magnitude of headroom on both sides.
STREAMING_RATIO = 1 / 3

#: The stub graph streams for at least this long. #1673's whole point is that a
#: sub-second turn cannot distinguish buffered from streamed, so the turn has to
#: be genuinely multi-second.
MIN_TURN_SECONDS = 3.0

#: ~2.6 KB per token chunk x 30 chunks x 150 ms => a ~4.6 s turn of ~240 KB.
#: Sized against the real thing: the production turn measured for #1673 was
#: 181,125 bytes over 81 chunks in 7.7 s.
_CHUNK_TEXT = "Kisqali TRx grew twelve percent quarter over quarter. " * 50
_N_CHUNKS = 30
_CHUNK_DELAY = 0.15


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# The stub graph — controls the event CADENCE, nothing else
# ---------------------------------------------------------------------------


class SlowStreamingChatModel(BaseChatModel):
    """A real ``BaseChatModel`` that PACES its token stream.

    ``GenericFakeChatModel`` (what the #1667/#1669 harness uses) emits its whole
    message as fast as the loop will run. That is fine for asserting *which*
    events appear, but useless here: a turn that finishes in milliseconds looks
    identical whether the proxy buffered it or not. Buffering is only observable
    against elapsed time, so the stub has to spend real time.

    It is a genuine ``BaseChatModel`` with a real ``_astream``, so it drives real
    ``on_chat_model_start`` / ``on_chat_model_stream`` / ``on_chat_model_end``
    callbacks into ``astream_events``, which is what the AG-UI translation layer
    consumes. Verified end to end through the real ``LangGraphAgent.execute``:
    66 frames over 3.4 s, of which 24 were ``TEXT_MESSAGE_CONTENT``.
    """

    n_chunks: int = _N_CHUNKS
    delay: float = _CHUNK_DELAY
    chunk_text: str = _CHUNK_TEXT

    @property
    def _llm_type(self) -> str:
        return "slow-streaming-stub"

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for _ in range(self.n_chunks):
            await asyncio.sleep(self.delay)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=self.chunk_text))
            if run_manager:
                await run_manager.on_llm_new_token(self.chunk_text, chunk=chunk)
            yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.chunk_text * self.n_chunks))]
        )


class _StubState(TypedDict):
    messages: Annotated[list, add_messages]
    tools: list


def _build_streaming_graph():
    """A real compiled ``StateGraph`` whose single node streams over several seconds."""
    llm = SlowStreamingChatModel()

    async def chat(state: _StubState) -> dict:
        return {"messages": [await llm.ainvoke(state["messages"])]}

    workflow = StateGraph(_StubState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# The real edge: real nginx -> real uvicorn -> real route handler
# ---------------------------------------------------------------------------

#: Copied VERBATIM from the live edge. The http-block gzip directives come from
#: ``/etc/nginx/nginx.conf`` and the server-block gzip plus the whole
#: ``location /api/`` proxy stanza from ``/etc/nginx/sites-enabled/e2i-analytics``
#: (which matches the tracked ``docker/nginx/host-nginx.conf`` modulo two
#: comment lines, verified by diff). Deviations, all deliberate: cleartext on a
#: loopback port instead of ``listen 127.0.0.1:4443 ssl`` because TLS is
#: transport-layer and cannot reach the gzip or proxy filters; ``limit_req``
#: dropped because rate limiting would only add noise; temp paths and logs
#: redirected into the test's tmp dir so nginx runs unprivileged.
_NGINX_CONF = """
daemon off;
worker_processes 1;
pid {tmp}/nginx.pid;
error_log {tmp}/error.log warn;
events {{ worker_connections 1024; multi_accept on; }}
http {{
    access_log off;
    client_body_temp_path {tmp}/client_body;
    proxy_temp_path {tmp}/proxy_temp;
    fastcgi_temp_path {tmp}/fastcgi_temp;
    uwsgi_temp_path {tmp}/uwsgi_temp;
    scgi_temp_path {tmp}/scgi_temp;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    default_type application/octet-stream;
    keepalive_timeout 65;

    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_min_length 256;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript application/xml+rss font/truetype font/opentype image/svg+xml;

    server {{
        listen 127.0.0.1:{edge_port};
        server_name eznomics.site;

        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;

        location /api/ {{
            proxy_pass http://127.0.0.1:{app_port}/api/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 32k;
            proxy_busy_buffers_size 64k;
        }}
    }}
}}
"""


def _build_app():
    """A FastAPI app whose ``/api/copilotkit/{path}`` is the REAL handler.

    ``add_copilotkit_routes`` is not reused because it calls
    ``create_copilotkit_sdk()``, which wires the production chat graph and would
    make paid LLM calls. Everything it does that MATTERS here is reproduced:
    the same ``copilotkit_custom_handler``, the same route shape, and a real
    ``CopilotKitRemoteEndpoint`` so the third-party dispatch
    (``copilotkit/sdk.py::execute_agent`` -> ``handle_execute_agent`` ->
    ``StreamingResponse``) runs exactly as in production.
    """
    from copilotkit import CopilotKitRemoteEndpoint
    from fastapi import FastAPI

    from src.api.routes.copilotkit import LangGraphAgent, copilotkit_custom_handler

    agent = LangGraphAgent(
        name="default",
        description="stub agent for edge buffering measurement",
        graph=_build_streaming_graph(),
    )
    sdk = CopilotKitRemoteEndpoint(agents=[agent], actions=[])

    app = FastAPI()

    async def handler(request: Request, path: str = ""):
        return await copilotkit_custom_handler(request, sdk, path)

    app.add_api_route(
        "/api/copilotkit/{path:path}",
        handler,
        methods=["GET", "POST"],
        include_in_schema=False,
    )
    return app


class _Edge:
    """A running nginx + uvicorn pair. ``base`` is the client-facing origin."""

    def __init__(self, base: str, app_base: str):
        self.base = base
        self.app_base = app_base


@pytest.fixture(scope="module")
def edge(tmp_path_factory):
    """Start a real nginx in front of a real uvicorn; tear both down after."""
    nginx_bin = shutil.which("nginx")
    if not nginx_bin:
        pytest.skip(
            "nginx binary not on PATH — this test measures buffering at the real "
            "proxy and there is nothing faithful to substitute for it"
        )

    import uvicorn

    tmp = tmp_path_factory.mktemp("agui_edge")
    app_port = _free_port()
    edge_port = _free_port()

    config = uvicorn.Config(
        _build_app(), host="127.0.0.1", port=app_port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    conf_path = Path(tmp) / "nginx.conf"
    conf_path.write_text(_NGINX_CONF.format(tmp=str(tmp), edge_port=edge_port, app_port=app_port))
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [nginx_bin, "-p", str(tmp), "-c", str(conf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not server.started:
            time.sleep(0.05)
            continue
        try:
            with socket.create_connection(("127.0.0.1", edge_port), timeout=0.5):
                break
        except OSError:
            if proc.poll() is not None:
                err = (
                    (Path(tmp) / "error.log").read_text()
                    if (Path(tmp) / "error.log").exists()
                    else ""
                )
                pytest.skip(f"replica nginx failed to start: {err[:400]}")
            time.sleep(0.05)
    else:
        pytest.skip("replica edge did not come up within 30s")

    try:
        yield _Edge(f"http://127.0.0.1:{edge_port}", f"http://127.0.0.1:{app_port}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
        server.should_exit = True
        thread.join(timeout=15)


class Timeline:
    """When each byte actually arrived, which is the only thing #1673 is about."""

    def __init__(self, status: int, headers: Dict[str, str], arrivals: List[float], nbytes: int):
        self.status = status
        self.headers = headers
        self.arrivals = arrivals
        self.nbytes = nbytes
        self.total = arrivals[-1] if arrivals else 0.0

    @property
    def ttfb(self) -> float:
        return self.arrivals[0] if self.arrivals else float("inf")

    @property
    def chunks(self) -> int:
        return len(self.arrivals)

    @property
    def ratio(self) -> float:
        return self.ttfb / self.total if self.total else float("inf")

    def __repr__(self) -> str:
        return (
            f"Timeline(status={self.status}, ttfb={self.ttfb:.3f}s, total={self.total:.3f}s, "
            f"ttfb/total={self.ratio:.4f}, chunks={self.chunks}, bytes={self.nbytes}, "
            f"content-encoding={self.headers.get('content-encoding', '-')!r}, "
            f"content-type={self.headers.get('content-type', '-')!r})"
        )


def _drive(origin: str, accept_encoding: str = "gzip") -> Timeline:
    """One real AG-UI turn over real HTTP, timing every chunk off the socket."""
    payload = {
        "threadId": str(uuid.uuid4()),
        "state": {},
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "type": "TextMessage",
                "role": "user",
                "content": "What is the current TRx for Kisqali?",
            }
        ],
        "actions": [],
    }
    arrivals: List[float] = []
    nbytes = 0
    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
        with client.stream(
            "POST",
            f"{origin}/api/copilotkit/agent/default",
            headers={
                "Content-Type": "application/json",
                "Accept-Encoding": accept_encoding,
                "Authorization": "Bearer edge-buffering-probe",
            },
            json=payload,
        ) as resp:
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            for raw in resp.iter_raw():
                if raw:
                    arrivals.append(time.monotonic() - t0)
                    nbytes += len(raw)
    return Timeline(status, headers, arrivals, nbytes)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


class TestTheTurnIsLongEnoughToJudge:
    """A sub-second turn cannot distinguish buffered from streamed."""

    def test_upstream_turn_is_multi_second(self, edge):
        """Guards the guard: if the stub graph gets fast, every assertion below
        becomes vacuously true without failing."""
        timeline = _drive(edge.app_base, accept_encoding="identity")

        assert timeline.status == 200, f"upstream did not answer: {timeline!r}"
        assert timeline.total >= MIN_TURN_SECONDS, (
            f"turn lasted only {timeline.total:.2f}s — under {MIN_TURN_SECONDS}s a "
            f"buffered response is indistinguishable from a streamed one, so the "
            f"TTFB assertions below would pass for the wrong reason ({timeline!r})"
        )


class TestApplicationDoesNotBuffer:
    """Establish the control BEFORE blaming the proxy."""

    def test_direct_to_app_streams(self, edge):
        """#1673's direct-to-app control reported ``frames=1``, which would have
        implicated the app. It does not reproduce: the app streams cleanly.

        This is not a formality. If the app buffered, the header fix would be
        pointless and the whole diagnosis would be wrong.
        """
        timeline = _drive(edge.app_base)

        assert timeline.status == 200, f"upstream did not answer: {timeline!r}"
        assert timeline.chunks > 1, (
            f"the application itself delivered the turn in one chunk — nginx is not "
            f"the only thing buffering and the fix is misplaced ({timeline!r})"
        )
        assert timeline.ratio < STREAMING_RATIO, (
            f"the application buffered: first byte at {timeline.ttfb:.3f}s of a "
            f"{timeline.total:.3f}s turn ({timeline!r})"
        )


class TestEdgeStreams:
    """The defect, and the property that was actually broken."""

    def test_ttfb_is_far_below_total_through_nginx(self, edge):
        """RED before the fix: TTFB == total, one chunk, the whole turn at the end.

        Deliberately asserted on TTFB rather than on the response header. nginx
        CONSUMES ``X-Accel-Buffering`` and never forwards it, so a client cannot
        see it — and a header assertion would pass against a proxy that ignored
        the header entirely. This measures the delivery, not the instruction.
        """
        timeline = _drive(edge.base)

        assert timeline.status == 200, f"edge did not answer: {timeline!r}"
        assert timeline.total >= MIN_TURN_SECONDS, f"turn too short to judge: {timeline!r}"
        assert timeline.ratio < STREAMING_RATIO, (
            f"THE TURN IS BUFFERED AT THE EDGE: first byte at {timeline.ttfb:.3f}s of a "
            f"{timeline.total:.3f}s turn (ttfb/total={timeline.ratio:.4f}). The user "
            f"stares at nothing for the whole turn and then receives it all at once. "
            f"Note the answer is CORRECT and the status is 200 — status codes and "
            f"frame counts cannot see this. {timeline!r}"
        )

    def test_it_streams_while_still_gzipped(self, edge):
        """The fix must not work by accidentally disabling compression.

        ``gzip off`` in the location would also make the turn stream, at the cost
        of every JSON response under ``location /api/``. If compression silently
        stopped, this passes for the wrong reason and we would have paid a
        bandwidth regression we never chose.
        """
        timeline = _drive(edge.base, accept_encoding="gzip")

        assert timeline.status == 200, f"edge did not answer: {timeline!r}"
        assert timeline.headers.get("content-encoding") == "gzip", (
            f"the response is no longer compressed — the stream may be flowing only "
            f"because gzip stopped applying, which is a different (unchosen) change "
            f"({timeline!r})"
        )
        assert timeline.ratio < STREAMING_RATIO, f"still buffered with gzip active: {timeline!r}"
        assert timeline.chunks > 1, f"gzip still collapsed the turn to one flush: {timeline!r}"


class TestGzipIsTheMechanism:
    """Pins the DIAGNOSIS, so a future reader does not re-derive it from the
    issue's original (wrong) ``proxy_buffering on`` hypothesis."""

    def test_identity_encoding_streams_even_when_the_fix_is_absent(self, edge):
        """``Accept-Encoding: identity`` is the control that identified gzip.

        It streams regardless of the fix, because ``proxy_buffering on`` alone
        does not hold a chunked response — nginx forwards buffers as they fill.
        Kept as an executable record of the cheapest disproof: it is what
        eliminated ``proxy_buffering`` as the cause and pointed at the gzip
        filter, which is why the fix targets the response header rather than the
        buffer directives #1673 named.
        """
        timeline = _drive(edge.base, accept_encoding="identity")

        assert timeline.status == 200, f"edge did not answer: {timeline!r}"
        assert timeline.headers.get("content-encoding") is None, (
            f"expected an uncompressed response for the control: {timeline!r}"
        )
        assert timeline.ratio < STREAMING_RATIO, (
            f"uncompressed AND buffered — the cause is not gzip after all and this "
            f"whole diagnosis needs redoing ({timeline!r})"
        )
