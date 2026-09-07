"""#1454: per-request node-span instrumentation for the /chat and /chat/stream paths.

The measured cold-start (~68s of an ~80s first request) was unattributable
because the streaming path never activates the chatbot trace context, per-node
durations only ever reached DEBUG logs, and nothing surfaced them to the SSE
consumer, the DB, or MLflow. These tests pin the instrumentation:

1. ``_timed_node`` graph wrapper records FULL-node wall time (not just the
   fragment each node happens to wrap in ``trace_node``) into
   ``ChatbotTraceContext.node_wall_ms``, accumulating across repeat visits
   (the tools<->generate loop) and surviving node exceptions.
2. ``stream_chatbot`` activates the trace context (parity with ``run_chatbot``),
   emits a final synthetic ``__latency_span__`` item, logs one INFO span line
   per request, and always clears the context.
3. ``_stream_chat_response`` surfaces the span in the ``dispatch_info`` SSE
   event so live probes can read latency attribution without container logs.
4. ``finalize_node`` persists ``node_wall_ms`` into the assistant message
   metadata JSONB (no migration needed).
5. ``run_chatbot``'s MLflow metrics include per-node wall times.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

import src.api.routes.chatbot_graph as g
import src.api.routes.copilotkit as ck
from src.api.routes.chatbot_state import ChatbotState
from src.api.routes.chatbot_tracer import ChatbotTraceContext

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _isolate_mlflow_metrics(monkeypatch):
    """Keep every test in this module off the real MLflow HTTP path.

    Same rationale as test_chatbot_graph.py: in CI no MLflow server listens,
    and the client's retry backoff outlives the per-test timeout.
    """
    monkeypatch.setattr(g, "CHATBOT_MLFLOW_METRICS_ENABLED", False)
    monkeypatch.setattr(g, "_mlflow_experiment_id", None)
    monkeypatch.setattr(g, "_mlflow_connector", None)


@pytest.fixture(autouse=True)
def _reset_worker_cold_flag(monkeypatch):
    """Each test starts as if the worker had already served a request, so the
    cold-worker flag is opt-in per test (tests that need a cold worker set the
    module global themselves)."""
    monkeypatch.setattr(g, "_worker_first_request_pending", False, raising=False)


def _ctx() -> ChatbotTraceContext:
    return ChatbotTraceContext(trace_id="trace-t", span_id="span-s", query="q")


# =============================================================================
# 1. ChatbotTraceContext.node_wall_ms + _timed_node wrapper
# =============================================================================


class TestNodeWallTimeLedger:
    def test_record_node_wall_time_accumulates(self):
        ctx = _ctx()
        ctx.record_node_wall_time("tools", 10.0)
        ctx.record_node_wall_time("tools", 5.5)
        assert ctx.node_wall_ms["tools"] == pytest.approx(15.5)

    def test_node_wall_ms_starts_empty(self):
        assert _ctx().node_wall_ms == {}


class TestTimedNodeWrapper:
    @pytest.mark.asyncio
    async def test_records_full_node_wall_time(self):
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:

            async def slow_node(state):
                await asyncio.sleep(0.05)
                return {"intent": "greeting"}

            wrapped = g._timed_node("slow", slow_node)
            update = await wrapped({"query": "q"})
            assert update == {"intent": "greeting"}
            assert ctx.node_wall_ms["slow"] >= 45.0
        finally:
            g._active_trace_context.reset(token)

    @pytest.mark.asyncio
    async def test_accumulates_across_repeat_visits(self):
        """The tools<->generate loop revisits nodes; attribution must sum."""
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:

            async def node(state):
                await asyncio.sleep(0.02)
                return {}

            wrapped = g._timed_node("tools", node)
            await wrapped({})
            await wrapped({})
            assert ctx.node_wall_ms["tools"] >= 35.0
        finally:
            g._active_trace_context.reset(token)

    @pytest.mark.asyncio
    async def test_no_trace_context_is_safe(self):
        async def node(state):
            return {"intent": "help"}

        wrapped = g._timed_node("n", node)
        assert await wrapped({}) == {"intent": "help"}

    @pytest.mark.asyncio
    async def test_records_wall_time_when_node_raises(self):
        """A hung-then-failed node is exactly the case latency attribution
        must survive — record before re-raising."""
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:

            async def bad_node(state):
                await asyncio.sleep(0.02)
                raise ValueError("boom")

            wrapped = g._timed_node("bad", bad_node)
            with pytest.raises(ValueError, match="boom"):
                await wrapped({})
            assert ctx.node_wall_ms["bad"] >= 15.0
        finally:
            g._active_trace_context.reset(token)

    @pytest.mark.asyncio
    async def test_wraps_runnable_with_ainvoke(self):
        """ToolNode is a Runnable, not a coroutine function — the wrapper must
        time it through .ainvoke."""

        class FakeRunnable:
            async def ainvoke(self, state, config=None):
                await asyncio.sleep(0.02)
                return {"messages": []}

        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:
            wrapped = g._timed_node("tools", FakeRunnable())
            update = await wrapped({})
            assert update == {"messages": []}
            assert ctx.node_wall_ms["tools"] >= 15.0
        finally:
            g._active_trace_context.reset(token)


def test_every_production_graph_node_is_timed():
    """Structural pin: adding a node to the graph without timing it silently
    reopens the #1454 attribution hole."""
    graph_nodes = set(g.e2i_chatbot_graph.get_graph().nodes) - {"__start__", "__end__"}
    assert graph_nodes == set(g.TIMED_NODE_NAMES)


# =============================================================================
# 2. stream_chatbot: trace context wiring + synthetic span item + INFO log
# =============================================================================


def _mini_graph(fail_second_node: bool = False):
    """A REAL compiled LangGraph over ChatbotState — the instrumentation under
    test runs against genuine langgraph machinery, not a stand-in."""
    from langgraph.graph import END, StateGraph

    wf = StateGraph(ChatbotState)

    async def alpha(state):
        await asyncio.sleep(0.02)
        return {"intent": "greeting"}

    async def beta(state):
        await asyncio.sleep(0.02)
        if fail_second_node:
            raise RuntimeError("beta exploded")
        return {"response_text": "hi", "streaming_complete": True}

    wf.add_node("alpha", g._timed_node("alpha", alpha))
    wf.add_node("beta", g._timed_node("beta", beta))
    wf.set_entry_point("alpha")
    wf.add_edge("alpha", "beta")
    wf.add_edge("beta", END)
    return wf.compile(checkpointer=False)


async def _consume_stream(**kwargs):
    items = []
    async for item in g.stream_chatbot(
        query=kwargs.pop("query", "hello"),
        user_id=kwargs.pop("user_id", str(uuid.uuid4())),
        request_id=kwargs.pop("request_id", "req-span-test"),
        **kwargs,
    ):
        items.append(item)
    return items


class TestStreamChatbotSpan:
    @pytest.mark.asyncio
    async def test_emits_latency_span_as_final_item(self, monkeypatch, caplog):
        monkeypatch.setattr(g, "e2i_chatbot_graph", _mini_graph())
        monkeypatch.setattr(g, "_worker_first_request_pending", True)
        caplog.set_level(logging.INFO)

        items = await _consume_stream(request_id="req-span-1")

        span_items = [it for it in items if g.LATENCY_SPAN_KEY in it]
        assert len(span_items) == 1, f"expected exactly one span item, got {items}"
        assert items[-1] is span_items[0], "span item must be the final yield"
        payload = span_items[0][g.LATENCY_SPAN_KEY]
        assert set(payload["node_wall_ms"]) == {"alpha", "beta"}
        assert payload["node_wall_ms"]["alpha"] >= 15.0
        node_sum = sum(payload["node_wall_ms"].values())
        assert payload["graph_total_ms"] >= node_sum - 1.0
        assert payload["untimed_overhead_ms"] >= -1.0
        assert payload["first_request_in_worker"] is True
        assert payload["request_id"] == "req-span-1"

    @pytest.mark.asyncio
    async def test_trace_context_set_during_and_cleared_after(self, monkeypatch):
        seen: dict = {}

        from langgraph.graph import END, StateGraph

        wf = StateGraph(ChatbotState)

        async def probe(state):
            seen["ctx"] = g._active_trace_context.get()
            return {"response_text": "ok"}

        wf.add_node("probe", g._timed_node("probe", probe))
        wf.set_entry_point("probe")
        wf.add_edge("probe", END)
        monkeypatch.setattr(g, "e2i_chatbot_graph", wf.compile(checkpointer=False))

        await _consume_stream()

        assert seen["ctx"] is not None, (
            "nodes on the streaming path must see an active trace context "
            "(this was the #1454 root cause: stream_chatbot never set it)"
        )
        assert g._active_trace_context.get() is None, "context must be cleared after the stream"

    @pytest.mark.asyncio
    async def test_second_request_is_not_cold(self, monkeypatch):
        monkeypatch.setattr(g, "e2i_chatbot_graph", _mini_graph())
        monkeypatch.setattr(g, "_worker_first_request_pending", True)

        first = await _consume_stream()
        second = await _consume_stream()

        assert first[-1][g.LATENCY_SPAN_KEY]["first_request_in_worker"] is True
        assert second[-1][g.LATENCY_SPAN_KEY]["first_request_in_worker"] is False

    @pytest.mark.asyncio
    async def test_span_logged_even_when_graph_raises(self, monkeypatch, caplog):
        """A request that dies mid-graph is the one whose latency you need."""
        monkeypatch.setattr(g, "e2i_chatbot_graph", _mini_graph(fail_second_node=True))
        caplog.set_level(logging.INFO)

        with pytest.raises(RuntimeError, match="beta exploded"):
            await _consume_stream(request_id="req-span-err")

        assert g._active_trace_context.get() is None
        span_logs = [r.message for r in caplog.records if "request span" in r.message]
        assert any("req-span-err" in m and "alpha" in m for m in span_logs), (
            f"span log with partial node timings expected on failure; got {span_logs}"
        )

    @pytest.mark.asyncio
    async def test_info_span_log_line(self, monkeypatch, caplog):
        monkeypatch.setattr(g, "e2i_chatbot_graph", _mini_graph())
        caplog.set_level(logging.INFO)

        await _consume_stream(request_id="req-span-log")

        span_logs = [
            r for r in caplog.records if "request span" in r.message and r.levelno == logging.INFO
        ]
        assert span_logs, "one INFO-level span line per request"
        msg = span_logs[-1].message
        for needle in ("req-span-log", "alpha", "beta", "total_ms"):
            assert needle in msg, f"{needle!r} missing from span log: {msg}"


# =============================================================================
# 3. _stream_chat_response surfaces the span in dispatch_info
# =============================================================================


class TestDispatchInfoSpan:
    @pytest.mark.asyncio
    async def test_dispatch_info_carries_span_fields(self, monkeypatch):
        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": "hello there"}}
            yield {
                g.LATENCY_SPAN_KEY: {
                    "request_id": "r1",
                    "node_wall_ms": {"init": 5.0, "classify_intent": 28000.0},
                    "graph_total_ms": 42000.0,
                    "untimed_overhead_ms": 13995.0,
                    "first_request_in_worker": True,
                }
            }

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        req = ck.ChatRequest(query="hi", user_id="u", request_id="r1", session_id="s1")

        events = []
        async for chunk in ck._stream_chat_response(req, "auth-user"):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))

        dispatch = [e for e in events if e["type"] == "dispatch_info"]
        assert len(dispatch) == 1
        di = dispatch[0]["data"]
        assert di["node_wall_ms"] == {"init": 5.0, "classify_intent": 28000.0}
        assert di["graph_total_ms"] == 42000.0
        assert di["untimed_overhead_ms"] == 13995.0
        assert di["first_request_in_worker"] is True

    @pytest.mark.asyncio
    async def test_span_item_produces_no_text_event(self, monkeypatch):
        """The synthetic item must never leak into the visible answer."""

        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": "the answer"}}
            yield {
                g.LATENCY_SPAN_KEY: {
                    "request_id": "r2",
                    "node_wall_ms": {"init": 5.0},
                    "graph_total_ms": 10.0,
                    "untimed_overhead_ms": 5.0,
                    "first_request_in_worker": False,
                }
            }

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        req = ck.ChatRequest(query="hi", user_id="u", request_id="r2", session_id="s2")

        events = []
        async for chunk in ck._stream_chat_response(req, "auth-user"):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))

        error_events = [e for e in events if e["type"] == "error"]
        assert error_events == [], f"stream errored: {error_events}"
        text_events = [e for e in events if e["type"] == "text"]
        assert [e["data"] for e in text_events] == ["the answer"]


# =============================================================================
# 4. finalize_node persists node_wall_ms in assistant message metadata
# =============================================================================


class TestFinalizePersistsSpan:
    @pytest.mark.asyncio
    async def test_assistant_metadata_includes_node_wall_ms(self, monkeypatch):
        monkeypatch.setattr(g, "CHATBOT_SIGNAL_COLLECTION_ENABLED", False)
        ctx = _ctx()
        ctx.record_node_wall_time("classify_intent", 28000.0)
        ctx.record_node_wall_time("orchestrator", 34000.0)
        token = g._active_trace_context.set(ctx)

        mock_client = AsyncMock()
        mock_msg_repo = AsyncMock()
        try:
            with patch(
                "src.api.routes.chatbot_graph.get_async_supabase_client",
                return_value=mock_client,
            ):
                with patch(
                    "src.api.routes.chatbot_graph.get_chatbot_message_repository",
                    return_value=mock_msg_repo,
                ):
                    with patch(
                        "src.api.routes.chatbot_graph._save_to_episodic_memory",
                        new=AsyncMock(return_value=None),
                    ):
                        state = {
                            "messages": [AIMessage(content="the answer")],
                            "session_id": "u1~s1",
                            "query": "q",
                            "request_id": "req-span-db",
                        }
                        await g.finalize_node(state)
        finally:
            g._active_trace_context.set(None)
            del token

        assistant_calls = [
            c
            for c in mock_msg_repo.add_message.await_args_list
            if c.kwargs.get("role") == "assistant"
        ]
        assert assistant_calls, "assistant message was not persisted"
        metadata = assistant_calls[0].kwargs["metadata"]
        assert metadata["node_wall_ms"] == {
            "classify_intent": 28000.0,
            "orchestrator": 34000.0,
        }

    @pytest.mark.asyncio
    async def test_no_trace_context_persists_null_not_fabrication(self, monkeypatch):
        """Without a trace context there is no measurement — the metadata field
        must be an honest None, never {} pretending to be a measured empty."""
        monkeypatch.setattr(g, "CHATBOT_SIGNAL_COLLECTION_ENABLED", False)
        assert g._active_trace_context.get() is None

        mock_client = AsyncMock()
        mock_msg_repo = AsyncMock()
        with patch(
            "src.api.routes.chatbot_graph.get_async_supabase_client",
            return_value=mock_client,
        ):
            with patch(
                "src.api.routes.chatbot_graph.get_chatbot_message_repository",
                return_value=mock_msg_repo,
            ):
                with patch(
                    "src.api.routes.chatbot_graph._save_to_episodic_memory",
                    new=AsyncMock(return_value=None),
                ):
                    state = {
                        "messages": [AIMessage(content="a")],
                        "session_id": "u1~s2",
                        "query": "q",
                        "request_id": "req-span-null",
                    }
                    await g.finalize_node(state)

        assistant_calls = [
            c
            for c in mock_msg_repo.add_message.await_args_list
            if c.kwargs.get("role") == "assistant"
        ]
        assert assistant_calls[0].kwargs["metadata"]["node_wall_ms"] is None


# =============================================================================
# 4b. run_chatbot span totals use the monotonic clock (codex iter-1 MED)
# =============================================================================


class TestRunChatbotSpanClock:
    @pytest.mark.asyncio
    async def test_span_total_uses_monotonic_clock(self, monkeypatch, caplog):
        """codex iter-1 MED: node timings use perf_counter; if /chat's span
        total comes from time.time(), a stepped wall clock fabricates or hides
        untimed_overhead_ms. Freeze time.time() — the span total must still
        measure the real ~30ms of graph work."""
        caplog.set_level(logging.INFO)
        monkeypatch.setattr("time.time", lambda: 1_754_300_000.0)

        async def slow_ainvoke(state, config=None):
            await asyncio.sleep(0.03)
            return {"response_text": "ok", "metadata": {}}

        with patch.object(g.e2i_chatbot_graph, "ainvoke", new=AsyncMock(side_effect=slow_ainvoke)):
            await g.run_chatbot(query="hi", user_id="u", request_id="req-clock")

        span_logs = [
            r.message
            for r in caplog.records
            if "request span" in r.message and "req-clock" in r.message
        ]
        assert span_logs, "run_chatbot must emit the request-span log line"
        match = re.search(r"total_ms=([0-9.]+)", span_logs[-1])
        assert match, span_logs[-1]
        assert float(match.group(1)) >= 25.0, (
            f"span total came from the frozen wall clock, not perf_counter: {span_logs[-1]}"
        )


# =============================================================================
# 5. run_chatbot MLflow metrics include per-node wall times
# =============================================================================


class TestMlflowPerNodeMetrics:
    def test_metrics_include_node_wall_times(self):
        ctx = _ctx()
        ctx.record_node_wall_time("classify_intent", 28000.0)
        ctx.record_node_wall_time("orchestrator", 34000.0)
        metrics = g._build_chat_mlflow_metrics(
            result={
                "response_text": "x",
                "metadata": {"total_tokens": 12},
                "tool_results": [],
                "rag_context": [],
                "intent": "kpi_query",
            },
            latency_ms=80000.0,
            error_occurred=False,
            trace_ctx=ctx,
        )
        assert metrics["latency_ms"] == 80000.0
        assert metrics["node_classify_intent_ms"] == 28000.0
        assert metrics["node_orchestrator_ms"] == 34000.0
        # Pre-existing keys preserved by the extraction
        assert metrics["total_tokens"] == 12
        assert metrics["response_length"] == 1
        assert metrics["intent_kpi_query"] == 1
        assert metrics["tool_calls_count"] == 0
        assert metrics["rag_result_count"] == 0
        assert metrics["is_error"] == 0

    def test_metrics_without_trace_ctx_add_no_node_keys(self):
        metrics = g._build_chat_mlflow_metrics(
            result={"response_text": "", "metadata": {}, "tool_results": [], "rag_context": []},
            latency_ms=100.0,
            error_occurred=True,
            trace_ctx=None,
        )
        assert not any(k.startswith("node_") for k in metrics)
        assert metrics["is_error"] == 1


# =============================================================================
# 6. worker_pid on the span (#1454 warm lane)
# =============================================================================


class TestWorkerPidSpanField:
    """The warm task logs its completion with a pid; the probe needs the SAME
    identifier on the request side to prove the worker it hit is one whose warm
    completed. Derived at emission time — never carried on a ChatbotState
    channel (the #1442 checkpointer-replay class)."""

    def test_span_payload_carries_the_emitting_worker_pid(self):
        payload = g._build_latency_span_payload("req-pid", None, 12.0, True)
        assert payload["worker_pid"] == os.getpid()

    def test_span_log_line_carries_worker_pid(self, caplog):
        """The non-streaming /chat path surfaces the span ONLY through this log
        line, so the pid has to be on it for warm/request correlation."""
        caplog.set_level(logging.INFO)
        g._log_request_span(g._build_latency_span_payload("req-pid-log", None, 12.0, True))

        lines = [r.getMessage() for r in caplog.records if "request span" in r.getMessage()]
        assert lines, "span line must be logged"
        assert f"worker_pid={os.getpid()}" in lines[-1]

    def test_worker_pid_is_not_a_chatbot_state_channel(self):
        assert "worker_pid" not in ChatbotState.__annotations__, (
            "#1442 class: per-request worker identity must not ride graph state"
        )

    @pytest.mark.asyncio
    async def test_dispatch_info_carries_worker_pid(self, monkeypatch):
        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": "hello"}}
            yield {
                g.LATENCY_SPAN_KEY: {
                    "request_id": "r3",
                    "node_wall_ms": {"init": 5.0},
                    "graph_total_ms": 40.0,
                    "untimed_overhead_ms": 35.0,
                    "first_request_in_worker": True,
                    "worker_pid": 4242,
                }
            }

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        req = ck.ChatRequest(query="hi", user_id="u", request_id="r3", session_id="s3")

        events = []
        async for chunk in ck._stream_chat_response(req, "auth-user"):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))

        dispatch = [e for e in events if e["type"] == "dispatch_info"]
        assert len(dispatch) == 1
        assert dispatch[0]["data"]["worker_pid"] == 4242


# =============================================================================
# 6. #1933: the generate span must report the model that was actually called
# =============================================================================


class _StubLLM:
    """A chat client shaped like what ``get_chat_llm`` returns.

    Deliberately NOT an ``AsyncMock``: ``getattr(mock, "model")`` auto-creates a
    truthy child mock, so a mock would let a broken model lookup pass while
    asserting nothing. The attribute has to be a real string for these tests to
    mean anything.
    """

    def __init__(self, model: str, response: AIMessage):
        self.model = model
        self._response = response

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return self._response


def _capture_node_spans(ctx: ChatbotTraceContext) -> dict:
    """Record the NodeSpanContext each ``trace_node`` yields, by node name.

    ``log_generate`` writes into that object's ``metadata``, which is what
    reaches Opik via ``span.set_output`` — so the metadata dict is the thing
    under test, not a log line.
    """
    from contextlib import asynccontextmanager

    captured: dict = {}
    real = ctx.trace_node

    @asynccontextmanager
    async def spy(node_name, metadata=None):
        async with real(node_name, metadata) as node_span:
            captured[node_name] = node_span
            yield node_span

    ctx.trace_node = spy  # instance attribute shadows the bound method
    return captured


def _llm_state():
    from src.api.routes.chatbot_state import create_initial_state

    state = create_initial_state(
        user_id="u-1933",
        query="What is the TRx for Kisqali?",
        request_id="req-1933",
        session_id="u-1933~s-1933",
    )
    state["messages"] = [g.HumanMessage(content="What is the TRx for Kisqali?")]
    return state


async def _run_generate_with_span(state, monkeypatch, *, llm=None, synthesis=None):
    """Drive ``generate_node`` under a live trace context; return generate's metadata."""
    ctx = _ctx()
    captured = _capture_node_spans(ctx)

    if synthesis is None:
        monkeypatch.setattr(g, "CHATBOT_DSPY_SYNTHESIS_ENABLED", False)
        monkeypatch.setattr(g, "get_chat_llm", lambda **kwargs: llm)
        monkeypatch.setattr(g, "get_llm_provider", lambda: "anthropic")
    else:
        monkeypatch.setattr(g, "CHATBOT_DSPY_SYNTHESIS_ENABLED", True)
        monkeypatch.setattr(g, "synthesize_response_dspy", synthesis)

    token = g._active_trace_context.set(ctx)
    try:
        await g.generate_node(state)
    finally:
        g._active_trace_context.reset(token)

    assert "generate" in captured, "generate node never opened its trace span"
    return captured["generate"].metadata


class TestGenerateSpanReportsTheRealModel:
    """#1933: the span attributed cost/latency to a model that is never invoked.

    ``ANTHROPIC_MODEL`` is deliberately NOT forwarded into the containers
    (docker/docker-compose.yml states the policy), so inside `e2i_api` the read
    could only ever return its own hardcoded fallback — a plausible-looking id
    that nothing had called. Meanwhile the factory resolves the tier through
    ``MODEL_MAPPINGS`` and calls something else entirely. Measured 2026-09-07::

        MODEL_MAPPINGS["anthropic"]["standard"]           -> claude-sonnet-5
        os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6") -> claude-sonnet-4-6

    Latent while Opik is stopped, live the moment it is switched back on.
    """

    @pytest.mark.asyncio
    async def test_span_records_the_model_of_the_client_that_was_called(self, monkeypatch):
        llm = _StubLLM("claude-sonnet-5", AIMessage(content="answer"))

        metadata = await _run_generate_with_span(_llm_state(), monkeypatch, llm=llm)

        assert metadata["model"] == "claude-sonnet-5", (
            "the generate span must report the model of the client actually "
            "invoked, read off the constructed object"
        )
        assert metadata["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_span_ignores_the_unforwarded_anthropic_model_env(self, monkeypatch):
        """Even when the host sets it, it must not reach the span.

        The variable is host-side-only by policy. Honouring it here would
        reintroduce exactly the divergence #1933 is about: the span naming a
        model the factory never selected.
        """
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-1-interactive")
        llm = _StubLLM("claude-sonnet-5", AIMessage(content="answer"))

        metadata = await _run_generate_with_span(_llm_state(), monkeypatch, llm=llm)

        assert metadata["model"] == "claude-sonnet-5"
        assert metadata["model"] != "claude-opus-4-1-interactive"

    @pytest.mark.asyncio
    async def test_span_never_reports_the_old_hardcoded_fallback(self, monkeypatch):
        """Regression pin on the literal that was being recorded in production."""
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        llm = _StubLLM("claude-sonnet-5", AIMessage(content="answer"))

        metadata = await _run_generate_with_span(_llm_state(), monkeypatch, llm=llm)

        assert metadata["model"] != "claude-sonnet-4-6", (
            "claude-sonnet-4-6 was the unreachable fallback the span recorded "
            "for every containerised request"
        )

    def test_anthropic_model_is_never_read_from_the_environment_here(self):
        """The read itself is the defect, not just the value it returned.

        Compose forwards nothing under this name, so any reader inside a
        container observes only its own default. Asserted against the source so
        a future edit cannot quietly reintroduce it — and scoped to the *read*
        rather than the identifier, because naming the variable in a comment to
        explain why it must not be read is the correct thing to do.
        """
        import inspect

        source = inspect.getsource(g)
        reads = re.findall(
            r"os\.(?:environ\.get|getenv|environ)\s*[(\[]\s*[\"']ANTHROPIC_MODEL[\"']", source
        )
        assert not reads, (
            "src/api/routes/chatbot_graph.py must not read ANTHROPIC_MODEL from the "
            "environment: it is deliberately not forwarded into the containers, so "
            "the read can only return the caller's own fallback (#1933). Read the "
            "model off the client returned by get_chat_llm instead."
        )


class TestGenerateSpanOperatorPrecedence:
    """#1933 defect B: ``a or b if c else d`` binds as ``(a or b) if c else d``.

    So the plain-LLM path — every request that does not synthesize — recorded
    ``"unknown"`` for both model and provider, discarding values it already had.
    """

    @pytest.mark.asyncio
    async def test_non_synthesis_path_does_not_record_unknown(self, monkeypatch):
        llm = _StubLLM("claude-sonnet-5", AIMessage(content="answer"))

        metadata = await _run_generate_with_span(_llm_state(), monkeypatch, llm=llm)

        assert metadata["model"] != "unknown"
        assert metadata["provider"] != "unknown"

    @pytest.mark.asyncio
    async def test_synthesis_path_still_reports_dspy(self, monkeypatch):
        """The parenthesisation must not disturb the branch that was correct.

        On the synthesis path ``model_name`` is never assigned, so the old
        expression happened to yield "dspy_synthesis" via the ``or``. The fixed
        one must reach the same answer deliberately rather than by accident.
        """

        class _Synth:
            response = "synthesized answer"
            synthesis_method = "evidence_synthesis"
            confidence_statement = "high confidence"
            confidence_level = "high"
            evidence_citations = ["c1"]
            follow_up_suggestions = ["f1"]

        async def fake_synthesize(**kwargs):
            return _Synth()

        state = _llm_state()
        state["rag_context"] = [{"content": "evidence", "source": "s", "relevance_score": 0.9}]

        metadata = await _run_generate_with_span(state, monkeypatch, synthesis=fake_synthesize)

        assert metadata["model"] == "dspy_synthesis"
        assert metadata["provider"] == "dspy"

    @pytest.mark.asyncio
    async def test_llm_failure_records_unknown_because_no_client_exists(self, monkeypatch):
        """The fallback path keeps "unknown" — there honestly is no model to name."""

        def _boom(**kwargs):
            raise RuntimeError("LLM failed")

        monkeypatch.setattr(g, "CHATBOT_DSPY_SYNTHESIS_ENABLED", False)
        monkeypatch.setattr(g, "get_chat_llm", _boom)
        monkeypatch.setattr(g, "get_llm_provider", lambda: "anthropic")

        ctx = _ctx()
        captured = _capture_node_spans(ctx)
        token = g._active_trace_context.set(ctx)
        try:
            await g.generate_node(_llm_state())
        finally:
            g._active_trace_context.reset(token)

        metadata = captured["generate"].metadata
        assert metadata["is_fallback"] is True
        assert metadata["model"] == "unknown", (
            "no client was constructed, so there is no model to name — 'unknown' "
            "is the honest value here, unlike on the working LLM path"
        )
