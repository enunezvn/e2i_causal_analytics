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
