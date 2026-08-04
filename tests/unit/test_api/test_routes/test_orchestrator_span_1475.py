"""#1475 WI1: orchestrator-internal span attribution.

#1454 / PR #1471 attributed request latency down to the chatbot graph's node
boundary and measured the warm floor at ~14.9s INSIDE the orchestrator node —
a black box. This mirrors #1471's pattern one level down:

1. ``ChatbotTraceContext`` gains an ``orchestrator_stage_ms`` ledger plus
   ``orchestrator_run_ms`` / ``orchestrator_untimed_ms``.
2. ``orchestrator_node`` activates the stage-timing contextvar ledger around
   ``orchestrator.run`` — the SHARED ``audited_node`` wrapper (perf_counter)
   fills it with ``{agent}.{node}`` wall times — and times its own legs
   (``get_orchestrator``, ``chat_bridge``) with the same clock.
3. The SAME #1471 surfaces publish it: the ``__latency_span__`` payload /
   ``[Chatbot] request span`` INFO log, MLflow metrics, and the SSE
   ``dispatch_info`` event.

Hard constraints carried over from shipped incidents:
- per-request timings ride the trace-context CONTEXTVAR, never a
  ChatbotState channel (checkpointer replay class #1442);
- every sub-timing and total shares time.perf_counter() — no time.time()
  mixing (fabricates untimed overhead under wall-clock steps);
- measurement only: no orchestrator behavior change.
"""

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import src.api.routes.chatbot_graph as g
import src.api.routes.copilotkit as ck
from src.api.routes.chatbot_state import IntentType, create_initial_state
from src.api.routes.chatbot_tracer import ChatbotTraceContext
from src.utils.stage_timing import get_active_stage_ledger, record_stage_wall_time


@pytest.fixture(autouse=True)
def _isolate_mlflow_metrics(monkeypatch):
    """Same rationale as test_chatbot_span_1454.py: never touch real MLflow."""
    monkeypatch.setattr(g, "CHATBOT_MLFLOW_METRICS_ENABLED", False)
    monkeypatch.setattr(g, "_mlflow_experiment_id", None)
    monkeypatch.setattr(g, "_mlflow_connector", None)


def _ctx() -> ChatbotTraceContext:
    return ChatbotTraceContext(trace_id="trace-t", span_id="span-s", query="q")


def _state():
    state = create_initial_state(
        user_id="test-user-123",
        query="What is the TRx for Kisqali?",
        request_id="req-1475",
        session_id="test-user-123~session-456",
        brand_context="Kisqali",
        region_context="US",
    )
    state["intent"] = IntentType.KPI_QUERY
    return state


# =============================================================================
# 1. ChatbotTraceContext orchestrator ledger
# =============================================================================


class TestTraceContextOrchestratorLedger:
    def test_starts_empty_and_unmeasured(self):
        ctx = _ctx()
        assert ctx.orchestrator_stage_ms == {}
        assert ctx.orchestrator_run_ms is None
        assert ctx.orchestrator_untimed_ms is None

    def test_stage_times_accumulate(self):
        ctx = _ctx()
        ctx.record_orchestrator_stage_time("orchestrator.dispatch", 10.0)
        ctx.record_orchestrator_stage_time("orchestrator.dispatch", 5.5)
        assert ctx.orchestrator_stage_ms["orchestrator.dispatch"] == pytest.approx(15.5)

    def test_run_and_untimed_accumulate(self):
        ctx = _ctx()
        ctx.record_orchestrator_run(100.0, 40.0)
        ctx.record_orchestrator_run(50.0, 10.0)
        assert ctx.orchestrator_run_ms == pytest.approx(150.0)
        assert ctx.orchestrator_untimed_ms == pytest.approx(50.0)


# =============================================================================
# 2. orchestrator_node stage attribution
# =============================================================================


def _fake_orchestrator(run_result=None, sleep_s=0.03, ledger_stages=None, expect_ledger=True):
    """An orchestrator whose ``run`` proves the ledger is ACTIVE during the
    call by recording stages exactly as audited_node would.

    ``expect_ledger=False`` for the no-trace-context path: without a consumer
    the node deliberately activates no ledger."""

    async def run(input_data):
        if expect_ledger:
            assert get_active_stage_ledger() is not None, (
                "orchestrator_node must activate the stage ledger around orchestrator.run"
            )
        for stage, ms in (ledger_stages or {}).items():
            record_stage_wall_time(stage, ms)
        await asyncio.sleep(sleep_s)
        return run_result or {
            "response_text": "answer",
            "response_confidence": 0.9,
            "agents_dispatched": ["causal_impact"],
            "successful_agents": ["causal_impact"],
            "failed_agents": [],
            "status": "completed",
        }

    orch = SimpleNamespace()
    orch.run = run
    return orch


class TestOrchestratorNodeStageAttribution:
    @pytest.mark.asyncio
    async def test_inside_run_stages_land_on_the_trace_context(self):
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:
            orch = _fake_orchestrator(
                ledger_stages={"orchestrator.classify": 10.0, "orchestrator.dispatch": 5.0}
            )
            with (
                patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
                patch.object(g, "get_orchestrator", return_value=orch),
            ):
                result = await g.orchestrator_node(_state())
        finally:
            g._active_trace_context.reset(token)

        assert result["orchestrator_used"] is True
        assert ctx.orchestrator_stage_ms["orchestrator.classify"] == pytest.approx(10.0)
        assert ctx.orchestrator_stage_ms["orchestrator.dispatch"] == pytest.approx(5.0)
        # run wall time is a REAL perf_counter measurement of the ~30ms run
        assert ctx.orchestrator_run_ms is not None and ctx.orchestrator_run_ms >= 25.0
        # untimed = run - (orchestrator.* stages) = ~30ms - 15ms
        assert ctx.orchestrator_untimed_ms is not None
        assert ctx.orchestrator_untimed_ms == pytest.approx(
            ctx.orchestrator_run_ms - 15.0, abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_untimed_overhead_excludes_nested_agent_stages(self):
        """Stages of DISPATCHED agents (``causal_impact.*``) are attribution
        WITHIN orchestrator.dispatch — subtracting them too would double-count
        and drive untimed negative."""
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:
            orch = _fake_orchestrator(
                ledger_stages={
                    "orchestrator.dispatch": 5.0,
                    "causal_impact.analyze": 999_999.0,  # nested, huge on purpose
                }
            )
            with (
                patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
                patch.object(g, "get_orchestrator", return_value=orch),
            ):
                await g.orchestrator_node(_state())
        finally:
            g._active_trace_context.reset(token)

        assert ctx.orchestrator_stage_ms["causal_impact.analyze"] == pytest.approx(999_999.0)
        assert ctx.orchestrator_untimed_ms is not None
        assert ctx.orchestrator_untimed_ms >= 0.0, (
            "nested agent stages must not be subtracted from the run wall time"
        )

    @pytest.mark.asyncio
    async def test_get_orchestrator_leg_is_timed(self):
        """The registry build (~3.4-4s cold) happens inside get_orchestrator();
        it needs its own stage so a cold hit is attributable."""
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:

            def slow_get_orchestrator():
                time.sleep(0.02)  # sync singleton build, as in production
                return _fake_orchestrator(sleep_s=0.0)

            with (
                patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
                patch.object(g, "get_orchestrator", slow_get_orchestrator),
            ):
                await g.orchestrator_node(_state())
        finally:
            g._active_trace_context.reset(token)

        assert ctx.orchestrator_stage_ms["get_orchestrator"] >= 15.0

    @pytest.mark.asyncio
    async def test_chat_bridge_leg_is_timed_on_complete_failure(self):
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:
            orch = _fake_orchestrator(
                run_result={
                    "response_text": "",
                    "response_confidence": 0.0,
                    "agents_dispatched": ["causal_impact"],
                    "successful_agents": [],
                    "failed_agents": ["causal_impact"],
                    "failure_details": [],
                    "status": "failed",
                },
                sleep_s=0.0,
            )

            async def slow_bridge(**kwargs):
                await asyncio.sleep(0.02)
                return SimpleNamespace(text="bridge answer", tool_grounded=True)

            with (
                patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
                patch.object(g, "get_orchestrator", return_value=orch),
                patch.object(g, "run_conversational_bridge", slow_bridge),
            ):
                result = await g.orchestrator_node(_state())
        finally:
            g._active_trace_context.reset(token)

        assert result["agent_name"] == "chat_bridge"
        assert ctx.orchestrator_stage_ms["chat_bridge"] >= 15.0

    @pytest.mark.asyncio
    async def test_stage_ledger_is_deactivated_after_the_node(self):
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:
            with (
                patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
                patch.object(g, "get_orchestrator", return_value=_fake_orchestrator(sleep_s=0.0)),
            ):
                await g.orchestrator_node(_state())
        finally:
            g._active_trace_context.reset(token)
        assert get_active_stage_ledger() is None

    @pytest.mark.asyncio
    async def test_run_failure_still_attributes_run_wall_time(self):
        """The request whose orchestrator dies is the one whose span you need."""
        ctx = _ctx()
        token = g._active_trace_context.set(ctx)
        try:

            async def failing_run(input_data):
                record_stage_wall_time("orchestrator.classify", 7.0)
                await asyncio.sleep(0.02)
                raise RuntimeError("orchestrator exploded")

            orch = SimpleNamespace(run=failing_run)
            with (
                patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
                patch.object(g, "get_orchestrator", return_value=orch),
            ):
                result = await g.orchestrator_node(_state())  # node swallows, falls through
        finally:
            g._active_trace_context.reset(token)

        assert result.get("orchestrator_used") is not True
        assert ctx.orchestrator_stage_ms["orchestrator.classify"] == pytest.approx(7.0)
        assert ctx.orchestrator_run_ms is not None and ctx.orchestrator_run_ms >= 15.0

    @pytest.mark.asyncio
    async def test_no_trace_context_is_safe_and_records_nothing(self):
        assert g._active_trace_context.get() is None
        orch = _fake_orchestrator(sleep_s=0.0, expect_ledger=False)
        with (
            patch.object(g, "CHATBOT_ORCHESTRATOR_ENABLED", True),
            patch.object(g, "get_orchestrator", return_value=orch),
        ):
            result = await g.orchestrator_node(_state())
        assert result["orchestrator_used"] is True
        assert get_active_stage_ledger() is None, "no consumer -> no ledger left active"


# =============================================================================
# 3. Span payload + INFO log surfaces
# =============================================================================


class TestSpanPayloadOrchestratorFields:
    def test_payload_carries_orchestrator_attribution(self):
        ctx = _ctx()
        ctx.record_orchestrator_stage_time("orchestrator.dispatch", 12000.04)
        ctx.record_orchestrator_stage_time("get_orchestrator", 3.0)
        ctx.record_orchestrator_run(14900.0, 2896.96)

        payload = g._build_latency_span_payload("req-orch", ctx, 16000.0, False)

        assert payload["orchestrator_stage_ms"] == {
            "orchestrator.dispatch": 12000.0,
            "get_orchestrator": 3.0,
        }
        assert payload["orchestrator_run_ms"] == pytest.approx(14900.0)
        assert payload["orchestrator_untimed_ms"] == pytest.approx(2897.0)

    def test_unmeasured_payload_is_honest(self):
        """No orchestrator measurement -> {} / None, never fabricated zeros."""
        payload = g._build_latency_span_payload("req-none", _ctx(), 100.0, False)
        assert payload["orchestrator_stage_ms"] == {}
        assert payload["orchestrator_run_ms"] is None
        assert payload["orchestrator_untimed_ms"] is None

    def test_no_trace_ctx_payload_is_honest(self):
        payload = g._build_latency_span_payload("req-nctx", None, 100.0, False)
        assert payload["orchestrator_stage_ms"] == {}
        assert payload["orchestrator_run_ms"] is None
        assert payload["orchestrator_untimed_ms"] is None


class TestSpanLogOrchestratorFields:
    def test_span_log_line_carries_orchestrator_attribution(self, caplog):
        caplog.set_level(logging.INFO)
        ctx = _ctx()
        ctx.record_orchestrator_stage_time("orchestrator.dispatch", 12000.0)
        ctx.record_orchestrator_run(14900.0, 2900.0)

        g._log_request_span(g._build_latency_span_payload("req-orch-log", ctx, 16000.0, False))

        lines = [r.getMessage() for r in caplog.records if "request span" in r.getMessage()]
        assert lines, "span line must be logged"
        msg = lines[-1]
        for needle in (
            "orchestrator_run_ms=14900.0",
            "orchestrator_untimed_ms=2900.0",
            "orchestrator.dispatch",
        ):
            assert needle in msg, f"{needle!r} missing from span log: {msg}"


# =============================================================================
# 4. MLflow metrics surface
# =============================================================================


class TestMlflowOrchestratorMetrics:
    def test_metrics_include_orchestrator_stages(self):
        ctx = _ctx()
        ctx.record_orchestrator_stage_time("orchestrator.dispatch", 12000.0)
        ctx.record_orchestrator_stage_time("causal_impact.analyze", 9000.0)
        ctx.record_orchestrator_run(14900.0, 2900.0)

        metrics = g._build_chat_mlflow_metrics(
            result={"response_text": "x", "metadata": {}, "tool_results": [], "rag_context": []},
            latency_ms=16000.0,
            error_occurred=False,
            trace_ctx=ctx,
        )

        assert metrics["orch_run_ms"] == pytest.approx(14900.0)
        assert metrics["orch_untimed_ms"] == pytest.approx(2900.0)
        assert metrics["orch_orchestrator_dispatch_ms"] == pytest.approx(12000.0)
        assert metrics["orch_causal_impact_analyze_ms"] == pytest.approx(9000.0)

    def test_unmeasured_metrics_add_no_orch_keys(self):
        metrics = g._build_chat_mlflow_metrics(
            result=None,
            latency_ms=100.0,
            error_occurred=True,
            trace_ctx=_ctx(),
        )
        assert not any(k.startswith("orch_") for k in metrics)


# =============================================================================
# 5. SSE dispatch_info surface
# =============================================================================


class TestDispatchInfoOrchestratorSpan:
    @pytest.mark.asyncio
    async def test_dispatch_info_carries_orchestrator_span_fields(self, monkeypatch):
        async def fake_stream(**kwargs):
            yield {"finalize": {"response_text": "hello"}}
            yield {
                g.LATENCY_SPAN_KEY: {
                    "request_id": "r-orch",
                    "node_wall_ms": {"orchestrator": 14900.0},
                    "graph_total_ms": 16000.0,
                    "untimed_overhead_ms": 1100.0,
                    "first_request_in_worker": False,
                    "worker_pid": 4242,
                    "orchestrator_stage_ms": {"orchestrator.dispatch": 12000.0},
                    "orchestrator_run_ms": 14900.0,
                    "orchestrator_untimed_ms": 2900.0,
                }
            }

        monkeypatch.setattr(g, "stream_chatbot", fake_stream)
        req = ck.ChatRequest(query="hi", user_id="u", request_id="r-orch", session_id="s-orch")

        events = []
        async for chunk in ck._stream_chat_response(req, "auth-user"):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))

        dispatch = [e for e in events if e["type"] == "dispatch_info"]
        assert len(dispatch) == 1
        di = dispatch[0]["data"]
        assert di["orchestrator_stage_ms"] == {"orchestrator.dispatch": 12000.0}
        assert di["orchestrator_run_ms"] == 14900.0
        assert di["orchestrator_untimed_ms"] == 2900.0


# =============================================================================
# 6. OrchestratorAgent memory legs are stage-timed
# =============================================================================


class TestOrchestratorAgentMemoryStages:
    @pytest.mark.asyncio
    async def test_memory_read_and_contribute_are_recorded(self):
        """orchestrator.run does two non-graph awaits (history hydration and
        memory contribution) — without their own stages they would land in
        untimed overhead unexplained."""
        from src.agents.orchestrator.agent import OrchestratorAgent
        from src.utils.stage_timing import activate_stage_ledger, deactivate_stage_ledger

        agent = OrchestratorAgent(agent_registry={}, enable_opik=False, enable_memory=True)

        final_state = {
            "status": "completed",
            "agent_results": [],
            "synthesized_response": "ok",
        }

        hooks = SimpleNamespace(
            get_conversation_history=AsyncMock(return_value=[{"role": "user", "content": "hi"}])
        )
        agent._memory_hooks = hooks

        ledger, token = activate_stage_ledger()
        try:
            with (
                patch.object(agent.graph, "ainvoke", new=AsyncMock(return_value=final_state)),
                patch(
                    "src.agents.orchestrator.agent.contribute_to_memory",
                    new=AsyncMock(return_value={}),
                ),
            ):
                await agent.run({"query": "q", "session_id": "s-1475"})
        finally:
            deactivate_stage_ledger(token)

        assert "orchestrator.memory_read" in ledger
        assert "orchestrator.memory_contribute" in ledger
