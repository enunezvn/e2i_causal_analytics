"""Red-first tests for #1475 target 1: the bridge's chat/tool-selection leg
runs on the fast tier.

Measured mechanism (2026-08-04, live probes + faithful same-box/key/prompt
experiments — see PR body):

- ``chat_bridge`` = two sequential claude-sonnet-5 legs: the chat leg (~4s in
  prod) and the synthesize leg (~3s). Everything else in the bridge is <1s.
- On bridged turns the chat leg is purely a TOOL ROUTER: sonnet emitted
  0 content chars and an ~86-token tool call in every measured run.
- Thinking is NOT the cost (medium 2.9-3.2s vs none 2.5-2.6s) and prompt
  caching does NOT cut TTFT (cache_read hits: 2.07-2.32s vs uncached 2.19s).
- The fast tier (claude-haiku-4-5) selected the IDENTICAL tool with
  equivalent args on 3/3 real bridged queries at 1.17-1.33s vs sonnet's
  3.12-5.79s — a measured 2-4.5s saving per bridged turn.

Contract pins:

- ``run_conversational_bridge`` builds its per-call graph with the fast
  chat leg (``chat_llm_tier="fast"``, ``chat_llm_reasoning_effort="none"``).
- The AG-UI brain is untouched: ``create_e2i_chat_agent()`` defaults keep
  the chat leg on standard/medium byte-for-byte (two-brain separation).
- The synthesize leg (the user-facing prose author) stays on the standard
  tier in BOTH brains — the quality surface does not change.
- Persisted/analytics ``configured_model`` metadata reports the model that
  actually ran the chat leg — never a hardcoded 'standard' claim (#1257
  provenance-honesty class).
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import src.api.routes.copilotkit as copilotkit_mod
from src.api.routes.chat_bridge import run_conversational_bridge


class _FakeGraph:
    """Plain stand-in for the compiled AG-UI graph (no MagicMock — MagicMock
    fakes hasattr and can mask attribute-shape bugs)."""

    def __init__(self, final_state=None):
        self.final_state = final_state or {}
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        return self.final_state


class TestBridgeUsesFastToolLeg:
    """The bridge's per-call graph is built with the fast chat leg."""

    async def test_bridge_builds_agent_with_fast_tool_leg(self):
        graph = _FakeGraph(final_state={"messages": [AIMessage(content="answer")]})
        with patch(
            "src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph
        ) as factory:
            answer = await run_conversational_bridge(query="q", session_id="u~s")

        assert answer.text == "answer"
        factory.assert_called_once_with(chat_llm_tier="fast", chat_llm_reasoning_effort="none")


class _RecordingFakeLLM:
    """Boundary fake for get_chat_llm: streams one direct-answer chunk."""

    def bind_tools(self, tools, tool_choice=None):
        return self

    async def astream(self, messages):
        yield AIMessageChunk(content="direct answer")


def _patched_chat_node_boundaries():
    """Patch chat_node's side-effect boundaries (DB persistence, analytics,
    CoAgent emits) so the graph can run as a unit under test."""
    return (
        patch.object(copilotkit_mod, "_ensure_conversation_exists", AsyncMock(return_value=False)),
        patch.object(copilotkit_mod, "_persist_message_sync", MagicMock(return_value=None)),
        patch.object(copilotkit_mod, "_record_analytics_sync", MagicMock(return_value=None)),
        patch.object(copilotkit_mod, "_collect_copilot_learning_signal", AsyncMock()),
        patch.object(copilotkit_mod, "copilotkit_emit_state", AsyncMock()),
        patch.object(copilotkit_mod, "copilotkit_emit_message", AsyncMock()),
    )


async def _run_direct_answer_turn(recorder, **agent_kwargs):
    """Build the agent with agent_kwargs and drive one direct-answer turn."""

    def _fake_get_chat_llm(**kwargs):
        recorder.append(kwargs)
        return _RecordingFakeLLM()

    patches = _patched_chat_node_boundaries()
    with (
        patch.object(copilotkit_mod, "get_chat_llm", _fake_get_chat_llm),
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        graph = copilotkit_mod.create_e2i_chat_agent(**agent_kwargs)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is TRx for Kisqali?")]},
            config={"configurable": {"thread_id": "test-1475"}},
        )
    return result


class TestChatLegTierThreading:
    """create_e2i_chat_agent threads the chat-leg tier to get_chat_llm."""

    async def test_fast_tier_reaches_get_chat_llm(self):
        calls = []
        result = await _run_direct_answer_turn(
            calls, chat_llm_tier="fast", chat_llm_reasoning_effort="none"
        )

        assert result["messages"][-1].content == "direct answer"
        assert len(calls) == 1
        assert calls[0]["model_tier"] == "fast"
        assert calls[0]["reasoning_effort"] == "none"

    async def test_default_agent_keeps_standard_medium(self):
        # Two-brain separation: the AG-UI brain (built with no args, both the
        # module singleton and the per-request graph_factory) must keep the
        # chat leg on standard/medium byte-for-byte.
        calls = []
        result = await _run_direct_answer_turn(calls)

        assert result["messages"][-1].content == "direct answer"
        assert len(calls) == 1
        assert calls[0]["model_tier"] == "standard"
        assert calls[0]["reasoning_effort"] == "medium"


class TestConfiguredModelProvenance:
    """configured_model metadata must name the model that actually ran the
    chat leg (#1257 class: never a stronger/wrong provenance claim)."""

    async def test_direct_answer_analytics_reports_actual_tier(self):
        calls = []
        recorded = []

        def _fake_get_chat_llm(**kwargs):
            calls.append(kwargs)
            return _RecordingFakeLLM()

        def _fake_record_analytics(**kwargs):
            recorded.append(kwargs)

        patches = _patched_chat_node_boundaries()
        with (
            patch.object(copilotkit_mod, "get_chat_llm", _fake_get_chat_llm),
            patches[0],
            patch.object(copilotkit_mod, "_persist_message_sync", MagicMock(return_value=None)),
            patch.object(copilotkit_mod, "_record_analytics_sync", _fake_record_analytics),
            patches[3],
            patches[4],
            patches[5],
        ):
            graph = copilotkit_mod.create_e2i_chat_agent(
                chat_llm_tier="fast", chat_llm_reasoning_effort="none"
            )
            await graph.ainvoke(
                {"messages": [HumanMessage(content="What is TRx for Kisqali?")]},
                config={"configurable": {"thread_id": "test-1475-prov"}},
            )

        assert len(recorded) == 1
        configured = recorded[0]["metadata"]["configured_model"]
        fast_model = copilotkit_mod.MODEL_MAPPINGS["anthropic"]["fast"]
        standard_model = copilotkit_mod.MODEL_MAPPINGS["anthropic"]["standard"]
        assert fast_model in configured
        assert standard_model not in configured


class TestSynthesizeLegStaysStandard:
    """The synthesize leg authors the user-facing prose — it must stay on the
    standard tier in both brains. Source-level check because the nodes are
    closures inside create_e2i_chat_agent (established idiom: see
    test_graph_nodes_wire_signal_collection); the live flow is verified
    post-deploy."""

    def test_synthesize_call_site_pins_standard_tier(self):
        source = inspect.getsource(copilotkit_mod.create_e2i_chat_agent)
        synth_source = source.split("async def synthesize_node", 1)[1]
        assert 'model_tier="standard"' in synth_source
        # and the chat leg's call site is parameterized, not hardcoded
        chat_source = source.split("async def chat_node", 1)[1].split(
            "async def synthesize_node", 1
        )[0]
        assert "model_tier=chat_llm_tier" in chat_source
        assert 'model_tier="standard"' not in chat_source
