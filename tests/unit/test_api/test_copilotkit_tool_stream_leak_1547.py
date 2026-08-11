"""Regression tests for #1547: tool-internal LLM streams must never render as answer text.

Root cause (eval 2026-08-11, turn 2.6): LangGraph's ``astream_events`` propagates
callbacks into LLM calls made INSIDE tools (async contextvars), and
ag_ui_langgraph's ``_handle_single_event`` translates EVERY
``on_chat_model_stream`` event into a TEXT_MESSAGE lifecycle. When
``tool_composer_tool`` ran, its decompose/plan phases each ``ainvoke``d a
LangChain chat model, so their raw planner JSON generations streamed into the
delivered answer as TEXT_MESSAGE_CONTENT — two blobs (~3,000 chars), the second
truncated mid-generation, before any prose. Measured in
``docs/demos/results/2026-08-11_copilot_chat_perf/raw_agui.jsonl``: the leaked
TEXT_MESSAGE lifecycles carry ``rawEvent.metadata.langgraph_node == "tools"``,
while the legitimate answer streams carry ``"chat"`` / ``"synthesize"``.

Contract under test: ``LangGraphAgent._handle_single_event`` drops
``on_chat_model_*`` events whose ``metadata.langgraph_node`` is the tools node
BEFORE translation, and passes every other node's stream through unchanged.
"""

from typing import Any, Dict, List, Optional

import pytest

# NOTE: importing the route module is heavyweight but is the established
# pattern for this module's unit tests (see test_copilotkit_stream_toolcalls).
from src.api.routes.copilotkit import _TOOL_NODE_NAME, LangGraphAgent

pytestmark = pytest.mark.unit


class _FakeChunk:
    """Minimal AIMessageChunk stand-in (only the fields the translator reads)."""

    def __init__(self, content: Any = "", chunk_id: str = "lc_run--test"):
        self.content = content
        self.id = chunk_id
        self.response_metadata: Dict[str, Any] = {}
        self.tool_call_chunks: List[Dict[str, Any]] = []
        self.additional_kwargs: Dict[str, Any] = {}


def _bare_agent() -> LangGraphAgent:
    """Construct the wrapper without a compiled graph (pattern from
    tests/unit/test_patches/test_agui_langgraph_agent.py)."""
    agent = object.__new__(LangGraphAgent)
    agent.messages_in_process = {}
    agent.active_run = {"id": "run-1"}
    return agent


def _stream_event(node: Optional[str], content: str, chunk_id: str = "lc_run--test") -> dict:
    """An astream_events-v2 ``on_chat_model_stream`` event as ag_ui sees it."""
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node} if node is not None else {},
        "data": {"chunk": _FakeChunk(content=content, chunk_id=chunk_id)},
    }


def _end_event(node: str) -> dict:
    return {
        "event": "on_chat_model_end",
        "metadata": {"langgraph_node": node},
        "data": {},
    }


async def _collect(agent: LangGraphAgent, event: dict) -> list:
    return [e async for e in agent._handle_single_event(event, {})]


LEAKED_PLANNER_JSON = (
    '{\n  "reasoning": "The query asks for a resource allocation optimization '
    'across three levers..."'
)


class TestToolNodeStreamSuppressed:
    async def test_tools_node_chat_model_stream_emits_nothing(self):
        """The 2.6 leak: a planner JSON generation streaming from inside the
        tools node must not produce ANY AG-UI events."""
        agent = _bare_agent()
        events = await _collect(agent, _stream_event("tools", LEAKED_PLANNER_JSON))
        assert events == [], (
            "tool-internal chat-model stream leaked into the answer stream: "
            f"{[getattr(e, 'type', e) for e in events]}"
        )

    async def test_tools_node_stream_leaves_no_message_in_progress(self):
        """A dropped tool-internal stream must not corrupt the message
        lifecycle state the real chat/synthesize streams rely on."""
        agent = _bare_agent()
        await _collect(agent, _stream_event("tools", LEAKED_PLANNER_JSON))
        assert agent.messages_in_process.get("run-1") is None

    async def test_tools_node_chat_model_end_emits_nothing(self):
        agent = _bare_agent()
        events = await _collect(agent, _end_event("tools"))
        assert events == []


class TestLegitimateStreamsUntouched:
    @pytest.mark.parametrize("node", ["chat", "synthesize"])
    async def test_answer_nodes_still_stream(self, node: str):
        agent = _bare_agent()
        events = await _collect(agent, _stream_event(node, "Here is the regional picture."))
        types = [str(getattr(e, "type", "")) for e in events]
        assert any("TEXT_MESSAGE_START" in t for t in types), types
        assert any("TEXT_MESSAGE_CONTENT" in t for t in types), types
        content = next(e for e in events if "TEXT_MESSAGE_CONTENT" in str(e.type))
        assert content.delta == "Here is the regional picture."

    async def test_node_metadata_missing_still_streams(self):
        """Fail open for events with no node metadata — only a POSITIVE match
        on the tools node may suppress (never silence unknown streams)."""
        agent = _bare_agent()
        events = await _collect(agent, _stream_event(None, "prose"))
        types = [str(getattr(e, "type", "")) for e in events]
        assert any("TEXT_MESSAGE_CONTENT" in t for t in types), types


class TestEvalTurn26Shape:
    async def test_composer_run_yields_prose_only(self):
        """Replay the 2.6 event shape: decomposer blob (tools) + planner blob
        (tools) + synthesized prose (synthesize). Delivered text must be the
        prose alone — no JSON blobs, no truncated fragment."""
        agent = _bare_agent()
        prose = "The composer could not complete planning; here is the grounded view."
        delivered: List[str] = []
        for ev in (
            _stream_event("tools", '{\n  "reasoning": "The query asks for..."', "lc_run--dec"),
            _stream_event("tools", '{\n  "reasoning": "sq_1 and sq_2 are DES', "lc_run--plan"),
            _stream_event("synthesize", prose, "lc_run--synth"),
        ):
            for out in await _collect(agent, ev):
                if "TEXT_MESSAGE_CONTENT" in str(getattr(out, "type", "")):
                    delivered.append(out.delta)
        text = "".join(delivered)
        assert text == prose
        assert '"reasoning"' not in text


class TestGraphCoupling:
    def test_tool_node_name_constant_matches_filter(self):
        """The filter keys on the graph's tools-node name; keep them coupled
        through the shared constant."""
        assert _TOOL_NODE_NAME == "tools"
        assert LangGraphAgent._is_tool_internal_llm_event(
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": _TOOL_NODE_NAME}}
        )

    def test_non_model_events_from_tools_node_not_matched(self):
        """Only chat-model callback events are dropped — tool lifecycle events
        (on_tool_start/end, custom events) from the tools node must pass."""
        assert not LangGraphAgent._is_tool_internal_llm_event(
            {"event": "on_tool_start", "metadata": {"langgraph_node": "tools"}}
        )
        assert not LangGraphAgent._is_tool_internal_llm_event(
            {"event": "on_custom_event", "metadata": {"langgraph_node": "tools"}}
        )

    def test_non_dict_event_not_matched(self):
        assert not LangGraphAgent._is_tool_internal_llm_event("RUN_STARTED")
