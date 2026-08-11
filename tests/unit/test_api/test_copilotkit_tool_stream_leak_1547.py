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
    async def test_composer_run_yields_one_prose_lifecycle(self):
        """Replay the 2.6 event shape INCLUDING the on_chat_model_end events:
        decomposer blob (tools) + end, planner blob (tools) + end, synthesized
        prose (synthesize) + end. The delivered stream must be exactly ONE
        paired TEXT_MESSAGE_START/END lifecycle with the prose in between —
        no JSON deltas, no extra lifecycles from the suppressed tool streams
        (this pins the "no lifecycle corruption" claim)."""
        agent = _bare_agent()
        prose = "The composer could not complete planning; here is the grounded view."
        emitted: List[Any] = []
        for ev in (
            _stream_event("tools", '{\n  "reasoning": "The query asks for..."', "lc_run--dec"),
            _end_event("tools"),
            _stream_event("tools", '{\n  "reasoning": "sq_1 and sq_2 are DES', "lc_run--plan"),
            _end_event("tools"),
            _stream_event("synthesize", prose, "lc_run--synth"),
            _end_event("synthesize"),
        ):
            emitted.extend(await _collect(agent, ev))

        types = [str(getattr(e, "type", "")) for e in emitted]
        starts = [e for e, t in zip(emitted, types, strict=True) if "TEXT_MESSAGE_START" in t]
        contents = [e for e, t in zip(emitted, types, strict=True) if "TEXT_MESSAGE_CONTENT" in t]
        ends = [e for e, t in zip(emitted, types, strict=True) if "TEXT_MESSAGE_END" in t]

        # Exactly one lifecycle — the synthesize stream. The unpatched library
        # produces THREE (one per model call), the first two carrying JSON.
        assert len(starts) == 1, types
        assert len(ends) == 1, types
        assert [starts[0].message_id, ends[0].message_id] == ["lc_run--synth", "lc_run--synth"]

        # Pairing/order: START, then all CONTENT, then END.
        i_start = types.index(next(t for t in types if "TEXT_MESSAGE_START" in t))
        i_end = types.index(next(t for t in types if "TEXT_MESSAGE_END" in t))
        i_contents = [i for i, t in enumerate(types) if "TEXT_MESSAGE_CONTENT" in t]
        assert i_contents and i_start < min(i_contents) and max(i_contents) < i_end, types

        text = "".join(c.delta for c in contents)
        assert text == prose
        assert '"reasoning"' not in text
        assert all(c.message_id == "lc_run--synth" for c in contents)

        # And the ended lifecycle left clean bookkeeping behind.
        assert agent.messages_in_process.get("run-1") is None


class TestDispatcherCoupling:
    """The filter only works while the library dispatches events through the
    overridden method name. Pin that coupling so a future ag_ui_langgraph /
    copilotkit upgrade that renames the dispatcher fails HERE (loudly) instead
    of silently turning the override into dead code and resurrecting the leak.
    """

    def test_override_is_defined_on_our_subclass(self):
        """Removing the override (or renaming it away from the dispatcher's
        method name) must fail this pin."""
        assert "_handle_single_event" in LangGraphAgent.__dict__
        assert callable(LangGraphAgent.__dict__["_handle_single_event"])

    def test_base_chain_defines_the_overridden_method(self):
        """Every base up the chain that we delegate through must still define
        ``_handle_single_event`` — i.e. our method genuinely OVERRIDES an
        inherited implementation rather than dangling on its own."""
        bases_defining = [
            klass
            for klass in LangGraphAgent.__mro__[1:]
            if "_handle_single_event" in klass.__dict__
        ]
        assert bases_defining, "no base class defines _handle_single_event (library renamed it?)"
        inherited = bases_defining[0].__dict__["_handle_single_event"]
        assert callable(inherited)
        assert LangGraphAgent.__dict__["_handle_single_event"] is not inherited

    def test_library_run_loop_still_dispatches_via_the_overridden_name(self):
        """The installed ag_ui_langgraph run loop dispatches every stream event
        via ``self._handle_single_event(...)`` (agent.py:218 at pin time) —
        that dynamic ``self.`` dispatch is what makes our override effective.
        If an upgrade renames the call site, this pin fails."""
        import inspect

        import ag_ui_langgraph.agent as _agui_agent_module

        assert "self._handle_single_event(" in inspect.getsource(_agui_agent_module)


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
