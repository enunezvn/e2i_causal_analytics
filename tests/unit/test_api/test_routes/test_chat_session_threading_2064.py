"""#2064: chat tools use the real chat session instead of inventing one.

``orchestrator_tool`` and ``tool_composer_tool`` are invoked by LangGraph's
ToolNode with the model's arguments only. The model cannot know the frontend
thread id, so the tools used to fill the gap with ``chatbot-<timestamp>`` /
``composer-<timestamp>`` / ``fallback-<timestamp>`` — ids that look like a
session but belong to no conversation. Once #2069 threaded the orchestrator's
session into ``composer_episodes``, those values would have made unattributable
compositions look attributed.

The real session is bound by each chat brain before its tools run:

* AG-UI (``copilotkit.py`` execute) sets ``_session_id_context``;
* ``/chat/stream`` (``chatbot_graph``) binds ``state["session_id"]`` around its
  ``tools`` node.

Every test here drives the production entry (the tool, a real ToolNode, or the
compiled chatbot graph's own ``tools`` node); only the orchestrator and the
composer behind the tools are faked.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.api.routes import chatbot_tools
from src.api.routes.copilotkit import _session_id_context

REAL_SESSION = "0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21"
COMPOSITE_SESSION = "4a0c1f1e-9d7b-4f5e-8a2d-6b3c2e1d0f99~" + REAL_SESSION
# The shapes the tools used to invent.
FABRICATED = re.compile(r"^(chatbot|composer|fallback)-\d{14}$")


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def run(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "response_text": "ok",
            "response_confidence": 0.9,
            "agents_dispatched": [],
            "status": "completed",
        }


@pytest.fixture
def orchestrator(monkeypatch) -> _FakeOrchestrator:
    fake = _FakeOrchestrator()
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: fake)
    return fake


@pytest.fixture
def failing_composer(monkeypatch) -> list[dict]:
    """Capture the composer context, then fail so the orchestrator fallback runs."""
    contexts: list[dict] = []

    async def compose_query(*, query, context):
        contexts.append(context)
        raise RuntimeError("composition failed")

    monkeypatch.setattr(chatbot_tools, "compose_query", compose_query)
    monkeypatch.setattr(chatbot_tools.kpi_resolution, "recognize_kpi", lambda query: None)
    monkeypatch.setattr(chatbot_tools, "_resolve_cohort_frame", lambda *args: None)
    return contexts


def _tool_call_message(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-2064"}])


# ------------------------------------------------------------ orchestrator_tool


async def test_orchestrator_tool_uses_the_bound_chat_session(orchestrator):
    token = _session_id_context.set(REAL_SESSION)
    try:
        result = await chatbot_tools.orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})
    finally:
        _session_id_context.reset(token)

    assert orchestrator.payloads[0]["session_id"] == REAL_SESSION
    assert result["context"]["session_id"] == REAL_SESSION


async def test_a_model_supplied_session_does_not_override_the_bound_one(orchestrator):
    """The model can only guess (the schema example even offers ``sess_abc123``)."""
    token = _session_id_context.set(REAL_SESSION)
    try:
        await chatbot_tools.orchestrator_tool.ainvoke(
            {"query": "Why is TRx moving?", "session_id": "sess_abc123"}
        )
    finally:
        _session_id_context.reset(token)

    assert orchestrator.payloads[0]["session_id"] == REAL_SESSION


async def test_orchestrator_tool_invents_no_session_when_none_is_bound(orchestrator):
    result = await chatbot_tools.orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})

    assert orchestrator.payloads[0]["session_id"] is None
    assert result["context"]["session_id"] is None


async def test_a_real_tool_node_carries_the_bound_session_into_the_tool(orchestrator):
    """LangGraph's ToolNode, run by a compiled graph (as AG-UI's chat graph runs it),
    sees the binding the handler made before the run started."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("tools", ToolNode(chatbot_tools.E2I_CHATBOT_TOOLS))
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    graph = workflow.compile()

    token = _session_id_context.set(REAL_SESSION)
    try:
        await graph.ainvoke({"messages": [_tool_call_message("orchestrator_tool", {"query": "q"})]})
    finally:
        _session_id_context.reset(token)

    assert orchestrator.payloads[0]["session_id"] == REAL_SESSION


# ------------------------------------------------------------ tool_composer_tool


async def test_tool_composer_tool_uses_the_bound_chat_session(orchestrator, failing_composer):
    token = _session_id_context.set(REAL_SESSION)
    try:
        await chatbot_tools.tool_composer_tool.ainvoke(
            {"query": "Compare TRx and explain drivers", "session_id": "sess_abc123"}
        )
    finally:
        _session_id_context.reset(token)

    assert failing_composer[0]["session_id"] == REAL_SESSION
    # The orchestrator fallback behind a failed composition gets it too.
    assert orchestrator.payloads[0]["session_id"] == REAL_SESSION


async def test_tool_composer_tool_invents_no_session_when_none_is_bound(
    orchestrator, failing_composer
):
    await chatbot_tools.tool_composer_tool.ainvoke({"query": "Compare TRx and explain drivers"})

    assert failing_composer[0]["session_id"] is None
    assert orchestrator.payloads[0]["session_id"] is None


def test_composer_context_passes_an_absent_session_through_as_none():
    context = chatbot_tools._composer_context(
        brand=None, region=None, session_id=None, max_parallel=3
    )
    assert context["session_id"] is None


# ------------------------------------------------------------ /chat/stream brain


async def test_chatbot_graph_tools_node_binds_the_state_session(orchestrator):
    """The compiled /chat/stream graph's own ``tools`` node callable, not a rebuilt copy.

    It is mounted in a one-node graph only so LangGraph supplies the runtime
    config ToolNode requires; no session is bound by the test itself.
    """
    from src.api.routes import chatbot_graph
    from src.api.routes.chatbot_state import ChatbotState

    workflow = StateGraph(ChatbotState)
    workflow.add_node("tools", chatbot_graph.e2i_chatbot_graph.nodes["tools"].bound)
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    graph = workflow.compile()
    before = _session_id_context.get()

    await graph.ainvoke(
        {
            "session_id": COMPOSITE_SESSION,
            "messages": [_tool_call_message("orchestrator_tool", {"query": "q"})],
        }
    )

    session = orchestrator.payloads[0]["session_id"]
    assert session == COMPOSITE_SESSION, session
    assert not FABRICATED.match(session or "")
    # The binding is scoped to the node: nothing leaks into the caller's context.
    assert _session_id_context.get() == before
