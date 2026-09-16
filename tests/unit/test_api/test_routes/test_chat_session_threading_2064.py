"""#2064: chat tools use the real chat session instead of inventing one.

``orchestrator_tool`` and ``tool_composer_tool`` are invoked by LangGraph's
ToolNode with the model's arguments only. The model cannot know the frontend
thread id, so the tools used to fill the gap with ``chatbot-<timestamp>`` /
``composer-<timestamp>`` / ``fallback-<timestamp>`` — ids that look like a
session but belong to no conversation. Once #2069 threaded the orchestrator's
session into ``composer_episodes``, those values would have made unattributable
compositions look attributed.

The real session is bound by each chat brain's ``tools`` node
(``SessionBoundToolNode``), from ``state["session_id"]``: AG-UI's ``execute()``
also sets ``_session_id_context``, which the handler's keepalive wrapper used to
drop by pulling each frame in a fresh task (repaired by #2100, which is why
these tests assert the VALUE rather than which channel delivered it).

The AG-UI and ``/chat/stream`` graph tests drive the real graph entry points
(``execute()`` under ``with_sse_keepalive``, or the compiled graph's own
``tools`` node) and bind no session themselves, except to plant a stale outer
binding. The resolver unit tests and the chat-bridge model test set the var
directly. Only the orchestrator and the composer behind the tools are faked.
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
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
    """LangGraph's ToolNode, run by a compiled graph, sees a binding made before the run.

    This models the chat bridge (``chat_bridge.py:~206``), which sets the var and
    calls ``graph.ainvoke`` with no keepalive wrapper. It does not model AG-UI,
    whose binding is made inside the wrapper (see the AG-UI tests below).
    """
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
        brand=None, region=None, session_id=None, user_id=None, max_parallel=3
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


# ------------------------------------------------------------ AG-UI brain


class _ScriptedChatModel:
    """The chat leg asks for ``orchestrator_tool``; the synthesis leg answers in text."""

    def __init__(self, calls_tool: bool = False) -> None:
        self._calls_tool = calls_tool

    def bind_tools(self, tools, **kwargs) -> _ScriptedChatModel:
        return _ScriptedChatModel(calls_tool=True)

    async def astream(self, messages, *args, **kwargs):
        if self._calls_tool:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "orchestrator_tool",
                        "args": json.dumps({"query": "system health score"}),
                        "id": "call-2064",
                        "index": 0,
                    }
                ],
            )
        else:
            yield AIMessageChunk(content="The system health score is available.")


async def test_an_agui_turn_gives_its_tools_the_thread_session(orchestrator, monkeypatch):
    """The browser route end to end: ``execute()`` under the handler's keepalive wrapper.

    The handler streams ``with_sse_keepalive(...)`` around ``execute()``. It used
    to pull every frame in a fresh task, so the session ``execute()`` binds before
    its first frame never reached the graph's nodes (production: ``chat_node …
    (source=state)``, and ``classification_logs.session_id`` NULL) — #2100
    repaired that channel. This asserts the VALUE the tool sees, not which
    channel delivered it, so it holds either way: the tools node binds the thread
    session from graph state itself.
    """
    from src.api.routes import copilotkit
    from src.api.utils.sse_keepalive import with_sse_keepalive

    monkeypatch.setattr(copilotkit, "get_chat_llm", lambda **kwargs: _ScriptedChatModel())
    monkeypatch.setattr("src.api.dependencies.supabase_client.get_supabase", lambda: None)

    async def _no_learning_signal(**kwargs) -> None:
        return None

    monkeypatch.setattr(copilotkit, "_collect_copilot_learning_signal", _no_learning_signal)
    agent = copilotkit.LangGraphAgent(
        name="default",
        description="#2064",
        graph=copilotkit.e2i_chat_graph,
        graph_factory=copilotkit.create_e2i_chat_agent,
    )

    # The thread id execute() is given IS the session the tool must see (execute()
    # swaps in its own fresh checkpoint key), so one per-test uuid serves as both.
    session = str(uuid4())
    frames = with_sse_keepalive(
        agent.execute(
            thread_id=session,
            state={},
            messages=[{"id": "u-2064", "role": "user", "content": "What is the health score?"}],
            actions=[],
        )
    )
    async for _ in frames:
        pass

    assert len(orchestrator.payloads) == 1, "orchestrator_tool never ran"
    assert orchestrator.payloads[0]["session_id"] == session


def _agui_tools_graph():
    """The compiled AG-UI graph's own ``tools`` node, mounted alone in its own state."""
    from src.api.routes import copilotkit

    workflow = StateGraph(copilotkit.E2IAgentState)
    workflow.add_node("tools", copilotkit.e2i_chat_graph.nodes["tools"].bound)
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    return workflow.compile()


async def test_the_agui_tools_node_binds_nothing_without_a_state_session(orchestrator):
    """No session in state and none bound outside: none invented."""
    await _agui_tools_graph().ainvoke(
        {"messages": [_tool_call_message("orchestrator_tool", {"query": "q"})]}
    )

    assert orchestrator.payloads[0]["session_id"] is None


async def test_the_agui_tools_node_prefers_the_state_session_over_an_outer_binding(
    orchestrator,
):
    """State wins over a stale binding from outside the graph, which is restored after."""
    token = _session_id_context.set("stale")
    try:
        await _agui_tools_graph().ainvoke(
            {
                "session_id": REAL_SESSION,
                "messages": [_tool_call_message("orchestrator_tool", {"query": "q"})],
            }
        )
        after = _session_id_context.get()
    finally:
        _session_id_context.reset(token)

    assert orchestrator.payloads[0]["session_id"] == REAL_SESSION
    assert after == "stale"
