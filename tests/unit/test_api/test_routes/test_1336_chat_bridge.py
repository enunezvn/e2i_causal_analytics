"""Red-first tests for the #1336 D5 conversational bridge on /chat/stream.

Decision (owner-locked): BRIDGE. When the orchestrator fails completely
(zero successful agents — the case where /chat/stream today streams the
fail-closed "I was unable to complete the analysis" summary), route the turn
through the AG-UI chat brain (chat_node + tools, the surface that answers the
same questions with real grounded data) and return its answer behind an
honest preamble.

Contract pins:
- Bridge fires ONLY on complete failure (status == "failed"). Partial
  successes and successes keep today's behavior byte-for-byte.
- Bridge failure/disabled/timeout falls back to the original fail-closed
  summary — the bridge can only improve on the status quo, never mask it.
- The routing instrument loses nothing: original routed_agent, failed_agents,
  failure_details and orchestrator_status stay in metadata; bridge use is
  marked explicitly (#883 honesty discipline).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.api.routes.chat_bridge import (
    BRIDGE_PREAMBLE,
    _prepare_bridge_messages,
    run_conversational_bridge,
)
from src.api.routes.chatbot_graph import orchestrator_node

_FAIL_CLOSED_TEXT = (
    "I was unable to complete the analysis due to the following errors:\n"
    "- causal_impact: causal_impact could not produce a real result"
)

_ORCH_FAILED = {
    "response_text": _FAIL_CLOSED_TEXT,
    "response_confidence": 0.0,
    "agents_dispatched": ["causal_impact", "explainer"],
    "successful_agents": [],
    "failed_agents": ["causal_impact", "explainer"],
    "failure_details": [{"agent": "causal_impact", "error": "no substrate"}],
    "has_partial_failure": False,
    "status": "failed",
    "routing_pattern": "SINGLE_AGENT",
    "used_llm_layer": False,
}

_ORCH_PARTIAL = {
    **_ORCH_FAILED,
    "response_text": "Gap analysis found 2 gaps.",
    "successful_agents": ["gap_analyzer"],
    "failed_agents": ["causal_impact"],
    "has_partial_failure": True,
    "status": "partial_success",
}

_ORCH_OK = {
    **_ORCH_FAILED,
    "response_text": "Causal analysis complete.",
    "successful_agents": ["causal_impact"],
    "failed_agents": [],
    "failure_details": [],
    "status": "completed",
}


def _state(intent="causal_analysis"):
    return {
        "intent": intent,
        "query": "What is the causal impact of rep visits on Kisqali conversion?",
        "session_id": "user-1~sess-1",
        "user_id": "user-1",
        "rag_context": [],
        "progress_steps": [],
        "metadata": {},
        "messages": [HumanMessage(content="What is the causal impact?")],
    }


def _mock_orchestrator(result):
    orch = MagicMock()
    orch.run = AsyncMock(return_value=result)
    return orch


class TestOrchestratorNodeBridge:
    """orchestrator_node engages the bridge only on complete failure."""

    async def test_bridge_engages_on_complete_failure(self):
        with (
            patch(
                "src.api.routes.chatbot_graph.get_orchestrator",
                return_value=_mock_orchestrator(_ORCH_FAILED),
            ),
            patch(
                "src.api.routes.chatbot_graph.run_conversational_bridge",
                new=AsyncMock(return_value="TRx conversion for Kisqali is 48.3%."),
            ) as bridge,
        ):
            result = await orchestrator_node(_state())

        bridge.assert_awaited_once()
        assert result["response_text"].startswith(BRIDGE_PREAMBLE)
        assert "TRx conversion for Kisqali is 48.3%." in result["response_text"]
        # The streamed message must carry the bridged text, not the error summary
        assert result["messages"][0].content == result["response_text"]
        # Honesty + routing instrument intact
        assert result["metadata"]["bridge_used"] is True
        assert result["metadata"]["orchestrator_status"] == "failed"
        assert result["metadata"]["failed_agents"] == ["causal_impact", "explainer"]
        assert result["agent_name"] == "chat_bridge"
        assert result["routed_agent"] == "causal_impact"

    async def test_bridge_skipped_on_partial_success(self):
        with (
            patch(
                "src.api.routes.chatbot_graph.get_orchestrator",
                return_value=_mock_orchestrator(_ORCH_PARTIAL),
            ),
            patch(
                "src.api.routes.chatbot_graph.run_conversational_bridge",
                new=AsyncMock(return_value="should not be used"),
            ) as bridge,
        ):
            result = await orchestrator_node(_state())

        bridge.assert_not_awaited()
        assert result["response_text"].startswith("Gap analysis found 2 gaps.")
        assert result["metadata"]["bridge_used"] is False

    async def test_bridge_skipped_on_success(self):
        with (
            patch(
                "src.api.routes.chatbot_graph.get_orchestrator",
                return_value=_mock_orchestrator(_ORCH_OK),
            ),
            patch(
                "src.api.routes.chatbot_graph.run_conversational_bridge",
                new=AsyncMock(return_value="should not be used"),
            ) as bridge,
        ):
            result = await orchestrator_node(_state())

        bridge.assert_not_awaited()
        assert result["response_text"] == "Causal analysis complete."

    async def test_bridge_none_falls_back_to_fail_closed(self):
        with (
            patch(
                "src.api.routes.chatbot_graph.get_orchestrator",
                return_value=_mock_orchestrator(_ORCH_FAILED),
            ),
            patch(
                "src.api.routes.chatbot_graph.run_conversational_bridge",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await orchestrator_node(_state())

        assert result["response_text"] == _FAIL_CLOSED_TEXT
        assert result["metadata"]["bridge_used"] is False
        assert result["agent_name"] == "causal_impact"


class _FakeGraph:
    """Stand-in for the compiled AG-UI graph (plain object, no MagicMock —
    MagicMock fakes hasattr and can mask attribute-shape bugs)."""

    def __init__(self, final_state=None, delay_s=0.0, exc=None):
        self.final_state = final_state or {}
        self.delay_s = delay_s
        self.exc = exc
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.exc:
            raise self.exc
        return self.final_state


class TestRunConversationalBridge:
    async def test_returns_last_aimessage_text(self):
        graph = _FakeGraph(
            final_state={
                "messages": [
                    HumanMessage(content="q"),
                    AIMessage(content="grounded answer"),
                ]
            }
        )
        with patch(
            "src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph
        ) as factory:
            text = await run_conversational_bridge(query="q", session_id="u~s", history=None)

        assert text == "grounded answer"
        # Fresh instance per call: the module singleton's MemorySaver must not
        # accumulate bridged turns in a long-lived API process.
        factory.assert_called_once()
        state, config = graph.calls[0]
        assert config["configurable"]["thread_id"] == "bridge~u~s"
        assert state["session_id"] == "u~s"

    async def test_list_content_normalized(self):
        # sonnet-5 AIMessage.content can be a block list (#1350 class)
        graph = _FakeGraph(
            final_state={
                "messages": [AIMessage(content=[{"type": "text", "text": "block answer"}])]
            }
        )
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            text = await run_conversational_bridge(query="q", session_id="u~s")

        assert text == "block answer"

    async def test_timeout_returns_none(self):
        graph = _FakeGraph(final_state={"messages": [AIMessage(content="late")]}, delay_s=0.5)
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            text = await run_conversational_bridge(query="q", session_id="u~s", timeout_s=0.05)

        assert text is None

    async def test_exception_returns_none(self):
        graph = _FakeGraph(exc=RuntimeError("provider down"))
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            text = await run_conversational_bridge(query="q", session_id="u~s")

        assert text is None

    async def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("E2I_CHAT_BRIDGE_ENABLED", "false")
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent") as factory:
            text = await run_conversational_bridge(query="q", session_id="u~s")

        assert text is None
        factory.assert_not_called()

    async def test_no_aimessage_returns_none(self):
        graph = _FakeGraph(final_state={"messages": [HumanMessage(content="q")]})
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            text = await run_conversational_bridge(query="q", session_id="u~s")

        assert text is None


class TestPrepareBridgeMessages:
    def test_empty_history_becomes_query(self):
        msgs = _prepare_bridge_messages("the query", None)
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "the query"

    def test_history_not_ending_in_human_gets_query_appended(self):
        history = [HumanMessage(content="earlier"), AIMessage(content="answer")]
        msgs = _prepare_bridge_messages("the query", history)
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].content == "the query"
        assert len(msgs) == 3

    def test_history_ending_in_human_kept_as_is(self):
        history = [AIMessage(content="a"), HumanMessage(content="the query")]
        msgs = _prepare_bridge_messages("the query", history)
        assert msgs == history

    def test_history_capped(self):
        history = [HumanMessage(content=f"m{i}") for i in range(20)]
        msgs = _prepare_bridge_messages("q", history)
        assert len(msgs) <= 8
        assert msgs[-1].content == "m19"
