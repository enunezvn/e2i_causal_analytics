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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.api.routes.chat_bridge import (
    BRIDGE_PREAMBLE,
    BRIDGE_PREAMBLE_UNGROUNDED,
    BridgeAnswer,
    _prepare_bridge_messages,
    build_bridge_preamble,
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
                new=AsyncMock(
                    return_value=BridgeAnswer("TRx conversion for Kisqali is 48.3%.", True)
                ),
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
            answer = await run_conversational_bridge(query="q", session_id="u~s", history=None)

        assert answer.text == "grounded answer"
        # Fresh instance per call: the module singleton's MemorySaver must not
        # accumulate bridged turns in a long-lived API process.
        factory.assert_called_once()
        state, config = graph.calls[0]
        assert config["configurable"]["thread_id"] == "bridge~u~s"
        # Shadow session id — see TestBridgePersistenceIsolation
        assert state["session_id"] == "u~s~bridge"

    async def test_list_content_normalized(self):
        # sonnet-5 AIMessage.content can be a block list (#1350 class)
        graph = _FakeGraph(
            final_state={
                "messages": [AIMessage(content=[{"type": "text", "text": "block answer"}])]
            }
        )
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            answer = await run_conversational_bridge(query="q", session_id="u~s")

        assert answer.text == "block answer"

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


class TestBridgeNeverRaises:
    """Codex iter-1 HIGH: pre-try failures must not escape — an escaped
    exception is swallowed by orchestrator_node's broad except and the turn
    falls through to generate_node instead of keeping the fail-closed
    summary (a behavior change, not fail-open-to-status-quo)."""

    async def test_import_surface_failure_returns_none(self):
        with patch(
            "src.api.routes.copilotkit.create_e2i_chat_agent",
            side_effect=ImportError("copilotkit surface shifted"),
        ):
            text = await run_conversational_bridge(query="q", session_id="u~s")

        assert text is None

    async def test_bad_timeout_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("E2I_CHAT_BRIDGE_TIMEOUT_S", "not-a-number")
        graph = _FakeGraph(final_state={"messages": [AIMessage(content="answer")]})
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            answer = await run_conversational_bridge(query="q", session_id="u~s")

        assert answer.text == "answer"


class TestBridgePersistenceIsolation:
    """chat_node persists messages via _persist_message_sync keyed on the
    session ctxvar (observed in a REAL local bridge run: '[CopilotKit]
    Persisted user message'). Without isolation a bridged turn double-writes
    the real session: chat_node's raw answer + finalize's preambled answer.
    The bridge must run under a shadow session id — same user prefix (the
    computed_user_id column splits on the first '~') but distinct from the
    real session so UI history loads stay clean."""

    async def test_bridge_runs_under_shadow_session(self):
        from src.api.routes.copilotkit import _session_id_context

        observed = {}

        class _CtxCapturingGraph(_FakeGraph):
            async def ainvoke(self, state, config=None):
                observed["ctxvar"] = _session_id_context.get()
                return await super().ainvoke(state, config)

        graph = _CtxCapturingGraph(final_state={"messages": [AIMessage(content="answer")]})
        before = _session_id_context.get()
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            answer = await run_conversational_bridge(query="q", session_id="u~s")

        assert answer.text == "answer"
        assert observed["ctxvar"] == "u~s~bridge"
        state, _config = graph.calls[0]
        assert state["session_id"] == "u~s~bridge"
        # user prefix preserved for attribution, real session id not reused
        assert observed["ctxvar"].split("~")[0] == "u"
        # ctxvar restored after the call
        assert _session_id_context.get() == before


# ---------------------------------------------------------------------------
# #1451 — a rescued turn must READ as an answer, not as an apology
# ---------------------------------------------------------------------------
#
# Measured 2026-08-03: A.5 returned the correct grounded TRx = 12,867, 5.7 a
# correct refutation answer and 4.3 a correct scoping answer — every one of
# them opened with "The full analysis pipeline couldn't complete...". The
# preamble described the INTERNAL pipeline's outcome in the pipeline's own
# terms and buried a good answer under an apology.
#
# What must NOT regress (#883 / #1336 honesty): the preamble still has to say
# the deeper multi-agent analysis did not run, and it must not assert live
# platform data for a turn where no tool actually executed.

_CAUSAL_USER_ACTION = (
    "To run the full causal analysis, name a treatment and an outcome "
    "(plus any confounders) — candidates from the causal knowledge graph "
    "are treatments: rep_visits, samples; outcomes: trx, nrx."
)
_EXPLAINER_USER_ACTION = (
    "Run an analysis first (a causal, gap or segmentation question), then ask me to explain it."
)

_FAILURE_DETAILS_WITH_ACTION = [
    {
        "agent_name": "causal_impact",
        "error": "causal_impact needs structured inputs that could not be grounded",
        "latency_ms": 12,
        "user_action": _CAUSAL_USER_ACTION,
    },
    {
        "agent_name": "explainer",
        "error": "explainer needs structured inputs that could not be grounded",
        "latency_ms": 3,
        "user_action": _EXPLAINER_USER_ACTION,
    },
]

_ORCH_FAILED_WITH_ACTION = {
    **_ORCH_FAILED,
    "failure_details": _FAILURE_DETAILS_WITH_ACTION,
}


class TestBridgePreambleReadsAsAnAnswer:
    """The preamble leads with what the answer IS, then discloses what did not run."""

    def test_preamble_leads_with_provenance_not_with_the_pipeline_failure(self):
        low = BRIDGE_PREAMBLE.lower()
        # what the answer IS comes first...
        assert "live platform data" in low
        # ...and the disclosure of what did not run comes after it
        assert low.index("live platform data") < low.index("did not run")
        # the old apology opener is gone
        assert "couldn't complete" not in low
        assert not low.startswith("the full analysis pipeline")

    def test_preamble_still_discloses_that_the_full_analysis_did_not_run(self):
        low = BRIDGE_PREAMBLE.lower()
        assert "multi-agent analysis" in low
        assert "did not run" in low

    def test_ungrounded_preamble_never_claims_live_platform_data(self):
        # A bridged turn where no tool executed (e.g. the measured 4.3 scoping
        # answer) must not be dressed up as a data lookup — that would be a
        # fabricated provenance claim on a user-facing surface.
        low = BRIDGE_PREAMBLE_UNGROUNDED.lower()
        assert "live platform data" not in low
        assert "multi-agent analysis" in low
        assert "did not run" in low

    def test_neither_preamble_implies_the_full_analysis_ran(self):
        for preamble in (BRIDGE_PREAMBLE, BRIDGE_PREAMBLE_UNGROUNDED):
            low = preamble.lower()
            assert "multi-agent analysis did not run" in low


class TestBridgePreambleCarriesTheActionableInvitation:
    """The dispatcher's fail-closed message names exactly what is missing —
    surface it instead of discarding it for a generic apology."""

    def test_first_available_user_action_is_appended(self):
        text = build_bridge_preamble(
            tool_grounded=True, failure_details=_FAILURE_DETAILS_WITH_ACTION
        )
        assert text.startswith(BRIDGE_PREAMBLE)
        assert _CAUSAL_USER_ACTION in text
        # the PRIMARY failed agent's invitation only — two invitations for one
        # turn would contradict each other
        assert _EXPLAINER_USER_ACTION not in text

    def test_no_user_action_leaves_the_bare_preamble(self):
        assert build_bridge_preamble(tool_grounded=True, failure_details=None) == BRIDGE_PREAMBLE
        assert (
            build_bridge_preamble(
                tool_grounded=True,
                failure_details=[{"agent_name": "causal_impact", "error": "boom"}],
            )
            == BRIDGE_PREAMBLE
        )
        assert (
            build_bridge_preamble(tool_grounded=False, failure_details=None)
            == BRIDGE_PREAMBLE_UNGROUNDED
        )

    def test_malformed_failure_details_are_tolerated(self):
        # failure_details is orchestrator-supplied; a shape change must not
        # crash the rescue path (fail open to the bare preamble).
        assert (
            build_bridge_preamble(tool_grounded=True, failure_details=["not-a-dict", None, {}])
            == BRIDGE_PREAMBLE
        )


class TestOrchestratorNodeSurfacesTheInvitation:
    async def test_rescued_turn_leads_with_provenance_and_invites_the_scoped_run(self):
        with (
            patch(
                "src.api.routes.chatbot_graph.get_orchestrator",
                return_value=_mock_orchestrator(_ORCH_FAILED_WITH_ACTION),
            ),
            patch(
                "src.api.routes.chatbot_graph.run_conversational_bridge",
                new=AsyncMock(return_value=BridgeAnswer("Kisqali TRx = 12,867.", True)),
            ),
        ):
            result = await orchestrator_node(_state())

        text = result["response_text"]
        assert text.startswith(BRIDGE_PREAMBLE)
        assert _CAUSAL_USER_ACTION in text
        assert "Kisqali TRx = 12,867." in text
        # provenance before the disclosure, both before the answer
        low = text.lower()
        assert low.index("live platform data") < low.index("did not run")
        assert low.index("did not run") < low.index("kisqali trx = 12,867.")
        # honesty instrument untouched
        assert result["metadata"]["bridge_used"] is True
        assert result["metadata"]["orchestrator_status"] == "failed"

    async def test_answer_without_a_tool_call_gets_the_ungrounded_preamble(self):
        with (
            patch(
                "src.api.routes.chatbot_graph.get_orchestrator",
                return_value=_mock_orchestrator(_ORCH_FAILED_WITH_ACTION),
            ),
            patch(
                "src.api.routes.chatbot_graph.run_conversational_bridge",
                new=AsyncMock(return_value=BridgeAnswer("Here is how I would scope that.", False)),
            ),
        ):
            result = await orchestrator_node(_state())

        text = result["response_text"]
        assert text.startswith(BRIDGE_PREAMBLE_UNGROUNDED)
        assert "live platform data" not in text.lower()
        assert "Here is how I would scope that." in text


class TestBridgeReportsToolGrounding:
    """``tool_grounded`` is real evidence (an executed ToolMessage), not a guess."""

    async def test_tool_message_makes_the_answer_tool_grounded(self):
        graph = _FakeGraph(
            final_state={
                "messages": [
                    HumanMessage(content="q"),
                    AIMessage(content=""),
                    ToolMessage(content="12867", tool_call_id="call-1"),
                    AIMessage(content="Kisqali TRx = 12,867."),
                ]
            }
        )
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            answer = await run_conversational_bridge(query="q", session_id="u~s")

        assert answer.text == "Kisqali TRx = 12,867."
        assert answer.tool_grounded is True

    async def test_no_tool_message_is_not_tool_grounded(self):
        graph = _FakeGraph(
            final_state={
                "messages": [
                    HumanMessage(content="q"),
                    AIMessage(content="Here is how I would scope that."),
                ]
            }
        )
        with patch("src.api.routes.copilotkit.create_e2i_chat_agent", return_value=graph):
            answer = await run_conversational_bridge(query="q", session_id="u~s")

        assert answer.tool_grounded is False


class TestActionableInvitationPlumbing:
    """The invitation text must actually reach the bridge call site — the
    dispatcher authors it, ``_build_output`` carries it in failure_details."""

    def test_causal_impact_fail_closed_authors_a_user_facing_invitation(self):
        from src.agents.orchestrator.nodes.dispatcher import (
            NeedsStructuredInput,
            _resolve_causal_impact_input,
        )

        needs = _resolve_causal_impact_input({"query": "why did it drop?"}, {"parameters": {}})

        assert isinstance(needs, NeedsStructuredInput)
        assert needs.user_action
        low = needs.user_action.lower()
        assert "treatment" in low and "outcome" in low
        # user-facing: no pipeline jargon leaks into the invitation
        for jargon in ("fail", "structured inputs", "substrate", "fabricat"):
            assert jargon not in low

    def test_explainer_fail_closed_authors_a_user_facing_invitation(self):
        from src.agents.orchestrator.nodes.dispatcher import (
            NeedsStructuredInput,
            _resolve_explainer_input,
        )

        needs = _resolve_explainer_input({"query": "explain that"}, {"parameters": {}})

        assert isinstance(needs, NeedsStructuredInput)
        assert needs.user_action
        assert "analysis" in needs.user_action.lower()

    def test_build_output_carries_user_action_into_failure_details(self):
        from src.agents.orchestrator.agent import OrchestratorAgent

        orchestrator = OrchestratorAgent(agent_registry={}, enable_checkpointing=False)
        output = orchestrator._build_output(
            {
                "query_id": "q-1451",
                "status": "failed",
                "agent_results": [
                    {
                        "agent_name": "causal_impact",
                        "success": False,
                        "error": "fail-closed",
                        "latency_ms": 12,
                        "user_action": _CAUSAL_USER_ACTION,
                    }
                ],
            }
        )

        assert output["failure_details"][0]["user_action"] == _CAUSAL_USER_ACTION
