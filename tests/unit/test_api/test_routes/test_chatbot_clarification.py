"""Unit tests for the stateful multi-turn clarification feature (#1407).

Covers the ``/chat/stream`` LangGraph clarification path:

- ``_detect_missing_slots`` on the verified disproof set (CLARIFY vs NO-clarify)
- ``route_after_classify`` (flag on/off, needs_clarification true/false)
- ``_is_slot_like`` / ``_classify_pending_followup`` / ``_pending_is_expired``
- ``clarify_node`` (LLM ask-back, LLM-unavailable canned fallback, DB error)
- ``load_context_node`` RESUME (answer merges+clears / pivot clears-not-resumes /
  expired clears)
- ``classify_intent_node`` (underspecified -> needs_clarification; resumed turn
  HARD-suppresses re-detection; classifies on the merged query)
- graph-level: an underspecified turn visits ``clarify`` NOT ``orchestrator``,
  and ``run_conversational_bridge`` is NEVER called (#883 fail-closed bridge
  lives in the skipped orchestrator_node).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.api.routes.chatbot_graph as g
from src.api.routes.chatbot_graph import (
    _classify_pending_followup,
    _detect_missing_slots,
    _generate_fallback_response,
    _is_slot_like,
    _pending_is_expired,
    clarify_node,
    classify_intent_node,
    load_context_node,
    route_after_classify,
)
from src.api.routes.chatbot_state import IntentType, create_initial_state

# =============================================================================
# _detect_missing_slots — the VERIFIED disproof set (real extractors)
# =============================================================================

# These queries are underspecified analytical asks (no brand AND no metric AND
# no prior referent) -> BOTH slots missing -> clarify.
CLARIFY_QUERIES = ["Why did it drop?", "causal analysis", "run both"]

# These carry a real brand and/or a recognizable metric -> at least one slot
# present -> NO clarify.
NO_CLARIFY_QUERIES = [
    "What is the TRx for Kisqali?",
    "How is Kisqali doing?",  # brand present
    "show me NRx",  # metric present
    "What's the conversion rate for Fabhalta?",
    "Did the Kisqali speaker program increase TRx?",
    "Kisqali TRx",
]


class TestDetectMissingSlots:
    """`_detect_missing_slots` on the disproof set + gating conditions."""

    @pytest.mark.parametrize("query", CLARIFY_QUERIES)
    def test_underspecified_causal_returns_both_slots(self, query):
        assert _detect_missing_slots(
            query, IntentType.CAUSAL_ANALYSIS, brand_context="", has_prior_referent=False
        ) == ["brand", "metric"]

    @pytest.mark.parametrize("query", CLARIFY_QUERIES)
    def test_underspecified_kpi_returns_both_slots(self, query):
        assert _detect_missing_slots(
            query, IntentType.KPI_QUERY, brand_context="", has_prior_referent=False
        ) == ["brand", "metric"]

    @pytest.mark.parametrize("query", NO_CLARIFY_QUERIES)
    def test_specified_queries_return_no_missing(self, query):
        assert (
            _detect_missing_slots(
                query, IntentType.KPI_QUERY, brand_context="", has_prior_referent=False
            )
            == []
        )

    def test_non_clarify_intent_never_clarifies(self):
        # greeting/help/agent_status/general/search/recommendation are out of scope
        for intent in (
            IntentType.GREETING,
            IntentType.HELP,
            IntentType.AGENT_STATUS,
            IntentType.GENERAL,
            IntentType.SEARCH,
            IntentType.RECOMMENDATION,
        ):
            assert _detect_missing_slots("why did it drop?", intent, "", False) == []

    def test_prior_referent_suppresses(self):
        assert (
            _detect_missing_slots(
                "why did it drop?", IntentType.CAUSAL_ANALYSIS, "", has_prior_referent=True
            )
            == []
        )

    def test_brand_context_fills_brand_slot(self):
        # An underspecified query text but a real brand carried in context -> no clarify
        assert (
            _detect_missing_slots(
                "why did it drop?", IntentType.CAUSAL_ANALYSIS, "Kisqali", has_prior_referent=False
            )
            == []
        )


# =============================================================================
# route_after_classify
# =============================================================================


class TestRouteAfterClassify:
    def test_routes_to_clarify_when_needed(self):
        state = create_initial_state("u", "why did it drop?", "r")
        state["needs_clarification"] = True
        with patch.object(g, "CHATBOT_CLARIFY_ENABLED", True):
            assert route_after_classify(state) == "clarify"

    def test_routes_to_rag_when_not_needed(self):
        state = create_initial_state("u", "what is TRx for Kisqali?", "r")
        state["needs_clarification"] = False
        with patch.object(g, "CHATBOT_CLARIFY_ENABLED", True):
            assert route_after_classify(state) == "retrieve_rag"

    def test_flag_off_forces_rag(self):
        state = create_initial_state("u", "why did it drop?", "r")
        state["needs_clarification"] = True
        with patch.object(g, "CHATBOT_CLARIFY_ENABLED", False):
            assert route_after_classify(state) == "retrieve_rag"


# =============================================================================
# _is_slot_like / _classify_pending_followup / _pending_is_expired
# =============================================================================


class TestIsSlotLike:
    @pytest.mark.parametrize(
        "text", ["Kisqali", "the northeast region", "last quarter", "Kisqali TRx"]
    )
    def test_slot_like_fragments(self, text):
        assert _is_slot_like(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What is the TRx?",  # question mark
            "Show me TRx",  # directive starter
            "why did the whole thing fall apart across every single region",  # > 6 words
        ],
    )
    def test_not_slot_like(self, text):
        assert _is_slot_like(text) is False


class TestClassifyPendingFollowup:
    def test_slot_supplying_fragment_is_answer(self):
        assert _classify_pending_followup("Kisqali", IntentType.CAUSAL_ANALYSIS) == "answer"

    def test_context_fragment_is_answer(self):
        # supplies neither brand nor metric but is a short slot-like fragment
        assert (
            _classify_pending_followup("the northeast region", IntentType.CAUSAL_ANALYSIS)
            == "answer"
        )

    def test_self_sufficient_question_is_pivot(self):
        assert (
            _classify_pending_followup("What is the TRx for Kisqali?", IntentType.CAUSAL_ANALYSIS)
            == "pivot"
        )

    def test_fresh_unrelated_question_is_pivot(self):
        assert (
            _classify_pending_followup("What is the weather today?", IntentType.CAUSAL_ANALYSIS)
            == "pivot"
        )


class TestPendingIsExpired:
    def test_recent_is_not_expired(self):
        pending = {"asked_at": datetime.now(timezone.utc).isoformat()}
        assert _pending_is_expired(pending) is False

    def test_old_is_expired(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=999)).isoformat()
        assert _pending_is_expired({"asked_at": old}) is True

    def test_missing_asked_at_is_expired(self):
        assert _pending_is_expired({}) is True

    def test_malformed_asked_at_is_expired(self):
        assert _pending_is_expired({"asked_at": "not-a-date"}) is True


# =============================================================================
# clarify_node
# =============================================================================


def _clarify_state():
    state = create_initial_state(
        user_id="u1", query="why did it drop?", request_id="r1", session_id="u1~s1"
    )
    state["intent"] = IntentType.CAUSAL_ANALYSIS
    state["needs_clarification"] = True
    state["missing_slots"] = ["brand", "metric"]
    return state


class TestClarifyNode:
    @pytest.mark.asyncio
    async def test_llm_produces_ask_back_and_persists_pending(self):
        state = _clarify_state()
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content='{"clarifying_questions": ["Which brand?", "Which metric?"], '
                '"assumed_interpretation": "x", "confidence_if_assumed": 0.6}'
            )
        )
        mock_conv_repo = AsyncMock()

        with patch.object(g, "get_chat_llm", return_value=llm):
            with patch.object(g, "get_async_supabase_client", AsyncMock(return_value=MagicMock())):
                with patch.object(
                    g, "get_chatbot_conversation_repository", return_value=mock_conv_repo
                ):
                    result = await clarify_node(state)

        assert result["agent_name"] == "clarifier"
        assert result["needs_clarification"] is True
        assert result["clarifying_questions"] == ["Which brand?", "Which metric?"]
        assert "Which brand?" in result["response_text"]
        assert "Which metric?" in result["response_text"]
        # last message is the ask-back
        assert isinstance(result["messages"][-1], AIMessage)
        # pending clarification persisted with the right schema
        mock_conv_repo.update_metadata.assert_awaited_once()
        call = mock_conv_repo.update_metadata.await_args
        assert call.args[0] == "u1~s1"
        pending = call.args[1]["pending_clarification"]
        assert pending["original_query"] == "why did it drop?"
        assert pending["missing_slots"] == ["brand", "metric"]
        assert pending["clarifying_questions"] == ["Which brand?", "Which metric?"]
        assert "asked_at" in pending

    @pytest.mark.asyncio
    async def test_llm_unavailable_falls_back_to_canned(self):
        state = _clarify_state()
        canned = _generate_fallback_response(state)["response_text"]
        mock_conv_repo = AsyncMock()

        with patch.object(g, "get_chat_llm", side_effect=RuntimeError("no llm")):
            with patch.object(g, "get_async_supabase_client", AsyncMock(return_value=MagicMock())):
                with patch.object(
                    g, "get_chatbot_conversation_repository", return_value=mock_conv_repo
                ):
                    result = await clarify_node(state)

        # canned per-intent fallback (CAUSAL) — no fabricated analytical content
        assert result["response_text"] == canned
        assert result["clarifying_questions"] == [canned]
        assert result["agent_name"] == "clarifier"
        mock_conv_repo.update_metadata.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_db_error_still_returns_ask_back(self):
        state = _clarify_state()
        mock_conv_repo = AsyncMock()
        mock_conv_repo.update_metadata.side_effect = RuntimeError("db down")

        with patch.object(g, "get_chat_llm", side_effect=RuntimeError("no llm")):
            with patch.object(g, "get_async_supabase_client", AsyncMock(return_value=MagicMock())):
                with patch.object(
                    g, "get_chatbot_conversation_repository", return_value=mock_conv_repo
                ):
                    result = await clarify_node(state)

        # DB failed but we still return an honest ask-back
        assert result["agent_name"] == "clarifier"
        assert result["response_text"]
        assert result["needs_clarification"] is True


# =============================================================================
# load_context_node RESUME logic (mocked DB)
# =============================================================================


def _conv_with_pending(original="why did it drop?", asked_at=None, missing=None):
    asked_at = asked_at or datetime.now(timezone.utc).isoformat()
    return {
        "title": "t",
        "brand_context": None,
        "region_context": None,
        "metadata": {
            "pending_clarification": {
                "original_query": original,
                "intent": IntentType.CAUSAL_ANALYSIS,
                "missing_slots": missing or ["brand", "metric"],
                "clarifying_questions": ["Which brand and metric?"],
                "asked_at": asked_at,
            }
        },
    }


async def _run_load_context(current_query, conv):
    state = create_initial_state(
        user_id="u1", query=current_query, request_id="r2", session_id="u1~s1"
    )
    mock_client = MagicMock()
    mock_msg_repo = AsyncMock()
    mock_msg_repo.get_recent_messages.return_value = []
    mock_conv_repo = AsyncMock()
    mock_conv_repo.get_by_session_id.return_value = conv

    with patch.object(g, "get_async_supabase_client", AsyncMock(return_value=mock_client)):
        with patch.object(g, "get_chatbot_message_repository", return_value=mock_msg_repo):
            with patch.object(
                g, "get_chatbot_conversation_repository", return_value=mock_conv_repo
            ):
                result = await load_context_node(state)
    return result, mock_conv_repo


class TestLoadContextResume:
    @pytest.mark.asyncio
    async def test_answer_merges_and_clears(self):
        result, repo = await _run_load_context("Kisqali", _conv_with_pending())
        assert result["resumed_from_clarification"] is True
        assert result["merged_query"] == "why did it drop? Kisqali"
        # pending cleared (None pops the key)
        repo.update_metadata.assert_awaited_once_with("u1~s1", {"pending_clarification": None})

    @pytest.mark.asyncio
    async def test_pivot_clears_without_resume(self):
        result, repo = await _run_load_context("What is the TRx for Kisqali?", _conv_with_pending())
        assert result.get("resumed_from_clarification") is False
        assert result.get("merged_query") is None
        repo.update_metadata.assert_awaited_once_with("u1~s1", {"pending_clarification": None})

    @pytest.mark.asyncio
    async def test_expired_clears_without_resume(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=999)).isoformat()
        result, repo = await _run_load_context("Kisqali", _conv_with_pending(asked_at=old))
        assert result.get("resumed_from_clarification") is False
        assert result.get("merged_query") is None
        repo.update_metadata.assert_awaited_once_with("u1~s1", {"pending_clarification": None})

    @pytest.mark.asyncio
    async def test_no_pending_no_resume(self):
        conv = {"title": "t", "metadata": {}}
        result, repo = await _run_load_context("Kisqali", conv)
        assert result.get("resumed_from_clarification") is False
        assert result.get("merged_query") is None
        repo.update_metadata.assert_not_awaited()


# =============================================================================
# classify_intent_node — detection + resume suppression
# =============================================================================


class TestClassifyIntentClarification:
    @pytest.mark.asyncio
    async def test_underspecified_sets_needs_clarification(self):
        state = create_initial_state("u1", "why did it drop?", "r1", session_id="u1~s1")
        # simulate init_node having added the current human message
        state["messages"] = [HumanMessage(content="why did it drop?")]
        with patch.object(
            g,
            "classify_intent_dspy",
            AsyncMock(return_value=(IntentType.CAUSAL_ANALYSIS, 0.9, "", "dspy")),
        ):
            with patch.object(
                g, "route_agent_hardcoded", return_value=("causal-impact", [], 0.9, "")
            ):
                result = await classify_intent_node(state)
        assert result["needs_clarification"] is True
        assert result["missing_slots"] == ["brand", "metric"]

    @pytest.mark.asyncio
    async def test_resumed_turn_hard_suppresses_detection(self):
        # A resumed turn whose merged text STILL lacks a slot must NOT re-clarify
        state = create_initial_state("u1", "the northeast", "r2", session_id="u1~s1")
        state["resumed_from_clarification"] = True
        state["merged_query"] = "why did it drop? the northeast"
        captured = {}

        async def _spy_classify(**kwargs):
            captured.update(kwargs)
            return (IntentType.CAUSAL_ANALYSIS, 0.9, "", "dspy")

        with patch.object(g, "classify_intent_dspy", _spy_classify):
            with patch.object(
                g, "route_agent_hardcoded", return_value=("causal-impact", [], 0.9, "")
            ):
                result = await classify_intent_node(state)
        assert result["needs_clarification"] is False
        assert result["missing_slots"] == []
        # classification ran on the MERGED query, not the raw turn
        assert captured["query"] == "why did it drop? the northeast"

    @pytest.mark.asyncio
    async def test_prior_referent_suppresses_clarification(self):
        # Turn 2 with a prior assistant turn -> "it" has an antecedent -> no clarify
        state = create_initial_state("u1", "why did it drop?", "r2", session_id="u1~s1")
        state["messages"] = [
            HumanMessage(content="why did it drop?"),
            HumanMessage(content="What is TRx for Kisqali?"),
            AIMessage(content="Kisqali TRx was 1234."),
        ]
        with patch.object(
            g,
            "classify_intent_dspy",
            AsyncMock(return_value=(IntentType.CAUSAL_ANALYSIS, 0.9, "", "dspy")),
        ):
            with patch.object(
                g, "route_agent_hardcoded", return_value=("causal-impact", [], 0.9, "")
            ):
                result = await classify_intent_node(state)
        assert result["needs_clarification"] is False


# =============================================================================
# graph-level: underspecified visits clarify NOT orchestrator; bridge never called
# =============================================================================


class TestGraphLevelClarification:
    @pytest.mark.asyncio
    async def test_underspecified_visits_clarify_not_orchestrator(self):
        from langgraph.checkpoint.memory import MemorySaver

        bridge = AsyncMock()
        orch_spy = MagicMock()

        with patch.object(g, "get_langgraph_checkpointer", return_value=MemorySaver()):
            with patch.object(g, "get_async_supabase_client", AsyncMock(return_value=None)):
                with patch.object(
                    g,
                    "classify_intent_dspy",
                    AsyncMock(return_value=(IntentType.CAUSAL_ANALYSIS, 0.9, "", "dspy")),
                ):
                    with patch.object(
                        g, "route_agent_hardcoded", return_value=("causal-impact", [], 0.9, "")
                    ):
                        with patch.object(g, "get_chat_llm", side_effect=RuntimeError("no llm")):
                            with patch.object(g, "run_conversational_bridge", bridge):
                                with patch.object(g, "get_orchestrator", orch_spy):
                                    with patch.object(
                                        g, "_calculate_significance_score", return_value=0.0
                                    ):
                                        with patch.object(
                                            g, "CHATBOT_SIGNAL_COLLECTION_ENABLED", False
                                        ):
                                            graph = g.create_e2i_chatbot_graph()
                                            state = create_initial_state(
                                                "u1",
                                                "why did it drop?",
                                                "r1",
                                                session_id="u1~s1",
                                            )
                                            config = {"configurable": {"thread_id": "u1~s1"}}
                                            result = await graph.ainvoke(state, config=config)

        assert result["agent_name"] == "clarifier"
        # the #883 fail-closed bridge lives in orchestrator_node, which was skipped
        bridge.assert_not_called()
        orch_spy.assert_not_called()
