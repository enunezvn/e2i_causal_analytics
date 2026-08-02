"""Real-DB 2-turn round-trip for the multi-turn clarification feature (#1407).

Faithful integration test against the local self-contained Supabase — NO
persistence mock. The clarification pending-state lives in
``chatbot_conversations.metadata`` (jsonb, keyed on session); this test proves
it is ACTUALLY written on an underspecified turn and ACTUALLY cleared when the
next turn supplies the missing slots, and that the orchestrator then dispatches
on the MERGED query.

Only non-persistence collaborators are stubbed for determinism (the DSPy intent
classifier -> CAUSAL, the ask-back LLM -> canned fallback, and the orchestrator
-> a spy that records the query it received). Every ``chatbot_conversations``
read/write goes through the real async Supabase client.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_chatbot_clarification_multiturn.py -p no:cacheprovider
"""

import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with local Supabase reachable",
)


class _SpyOrchestrator:
    """Records the query it was dispatched, returns a minimal success result."""

    def __init__(self):
        self.received_query = None

    async def run(self, payload):
        self.received_query = payload.get("query")
        return {
            "response_text": "Analysis complete.",
            "response_confidence": 0.9,
            "agents_dispatched": ["causal_impact"],
            "status": "completed",
        }


@pytest.mark.asyncio
async def test_clarification_two_turn_roundtrip_realdb():
    import src.api.routes.chatbot_graph as g
    from src.api.routes.chatbot_state import IntentType, create_initial_state
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.chatbot_conversation import get_chatbot_conversation_repository

    client = await get_async_supabase_client()
    assert client is not None, "local Supabase client unavailable"

    # Reuse an existing user profile to satisfy the
    # chatbot_conversations.user_id -> chatbot_user_profiles.id FK. The session
    # is a fresh throwaway uuid cleaned up at the end.
    prof = await client.table("chatbot_user_profiles").select("id").limit(1).execute()
    assert prof.data, "no chatbot_user_profiles row available to satisfy the FK"
    user_id = prof.data[0]["id"]
    session_id = f"{user_id}~{uuid.uuid4()}"
    conv_repo = get_chatbot_conversation_repository(client)

    # Parent conversation row (FK target for any message writes + metadata home)
    await conv_repo.create_conversation(
        user_id=user_id, session_id=session_id, query_type="causal_analysis"
    )

    try:
        # ---------------- TURN 1: underspecified analytical ask ----------------
        state1 = create_initial_state(
            user_id=user_id,
            query="Why did it drop?",
            request_id="itg-1407-1",
            session_id=session_id,
        )
        with patch.object(
            g,
            "classify_intent_dspy",
            AsyncMock(return_value=(IntentType.CAUSAL_ANALYSIS, 0.9, "", "dspy")),
        ):
            with patch.object(
                g, "route_agent_hardcoded", return_value=("causal-impact", [], 0.9, "")
            ):
                # ask-back LLM stubbed to fail -> deterministic canned fallback (no key)
                with patch.object(g, "get_chat_llm", side_effect=RuntimeError("no-llm")):
                    state1.update(await g.load_context_node(state1))
                    state1.update(await g.classify_intent_node(state1))
                    assert state1["needs_clarification"] is True
                    assert g.route_after_classify(state1) == "clarify"
                    state1.update(await g.clarify_node(state1))

        assert state1["agent_name"] == "clarifier"

        # pending_clarification ACTUALLY written to the real DB
        conv = await conv_repo.get_by_session_id(session_id)
        pending = (conv.get("metadata") or {}).get("pending_clarification")
        assert pending is not None, "pending_clarification was not persisted"
        assert pending["original_query"] == "Why did it drop?"
        assert pending["missing_slots"] == ["brand", "metric"]
        assert "asked_at" in pending

        # ---------------- TURN 2: slot-answer resumes the clarification --------
        state2 = create_initial_state(
            user_id=user_id,
            query="Kisqali TRx",
            request_id="itg-1407-2",
            session_id=session_id,
        )
        spy = _SpyOrchestrator()
        with patch.object(
            g,
            "classify_intent_dspy",
            AsyncMock(return_value=(IntentType.CAUSAL_ANALYSIS, 0.9, "", "dspy")),
        ):
            with patch.object(
                g, "route_agent_hardcoded", return_value=("causal-impact", [], 0.9, "")
            ):
                with patch.object(g, "get_orchestrator", return_value=spy):
                    # load_context reads pending from the real DB, decides ANSWER,
                    # merges + clears pending via the real update_metadata.
                    state2.update(await g.load_context_node(state2))
                    assert state2.get("resumed_from_clarification") is True
                    assert state2.get("merged_query") == "Why did it drop? Kisqali TRx"

                    state2.update(await g.classify_intent_node(state2))
                    # resumed turn HARD-suppresses re-detection -> no re-clarify
                    assert state2["needs_clarification"] is False
                    assert g.route_after_classify(state2) == "retrieve_rag"

                    state2.update(await g.orchestrator_node(state2))

        # orchestrator dispatched on the MERGED query
        assert spy.received_query == "Why did it drop? Kisqali TRx"

        # pending_clarification ACTUALLY cleared in the real DB
        conv2 = await conv_repo.get_by_session_id(session_id)
        assert (conv2.get("metadata") or {}).get("pending_clarification") is None, (
            "pending_clarification was not cleared after the answer turn"
        )
    finally:
        # Clean up the throwaway rows created by this test.
        try:
            await client.table("chatbot_messages").delete().eq("session_id", session_id).execute()
        except Exception:
            pass
        try:
            await (
                client.table("chatbot_conversations")
                .delete()
                .eq("session_id", session_id)
                .execute()
            )
        except Exception:
            pass
