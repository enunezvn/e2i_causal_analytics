"""Real-DB 2-turn guard against conversation-history duplication / replay (#1442).

The Redis checkpointer already restores prior turns into ``state["messages"]``
across requests. ``load_context_node`` used to ALSO re-load DB history into the
``messages`` channel every turn — and since the reducer is ``operator.add`` (no
dedup), that duplicated history and caused ``stream_chat`` to re-emit the prior
assistant message on turn 2+.

This test runs two REAL turns through the compiled graph (real LLM, real DB, real
checkpointer — NO persistence mock) on one thread and asserts the turn-1 assistant
message is present EXACTLY ONCE in the turn-2 accumulated state (was twice on the
bug). Message COUNT/identity is deterministic even though LLM text is not.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_chatbot_stream_no_history_replay.py -p no:cacheprovider
"""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with local Supabase reachable",
)


@pytest.mark.asyncio
async def test_turn2_does_not_duplicate_or_replay_history_realdb():
    import src.api.routes.chatbot_graph as g
    from src.api.routes.chatbot_state import create_initial_state
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    assert client is not None, "local Supabase client unavailable"

    prof = await client.table("chatbot_user_profiles").select("id").limit(1).execute()
    assert prof.data, "no chatbot_user_profiles row available to satisfy the FK"
    user_id = prof.data[0]["id"]
    session_id = f"{user_id}~{uuid.uuid4()}"
    graph = g.create_e2i_chatbot_graph()
    cfg = {"configurable": {"thread_id": session_id}}

    async def run_turn(query: str) -> list:
        state = create_initial_state(
            user_id=user_id,
            query=query,
            request_id=f"t1442-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
        )
        final: dict = {}
        async for values in graph.astream(state, config=cfg, stream_mode="values"):
            final = values
        return list(final.get("messages", []))

    try:
        after_t1 = await run_turn("What is the TRx for Kisqali?")
        # capture the turn-1 assistant content (the thing that used to get replayed)
        t1_ai = [m.content for m in after_t1 if isinstance(m, AIMessage) and m.content]
        assert t1_ai, "turn 1 produced no assistant message"
        t1_answer = t1_ai[-1]

        after_t2 = await run_turn("What about Fabhalta?")

        # The turn-1 assistant answer must appear EXACTLY ONCE after turn 2
        # (checkpointer's copy only) — never re-added by load_context.
        occurrences = sum(
            1 for m in after_t2 if isinstance(m, AIMessage) and m.content == t1_answer
        )
        assert occurrences == 1, (
            f"turn-1 answer duplicated {occurrences}x after turn 2 "
            f"(history replay regression #1442); seq="
            f"{''.join(type(m).__name__[0] for m in after_t2)}"
        )
        # Exactly one assistant message per turn -> two total, no dup HumanMessages.
        ai_count = sum(1 for m in after_t2 if isinstance(m, AIMessage))
        assert ai_count == 2, f"expected 2 assistant turns, got {ai_count}"
    finally:
        # Clean up throwaway rows (messages then conversation).
        await client.table("chatbot_messages").delete().eq("session_id", session_id).execute()
        await client.table("chatbot_conversations").delete().eq("session_id", session_id).execute()
