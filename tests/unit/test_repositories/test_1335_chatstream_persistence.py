"""Regression tests for issue #1335: /chat/stream session bookkeeping.

Two independent defects made scripted ``/api/copilotkit/chat/stream`` turns
silently non-persisted (see docs/demos/results/2026-07-29_copilot_chat_perf,
defect D4):

1. **chatbot_messages FK violation.**
   ``ChatbotConversationRepository.get_by_session_id`` used PostgREST
   ``.single()``, which RAISES ``APIError`` (PGRST116, "Cannot coerce the
   result to a single JSON object") on zero rows instead of returning ``None``
   -- its own documented "or None" contract. ``chatbot_graph.init_node``'s
   bootstrap ran the existence check and ``create_conversation`` inside one
   ``try/except``, so the raise on a brand-new session aborted the block
   *before* the parent ``chatbot_conversations`` row was created. The finalize
   node's ``chatbot_messages`` insert then FK-violated
   (``chatbot_messages_session_id_fkey``). The AG-UI surface avoided this by
   deliberately using ``.limit(1)`` (copilotkit ``_ensure_conversation_exists``,
   "don't use .single() which throws on no results").

2. **Audit-chain init on the composite session id.**
   The orchestrator's ``audit_init`` genesis passed the composite
   ``{user_id}~{session_uuid}`` id straight into the ``uuid``-typed
   ``audit_chain_entries.session_id`` column ("invalid input syntax for type
   uuid: ..."), which raised inside ``commit_entry`` and aborted audit-chain
   genesis for every turn on this surface.

The faithful DB behaviour used by these tests was confirmed READ-ONLY against
the local prod Supabase: ``.single()`` on 0 rows raises PGRST116, ``.limit(1)``
returns ``data == []``, and ``'<uuid>~<uuid>'::uuid`` raises "invalid input
syntax for type uuid".
"""

import uuid
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from postgrest.exceptions import APIError

from src.repositories.chatbot_conversation import ChatbotConversationRepository
from src.utils.audit_chain import (
    AgentTier,
    AuditChainService,
    _coerce_session_uuid,
)


def _faithful_conversation_client(*, rows_for_limit):
    """Build a Supabase-client mock that models the REAL PostgREST contract.

    ``.single().execute()`` raises ``APIError`` PGRST116 (0-row behaviour);
    ``.limit(n).execute()`` returns an object whose ``.data`` is ``rows_for_limit``.
    A correct ``get_by_session_id`` must use the ``.limit`` path and never let the
    ``.single`` raise escape.
    """
    single_exec = AsyncMock(
        side_effect=APIError(
            {
                "message": "Cannot coerce the result to a single JSON object",
                "code": "PGRST116",
                "details": "The result contains 0 rows",
                "hint": None,
            }
        )
    )
    limit_result = MagicMock()
    limit_result.data = rows_for_limit
    limit_exec = AsyncMock(return_value=limit_result)

    eq_obj = MagicMock()
    eq_obj.single.return_value = MagicMock(execute=single_exec)
    eq_obj.limit.return_value = MagicMock(execute=limit_exec)

    select_obj = MagicMock()
    select_obj.eq.return_value = eq_obj
    table_obj = MagicMock()
    table_obj.select.return_value = select_obj
    client = MagicMock()
    client.table.return_value = table_obj
    return client


class TestDefect1GetBySessionIdContract:
    """get_by_session_id must honour its "or None" contract on 0 rows."""

    @pytest.mark.asyncio
    async def test_returns_none_on_zero_rows_without_raising(self):
        """A missing session yields None -- it must NOT propagate PGRST116.

        RED before the fix: get_by_session_id calls ``.single()`` whose
        ``.execute()`` raises APIError, and the method has no guard, so the
        raise escapes. GREEN after: it uses ``.limit(1)`` -> ``data == []`` ->
        None.
        """
        client = _faithful_conversation_client(rows_for_limit=[])
        repo = ChatbotConversationRepository(client)

        result = await repo.get_by_session_id(f"{uuid.uuid4()}~{uuid.uuid4()}")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_row_when_present(self):
        """A present session returns the conversation dict."""
        row = {
            "session_id": "u~s",
            "user_id": "00000000-0000-0000-0000-000000000000",
            "title": "t",
        }
        client = _faithful_conversation_client(rows_for_limit=[row])
        repo = ChatbotConversationRepository(client)

        result = await repo.get_by_session_id("u~s")

        assert result is not None
        assert result["session_id"] == "u~s"


class TestDefect2AuditSessionUuidCoercion:
    """The audit chain must accept the platform's composite session-id format."""

    def test_coerce_composite_returns_trailing_session_uuid(self):
        session_uuid = uuid.uuid4()
        composite = f"{uuid.uuid4()}~{session_uuid}"

        assert _coerce_session_uuid(composite) == session_uuid

    def test_coerce_plain_uuid_string_is_preserved(self):
        u = uuid.uuid4()

        assert _coerce_session_uuid(str(u)) == u

    def test_coerce_uuid_object_is_passed_through(self):
        u = uuid.uuid4()

        assert _coerce_session_uuid(u) == u

    def test_coerce_none_is_none(self):
        assert _coerce_session_uuid(None) is None

    @pytest.mark.parametrize("bad", ["", "not-a-uuid", "a~b~c", "user~also-not-uuid"])
    def test_coerce_non_uuid_returns_none_not_fabricated(self, bad):
        assert _coerce_session_uuid(bad) is None

    def test_start_workflow_normalizes_composite_for_uuid_column(self):
        """start_workflow with a composite id must persist a valid uuid.

        RED before the fix: entry.session_id keeps the raw ``user~uuid`` string,
        so ``to_db_dict()["session_id"]`` is not uuid-parseable and the real
        insert into the uuid column raises. GREEN after: it is the trailing
        session uuid.
        """
        service = AuditChainService(MagicMock())
        session_uuid = uuid.uuid4()
        composite = f"{uuid.uuid4()}~{session_uuid}"

        entry = service.start_workflow(
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            action_type="workflow_start",
            session_id=composite,
            auto_commit=False,
        )

        assert entry.session_id == session_uuid
        # Must be a valid uuid string for the uuid-typed column (no raise).
        assert UUID(entry.to_db_dict()["session_id"]) == session_uuid

    def test_start_workflow_plain_uuid_still_stored(self):
        service = AuditChainService(MagicMock())
        u = uuid.uuid4()

        entry = service.start_workflow(
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            action_type="workflow_start",
            session_id=u,
            auto_commit=False,
        )

        assert entry.session_id == u

    def test_start_workflow_none_session_stays_none(self):
        service = AuditChainService(MagicMock())

        entry = service.start_workflow(
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
            action_type="workflow_start",
            session_id=None,
            auto_commit=False,
        )

        assert entry.session_id is None
        assert entry.to_db_dict()["session_id"] is None
