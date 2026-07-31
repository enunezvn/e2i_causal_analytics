"""Regression test for #1393 (2nd chat-path consumer): chatbot episodic write.

Codex iter-1 review of the orchestrator fix found a SECOND chat-path episodic
writer with the identical defect: ``chatbot_graph._save_to_episodic_memory``
forwarded the state's ``session_id`` -- composite ``{user}~{session}`` by default
(``chatbot_state.create_initial_state``) -- straight into the ``uuid``-typed
``episodic_memories.session_id`` column, so Postgres rejected it with 22P02 and
the surrounding ``except`` swallowed it. The chat interaction's episodic record
was silently lost, the same as the orchestrator hook (#1393).

Fix: coerce the composite id to the plain session uuid (shared
``coerce_session_uuid`` helper) before the insert. This regression asserts the
``session_id`` handed to the writer is a valid uuid and the correct one. RED
before the fix (raw composite passed through).

Placed in the CI-allowlisted ``tests/unit/test_api`` subdir (the sibling
``test_episodic_memory_bridge.py`` is explicitly ``--ignore``-d in
backend-tests.yml, so its coverage of this function does NOT run in CI).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from src.api.routes.chatbot_graph import _save_to_episodic_memory
from src.api.routes.chatbot_state import IntentType


def _state(session_id: str) -> dict:
    return {
        "query": "why did TRx drop in the northeast",
        "session_id": session_id,
        "intent": IntentType.CAUSAL_ANALYSIS,
        "brand_context": None,
        "region_context": None,
        "tool_results": [],
        "metadata": {},
    }


@pytest.mark.asyncio
@patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
async def test_save_to_episodic_coerces_composite_session_to_session_uuid(mock_insert) -> None:
    mock_insert.return_value = "mem-1"
    session_uuid = uuid.uuid4()
    composite = f"{uuid.uuid4()}~{session_uuid}"

    await _save_to_episodic_memory(
        state=_state(composite),
        response_text="TRx declined due to reduced call frequency.",
        tool_calls=[],
        significance_score=0.7,
    )

    passed = mock_insert.call_args.kwargs["session_id"]
    assert UUID(passed) == session_uuid  # valid uuid for the uuid column (no 22P02)
    assert passed == str(session_uuid)  # the session segment, not the user segment


@pytest.mark.asyncio
@patch("src.api.routes.chatbot_graph.insert_episodic_memory_with_text", new_callable=AsyncMock)
async def test_save_to_episodic_preserves_composite_in_raw_content(mock_insert) -> None:
    """The uuid column is coerced, but raw_content keeps the composite id verbatim."""
    mock_insert.return_value = "mem-2"
    session_uuid = uuid.uuid4()
    composite = f"{uuid.uuid4()}~{session_uuid}"

    await _save_to_episodic_memory(
        state=_state(composite),
        response_text="answer",
        tool_calls=[],
        significance_score=0.7,
    )

    memory_input = mock_insert.call_args.kwargs["memory"]
    assert memory_input.raw_content["session_id"] == composite
