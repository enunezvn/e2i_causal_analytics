"""Regression test for #1393: orchestrator episodic write on the chat path.

The chat surface passes a composite ``{user_uuid}~{session_uuid}`` session id.
``OrchestratorMemoryHooks.store_orchestration`` fed it straight into
``insert_episodic_memory_with_text`` -> the ``uuid``-typed
``episodic_memories.session_id`` column, which Postgres rejects with 22P02
("invalid input syntax for type uuid"). The ``except`` in ``store_orchestration``
swallowed the error, so EVERY chat-path orchestration silently lost its episodic
record (``episodic=0`` in "Memory contribution complete"). Agent-level episodic
writes were unaffected.

Fix: coerce the composite id to its trailing session uuid (via the shared
``coerce_session_uuid`` helper) before the episodic insert. The regression
asserts the ``session_id`` handed to the writer is a VALID uuid and the CORRECT
one (the session segment). RED before the fix (raw composite reaches the writer).

Placed in the CI-allowlisted ``tests/unit/test_memory`` subdir: the orchestrator
agent's own test dir (``tests/unit/test_agents/test_orchestrator``) is NOT in
either backend-tests.yml allowlist and so never runs in CI.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks


def _orchestration_result() -> dict:
    return {
        "query": "why did TRx drop in the northeast",
        "intent_classified": "causal_analysis",
        "agents_dispatched": ["gap_analyzer"],
        "response_text": "TRx declined due to reduced call frequency.",
        "response_confidence": 0.82,
        "total_latency_ms": 1234,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_store_orchestration_coerces_composite_session_to_session_uuid() -> None:
    """A ``{user}~{session}`` id reaches the episodic writer as the session uuid."""
    session_uuid = uuid.uuid4()
    composite = f"{uuid.uuid4()}~{session_uuid}"

    hooks = OrchestratorMemoryHooks()

    with patch(
        "src.memory.episodic_memory.insert_episodic_memory_with_text",
        new=AsyncMock(return_value="mem-1"),
    ) as mock_insert:
        await hooks.store_orchestration(session_id=composite, result=_orchestration_result())

    assert mock_insert.await_count == 1
    passed = mock_insert.await_args.kwargs["session_id"]
    # Must be a valid uuid string for the uuid-typed column (no 22P02) ...
    assert UUID(passed) == session_uuid
    # ... and specifically the SESSION segment, not the user segment.
    assert passed == str(session_uuid)


@pytest.mark.asyncio
async def test_store_orchestration_bridge_suffixed_composite() -> None:
    """A ``{user}~{session}~bridge`` id (PR #1394) still yields the session uuid.

    The bridge path does not itself reach this hook (the bridge runs the AG-UI
    graph, not the orchestrator), but the coercion must be robust to the suffix
    since it shares the hardened helper with the audit chain, which the bridge
    DOES exercise.
    """
    session_uuid = uuid.uuid4()
    bridged = f"{uuid.uuid4()}~{session_uuid}~bridge"

    hooks = OrchestratorMemoryHooks()

    with patch(
        "src.memory.episodic_memory.insert_episodic_memory_with_text",
        new=AsyncMock(return_value="mem-2"),
    ) as mock_insert:
        await hooks.store_orchestration(session_id=bridged, result=_orchestration_result())

    assert UUID(mock_insert.await_args.kwargs["session_id"]) == session_uuid


@pytest.mark.asyncio
async def test_store_orchestration_plain_uuid_unchanged() -> None:
    """A plain uuid (AG-UI threadId) is passed through unchanged."""
    session_uuid = uuid.uuid4()

    hooks = OrchestratorMemoryHooks()

    with patch(
        "src.memory.episodic_memory.insert_episodic_memory_with_text",
        new=AsyncMock(return_value="mem-3"),
    ) as mock_insert:
        await hooks.store_orchestration(
            session_id=str(session_uuid), result=_orchestration_result()
        )

    assert mock_insert.await_args.kwargs["session_id"] == str(session_uuid)
