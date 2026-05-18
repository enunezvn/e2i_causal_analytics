"""Regression tests for issue #355 (S9): causal_impact audit-chain kwarg mismatch.

Pre-fix, ``src/agents/causal_impact/graph.py:173-187`` called
``AuditChainService.add_entry(...)`` with kwargs that did not match the method
signature: ``input_hash``/``output_hash``/``user_id``/``session_id``/``brand``
and passed ``agent_tier`` as a string (``.value``) instead of the
:class:`AgentTier` enum. The mismatch was silenced by
``# type: ignore[call-arg]`` at static-check time, and by a
``try/except Exception`` that turned the runtime ``TypeError`` into a debug
warning. The net effect was that no audit-chain entries were written for any
causal_impact node execution — the chain's ``previous_hash`` invariant was
broken and expert_review repository lookups keyed on audit lineage could not
trace causal_impact ancestry.

These tests assert the *call shape* (kwargs by name) instead of inner state.
The kwargs that survive are exactly those declared by
:meth:`AuditChainService.add_entry` (see ``src/utils/audit_chain.py:288-301``).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.agents.base.audit_chain_mixin import set_audit_chain_service
from src.agents.causal_impact.graph import traced_node
from src.utils.audit_chain import AgentTier, AuditChainService


@pytest.fixture(autouse=True)
def _reset_audit_service():
    set_audit_chain_service(None)
    yield
    set_audit_chain_service(None)


@pytest.fixture
def mock_audit_service():
    svc = MagicMock(spec=AuditChainService)
    set_audit_chain_service(svc)
    return svc


@pytest.fixture
def mock_opik():
    """Patch get_opik_connector to a no-op async context manager."""
    mock = MagicMock()
    mock.is_enabled = True
    span = MagicMock()
    span.span_id = "span_test"
    span.set_output = MagicMock()
    span.set_attribute = MagicMock()
    mock.trace_agent = MagicMock()
    mock.trace_agent.return_value.__aenter__ = AsyncMock(return_value=span)
    mock.trace_agent.return_value.__aexit__ = AsyncMock(return_value=None)
    with patch("src.agents.causal_impact.graph.get_opik_connector", return_value=mock):
        yield mock


@pytest.fixture
def base_state():
    return {
        "audit_workflow_id": uuid4(),
        "query": "Did dispatch X cause uplift?",
        "treatment_var": "spend",
        "outcome_var": "uplift",
        "current_phase": "estimation",
        "session_id": "sess-1",
        "user_id": "user-1",
        "brand": "BrandX",
        "query_id": "qid-1",
        "span_id": None,
        "dispatch_id": "dispatch-1",
    }


# ---------------------------------------------------------------------------
# Regression cases — call-shape assertions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traced_node_passes_input_data_not_input_hash(
    mock_audit_service, mock_opik, base_state
):
    """add_entry must receive ``input_data`` (in signature), not ``input_hash``."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    assert mock_audit_service.add_entry.called, (
        "add_entry should be called when workflow_id and audit_service present"
    )
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "input_data" in kwargs, (
        f"add_entry must use input_data (matches signature); got kwargs={sorted(kwargs)}"
    )
    assert "input_hash" not in kwargs, "input_hash is not in add_entry signature; remove it"


@pytest.mark.asyncio
async def test_traced_node_passes_output_data_not_output_hash(
    mock_audit_service, mock_opik, base_state
):
    """add_entry must receive ``output_data`` (in signature), not ``output_hash``."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "output_data" in kwargs, (
        f"add_entry must use output_data (matches signature); got kwargs={sorted(kwargs)}"
    )
    assert "output_hash" not in kwargs, "output_hash is not in add_entry signature; remove it"


@pytest.mark.asyncio
async def test_traced_node_passes_agent_tier_as_enum_not_string(
    mock_audit_service, mock_opik, base_state
):
    """add_entry's ``agent_tier`` parameter is typed AgentTier, not its .value."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "agent_tier" in kwargs
    assert isinstance(kwargs["agent_tier"], AgentTier), (
        f"agent_tier must be AgentTier enum, not {type(kwargs['agent_tier']).__name__}; "
        f"got value={kwargs['agent_tier']!r}"
    )
    assert kwargs["agent_tier"] is AgentTier.CAUSAL_ANALYTICS


@pytest.mark.asyncio
async def test_traced_node_does_not_pass_user_id_session_id_brand(
    mock_audit_service, mock_opik, base_state
):
    """user_id/session_id/brand are NOT in add_entry's signature — they are
    inherited from the workflow's genesis entry. Passing them caused TypeError
    pre-fix (silenced by ``# type: ignore[call-arg]`` and try/except)."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    for forbidden in ("user_id", "session_id", "brand"):
        assert forbidden not in kwargs, (
            f"{forbidden!r} is not in add_entry signature; remove it "
            f"(inherited from genesis entry via previous.{forbidden})"
        )


@pytest.mark.asyncio
async def test_traced_node_audit_entry_does_not_warning_log_failure(mock_opik, base_state, caplog):
    """End-to-end shape check: with a real :class:`AuditChainService` autospec,
    the add_entry call must succeed without exception. Pre-fix, the TypeError
    from bad kwargs was caught by ``except Exception`` and surfaced as
    ``logger.warning('Failed to record audit entry: ...')``. After fix, that
    warning must NOT appear."""
    from unittest.mock import create_autospec

    svc = create_autospec(AuditChainService, instance=True)
    set_audit_chain_service(svc)

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    with caplog.at_level(logging.WARNING, logger="src.agents.causal_impact.graph"):
        await _node(base_state)

    failure_warnings = [
        r for r in caplog.records if "Failed to record audit entry" in r.getMessage()
    ]
    assert not failure_warnings, (
        "add_entry raised an exception that was caught & warning-logged — "
        "indicates kwarg/signature mismatch is still present. "
        f"Warnings: {[r.getMessage() for r in failure_warnings]}"
    )
