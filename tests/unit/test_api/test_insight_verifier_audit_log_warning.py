"""Audit-log failure severity for JIT provenance verifier.

The ``audit_chain_verification_log`` is the regulatory trail. If the insert
fails, we cannot silently lose the entry — the severity must be at least
``warning`` with enough context to triage (insight_type, insight_id,
strict-mode, verdict).

PR #250's initial implementation used ``logger.debug``, which made audit-log
failures invisible in production log streams (most ops configs log at
INFO/WARNING level). PR #247 review flagged this; iter-1 of PR #250
addresses it.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_audit_log_failure_emits_warning_with_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When audit_chain_verification_log insert raises, emit warning level
    with insight_type + insight_id + verdict reason in the message.

    Falsifiability: reverting `logger.warning(...)` to `logger.debug(...)`
    trips this test (no WARNING-level record captured).
    """
    from src.api.middleware.insight_verifier import _log_verification

    fake_client = MagicMock()
    fake_client.table.return_value.insert.return_value.execute.side_effect = RuntimeError(
        "supabase RPC unreachable"
    )

    with patch(
        "src.memory.services.factories.get_supabase_client",
        return_value=fake_client,
    ):
        with caplog.at_level(logging.WARNING, logger="src.api.middleware.insight_verifier"):
            await _log_verification(
                insight_type="executive_insight",
                insight_id="ei-42",
                verdict={
                    "is_valid": False,
                    "reason": "ancestor causal_path overturned",
                    "depth_walked": 3,
                },
                strict=True,
            )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, (
        "no WARNING-level record emitted on audit-log failure — "
        "regulatory-grade audit must surface the failure, not swallow at DEBUG"
    )
    msg = warnings[0].getMessage()
    assert "executive_insight" in msg, f"missing insight_type in warning: {msg}"
    assert "ei-42" in msg, f"missing insight_id in warning: {msg}"


@pytest.mark.asyncio
async def test_audit_log_success_does_not_emit_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sanity: when audit-log insert succeeds, NO warning is emitted.

    Confirms the warning fires only on failure, not on every call.
    """
    from src.api.middleware.insight_verifier import _log_verification

    fake_client = MagicMock()
    fake_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[{"id": 1}]
    )

    with patch(
        "src.memory.services.factories.get_supabase_client",
        return_value=fake_client,
    ):
        with caplog.at_level(logging.WARNING, logger="src.api.middleware.insight_verifier"):
            await _log_verification(
                insight_type="executive_insight",
                insight_id="ei-43",
                verdict={"is_valid": True, "depth_walked": 0},
                strict=False,
            )

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "audit_chain_verification_log" in r.getMessage()
    ]
    assert not warnings, (
        f"unexpected warning(s) on success path: {[r.getMessage() for r in warnings]}"
    )
