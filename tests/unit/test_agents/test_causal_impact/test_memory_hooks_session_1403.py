"""#1403: causal_impact memory hook preserves the composite chat session id.

``contribute_to_memory`` used to parse ``session_id`` as a plain uuid and, on
failure (a composite ``{user_uuid}~{session_uuid}`` id never parses), mint a
*random* uuid. That destroyed session linkage — the stored session id correlated
with nothing. The hook must instead forward the RAW id; the episodic writer
coerces it to the real session uuid for the uuid column (#1404).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.causal_impact.memory_hooks import contribute_to_memory

_USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
_SESSION = "53f47dba-378e-4c39-96d9-ec3fda26e168"
_COMPOSITE = f"{_USER}~{_SESSION}"


@pytest.mark.asyncio
@patch("src.agents.causal_impact.memory_hooks.persist_agent_activity", return_value=None)
async def test_composite_session_id_forwarded_not_minted(_mock_activity):
    hooks = MagicMock()
    hooks.cache_causal_analysis = AsyncMock(return_value=True)
    hooks.store_causal_analysis = AsyncMock(return_value="mem-1")
    hooks.store_causal_path = AsyncMock(return_value=None)

    await contribute_to_memory(
        result={"status": "success", "ate_estimate": 0.1, "refutation_passed": False},
        state={"session_id": _COMPOSITE, "treatment_var": "x", "outcome_var": "y"},
        memory_hooks=hooks,
    )

    # The hook must forward the REAL composite id (the writer coerces it to the
    # session uuid downstream), NOT a freshly minted random uuid.
    assert hooks.store_causal_analysis.call_args.kwargs["session_id"] == _COMPOSITE
    assert hooks.cache_causal_analysis.call_args.args[0] == _COMPOSITE


@pytest.mark.asyncio
@patch("src.agents.causal_impact.memory_hooks.persist_agent_activity", return_value=None)
async def test_garbage_session_id_forwarded_not_minted(_mock_activity):
    hooks = MagicMock()
    hooks.cache_causal_analysis = AsyncMock(return_value=True)
    hooks.store_causal_analysis = AsyncMock(return_value="mem-1")
    hooks.store_causal_path = AsyncMock(return_value=None)

    await contribute_to_memory(
        result={"status": "success", "refutation_passed": False},
        state={"session_id": "sess_123"},
        memory_hooks=hooks,
    )

    # A non-uuid id is forwarded as-is (the writer coerces it to an honest NULL);
    # it is NOT replaced with a random uuid here.
    assert hooks.store_causal_analysis.call_args.kwargs["session_id"] == "sess_123"
