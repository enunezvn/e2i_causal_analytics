"""#2099: CausalImpactAgent must not mint a session id, nor pass the audit id as one.

``_contribute_to_memory`` computed
``str(UUID(str(candidate))) if candidate else str(uuid4())`` over
``audit_workflow_id or state["session_id"]``. Three things were wrong with that
one line:

* a session-less run minted a uuid belonging to no conversation (this issue);
* the audit workflow id WON over a real chat session, so the row recorded the
  audit chain's identity in the session column -- exactly the role confusion
  #2099 names;
* the local ``UUID()`` parse is the parse-or-mint that #1403 removed from the
  hook one level down. A composite ``{user}~{session}`` chat id never parses, so
  a real session was destroyed here before the writer's #1404 coercion (which
  recovers the session uuid, and NULLs anything else) could see it.

The agent now forwards the raw caller session, or None, and the audit id moves
into the episodic ``raw_content`` where it stays a correlation handle without
claiming to be a session.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from src.agents.causal_impact.agent import CausalImpactAgent

# Captured at import, before this directory's autouse fixture swaps the method
# for an AsyncMock (it keeps the synthetic run() sweep off the shared dev store).
# These tests are about that method, so they call the real one directly.
_REAL_CONTRIBUTE = CausalImpactAgent._contribute_to_memory

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
_SESSION = "53f47dba-378e-4c39-96d9-ec3fda26e168"
_COMPOSITE = f"{_USER}~{_SESSION}"
_AUDIT = "9f2a1d4e-7c3b-4a15-9e28-6b0d5a7c1e39"

_OUTPUT: Dict[str, Any] = {
    "status": "success",
    "ate_estimate": 0.12,
    "confidence": 0.8,
    "refutation_passed": False,
}


@pytest.fixture
def recorder(monkeypatch):
    """Record the session id the agent hands to the memory hook."""
    calls: List[Dict[str, Any]] = []

    async def _contribute(**kwargs: Any) -> Dict[str, int]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("src.agents.causal_impact.memory_hooks.contribute_to_memory", _contribute)
    return calls


async def _contribute(state: Dict[str, Any]) -> None:
    await _REAL_CONTRIBUTE(CausalImpactAgent(), _OUTPUT, state)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_session_less_run_hands_the_hook_none(recorder):
    """No session and no audit id -> None. Never a minted uuid."""
    await _contribute({"treatment_var": "x", "outcome_var": "y"})

    assert len(recorder) == 1
    got = recorder[0]["session_id"]
    assert not (isinstance(got, str) and _UUID_RE.match(got)), f"agent minted a session id: {got!r}"
    assert got is None


@pytest.mark.asyncio
async def test_audit_workflow_id_is_not_passed_as_the_session(recorder):
    """The audit chain's identity is not a conversation."""
    await _contribute({"audit_workflow_id": _AUDIT, "treatment_var": "x", "outcome_var": "y"})

    assert len(recorder) == 1
    assert recorder[0]["session_id"] is None, (
        f"the audit workflow id was stored as the session: {recorder[0]['session_id']!r}"
    )


@pytest.mark.asyncio
async def test_a_real_session_beats_the_audit_workflow_id(recorder):
    """A real chat session wins; the audit id no longer outranks it."""
    await _contribute({"session_id": _SESSION, "audit_workflow_id": _AUDIT, "treatment_var": "x"})

    assert recorder[0]["session_id"] == _SESSION


@pytest.mark.asyncio
async def test_composite_session_id_is_forwarded_raw_not_parsed_away(recorder):
    """#1403's contract, now honoured at the agent too.

    A composite id never parses as a uuid. Parsing it here destroyed the real
    session before the writer's coercion could recover it.
    """
    await _contribute({"session_id": _COMPOSITE, "treatment_var": "x"})

    assert recorder[0]["session_id"] == _COMPOSITE


@pytest.mark.asyncio
@patch("src.agents.causal_impact.memory_hooks.persist_agent_activity", return_value=None)
async def test_audit_workflow_id_survives_in_raw_content(_activity):
    """Dropping it from the session column must not drop it from the row."""
    captured: Dict[str, Any] = {}

    async def _insert(memory: Any, text_to_embed: str, session_id: Optional[str]) -> str:
        captured["raw_content"] = memory.raw_content
        captured["session_id"] = session_id
        return "mem-1"

    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    hooks = CausalImpactMemoryHooks()
    with patch("src.memory.episodic_memory.insert_episodic_memory_with_text", _insert):
        memory_id = await hooks.store_causal_analysis(
            session_id=None,
            result=_OUTPUT,
            state={
                "audit_workflow_id": _AUDIT,
                "treatment_var": "x",
                "outcome_var": "y",
                "query": "does x lift y",
            },
        )

    assert memory_id == "mem-1"
    assert captured["session_id"] is None
    assert captured["raw_content"].get("audit_workflow_id") == _AUDIT


@pytest.mark.asyncio
@patch("src.agents.causal_impact.memory_hooks.persist_agent_activity", return_value=None)
async def test_raw_content_audit_id_is_absent_not_invented(_activity):
    """No audit id in state -> no fabricated key."""
    captured: Dict[str, Any] = {}

    async def _insert(memory: Any, text_to_embed: str, session_id: Optional[str]) -> str:
        captured["raw_content"] = memory.raw_content
        return "mem-1"

    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    hooks = CausalImpactMemoryHooks()
    with patch("src.memory.episodic_memory.insert_episodic_memory_with_text", _insert):
        await hooks.store_causal_analysis(
            session_id=None,
            result=_OUTPUT,
            state={"treatment_var": "x", "outcome_var": "y", "query": "q"},
        )

    assert captured["raw_content"].get("audit_workflow_id") is None


@pytest.mark.asyncio
async def test_hook_failure_stays_non_blocking(recorder, monkeypatch):
    """The graceful-degradation posture is unchanged."""

    async def _boom(**_kwargs: Any) -> Dict[str, int]:
        raise RuntimeError("memory down")

    monkeypatch.setattr("src.agents.causal_impact.memory_hooks.contribute_to_memory", _boom)

    await _contribute({"treatment_var": "x"})  # must not raise


def test_agent_module_no_longer_mints_a_session_uuid():
    """A grep-level pin: the mint expression must not come back."""
    import inspect

    source = inspect.getsource(_REAL_CONTRIBUTE)
    assert "uuid4()" not in source, "the session mint is back in _contribute_to_memory"
