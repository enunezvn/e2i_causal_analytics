"""#876 focused unit tests: tool_composer episodic ``outcome_type`` must be VALID.

The DB ``memory_outcome_type`` enum is a generic outcome STATE
(success / partial_success / failure / pending / escalated). Before #876,
``store_composition`` passed the domain literal ``"composition_delivered"``,
which Postgres rejected (22P02) and the hook swallowed — so no tool_composer
episodic row was ever written. The decision (mirroring causal_impact #788 and
het #873) is to MAP to the state enum, keeping the domain signal in
``event_type='composition_completed'`` + ``agent_name``.

tool_composer is the one agent of the #876 family whose result carries a REAL
multi-state contract status (CompositionStatus: success / partial / failed /
timeout / blocked, plus the ``success`` quality-gate bool that is True for both
SUCCESS and PARTIAL), so the mapping is richer than the binary het pattern:

  * status 'partial'                      -> 'partial_success'  (mirrors the
    causal_impact #788 REVIEW-band precedent — completed, but degraded)
  * status 'failed' / 'timeout' / 'blocked' -> 'failure'
  * status 'success' (or success bool)    -> 'success'
  * nothing recognizable                  -> 'failure' (don't fabricate success)

These tests capture the ``EpisodicMemoryInput`` the hook builds (no DB; the
faithful persistence proof lives in
``tests/integration/test_agent_episodic_outcome_876.py``).
"""

import pytest

# Clearly-fake sentinel for the captured-input stub (unit scope only; the real-DB
# integration test is the persistence proof).
_FAKE_MEMORY_ID = "00000000-0000-0000-0000-000000000876"

_VALID_OUTCOME_TYPES = {"success", "partial_success", "failure", "pending", "escalated"}


def _result(status, success):
    base = {
        "composition_id": "comp-876",
        "query": "Compare TRx uplift by region",
        "decomposition": {"sub_questions": [{"q": "uplift"}]},
        "plan": {"steps": [{"tool_name": "kpi_query"}]},
        "execution": {"tools_executed": 1, "tools_succeeded": 1},
        "response": {"confidence": 0.9},
        "total_duration_ms": 1000,
    }
    if status is not None:
        base["status"] = status
    if success is not None:
        base["success"] = success
    return base


@pytest.fixture()
def captured(monkeypatch):
    """Capture the EpisodicMemoryInput store_composition hands to the insert."""
    box = {}

    async def _capture(memory, text_to_embed=None, session_id=None, cycle_id=None):
        box["memory"] = memory
        box["session_id"] = session_id
        return _FAKE_MEMORY_ID

    # store_composition imports this symbol inside the method body, so patching
    # the source module attribute intercepts the call.
    monkeypatch.setattr("src.memory.episodic_memory.insert_episodic_memory_with_text", _capture)
    return box


@pytest.mark.asyncio
async def test_success_status_maps_outcome_type_to_success(captured):
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    memory_id = await hooks.store_composition(
        session_id="session-876",
        result=_result(status="success", success=True),
    )

    assert memory_id == _FAKE_MEMORY_ID
    memory = captured["memory"]
    # STATE dimension: valid enum value, mapped — never the domain literal.
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
    # DOMAIN dimension: the "composition delivered" signal stays here.
    assert memory.event_type == "composition_completed"
    assert memory.agent_name == "tool_composer"


@pytest.mark.asyncio
async def test_partial_status_maps_outcome_type_to_partial_success(captured):
    """CompositionStatus.PARTIAL (some tools failed, synthesis still ran; the
    contract sets success=True for it) is degraded, not full success."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    await hooks.store_composition(
        session_id="session-876-partial",
        result=_result(status="partial", success=True),
    )

    memory = captured["memory"]
    assert memory.outcome_type == "partial_success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES


@pytest.mark.asyncio
async def test_failed_status_maps_outcome_type_to_failure(captured):
    """Defensive path: the composer only contributes memory on the synthesized
    path (status success/partial), but a failed result reaching the store must
    not be mislabeled 'success'."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    await hooks.store_composition(
        session_id="session-876-failed",
        result=_result(status="failed", success=False),
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES


@pytest.mark.asyncio
async def test_success_flag_without_status_maps_to_success(captured):
    """Legacy/partial dicts: the quality-gate bool alone still maps to success."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    await hooks.store_composition(
        session_id="session-876-flag",
        result=_result(status=None, success=True),
    )

    memory = captured["memory"]
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES


@pytest.mark.asyncio
async def test_no_status_no_success_claim_maps_to_failure(captured):
    """No recognizable status and no success claim: don't fabricate success."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    await hooks.store_composition(
        session_id="session-876-unknown",
        result=_result(status=None, success=None),
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
