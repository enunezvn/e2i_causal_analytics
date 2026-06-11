"""#876 focused unit tests: explainer episodic ``outcome_type`` must be a VALID enum value.

The DB ``memory_outcome_type`` enum is a generic outcome STATE
(success / partial_success / failure / pending / escalated). Before #876,
``store_explanation`` passed the domain literal ``"explanation_delivered"``,
which Postgres rejected (22P02) and the hook swallowed — so no explainer
episodic row was ever written. The decision (mirroring causal_impact #788 and
het #873) is to MAP to the state enum, keeping the domain signal in
``event_type='explanation_generated'`` + ``agent_name``.

These tests capture the ``EpisodicMemoryInput`` the hook builds (no DB; the
faithful persistence proof lives in
``tests/integration/test_agent_episodic_outcome_876.py``).
"""

import pytest

# Clearly-fake sentinel for the captured-input stub (unit scope only; the real-DB
# integration test is the persistence proof).
_FAKE_MEMORY_ID = "00000000-0000-0000-0000-000000000876"

_VALID_OUTCOME_TYPES = {"success", "partial_success", "failure", "pending", "escalated"}


@pytest.fixture()
def captured(monkeypatch):
    """Capture the EpisodicMemoryInput store_explanation hands to the insert."""
    box = {}

    async def _capture(memory, text_to_embed=None, session_id=None, cycle_id=None):
        box["memory"] = memory
        box["session_id"] = session_id
        return _FAKE_MEMORY_ID

    # store_explanation imports this symbol inside the method body, so patching
    # the source module attribute intercepts the call.
    monkeypatch.setattr("src.memory.episodic_memory.insert_episodic_memory_with_text", _capture)
    return box


@pytest.mark.asyncio
async def test_explanation_maps_outcome_type_to_success(captured):
    from src.agents.explainer.memory_hooks import ExplanationMemoryHooks

    hooks = ExplanationMemoryHooks()
    memory_id = await hooks.store_explanation(
        session_id="session-876",
        explanation={
            "query": "Why did TRx rise?",
            "executive_summary": "Engagement-led uplift",
            "output_format": "narrative",
        },
    )

    assert memory_id == _FAKE_MEMORY_ID
    memory = captured["memory"]
    # STATE dimension: valid enum value, mapped — never the domain literal.
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
    # DOMAIN dimension: the "explanation delivered" signal stays here.
    assert memory.event_type == "explanation_generated"
    assert memory.agent_name == "explainer"


@pytest.mark.asyncio
async def test_failed_explanation_maps_outcome_type_to_failure(captured):
    """Defensive path: callers gate on status != 'failed' (contribute_to_memory)
    or only store on the successful generation path (narrative_generator), but if
    a failed payload ever reaches the store it must not be mislabeled 'success'."""
    from src.agents.explainer.memory_hooks import ExplanationMemoryHooks

    hooks = ExplanationMemoryHooks()
    await hooks.store_explanation(
        session_id="session-876-failed",
        explanation={
            "query": "Why did TRx rise?",
            "executive_summary": "",
            "status": "failed",
        },
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
