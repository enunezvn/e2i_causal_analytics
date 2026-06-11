"""#873 focused unit tests: het episodic ``outcome_type`` must be a VALID enum value.

The DB ``memory_outcome_type`` enum is a generic outcome STATE
(success / partial_success / failure / pending / escalated). Before #873,
``store_cate_analysis`` passed the domain literal ``"cate_analysis_delivered"``,
which Postgres rejected (22P02) and the hook swallowed — so no het CATE episodic
row was ever written. The decision (mirroring the causal_impact #788 fix) is to
MAP to the state enum, keeping the domain signal in
``event_type='cate_analysis_completed'`` + ``agent_name``.

These tests capture the ``EpisodicMemoryInput`` the hook builds (no DB; the
faithful persistence proof lives in
``tests/integration/test_het_episodic_outcome_873.py``).
"""

import pytest

# Clearly-fake sentinel for the captured-input stub (unit scope only; the real-DB
# integration test is the persistence proof).
_FAKE_MEMORY_ID = "00000000-0000-0000-0000-000000000873"

_VALID_OUTCOME_TYPES = {"success", "partial_success", "failure", "pending", "escalated"}


@pytest.fixture()
def captured(monkeypatch):
    """Capture the EpisodicMemoryInput store_cate_analysis hands to the insert."""
    box = {}

    async def _capture(memory, text_to_embed=None, session_id=None, cycle_id=None):
        box["memory"] = memory
        box["session_id"] = session_id
        return _FAKE_MEMORY_ID

    # store_cate_analysis imports this symbol inside the method body, so patching
    # the source module attribute intercepts the call.
    monkeypatch.setattr("src.memory.episodic_memory.insert_episodic_memory_with_text", _capture)
    return box


@pytest.mark.asyncio
async def test_completed_analysis_maps_outcome_type_to_success(captured):
    from src.agents.heterogeneous_optimizer.memory_hooks import (
        HeterogeneousOptimizerMemoryHooks,
    )

    hooks = HeterogeneousOptimizerMemoryHooks()
    memory_id = await hooks.store_cate_analysis(
        session_id="session-873",
        analysis_result={
            "status": "completed",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "heterogeneity_score": 0.34,
            "high_responders": [],
            "low_responders": [],
        },
    )

    assert memory_id == _FAKE_MEMORY_ID
    memory = captured["memory"]
    # STATE dimension: valid enum value, mapped — never the domain literal.
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
    # DOMAIN dimension: the "CATE analysis delivered" signal stays here.
    assert memory.event_type == "cate_analysis_completed"
    assert memory.agent_name == "heterogeneous_optimizer"


@pytest.mark.asyncio
async def test_failed_analysis_maps_outcome_type_to_failure(captured):
    """Defensive path: callers gate on status != 'failed', but if a failed result
    ever reaches the store, it must not be mislabeled 'success'."""
    from src.agents.heterogeneous_optimizer.memory_hooks import (
        HeterogeneousOptimizerMemoryHooks,
    )

    hooks = HeterogeneousOptimizerMemoryHooks()
    await hooks.store_cate_analysis(
        session_id="session-873-failed",
        analysis_result={
            "status": "failed",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
        },
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
