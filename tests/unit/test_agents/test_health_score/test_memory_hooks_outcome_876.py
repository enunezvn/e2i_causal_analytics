"""#876 focused unit tests: health_score episodic ``outcome_type`` must be a VALID enum value.

The DB ``memory_outcome_type`` enum is a generic outcome STATE
(success / partial_success / failure / pending / escalated). Before #876,
``store_health_check`` passed the domain literal ``"health_assessment_delivered"``,
which Postgres rejected (22P02) and the hook swallowed — so no health_score
episodic row was ever written. (health_score had a DOUBLE bug: its event_type
``health_check_completed`` was also missing from ``memory_event_type`` until
database/migrations/070.) The decision (mirroring causal_impact #788 and het
#873) is to MAP to the state enum, keeping the domain signal in
``event_type='health_check_completed'`` + ``agent_name``.

NOTE on semantics: ``outcome_type`` reflects the health CHECK operation's
outcome, not the health of the system being checked — an unhealthy system
(grade F, critical issues) reported by a successfully-completed check is a
``success``; the severity lives in ``raw_content`` + ``importance_score``.

These tests capture the ``EpisodicMemoryInput`` the hook builds (no DB; the
faithful persistence proof lives in
``tests/integration/test_agent_episodic_outcome_876.py``).
"""

import pytest

# Clearly-fake sentinel for the captured-input stub (unit scope only; the real-DB
# integration test is the persistence proof).
_FAKE_MEMORY_ID = "00000000-0000-0000-0000-000000000876"

_VALID_OUTCOME_TYPES = {"success", "partial_success", "failure", "pending", "escalated"}

# Passes _is_significant_health_event (critical issues present) while remaining
# an obviously-degraded SYSTEM — the agent OPERATION still succeeded.
_SIGNIFICANT_RESULT = {
    "overall_health_score": 52.5,
    "health_grade": "D",
    "critical_issues": ["database connection pool exhausted"],
    "warnings": [],
    "component_health_score": 40.0,
    "model_health_score": 55.0,
    "pipeline_health_score": 60.0,
    "agent_health_score": 58.0,
    "total_latency_ms": 1234,
}


@pytest.fixture()
def captured(monkeypatch):
    """Capture the EpisodicMemoryInput store_health_check hands to the insert."""
    box = {}

    async def _capture(memory, text_to_embed=None, session_id=None, cycle_id=None):
        box["memory"] = memory
        box["session_id"] = session_id
        return _FAKE_MEMORY_ID

    # store_health_check imports this symbol inside the method body, so patching
    # the source module attribute intercepts the call.
    monkeypatch.setattr("src.memory.episodic_memory.insert_episodic_memory_with_text", _capture)
    return box


@pytest.mark.asyncio
async def test_completed_check_maps_outcome_type_to_success(captured):
    """An unhealthy SYSTEM found by a completed check is still operation success."""
    from src.agents.health_score.memory_hooks import HealthScoreMemoryHooks

    hooks = HealthScoreMemoryHooks()
    memory_id = await hooks.store_health_check(
        session_id="session-876",
        result=dict(_SIGNIFICANT_RESULT),
        state={"check_scope": "full", "query": "system health", "status": "completed"},
    )

    assert memory_id == _FAKE_MEMORY_ID
    memory = captured["memory"]
    # STATE dimension: valid enum value, mapped — never the domain literal.
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
    # DOMAIN dimension: the "health assessment delivered" signal stays here.
    assert memory.event_type == "health_check_completed"
    assert memory.agent_name == "health_score"


@pytest.mark.asyncio
async def test_failed_check_maps_outcome_type_to_failure(captured):
    """Defensive path: contribute_to_memory gates on status != 'failed', but if a
    failed state ever reaches the store it must not be mislabeled 'success'."""
    from src.agents.health_score.memory_hooks import HealthScoreMemoryHooks

    hooks = HealthScoreMemoryHooks()
    await hooks.store_health_check(
        session_id="session-876-failed",
        result=dict(_SIGNIFICANT_RESULT),
        state={"check_scope": "full", "query": "system health", "status": "failed"},
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
