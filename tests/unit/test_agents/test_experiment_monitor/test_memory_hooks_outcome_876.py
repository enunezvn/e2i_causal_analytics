"""#876 focused unit tests: experiment_monitor episodic ``outcome_type`` must be VALID.

The DB ``memory_outcome_type`` enum is a generic outcome STATE
(success / partial_success / failure / pending / escalated). Before #876, both
of experiment_monitor's episodic stores passed domain literals —
``store_alert`` used ``"alert_generated"`` and ``store_monitoring_check`` used
``"monitoring_completed"`` — which Postgres rejected (22P02) and the hooks
swallowed, so no experiment_monitor episodic row was ever written.
(experiment_monitor had a DOUBLE bug: its event_types
``experiment_alert_generated`` / ``experiment_monitoring_completed`` were also
missing from ``memory_event_type`` until database/migrations/070.) The decision
(mirroring causal_impact #788 and het #873) is to MAP to the state enum,
keeping the domain signal in ``event_type`` + ``agent_name`` +
``event_subtype`` (e.g. ``alert_srm``).

NOTE on semantics: ``outcome_type`` reflects the MONITORING operation's
outcome, not the health of the monitored experiments — a critical alert raised
by a successfully-completed check is a ``success``; the alert's own severity
lives in ``event_subtype`` / ``raw_content`` / ``importance_score``.

These tests capture the ``EpisodicMemoryInput`` the hooks build (no DB; the
faithful persistence proof lives in
``tests/integration/test_agent_episodic_outcome_876.py``).
"""

import pytest

# Clearly-fake sentinel for the captured-input stub (unit scope only; the real-DB
# integration test is the persistence proof).
_FAKE_MEMORY_ID = "00000000-0000-0000-0000-000000000876"

_VALID_OUTCOME_TYPES = {"success", "partial_success", "failure", "pending", "escalated"}

_ALERT = {
    "alert_id": "alert-876",
    "alert_type": "srm",
    "severity": "critical",
    "experiment_id": "exp-876",
    "experiment_name": "remi_engagement_ab",
    "message": "Sample ratio mismatch detected",
}

# Passes _is_significant_check (critical_count > 0).
_SIGNIFICANT_CHECK = {
    "experiments_checked": 7,
    "healthy_count": 5,
    "warning_count": 1,
    "critical_count": 1,
    "alerts": [{"alert_type": "srm", "severity": "critical"}],
    "check_latency_ms": 850,
    "monitor_summary": "1 critical experiment",
}


@pytest.fixture()
def captured(monkeypatch):
    """Capture the EpisodicMemoryInput the stores hand to the insert."""
    box = {}

    async def _capture(memory, text_to_embed=None, session_id=None, cycle_id=None):
        box["memory"] = memory
        box["session_id"] = session_id
        return _FAKE_MEMORY_ID

    # Both stores import this symbol inside the method body, so patching the
    # source module attribute intercepts the call.
    monkeypatch.setattr("src.memory.episodic_memory.insert_episodic_memory_with_text", _capture)
    return box


@pytest.mark.asyncio
async def test_store_alert_maps_outcome_type_to_success(captured):
    """A critical ALERT from a completed monitoring run is operation success."""
    from src.agents.experiment_monitor.memory_hooks import ExperimentMonitorMemoryHooks

    hooks = ExperimentMonitorMemoryHooks()
    memory_id = await hooks.store_alert(
        session_id="session-876",
        alert=dict(_ALERT),
        state={"status": "completed"},
    )

    assert memory_id == _FAKE_MEMORY_ID
    memory = captured["memory"]
    # STATE dimension: valid enum value, mapped — never the domain literal.
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
    # DOMAIN dimension: the "alert generated" signal stays here.
    assert memory.event_type == "experiment_alert_generated"
    assert memory.event_subtype == "alert_srm"
    assert memory.agent_name == "experiment_monitor"


@pytest.mark.asyncio
async def test_store_alert_failed_state_maps_outcome_type_to_failure(captured):
    """Defensive path: contribute_to_memory gates on status != 'failed', but if a
    failed state ever reaches the store it must not be mislabeled 'success'."""
    from src.agents.experiment_monitor.memory_hooks import ExperimentMonitorMemoryHooks

    hooks = ExperimentMonitorMemoryHooks()
    await hooks.store_alert(
        session_id="session-876-failed",
        alert=dict(_ALERT),
        state={"status": "failed"},
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES


@pytest.mark.asyncio
async def test_store_monitoring_check_maps_outcome_type_to_success(captured):
    from src.agents.experiment_monitor.memory_hooks import ExperimentMonitorMemoryHooks

    hooks = ExperimentMonitorMemoryHooks()
    memory_id = await hooks.store_monitoring_check(
        session_id="session-876",
        result=dict(_SIGNIFICANT_CHECK),
        state={"status": "completed", "query": "monitor experiments"},
    )

    assert memory_id == _FAKE_MEMORY_ID
    memory = captured["memory"]
    assert memory.outcome_type == "success"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
    assert memory.event_type == "experiment_monitoring_completed"
    assert memory.agent_name == "experiment_monitor"


@pytest.mark.asyncio
async def test_store_monitoring_check_failed_state_maps_outcome_type_to_failure(captured):
    from src.agents.experiment_monitor.memory_hooks import ExperimentMonitorMemoryHooks

    hooks = ExperimentMonitorMemoryHooks()
    await hooks.store_monitoring_check(
        session_id="session-876-failed",
        result=dict(_SIGNIFICANT_CHECK),
        state={"status": "failed", "query": "monitor experiments"},
    )

    memory = captured["memory"]
    assert memory.outcome_type == "failure"
    assert memory.outcome_type in _VALID_OUTCOME_TYPES
