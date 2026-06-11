"""#876 faithful integration: four agents' episodic stores must PERSIST.

Red-first proof for issue #876 (follow-up to het #873 / causal_impact #788/#785).
Before the fix, four agents' memory hooks built their ``EpisodicMemoryInput`` with
domain-event strings in ``outcome_type`` — not values of the DB
``memory_outcome_type`` enum (success / partial_success / failure / pending /
escalated) — so every insert raised 22P02, each hook's ``except Exception``
swallowed it into a ``logger.warning``, and the agents' episodic memories were
silently NEVER written (fail-open-by-logging):

  * explainer           ``outcome_type="explanation_delivered"``
  * health_score        ``outcome_type="health_assessment_delivered"``
  * experiment_monitor  ``outcome_type="alert_generated"`` (store_alert) and
                        ``outcome_type="monitoring_completed"`` (store_monitoring_check)
  * tool_composer       ``outcome_type="composition_delivered"``

health_score and experiment_monitor carried a DOUBLE bug: their ``event_type``
values (health_check_completed / experiment_alert_generated /
experiment_monitoring_completed) were also missing from ``memory_event_type``
until database/migrations/070, so their inserts 22P02'd on event_type BEFORE
outcome_type even mattered — and their event_type-filtered episodic SEARCHES
(get_score_history / get_srm_history / _get_alert_history / ...) hit the same
invalid-enum comparison on the read side (the migration-046 het lesson). Each
test here re-reads the row through an ``event_type``-filtered query so the
read-side enum comparison is exercised against the live DB too.

These tests drive the REAL store paths (real Supabase ``episodic_memories``
table, real embedding service — no mocks) and pin that the happy path actually
lands a row with the DOMAIN signal in ``event_type``+``agent_name`` and the
generic STATE in ``outcome_type`` (#876 convention: map, don't extend).

Each test inserts a uniquely-marked row and deletes it afterwards
(non-polluting). Gated like the other faithful real-DB tests; run with the
shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_agent_episodic_outcome_876.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB episodic test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _assert_persisted_and_readable(
    memory_id: str,
    session_id: str,
    expected_event_type: str,
    expected_agent: str,
    expected_outcome: str,
) -> None:
    """Row exists with the expected DOMAIN/STATE split AND is reachable through
    an event_type-filtered read (the 046 read-path lesson: an enum value missing
    from memory_event_type 22P02s the comparison itself, poisoning searches)."""
    from src.memory.episodic_memory import get_supabase_client

    client = get_supabase_client()
    resp = (
        client.table("episodic_memories")
        .select("memory_id, session_id, event_type, outcome_type, agent_name, description")
        .eq("memory_id", memory_id)
        .execute()
    )
    rows = resp.data or []
    assert len(rows) == 1, f"expected exactly one row for {memory_id}, got {len(rows)}"
    row = rows[0]
    assert str(row["session_id"]) == session_id
    # The DOMAIN signal lives in event_type + agent_name (#876 convention).
    assert row["event_type"] == expected_event_type
    assert row["agent_name"] == expected_agent
    # The STATE signal: generic memory_outcome_type enum value (map, don't extend).
    assert row["outcome_type"] == expected_outcome

    # Read path: filtering BY the event_type value compares against the live
    # memory_event_type enum server-side — pre-070 this raised 22P02 for the
    # health_score / experiment_monitor values (same class of failure as the
    # hooks' search_episodic_by_text filter_event_type RPC parameter).
    read = (
        client.table("episodic_memories")
        .select("memory_id")
        .eq("event_type", expected_event_type)
        .eq("memory_id", memory_id)
        .execute()
    )
    assert len(read.data or []) == 1, (
        f"event_type-filtered read for '{expected_event_type}' did not return the row"
    )


def _cleanup(memory_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("memory_id", memory_id).execute()


@pytest.mark.asyncio
async def test_explainer_store_explanation_persists_episodic_row():
    """Red before fix: memory_outcome_type rejects 'explanation_delivered' (22P02)."""
    from src.agents.explainer.memory_hooks import ExplanationMemoryHooks

    hooks = ExplanationMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"876-explainer-{uuid.uuid4()}"
    explanation = {
        "query": "Why did Remibrutinib TRx rise in the northeast?",
        "executive_summary": f"TRx uplift driven by HCP engagement ({marker})",
        "detailed_explanation": "Engagement-led uplift concentrated in high-decile HCPs.",
        "insights": [{"theme": "engagement", "impact": "positive"}],
        "audience": "analyst",
        "output_format": "narrative",
    }

    memory_id = await hooks.store_explanation(
        session_id=session_id,
        explanation=explanation,
        brand="remibrutinib",
        region="northeast",
    )
    assert memory_id, (
        "store_explanation returned None — the episodic write swallowed an error "
        "(#876: memory_outcome_type enum rejected 'explanation_delivered'; see the "
        "captured 'Failed to store explanation in episodic memory' warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="explanation_generated",
            expected_agent="explainer",
            expected_outcome="success",
        )
    finally:
        _cleanup(memory_id)


@pytest.mark.asyncio
async def test_health_score_store_health_check_persists_episodic_row():
    """DOUBLE bug: event_type 'health_check_completed' missing pre-070 (22P02 first),
    then memory_outcome_type rejects 'health_assessment_delivered' (22P02 second)."""
    from src.agents.health_score.memory_hooks import HealthScoreMemoryHooks

    hooks = HealthScoreMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"876-health-{uuid.uuid4()}"
    # Payload must pass _is_significant_health_event: critical issues present.
    result = {
        "overall_health_score": 52.5,
        "health_grade": "D",
        "critical_issues": [f"database connection pool exhausted ({marker})"],
        "warnings": ["model staleness above threshold"],
        "component_health_score": 40.0,
        "model_health_score": 55.0,
        "pipeline_health_score": 60.0,
        "agent_health_score": 58.0,
        "total_latency_ms": 1234,
    }
    state = {"check_scope": "full", "query": "system health", "status": "completed"}

    memory_id = await hooks.store_health_check(
        session_id=session_id,
        result=result,
        state=state,
    )
    assert memory_id, (
        "store_health_check returned None — the episodic write swallowed an error "
        "(#876: pre-070 the memory_event_type enum rejected 'health_check_completed'; "
        "post-070/pre-fix the memory_outcome_type enum rejected "
        "'health_assessment_delivered'; see the captured warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="health_check_completed",
            expected_agent="health_score",
            expected_outcome="success",
        )
    finally:
        _cleanup(memory_id)


@pytest.mark.asyncio
async def test_experiment_monitor_store_alert_persists_episodic_row():
    """DOUBLE bug: event_type 'experiment_alert_generated' missing pre-070,
    then memory_outcome_type rejects 'alert_generated'."""
    from src.agents.experiment_monitor.memory_hooks import ExperimentMonitorMemoryHooks

    hooks = ExperimentMonitorMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"876-em-alert-{uuid.uuid4()}"
    alert = {
        "alert_id": str(uuid.uuid4()),
        "alert_type": "srm",
        "severity": "critical",
        "experiment_id": str(uuid.uuid4()),
        "experiment_name": f"remi_engagement_ab ({marker})",
        "message": "Sample ratio mismatch detected: p < 0.001",
        "details": {"p_value": 0.0004},
        "recommended_action": "Pause assignment and audit randomization",
    }
    state = {"status": "completed", "query": "monitor experiments"}

    memory_id = await hooks.store_alert(
        session_id=session_id,
        alert=alert,
        state=state,
    )
    assert memory_id, (
        "store_alert returned None — the episodic write swallowed an error "
        "(#876: pre-070 the memory_event_type enum rejected "
        "'experiment_alert_generated'; post-070/pre-fix the memory_outcome_type "
        "enum rejected 'alert_generated'; see the captured warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="experiment_alert_generated",
            expected_agent="experiment_monitor",
            expected_outcome="success",
        )
    finally:
        _cleanup(memory_id)


@pytest.mark.asyncio
async def test_experiment_monitor_store_monitoring_check_persists_episodic_row():
    """DOUBLE bug: event_type 'experiment_monitoring_completed' missing pre-070,
    then memory_outcome_type rejects 'monitoring_completed'."""
    from src.agents.experiment_monitor.memory_hooks import ExperimentMonitorMemoryHooks

    hooks = ExperimentMonitorMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"876-em-check-{uuid.uuid4()}"
    # Payload must pass _is_significant_check: critical_count > 0.
    result = {
        "experiments_checked": 7,
        "healthy_count": 5,
        "warning_count": 1,
        "critical_count": 1,
        "alerts": [{"alert_type": "srm", "severity": "critical"}],
        "check_latency_ms": 850,
        "monitor_summary": f"1 critical experiment ({marker})",
    }
    state = {"status": "completed", "query": "monitor experiments"}

    memory_id = await hooks.store_monitoring_check(
        session_id=session_id,
        result=result,
        state=state,
    )
    assert memory_id, (
        "store_monitoring_check returned None — the episodic write swallowed an "
        "error (#876: pre-070 the memory_event_type enum rejected "
        "'experiment_monitoring_completed'; post-070/pre-fix the "
        "memory_outcome_type enum rejected 'monitoring_completed')"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="experiment_monitoring_completed",
            expected_agent="experiment_monitor",
            expected_outcome="success",
        )
    finally:
        _cleanup(memory_id)


@pytest.mark.asyncio
async def test_tool_composer_store_composition_persists_episodic_row():
    """Red before fix: memory_outcome_type rejects 'composition_delivered' (22P02)."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"876-composer-{uuid.uuid4()}"
    result = {
        "composition_id": str(uuid.uuid4()),
        "query": f"Compare Remibrutinib TRx uplift vs Fabhalta by region ({marker})",
        "decomposition": {
            "sub_questions": [{"q": "TRx uplift Remi"}, {"q": "TRx uplift Fabhalta"}]
        },
        "plan": {"steps": [{"tool_name": "kpi_query"}, {"tool_name": "causal_effect_estimator"}]},
        "execution": {"tools_executed": 2, "tools_succeeded": 2},
        "response": {"confidence": 0.91},
        "status": "success",
        "success": True,
        "total_duration_ms": 4200,
    }

    memory_id = await hooks.store_composition(
        session_id=session_id,
        result=result,
        brand="remibrutinib",
        region="northeast",
    )
    assert memory_id, (
        "store_composition returned None — the episodic write swallowed an error "
        "(#876: memory_outcome_type enum rejected 'composition_delivered'; see the "
        "captured 'Failed to store composition in episodic memory' warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="composition_completed",
            expected_agent="tool_composer",
            expected_outcome="success",
        )
    finally:
        _cleanup(memory_id)
