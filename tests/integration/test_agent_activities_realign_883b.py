"""#883 PR B §5 faithful integration: agent_activities schema realignment.

Both experiment_designer and drift_monitor memory hooks inserted payloads with
FIVE nonexistent columns (``agent_type`` / ``session_id`` / ``query_text`` /
``result_summary`` / ``metadata``) into ``agent_activities`` and omitted the
defaultless NOT NULL PK ``activity_id`` + NOT NULL ``activity_timestamp`` —
PGRST204/23502 on every call, swallowed into ``logger.warning -> None``
(schema SSOT: database/core/e2i_ml_complete_v3_schema.sql:608, re-verified on
the live docker DB this session). The READ paths filtered on the same
nonexistent columns, so they were equally dead.

Compounding (experiment_designer only): ``store_validity_threats`` constructed
``ValidityThreatRecord`` objects, made NO database call at all, yet
incremented and returned ``stored_count`` — a fabricated "N threats stored"
success.

These tests drive the REAL hook write/read paths against the live DB,
red-first:
  * RED  — store returns None (insert rejected, swallowed) / stored_count
           fabricated with zero rows / readers return [] forever.
  * GREEN — payload realigned to the REAL columns (generated activity_id,
           agent_name, activity_timestamp, activity_type, JSONB
           input_data/analysis_results), the row LANDS, the readers return
           it, and store_validity_threats' count equals rows actually
           persisted.

drift_monitor's hooks remain UNWIRED by design (stateless-by-design decision,
see the module docstring of src/agents/drift_monitor/memory_hooks.py) — the
direct-call proof here ensures the intentional placeholder is no longer a
guaranteed-PGRST204 landmine for any future wiring.

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_agent_activities_realign_883b.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB agent_activities test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _activities_for_session(session_id: str, agent_name: str) -> list:
    from src.memory.episodic_memory import get_supabase_client

    return (
        get_supabase_client()
        .table("agent_activities")
        .select(
            "activity_id, agent_name, agent_tier, activity_type, status, "
            "input_data, analysis_results, records_processed"
        )
        .eq("agent_name", agent_name)
        .contains("input_data", {"session_id": session_id})
        .execute()
    ).data or []


def _cleanup_activities(session_id: str, agent_name: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("agent_activities").delete().eq("agent_name", agent_name).contains(
        "input_data", {"session_id": session_id}
    ).execute()


_DESIGN_RESULT = {
    "design_type": "RCT",
    "design_rationale": "randomization feasible at HCP level",
    "randomization_unit": "hcp",
    "randomization_method": "stratified",
    "power_analysis": {"required_sample_size": 420, "achieved_power": 0.82},
    "duration_estimate_days": 60,
    "overall_validity_score": 0.78,
    "validity_confidence": "medium",
    "redesign_iterations": 1,
    "total_latency_ms": 1234,
    "warnings": [],
    "treatments": [{"name": "rep_visit_boost"}],
    "outcomes": [{"name": "trx_lift"}],
}


def _design_threats(marker: str) -> list:
    return [
        {
            "threat_type": "internal",
            "threat_name": f"selection_bias_{marker}",
            "severity": "high",
            "description": "non-random HCP opt-in",
            "mitigation_possible": True,
            "mitigation_strategy": "stratify by decile",
        },
        {
            "threat_type": "external",
            "threat_name": f"seasonality_{marker}",
            "severity": "medium",
            "description": "allergy-season TRx inflation",
            "mitigation_possible": True,
            "mitigation_strategy": "extend washout window",
        },
    ]


# =============================================================================
# experiment_designer — store_experiment_design / store_validity_threats / read
# =============================================================================


@pytest.mark.asyncio
async def test_store_experiment_design_lands_real_row():
    """RED: the insert carries agent_type/session_id/query_text/result_summary/
    metadata (none exist) and omits activity_id/activity_timestamp (NOT NULL)
    -> PGRST204 swallowed -> None. GREEN: a schema-correct row lands."""
    from src.agents.experiment_designer.memory_hooks import (
        ExperimentDesignerMemoryHooks,
    )

    hooks = ExperimentDesignerMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = uuid.uuid4().hex[:8]
    threats = _design_threats(marker)

    try:
        record_id = await hooks.store_experiment_design(
            session_id=session_id,
            result={**_DESIGN_RESULT, "validity_threats": threats},
            state={
                "business_question": f"Does rep frequency lift TRx? ({marker})",
                "constraints": {"budget": 100000},
            },
            brand="remibrutinib",
        )
        assert record_id, (
            "store_experiment_design returned None — the agent_activities insert "
            "was rejected and swallowed (#883 §5: 5 nonexistent columns + missing "
            "NOT NULL activity_id/activity_timestamp; see the captured warning)"
        )

        rows = _activities_for_session(session_id, "experiment_designer")
        assert len(rows) == 1, "expected exactly one experiment_design activity row"
        row = rows[0]
        assert row["activity_type"] == "experiment_design"
        assert row["agent_tier"] == "monitoring"
        assert row["status"] == "completed"
        assert marker in row["input_data"]["business_question"]
        analysis = row["analysis_results"]
        assert analysis["design_type"] == "RCT"
        assert analysis["overall_validity_score"] == pytest.approx(0.78)
        assert [t["threat_name"] for t in analysis["validity_threats"]] == [
            t["threat_name"] for t in threats
        ]
    finally:
        _cleanup_activities(session_id, "experiment_designer")


@pytest.mark.asyncio
async def test_store_validity_threats_count_is_persisted_not_fabricated():
    """RED: store_validity_threats makes NO DB call yet returns len(threats)
    — a fabricated success. GREEN: the count equals threats actually
    persisted in a landed batch row (and 0 when nothing landed)."""
    from src.agents.experiment_designer.memory_hooks import (
        ExperimentDesignerMemoryHooks,
    )
    from src.memory.episodic_memory import get_supabase_client

    hooks = ExperimentDesignerMemoryHooks()
    marker = uuid.uuid4().hex[:8]
    record_id = f"rec_{marker}"
    threats = _design_threats(marker)

    client = get_supabase_client()
    try:
        stored = await hooks.store_validity_threats(
            experiment_record_id=record_id,
            validity_threats=threats,
            design_type="RCT",
            business_question=f"Does rep frequency lift TRx? ({marker})",
        )

        rows = (
            client.table("agent_activities")
            .select("activity_id, activity_type, analysis_results, records_processed")
            .eq("agent_name", "experiment_designer")
            .eq("activity_type", "validity_threats")
            .contains("input_data", {"experiment_record_id": record_id})
            .execute()
        ).data or []

        persisted = sum(len(r["analysis_results"].get("validity_threats", [])) for r in rows)
        assert stored == persisted, (
            f"store_validity_threats reported {stored} but {persisted} threats are "
            "actually persisted — the count was fabricated (#883 §5: the method "
            "constructed records, made no DB call, and returned len(threats))"
        )
        assert stored == len(threats), "the realigned batch insert should have landed"
        assert rows and rows[0]["records_processed"] == len(threats)
    finally:
        client.table("agent_activities").delete().eq("agent_name", "experiment_designer").contains(
            "input_data", {"experiment_record_id": record_id}
        ).execute()


@pytest.mark.asyncio
async def test_get_similar_validity_threats_returns_written_threats():
    """RED: the reader filters on nonexistent columns (agent_type / metadata)
    -> PostgREST error swallowed -> [] forever. GREEN: it returns the threats
    from a just-stored design."""
    from src.agents.experiment_designer.memory_hooks import (
        ExperimentDesignerMemoryHooks,
    )

    hooks = ExperimentDesignerMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = uuid.uuid4().hex[:8]
    threats = _design_threats(marker)
    # A design_type unique to this run so older rows cannot satisfy the read.
    design_type = f"RCT_{marker}"

    try:
        record_id = await hooks.store_experiment_design(
            session_id=session_id,
            result={**_DESIGN_RESULT, "design_type": design_type, "validity_threats": threats},
            state={"business_question": f"threat read-path probe ({marker})", "constraints": {}},
        )
        assert record_id, "write failed — cannot exercise the read path"

        found = await hooks.get_similar_validity_threats(design_type=design_type, max_threats=10)
        names = {t.get("threat_name") for t in found}
        assert {t["threat_name"] for t in threats} <= names, (
            "get_similar_validity_threats did not return the just-written threats "
            "(pre-fix: .eq('agent_type', ...) + .contains('metadata', ...) query "
            "nonexistent columns -> swallowed -> [])"
        )
    finally:
        _cleanup_activities(session_id, "experiment_designer")


@pytest.mark.asyncio
async def test_experiment_designer_episodic_context_reader_round_trip():
    """The other previously-dead reader (_get_episodic_context) must surface a
    just-stored design for the same brand."""
    from src.agents.experiment_designer.memory_hooks import (
        ExperimentDesignerMemoryHooks,
    )

    hooks = ExperimentDesignerMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = uuid.uuid4().hex[:8]
    brand = f"remibrutinib_{marker}"
    question = f"Does rep frequency lift TRx for {brand}?"

    try:
        record_id = await hooks.store_experiment_design(
            session_id=session_id,
            result=dict(_DESIGN_RESULT),
            state={"business_question": question, "constraints": {}},
            brand=brand,
        )
        assert record_id, "write failed — cannot exercise the read path"

        records = await hooks._get_episodic_context(
            business_question=question, brand=brand, max_records=5
        )
        assert any(
            (r.get("analysis_results") or {}).get("record_id") == record_id for r in records
        ), "_get_episodic_context did not return the just-written design row"
    finally:
        _cleanup_activities(session_id, "experiment_designer")


# =============================================================================
# drift_monitor — store_drift_detection realignment (hooks stay UNWIRED)
# =============================================================================


@pytest.mark.asyncio
async def test_drift_monitor_store_drift_detection_lands_real_row():
    """Same §5 schema-mismatch family, proven by direct call — the hooks file
    is an affirmed-stateless intentional placeholder (no production caller),
    but a placeholder must not be a guaranteed-PGRST204 landmine."""
    from src.agents.drift_monitor.memory_hooks import DriftMonitorMemoryHooks

    hooks = DriftMonitorMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = uuid.uuid4().hex[:8]

    try:
        record_id = await hooks.store_drift_detection(
            session_id=session_id,
            result={
                "overall_drift_score": 0.42,
                "features_with_drift": [f"feat_{marker}"],
                "data_drift_results": [{"feature": f"feat_{marker}", "psi": 0.31}],
                "model_drift_results": [],
                "concept_drift_results": [],
                "alerts": [{"severity": "medium", "feature": f"feat_{marker}"}],
                "drift_summary": f"PSI drift on feat_{marker}",
                "recommended_actions": ["retrain"],
                "total_latency_ms": 950,
                "warnings": [],
            },
            state={
                "query": f"drift check ({marker})",
                "model_id": "m_883b",
                "features_to_monitor": [f"feat_{marker}"],
                "time_window": "7d",
                "brand": "remibrutinib",
            },
        )
        assert record_id, (
            "store_drift_detection returned None — the agent_activities insert was "
            "rejected and swallowed (#883 §5 schema mismatch)"
        )

        rows = _activities_for_session(session_id, "drift_monitor")
        assert len(rows) == 1
        row = rows[0]
        assert row["activity_type"] == "drift_detection"
        assert row["agent_tier"] == "monitoring"
        assert row["analysis_results"]["overall_drift_score"] == pytest.approx(0.42)
        assert row["analysis_results"]["max_severity"] == "medium"

        # The (placeholder) reader is realigned too: it must surface the row.
        records = await hooks._get_episodic_context(
            features=[f"feat_{marker}"], model_id="m_883b", max_records=5
        )
        assert any(
            (r.get("analysis_results") or {}).get("record_id") == record_id for r in records
        ), "_get_episodic_context did not return the just-written drift row"
    finally:
        _cleanup_activities(session_id, "drift_monitor")
