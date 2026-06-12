"""#883 PR A faithful integration: stop the silent data loss.

Red-first proof for issue #883 PR A (follow-up to #876 / het #873 /
causal_impact #788/#785). Before the fix, five write paths and two read paths
silently dropped every row (each failure swallowed by a broad ``except`` into a
``logger.warning`` / ``logger.error``):

  * gap_analyzer          DOUBLE bug: ``event_type="gap_analysis_completed"``
                          missing from ``memory_event_type`` until
                          database/migrations/071 (22P02 FIRST), then
                          ``outcome_type="gap_analysis_delivered"`` invalid
                          against ``memory_outcome_type`` (22P02 SECOND). The
                          three event_type-FILTERED reads (_get_episodic_context
                          / get_historical_roi_data / get_opportunity_benchmarks)
                          22P02'd server-side too (the migration-046 read-path
                          lesson).
  * orchestrator          DOUBLE bug (latent — hooks unwired until PR B):
                          ``event_type="orchestration_completed"`` missing
                          pre-071, then ``outcome_type="response_delivered"``.
  * prediction_synthesizer ``outcome_type="prediction_delivered"`` — sole
                          blocker (event_type ``prediction_completed`` IS valid
                          via migration 065); wired via agent.py.
  * resource_optimizer    ``outcome_type="optimization_delivered"`` on the
                          episodic write (event_type ``optimization_completed``
                          valid via mem 020), AND
                          ``procedure_type="optimization_pattern"`` invalid
                          against ``procedure_type`` — the 22P02 fires at the
                          dedup ``find_relevant_procedures`` RPC (enum-typed
                          ``filter_type``) BEFORE the insert. The pattern READ
                          (`_get_optimization_patterns`) was doubly dead: it
                          imported ``ProceduralSearchFilters`` /
                          ``search_procedures_by_text`` which never existed in
                          src/memory/procedural_memory.py (ImportError -> []).
  * cognitive reflector   ``signal_type="outcome_success"/"outcome_partial"``
                          invalid against ``learning_signal_type`` -> 22P02 on
                          every learning-signal insert (latent: CognitiveService
                          has no prod consumer today).
  * SignalCollector       ``signal_type="dspy_signal"`` — same enum, same 22P02.

These tests drive the REAL store/read paths (real Supabase, real embedding
service — no mocks) and pin the #876 convention: DOMAIN label in
``event_type``+``agent_name`` (extended additively via migration when missing),
generic STATE in ``outcome_type`` (map, never extend), and learning signals
mapped onto the existing ``learning_signal_type`` values with the domain label
preserved in ``signal_details``.

Each test inserts uniquely-marked rows and deletes them afterwards
(non-polluting). Gated like the other faithful real-DB tests; run with the
shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_agent_memory_persistence_883.py'
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
        reason="faithful real-DB memory-persistence test; set E2I_DB_INTEGRATION=1 + creds in .env",
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
    # memory_event_type enum server-side — pre-071 this raised 22P02 for the
    # gap_analyzer / orchestrator values.
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


def _cleanup_episodic(memory_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("memory_id", memory_id).execute()


def _cleanup_episodic_by_session(session_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("session_id", session_id).execute()


# =============================================================================
# gap_analyzer — DOUBLE bug (event_type pre-071, then outcome_type) + 3 reads
# =============================================================================

_GAP_QUERY = "Find TRx opportunity gaps and ROI for remibrutinib by region segment"


async def _store_gap_row(session_id: str, marker: str) -> str | None:
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    hooks = GapAnalyzerMemoryHooks()
    result = {
        "prioritized_opportunities": [{"opportunity_id": marker}],
        "total_addressable_value": 1_250_000,
        "quick_wins": [{"opportunity_id": marker}],
        "strategic_bets": [],
        "confidence": 0.82,
        "executive_summary": f"TRx gap concentrated in northeast region ({marker})",
        "key_insights": ["northeast underperforms benchmark by 12%"],
    }
    state = {
        "query": _GAP_QUERY,
        "brand": "remibrutinib",
        "metrics": ["trx_rate"],
        "segments": ["region"],
        "status": "completed",
    }
    return await hooks.store_gap_analysis(
        session_id=session_id,
        result=result,
        state=state,
        region="northeast",
    )


@pytest.mark.asyncio
async def test_gap_analyzer_store_gap_analysis_persists_episodic_row():
    """TWO-STAGE red: pre-071 the memory_event_type enum rejects
    'gap_analysis_completed' (22P02 first); post-071/pre-remap the
    memory_outcome_type enum rejects 'gap_analysis_delivered' (22P02 second)."""
    session_id = str(uuid.uuid4())
    marker = f"883-gap-{uuid.uuid4()}"

    memory_id = await _store_gap_row(session_id, marker)
    assert memory_id, (
        "store_gap_analysis returned None — the episodic write swallowed an error "
        "(#883: pre-071 the memory_event_type enum rejected 'gap_analysis_completed'; "
        "post-071/pre-remap the memory_outcome_type enum rejected "
        "'gap_analysis_delivered'; see the captured warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="gap_analysis_completed",
            expected_agent="gap_analyzer",
            expected_outcome="success",
        )
    finally:
        _cleanup_episodic(memory_id)


@pytest.mark.asyncio
async def test_gap_analyzer_event_type_read_filters_return_written_row():
    """The three event_type-filtered reads (memory_hooks _get_episodic_context /
    get_historical_roi_data / get_opportunity_benchmarks) must return a
    just-written row: each binds 'gap_analysis_completed' into the
    search_episodic_memory RPC's enum-typed filter_event_type parameter, which
    22P02'd server-side pre-071 (the migration-046 read-path lesson).

    NOTE: ``_get_episodic_context`` is called WITHOUT brand/metrics/segments —
    its client-side post-filter reads ``raw_content`` which the RPC does not
    return (pre-existing, out of #883 PR A scope); the enum read path under
    test here is the RPC filter itself.
    """
    from src.agents.gap_analyzer.memory_hooks import GapAnalyzerMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"883-gap-read-{uuid.uuid4()}"

    memory_id = await _store_gap_row(session_id, marker)
    assert memory_id, "store_gap_analysis failed — cannot exercise the read filters"

    hooks = GapAnalyzerMemoryHooks()
    try:
        episodic = await hooks._get_episodic_context(query=_GAP_QUERY)
        assert memory_id in [m.get("memory_id") for m in episodic], (
            "_get_episodic_context (filter_event_type='gap_analysis_completed') "
            "did not return the just-written row"
        )

        roi_rows = await hooks.get_historical_roi_data(brand="remibrutinib", metric="trx_rate")
        assert memory_id in [m.get("memory_id") for m in roi_rows], (
            "get_historical_roi_data (filter_event_type='gap_analysis_completed') "
            "did not return the just-written row"
        )

        benchmarks = await hooks.get_opportunity_benchmarks(segment="region", metric="trx_rate")
        assert memory_id in [m.get("memory_id") for m in benchmarks], (
            "get_opportunity_benchmarks (filter_event_type='gap_analysis_completed') "
            "did not return the just-written row"
        )
    finally:
        _cleanup_episodic(memory_id)


# =============================================================================
# prediction_synthesizer — outcome_type sole blocker (event_type valid via 065)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_outcome"),
    [
        ("completed", "success"),
        # "degraded" (#438): 3-5 of 5 context deps failed but the prediction is
        # honest — a partial success, NOT a failure (and the agent.py gate lets
        # degraded runs through to the memory write).
        ("degraded", "partial_success"),
    ],
)
async def test_prediction_synthesizer_store_prediction_persists_episodic_row(
    status, expected_outcome
):
    """Red before fix: memory_outcome_type rejects 'prediction_delivered' (22P02).
    event_type 'prediction_completed' is valid (migration 065), so the misplaced
    outcome literal was the SOLE blocker killing 100% of wired episodic writes."""
    from src.agents.prediction_synthesizer.memory_hooks import (
        PredictionSynthesizerMemoryHooks,
    )

    hooks = PredictionSynthesizerMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"883-pred-{uuid.uuid4()}"
    result = {
        "ensemble_prediction": {
            "point_estimate": 0.731,
            "prediction_interval_lower": 0.65,
            "prediction_interval_upper": 0.81,
            "confidence": 0.88,
            "model_agreement": 0.92,
            "ensemble_method": "weighted_average",
        },
        "models_succeeded": 2,
        "models_failed": 0,
        "total_latency_ms": 412,
    }
    # entity_type deliberately NOT "hcp": the hook maps an hcp entity_id into
    # e2i_refs.hcp_id, which is FK-constrained to hcp_profiles — a synthetic
    # test id would 23503 for reasons unrelated to #883.
    state = {
        "entity_id": f"TERR-{marker}",
        "entity_type": "territory",
        "prediction_target": "rx_conversion",
        "time_horizon": "30d",
        "query": f"Predict rx conversion for territory ({marker})",
        "status": status,
    }

    memory_id = await hooks.store_prediction(
        session_id=session_id,
        result=result,
        state=state,
    )
    assert memory_id, (
        "store_prediction returned None — the episodic write swallowed an error "
        "(#883: memory_outcome_type enum rejected 'prediction_delivered'; see the "
        "captured 'Failed to store prediction in episodic memory' warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="prediction_completed",
            expected_agent="prediction_synthesizer",
            expected_outcome=expected_outcome,
        )
    finally:
        _cleanup_episodic(memory_id)


# =============================================================================
# resource_optimizer — episodic outcome_type + procedural pattern round-trip
# =============================================================================


@pytest.mark.asyncio
async def test_resource_optimizer_store_optimization_persists_episodic_row():
    """Red before fix: memory_outcome_type rejects 'optimization_delivered'
    (22P02). event_type 'optimization_completed' is valid (mem 020)."""
    from src.agents.resource_optimizer.memory_hooks import ResourceOptimizerMemoryHooks

    hooks = ResourceOptimizerMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"883-ro-{uuid.uuid4()}"
    result = {
        "objective_value": 1532.5,
        "projected_roi": 1.42,
        "projected_total_outcome": 20100.0,
        "optimal_allocations": [
            {"entity_type": "territory", "change": 12.0, "change_percentage": 15.0}
        ],
        "solver_status": "optimal",
        "solve_time_ms": 240,
        "recommendations": [f"shift field effort to northeast ({marker})"],
    }
    state = {
        "resource_type": "field_force",
        "objective": "trx_uplift",
        "solver_type": "linear",
        "query": f"Optimize field force allocation ({marker})",
        "status": "completed",
    }

    memory_id = await hooks.store_optimization(
        session_id=session_id,
        result=result,
        state=state,
    )
    assert memory_id, (
        "store_optimization returned None — the episodic write swallowed an error "
        "(#883: memory_outcome_type enum rejected 'optimization_delivered'; see the "
        "captured 'Failed to store optimization in episodic memory' warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="optimization_completed",
            expected_agent="resource_optimizer",
            expected_outcome="success",
        )
    finally:
        _cleanup_episodic(memory_id)


@pytest.mark.asyncio
async def test_resource_optimizer_optimization_pattern_round_trip():
    """Red before fix (WRITE): procedure_type 'optimization_pattern' is not a
    procedure_type enum value — the 22P02 fires at the dedup
    find_relevant_procedures RPC (enum-typed filter_type) BEFORE the insert is
    even reached, swallowed into None. Red before fix (READ):
    _get_optimization_patterns imported ProceduralSearchFilters /
    search_procedures_by_text which do not exist (ImportError -> [] forever).
    Green: the pattern lands with the EXISTING enum member 'optimization'
    (hpo_pattern_memory precedent) and is read back via the real
    find_relevant_procedures_by_text API."""
    from src.agents.resource_optimizer.memory_hooks import ResourceOptimizerMemoryHooks
    from src.memory.episodic_memory import get_supabase_client

    hooks = ResourceOptimizerMemoryHooks()
    session_id = str(uuid.uuid4())
    # Unique objective so the >0.9-similarity dedup can never collapse this
    # pattern into a previous test run's row.
    objective = f"trx_uplift_{uuid.uuid4().hex[:8]}"
    result = {
        "solver_status": "optimal",
        "projected_roi": 1.55,
        "solve_time_ms": 199,
        "optimal_allocations": [
            {"entity_type": "territory", "change": 14.0, "change_percentage": 18.0},
            {"entity_type": "hcp_segment", "change": -6.0, "change_percentage": -12.0},
        ],
    }
    state = {
        "resource_type": "field_force",
        "objective": objective,
        "solver_type": "linear",
        "constraints": [{"constraint_type": "budget"}, {"constraint_type": "capacity"}],
        "status": "completed",
    }

    pattern_id = await hooks.store_optimization_pattern(
        session_id=session_id,
        result=result,
        state=state,
    )
    assert pattern_id, (
        "store_optimization_pattern returned None — the procedural write swallowed "
        "an error (#883: the procedure_type enum rejected 'optimization_pattern' at "
        "the dedup find_relevant_procedures RPC before the insert; see the captured "
        "'Failed to store optimization pattern' warning above)"
    )
    try:
        client = get_supabase_client()
        rows = (
            client.table("procedural_memories")
            .select("procedure_id, procedure_name, procedure_type")
            .eq("procedure_id", pattern_id)
            .execute()
        ).data or []
        assert len(rows) == 1, f"expected exactly one procedural row for {pattern_id}"
        assert rows[0]["procedure_type"] == "optimization"
        assert rows[0]["procedure_name"].startswith("optimization_pattern_")

        # Read path: the repaired _get_optimization_patterns must return the
        # just-written pattern through the REAL find_relevant_procedures RPC
        # (enum-typed filter_type='optimization').
        patterns = await hooks._get_optimization_patterns(
            resource_type="field_force",
            objective=objective,
            constraints=state["constraints"],
        )
        assert str(pattern_id) in [str(p.get("procedure_id")) for p in patterns], (
            "_get_optimization_patterns did not return the just-written pattern "
            "(pre-fix: dead ProceduralSearchFilters import -> ImportError -> [])"
        )
    finally:
        get_supabase_client().table("procedural_memories").delete().eq(
            "procedure_id", pattern_id
        ).execute()


# =============================================================================
# orchestrator — DOUBLE bug, LATENT (hooks unwired until PR B): direct hook call
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_outcome"),
    [
        ("completed", "success"),
        # OrchestratorState.status includes 'partial_success' — preserve it.
        ("partial_success", "partial_success"),
    ],
)
async def test_orchestrator_store_orchestration_persists_episodic_row(status, expected_outcome):
    """TWO-STAGE red: pre-071 the memory_event_type enum rejects
    'orchestration_completed' (22P02 first); post-071/pre-remap the
    memory_outcome_type enum rejects 'response_delivered' (22P02 second).
    LATENT site — the hooks are unwired today (wiring lands in PR B), proven
    via direct hook call per the #876 pattern."""
    from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks

    hooks = OrchestratorMemoryHooks()
    session_id = str(uuid.uuid4())
    marker = f"883-orch-{uuid.uuid4()}"
    query = f"Why did Remibrutinib TRx drop in the midwest? ({marker})"
    result = {
        "query": query,
        "intent_classified": "causal_analysis",
        "agents_dispatched": ["causal_impact"],
        "response_text": f"TRx drop attributable to access changes ({marker})",
        "response_confidence": 0.84,
        "total_latency_ms": 2150,
        "status": status,
    }

    memory_id = await hooks.store_orchestration(
        session_id=session_id,
        result=result,
        brand="remibrutinib",
        region="midwest",
    )
    assert memory_id, (
        "store_orchestration returned None — the episodic write swallowed an error "
        "(#883: pre-071 the memory_event_type enum rejected 'orchestration_completed'; "
        "post-071/pre-remap the memory_outcome_type enum rejected "
        "'response_delivered'; see the captured warning above)"
    )
    try:
        _assert_persisted_and_readable(
            memory_id=memory_id,
            session_id=session_id,
            expected_event_type="orchestration_completed",
            expected_agent="orchestrator",
            expected_outcome=expected_outcome,
        )
        # Read filter (:215): _get_episodic_context binds
        # 'orchestration_completed' into the RPC's enum-typed filter_event_type.
        episodic = await hooks._get_episodic_context(query=query)
        assert memory_id in [m.get("memory_id") for m in episodic], (
            "_get_episodic_context (filter_event_type='orchestration_completed') "
            "did not return the just-written row"
        )
    finally:
        _cleanup_episodic(memory_id)


# =============================================================================
# learning signals — reflector (cognitive_integration) + SignalCollector (rag)
# =============================================================================


def _create_cycle(session_id: str, user_query: str) -> str:
    """Create a real cognitive_cycles parent row (the schema-SSOT declares
    learning_signals.cycle_id as an FK to it)."""
    from src.memory.episodic_memory import get_supabase_client

    cycle_id = str(uuid.uuid4())
    get_supabase_client().table("cognitive_cycles").insert(
        {"cycle_id": cycle_id, "session_id": session_id, "user_query": user_query}
    ).execute()
    return cycle_id


def _delete_cycle(cycle_id: str) -> None:
    """Delete the signals explicitly, then the cycle — the LIVE DB does not
    enforce the SSOT's ON DELETE CASCADE on learning_signals.cycle_id (verified
    2026-06-12: signal rows survive their cycle's deletion), so relying on the
    cascade would leak test rows."""
    from src.memory.episodic_memory import get_supabase_client

    client = get_supabase_client()
    client.table("learning_signals").delete().eq("cycle_id", cycle_id).execute()
    client.table("cognitive_cycles").delete().eq("cycle_id", cycle_id).execute()


def _signals_for_cycle(cycle_id: str) -> list:
    from src.memory.episodic_memory import get_supabase_client

    return (
        get_supabase_client()
        .table("learning_signals")
        .select("*")
        .eq("cycle_id", cycle_id)
        .execute()
    ).data or []


def _signal_details(signal: dict) -> dict:
    """record_learning_signal json.dumps's signal_details, so the JSONB column
    holds a JSON-string scalar (pre-existing writer behavior, not #883 scope) —
    parse it back to a dict for assertions."""
    import json

    details = signal["signal_details"]
    if isinstance(details, str):
        details = json.loads(details)
    assert isinstance(details, dict)
    return details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("confidence", "domain_signal"),
    [
        (0.9, "outcome_success"),
        # 0.6 < confidence <= 0.7 was 'outcome_partial' — still a POSITIVE
        # direction signal (the reflector only learns from cycles above its
        # 0.6 worth-remembering bar); mapping it to implicit_negative would
        # invert the signal.
        (0.65, "outcome_partial"),
    ],
)
async def test_reflector_records_learning_signal(confidence, domain_signal):
    """Red before fix: learning_signal_type rejects 'outcome_success' /
    'outcome_partial' (22P02 swallowed by the reflector's broad except ->
    zero learning signals ever persisted). Green: mapped onto the EXISTING
    enum member 'implicit_positive' (machine-derived, positive direction) with
    the exact grade in signal_value=confidence and the original domain label
    in signal_details. LATENT site (CognitiveService has no prod consumer);
    proven via direct call per the #876 pattern.

    _store_to_graphiti is stubbed out (codex R2): it runs BEFORE the
    learning-signal write, has its own swallow, and is NOT the persistence
    under proof here — left real it would seed uncleaned FalkorDB graph
    episodes (or add external-service flake) on every run. The
    learning_signals + episodic writes stay fully real."""
    from unittest.mock import AsyncMock, patch

    from src.memory.cognitive_integration import CognitiveService

    session_id = str(uuid.uuid4())
    marker = f"883-reflector-{uuid.uuid4()}"
    query = f"What drove Fabhalta NBRx growth last quarter? ({marker})"
    cycle_id = _create_cycle(session_id, query)

    try:
        service = CognitiveService()
        with patch.object(service, "_store_to_graphiti", new_callable=AsyncMock):
            await service._run_reflector(
                session_id=session_id,
                cycle_id=cycle_id,
                query=query,
                query_type="causal",
                response=f"NBRx growth driven by new-writer adoption ({marker})",
                confidence=confidence,
                evidence=[{"source": "episodic", "id": marker}],
                agent_used="orchestrator",
            )

        signals = _signals_for_cycle(cycle_id)
        assert len(signals) == 1, (
            "no learning_signals row landed — the reflector swallowed an error "
            "(#883: learning_signal_type enum rejected "
            f"'{domain_signal}'; see the captured 'Reflector phase failed' "
            "error above)"
        )
        signal = signals[0]
        assert signal["signal_type"] == "implicit_positive"
        assert signal["signal_value"] == pytest.approx(confidence)
        # The domain label survives in signal_details (map, don't extend).
        assert _signal_details(signal).get("domain_signal") == domain_signal
    finally:
        _delete_cycle(cycle_id)  # also deletes the signals explicitly
        _cleanup_episodic_by_session(session_id)


@pytest.mark.asyncio
async def test_signal_collector_records_dspy_signal():
    """Red before fix: learning_signal_type rejects 'dspy_signal' (22P02
    swallowed into the pending-retry queue -> nothing ever lands). Green:
    mapped onto the EXISTING enum member 'rating' (a graded score — the DSPy
    metric grades the signature execution) with signal_value=metric and the
    domain label in signal_details."""
    from src.rag.cognitive_backends import SignalCollector

    session_id = str(uuid.uuid4())
    marker = f"883-dspy-{uuid.uuid4()}"
    cycle_id = _create_cycle(session_id, f"dspy signal collection ({marker})")

    try:
        collector = SignalCollector()
        await collector.collect(
            [
                {
                    "signature_name": "AgentRoutingSignature",
                    "input": f"route query ({marker})",
                    "output": "causal_impact",
                    "metric": 0.87,
                    "cycle_id": cycle_id,
                }
            ]
        )

        signals = _signals_for_cycle(cycle_id)
        assert len(signals) == 1, (
            "no learning_signals row landed — SignalCollector swallowed an error "
            "(#883: learning_signal_type enum rejected 'dspy_signal'; see the "
            "captured 'Failed to collect signal' warning above)"
        )
        signal = signals[0]
        assert signal["signal_type"] == "rating"
        assert signal["signal_value"] == pytest.approx(0.87)
        assert signal["dspy_metric_value"] == pytest.approx(0.87)
        assert signal["is_training_example"] is True
        assert _signal_details(signal).get("domain_signal") == "dspy_signal"
    finally:
        _delete_cycle(cycle_id)
        _cleanup_episodic_by_session(session_id)
