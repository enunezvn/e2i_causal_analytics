"""#883 PR B faithful integration: wire the unwired memory hooks (agent paths).

PR A (#883) proved each HOOK persists when called directly (enum remaps +
migration 071). PR B is about the CALLERS — these tests drive each agent's
REAL terminal flow (no direct hook calls) and assert the row lands in the
live DB, red-first:

  * orchestrator           agent.run() never imported memory_hooks (graph.py's
                           MemorySaver is an unrelated langgraph checkpointer;
                           CONTRACT_VALIDATION.md §10 marks memory integration
                           BLOCKING) -> zero episodic/working writes despite a
                           working hook. RED: no row for the run's session_id.
  * cohort_constructor     agent.run() (graph + direct modes) never called
                           contribute_to_memory -> the learning-loop readers
                           get_prior_cohorts / get_effective_rules_for_brand
                           had nothing to read, ever. RED: no episodic row.
  * experiment_designer    arun()/run() never called contribute_to_memory; the
                           hook itself ALSO wrote 5 nonexistent columns into
                           agent_activities (PGRST204 swallowed) and
                           store_validity_threats fabricated its stored_count
                           — those honesty fixes are proven in
                           test_experiment_designer_activities_883b.py; here
                           the AGENT path is proven end-to-end.
  * prediction_synthesizer PARTIAL: update_model_performance had no caller, so
                           the Redis key prediction_synthesizer:
                           model_performance:{target} was never written and
                           get_context().model_performance was permanently {}
                           in prod. RED: key absent after a successful
                           synthesize().

Each test inserts uniquely-marked rows keyed to a fresh session_id and deletes
them afterwards (non-polluting). Gated like the other faithful real-DB tests;
run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_agent_memory_wiring_883b.py'
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
        reason="faithful real-DB memory-wiring test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _episodic_rows_for_session(session_id: str, event_type: str) -> list:
    from src.memory.episodic_memory import get_supabase_client

    return (
        get_supabase_client()
        .table("episodic_memories")
        .select("memory_id, session_id, event_type, outcome_type, agent_name, description")
        .eq("session_id", session_id)
        .eq("event_type", event_type)
        .execute()
    ).data or []


def _cleanup_episodic_by_session(session_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("session_id", session_id).execute()


def _reset_redis_singletons() -> None:
    """The factories module caches an ASYNCIO Redis client bound to the first
    test's event loop; pytest-asyncio gives each test a fresh loop, so a
    Redis-touching test that runs second would see loop-closed errors. Reset
    the cached client + working-memory singleton so each test binds its own."""
    import src.memory.services.factories as factories
    from src.memory.working_memory import reset_working_memory

    factories._redis_client = None
    reset_working_memory()


# =============================================================================
# orchestrator — agent.run() must contribute (episodic + working) per turn
# =============================================================================


@pytest.mark.asyncio
async def test_orchestrator_run_lands_episodic_row():
    """RED before wiring: agent.run() completes but NO episodic row lands for
    the run's session_id — the hooks (proven persisting in PR A via direct
    call) simply have no caller. GREEN: one 'orchestration_completed' row,
    outcome mapped onto the generic state enum (post-071 / post-remap base)."""
    from src.agents.orchestrator.agent import OrchestratorAgent

    session_id = str(uuid.uuid4())
    marker = f"883b-orch-{uuid.uuid4().hex[:12]}"

    # allow_mock=True: the orchestrator's OWN flow (classify -> route ->
    # dispatch -> synthesize -> memory contribution) is fully real; only the
    # downstream specialist agents are the canned dispatcher scaffold — the
    # subject under test is the orchestrator's terminal memory wiring, not
    # the specialists.
    agent = OrchestratorAgent(allow_mock=True, enable_opik=False)
    try:
        result = await agent.run(
            {
                "query": f"Why did Remibrutinib TRx drop in the midwest? ({marker})",
                "session_id": session_id,
                "user_id": "883b-test",
            }
        )

        assert result["status"] in ("completed", "partial_success"), (
            f"orchestrator run did not complete: {result.get('status')}"
        )

        rows = _episodic_rows_for_session(session_id, "orchestration_completed")
        assert len(rows) == 1, (
            "agent.run() landed no 'orchestration_completed' episodic row for its "
            "session — the memory hooks have no caller (#883 PR B: orchestrator "
            "contribute_to_memory unwired; CONTRACT_VALIDATION.md §10 BLOCKING)"
        )
        row = rows[0]
        assert row["agent_name"] == "orchestrator"
        assert row["outcome_type"] in ("success", "partial_success")
        assert marker in (row["description"] or ""), (
            "episodic description should embed the run's query (store_orchestration "
            "builds it from result['query'])"
        )
    finally:
        _cleanup_episodic_by_session(session_id)


@pytest.mark.asyncio
async def test_orchestrator_run_stores_conversation_turn_when_redis_up():
    """The same single contribution covers the working-memory writes: the
    conversation turn becomes readable through the previously-dead reader
    get_conversation_history. Skipped honestly when Redis is unreachable
    (the episodic proof above is the hard gate; working memory is best-effort
    by design — its failure must never block the turn)."""
    from src.agents.orchestrator.agent import OrchestratorAgent
    from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks

    _reset_redis_singletons()
    probe = OrchestratorMemoryHooks()
    if not probe.working_memory:
        pytest.skip("working memory (Redis) not reachable in this environment")
    try:
        await probe.working_memory.get_messages("883b-probe", limit=1)
    except Exception:
        pytest.skip("working memory (Redis) not responding in this environment")

    session_id = str(uuid.uuid4())
    marker = f"883b-orch-wm-{uuid.uuid4().hex[:12]}"

    agent = OrchestratorAgent(allow_mock=True, enable_opik=False)
    try:
        result = await agent.run(
            {
                "query": f"Compare Fabhalta NBRx trends across regions ({marker})",
                "session_id": session_id,
            }
        )
        assert result["status"] in ("completed", "partial_success")

        history = await probe.get_conversation_history(session_id=session_id, limit=10)
        user_turns = [m for m in history if m.get("role") == "user"]
        assert any(marker in (m.get("content") or "") for m in user_turns), (
            "get_conversation_history (previously a dead reader) did not return "
            "the stored user turn for this session"
        )
        assert any(m.get("role") == "assistant" for m in history), (
            "the assistant response turn was not stored"
        )
    finally:
        _cleanup_episodic_by_session(session_id)


# =============================================================================
# cohort_constructor — agent.run() must contribute per construction
# =============================================================================


def _cohort_patient_frame():
    """Minimal frame satisfying the remibrutinib CSU config: all required
    fields present, inclusion criteria met, exclusions false, and observation
    windows covering lookback (180d) + followup (90d) around diagnosis_date."""
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    now = datetime.now(timezone.utc)
    diagnosis = now - timedelta(days=200)
    n = 40
    return pd.DataFrame(
        {
            "patient_journey_id": [f"883b-pj-{i}" for i in range(n)],
            "age_at_diagnosis": [30 + (i % 40) for i in range(n)],
            "diagnosis_code": ["L50.1"] * n,
            "diagnosis_date": [diagnosis.date().isoformat()] * n,
            "urticaria_severity_uas7": [20 + (i % 8) for i in range(n)],
            "prior_antihistamine_therapy": [True] * n,
            "active_autoimmune_condition": [False] * n,
            "concurrent_immunosuppressive": [False] * n,
            "pregnancy_status": [False] * n,
            "severe_hepatic_impairment": [False] * n,
            "first_observation_date": [(diagnosis - timedelta(days=200)).date().isoformat()] * n,
            "last_observation_date": [(diagnosis + timedelta(days=120)).date().isoformat()] * n,
        }
    )


@pytest.mark.asyncio
async def test_cohort_constructor_run_lands_episodic_row():
    """RED before wiring: a successful cohort construction leaves NO trace in
    episodic memory (contribute_to_memory has zero callers), so the learning
    readers (get_prior_cohorts) can never return anything. GREEN: the run
    lands one 'cohort_construction_completed' row AND get_prior_cohorts —
    the previously-dead reader — returns it."""
    from src.agents.cohort_constructor.agent import CohortConstructorAgent
    from src.agents.cohort_constructor.memory_hooks import (
        get_cohort_constructor_memory_hooks,
        reset_memory_hooks,
    )

    session_id = str(uuid.uuid4())

    agent = CohortConstructorAgent(use_graph=True, enable_observability=False)
    try:
        eligible_df, result = await agent.run(
            patient_df=_cohort_patient_frame(),
            brand="remibrutinib",
            session_id=session_id,
        )
        # Graph mode terminates in the state vocabulary ("completed"); direct
        # mode in the result vocabulary ("success") — both are success here.
        assert result.status in ("success", "completed"), (
            f"cohort construction failed: status={result.status} error={result.error_message}"
        )

        rows = _episodic_rows_for_session(session_id, "cohort_construction_completed")
        assert len(rows) == 1, (
            "agent.run() landed no 'cohort_construction_completed' episodic row — "
            "the cohort_constructor memory hooks have no caller (#883 PR B)"
        )
        assert rows[0]["agent_name"] == "cohort_constructor"

        # Previously-dead learning-loop reader returns the just-written row.
        reset_memory_hooks()
        hooks = get_cohort_constructor_memory_hooks()
        priors = await hooks.get_prior_cohorts(brand="remibrutinib", min_eligibility_rate=0.0)
        assert str(rows[0]["memory_id"]) in [str(p.get("memory_id")) for p in priors], (
            "get_prior_cohorts did not return the just-written cohort row "
            "(pre-fix: the RPC returns no raw_content, so the post-filter "
            "dropped every row)"
        )
    finally:
        _cleanup_episodic_by_session(session_id)
        reset_memory_hooks()


# =============================================================================
# experiment_designer — arun() must contribute per design
# =============================================================================


@pytest.mark.asyncio
async def test_experiment_designer_arun_lands_activity_row():
    """RED before wiring: a completed design leaves NO agent_activities trace
    (contribute_to_memory has zero callers; pre-fix the payload also wrote 5
    nonexistent columns — both bugs masked each other). GREEN: arun() lands a
    schema-correct 'experiment_design' activity row keyed to the session, and
    the previously-dead reader get_similar_validity_threats returns the
    design's threats."""
    from src.agents.experiment_designer.agent import (
        ExperimentDesignerAgent,
        ExperimentDesignerInput,
    )
    from src.memory.episodic_memory import get_supabase_client

    marker = f"883b-expd-{uuid.uuid4().hex[:12]}"
    agent = ExperimentDesignerAgent(enable_mlflow=False)
    session_id = None
    client = get_supabase_client()
    try:
        output = await agent.arun(
            ExperimentDesignerInput(
                business_question=(
                    f"Does increasing rep visit frequency lift Remibrutinib TRx ({marker})?"
                ),
                constraints={"expected_effect_size": 0.15, "alpha": 0.05, "power": 0.8},
                enable_validity_audit=True,
            )
        )
        session_id = agent.last_memory_session_id
        assert session_id, (
            "arun() recorded no memory session id — the memory contribution "
            "did not run (#883 PR B: experiment_designer wiring)"
        )

        rows = (
            client.table("agent_activities")
            .select("activity_id, agent_name, activity_type, input_data, analysis_results")
            .eq("agent_name", "experiment_designer")
            .eq("activity_type", "experiment_design")
            .contains("input_data", {"session_id": session_id})
            .execute()
        ).data or []
        assert len(rows) == 1, (
            "arun() landed no schema-correct experiment_design row in "
            "agent_activities (#883 PR B: contribute_to_memory unwired and/or "
            "the payload wrote nonexistent columns — PGRST204 swallowed)"
        )
        row = rows[0]
        assert marker in (row["input_data"].get("business_question") or "")
        assert row["analysis_results"].get("design_type") == output.design_type

        # The previously-dead reader path: threats from the just-stored design
        # are retrievable by design_type.
        if output.validity_threats:
            from src.agents.experiment_designer.memory_hooks import (
                get_experiment_designer_memory_hooks,
                reset_memory_hooks,
            )

            reset_memory_hooks()
            hooks = get_experiment_designer_memory_hooks()
            threats = await hooks.get_similar_validity_threats(
                design_type=output.design_type, max_threats=50
            )
            stored_names = {t.get("threat_name") for t in threats}
            # output.validity_threats are pydantic models, not dicts.
            assert any(t.threat_name in stored_names for t in output.validity_threats), (
                "get_similar_validity_threats did not surface the just-stored threats"
            )
            reset_memory_hooks()
    finally:
        if session_id:
            client.table("agent_activities").delete().eq(
                "agent_name", "experiment_designer"
            ).contains("input_data", {"session_id": session_id}).execute()


# =============================================================================
# prediction_synthesizer — model-performance Redis key must be written
# =============================================================================


class _StaticRegistry:
    """Registry stub exposing the same surface as LiveChampionModelRegistry.

    The REAL metrics source (ml_model_registry's measured auc/pr_auc/
    brier_score/calibration_slope via the FK-embed join) is proven live-DB in
    test_model_registry_performance_883b; here the subject is the AGENT
    wiring: synthesize() -> update_model_performance -> Redis key ->
    get_context round-trip, against REAL Redis working memory.
    """

    def __init__(self, names, perf):
        self._names = names
        self._perf = perf

    async def get_models_for_target(self, target: str, entity_type: str = ""):
        return list(self._names)

    async def get_model_performance_for_target(self, target: str):
        return dict(self._perf)


class _StaticClient:
    def __init__(self, value: float):
        self._value = value

    async def predict(self, entity_id, features, time_horizon):
        return {
            "prediction": self._value,
            "model_type": "classifier",
            "confidence": 0.9,
            "features_used": list(features.keys()),
        }


@pytest.mark.asyncio
async def test_prediction_synthesizer_writes_model_performance_key():
    """RED before wiring: after a successful synthesize() the Redis key
    prediction_synthesizer:model_performance:{target} does not exist and
    get_context().model_performance == {} (update_model_performance has no
    caller — the live reader is starved). GREEN: the key holds the registry's
    measured metrics for the run's models and get_context returns them."""
    from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent
    from src.agents.prediction_synthesizer.memory_hooks import (
        PredictionSynthesizerMemoryHooks,
        reset_memory_hooks,
    )

    _reset_redis_singletons()
    hooks_probe = PredictionSynthesizerMemoryHooks()
    if not hooks_probe.working_memory:
        pytest.skip("working memory (Redis) not reachable in this environment")
    try:
        redis = await hooks_probe.working_memory.get_client()
        await redis.ping()
    except Exception:
        pytest.skip("working memory (Redis) not responding in this environment")

    target = f"trx_883b_{uuid.uuid4().hex[:8]}"
    perf = {
        "m883b_a": {"auc": 0.83, "pr_auc": 0.61, "brier_score": 0.12, "calibration_slope": 1.02},
        "m883b_b": {"auc": 0.78, "pr_auc": 0.55, "brier_score": 0.15, "calibration_slope": 0.96},
    }
    registry = _StaticRegistry(list(perf.keys()), perf)
    clients = {"m883b_a": _StaticClient(0.7), "m883b_b": _StaticClient(0.6)}

    session_id = str(uuid.uuid4())
    perf_key = f"prediction_synthesizer:model_performance:{target}"

    reset_memory_hooks()
    agent = PredictionSynthesizerAgent(
        model_registry=registry,
        model_clients=clients,
        enable_opik=False,
        enable_memory=True,
        enable_dspy=False,
    )
    try:
        output = await agent.synthesize(
            entity_id=f"TERR-883b-{uuid.uuid4().hex[:8]}",
            entity_type="territory",
            prediction_target=target,
            features={"f1": 1.0},
            include_context=False,
            session_id=session_id,
        )
        assert output.status != "failed", f"synthesize failed: {output.errors}"
        assert output.models_succeeded == 2

        raw = await redis.get(perf_key)
        assert raw, (
            f"Redis key {perf_key} was not written — update_model_performance "
            "has no caller (#883 PR B: prediction_synthesizer PARTIAL wiring)"
        )

        context = await hooks_probe.get_context(
            session_id=session_id,
            entity_id="TERR-context-probe",
            entity_type="territory",
            prediction_target=target,
        )
        assert set(context.model_performance.keys()) == set(perf.keys()), (
            "get_context did not return the per-model performance map (the "
            "previously-starved live reader)"
        )
        assert context.model_performance["m883b_a"]["auc"] == pytest.approx(0.83)
        assert context.model_performance["m883b_a"]["brier_score"] == pytest.approx(0.12)
    finally:
        try:
            await redis.delete(perf_key)
        except Exception:
            pass
        _cleanup_episodic_by_session(session_id)
        reset_memory_hooks()


@pytest.mark.asyncio
async def test_model_registry_performance_883b():
    """Live-DB proof of the REAL metrics source: ml_model_registry's measured
    columns (auc / pr_auc / brier_score / calibration_slope) are retrievable
    per deployable serving model through the same FK-embed join as
    get_models_for_target. RED before fix: the repo method does not exist.
    Seeds nothing: asserts shape-consistency against whatever serving models
    exist; when none exist the honest result is {} (fail-closed, never
    fabricated)."""
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.ml_experiment import MLModelRegistryRepository

    client = await get_async_supabase_client()
    repo = MLModelRegistryRepository(supabase_client=client)

    names = await repo.get_models_for_target("risk_score")
    perf = await repo.get_model_performance_for_target("risk_score")

    assert set(perf.keys()) == set(names), (
        "performance map keys must be exactly the deployable serving models "
        "for the target (same membership test as get_models_for_target)"
    )
    for model_name, metrics in perf.items():
        assert set(metrics.keys()) >= {"auc", "pr_auc", "brier_score", "calibration_slope"}, (
            f"missing measured-metric keys for {model_name}: {metrics}"
        )
