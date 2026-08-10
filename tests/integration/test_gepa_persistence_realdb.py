"""Faithful (real-DB, NO mocks) tests for the GEPA persistence wiring.

Migration 023 created five prompt-optimization tables that stayed unwired
("unwired-but-roadmapped" per database/memory/033_drop_orphan_dspy_tables.sql):
prompt_optimization_runs, optimized_instructions, optimized_tool_descriptions,
prompt_ab_tests, prompt_ab_test_observations — all 0 rows since December 2025.
These tests pin the wiring: src/repositories/prompt_optimization.py (repos +
never-raise recorder seam) writing REAL rows to the live local supabase.

Three tests additionally pin migration database/ml/035, which the wiring
requires (all three verified against the live schema before writing):
  * unique_active_instruction UNIQUE NULLS NOT DISTINCT (agent, predictor,
    is_active) caps history at one inactive row per predictor — versioned
    history (the table's stated purpose) is impossible under it.
  * idx_opt_instructions_hash is GLOBALLY unique — two agents producing the
    same instruction text (e.g. dspy's default signature docstring) collide.
  * version is VARCHAR(50), but real version ids
    (``gepa_v1_feedback_learner_recommendation_YYYYMMDD_HHMMSS``) are 55+
    chars — every real save of that agent would fail the insert.

Opt-in (real docker supabase-db required), skipped in CI by default:

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_gepa_persistence_realdb.py -p no:cacheprovider

Every row this file writes is namespaced ``itest_gepa_`` (agent_name /
test_name / created_by) and deleted again in the module teardown, FK order.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

PREFIX = "itest_gepa_"


def _ag(suffix: str) -> str:
    """Namespaced agent_name for this test module."""
    return f"{PREFIX}{suffix}"


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the cached async client so each test builds one on its OWN event
    loop (the global cache binds httpx.AsyncClient to the creating loop)."""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


@pytest.fixture
async def client():
    from src.memory.services.factories import get_async_supabase_client

    c = await get_async_supabase_client()
    yield c
    # Teardown per test, FK order: observations -> tests -> instructions ->
    # tool descriptions -> runs. like() keeps it scoped to this module's rows.
    tests = (
        await c.table("prompt_ab_tests").select("test_id").like("test_name", f"{PREFIX}%").execute()
    )
    test_ids = [row["test_id"] for row in tests.data or []]
    if test_ids:
        await c.table("prompt_ab_test_observations").delete().in_("test_id", test_ids).execute()
        await c.table("prompt_ab_tests").delete().in_("test_id", test_ids).execute()
    await c.table("optimized_instructions").delete().like("agent_name", f"{PREFIX}%").execute()
    await c.table("optimized_tool_descriptions").delete().like("agent_name", f"{PREFIX}%").execute()
    await c.table("prompt_optimization_runs").delete().like("agent_name", f"{PREFIX}%").execute()


# ---------------------------------------------------------------------------
# prompt_optimization_runs
# ---------------------------------------------------------------------------


async def test_start_run_inserts_running_row(client):
    """start_run must persist a 'running' row with the identity + config the
    optimizer actually used, and return it (run_id included)."""
    from src.repositories.prompt_optimization import PromptOptimizationRunRepository

    repo = PromptOptimizationRunRepository(client)
    row = await repo.start_run(
        agent_name=_ag("feedback_learner_pattern"),
        agent_tier=5,
        agent_type="deep",
        optimizer_type="gepa",
        budget_preset="light",
        trainset_size=7,
        valset_size=2,
        reflection_model="anthropic/claude-sonnet-4-6",
        created_by="itest",
    )

    assert row is not None and row["run_id"]
    fetched = await repo.get_by_id(row["run_id"])
    assert fetched["status"] == "running"
    assert fetched["agent_name"] == _ag("feedback_learner_pattern")
    assert fetched["agent_tier"] == 5
    assert fetched["agent_type"] == "deep"
    assert fetched["optimizer_type"] == "gepa"
    assert fetched["budget_preset"] == "light"
    assert fetched["trainset_size"] == 7
    assert fetched["started_at"] is not None


async def test_start_run_resolves_profile_when_not_given(client):
    """Without an explicit tier/type, start_run resolves them from the agent
    profile registry (unknown itest agents get the documented default)."""
    from src.repositories.prompt_optimization import (
        DEFAULT_AGENT_PROFILE,
        PromptOptimizationRunRepository,
    )

    repo = PromptOptimizationRunRepository(client)
    row = await repo.start_run(
        agent_name=_ag("unknown_agent"),
        optimizer_type="gepa",
        budget_preset="light",
        trainset_size=3,
    )

    fetched = await repo.get_by_id(row["run_id"])
    assert fetched["agent_tier"] == DEFAULT_AGENT_PROFILE[0]
    assert fetched["agent_type"] == DEFAULT_AGENT_PROFILE[1]


async def test_complete_run_records_measured_results(client):
    """complete_run flips status to completed and stores ONLY measured values:
    scores, counts, artifacts — improvement is percentage POINTS (x100 of the
    0-1 score delta), never a relative percentage."""
    from src.repositories.prompt_optimization import PromptOptimizationRunRepository

    repo = PromptOptimizationRunRepository(client)
    row = await repo.start_run(
        agent_name=_ag("complete"),
        optimizer_type="gepa",
        budget_preset="custom",
        max_metric_calls=40,
        trainset_size=5,
    )

    updated = await repo.complete_run(
        row["run_id"],
        baseline_score=0.41,
        optimized_score=0.55,
        total_metric_calls=38,
        num_candidates_explored=4,
        pareto_frontier_size=2,
        best_candidate_idx=3,
        log_dir="./gepa_logs/itest",
        mlflow_run_id="itest-mlflow-1",
    )

    assert updated["status"] == "completed"
    assert float(updated["baseline_score"]) == pytest.approx(0.41)
    assert float(updated["optimized_score"]) == pytest.approx(0.55)
    assert float(updated["improvement_percent"]) == pytest.approx(14.0)
    assert updated["total_metric_calls"] == 38
    assert updated["num_candidates_explored"] == 4
    assert updated["best_candidate_idx"] == 3
    assert updated["completed_at"] is not None
    assert updated["duration_seconds"] is not None and updated["duration_seconds"] >= 0


async def test_complete_run_without_scores_leaves_them_null(client):
    """A run whose optimizer exposed no stats must persist NULL scores — the
    wiring records measurements, it never fabricates them."""
    from src.repositories.prompt_optimization import PromptOptimizationRunRepository

    repo = PromptOptimizationRunRepository(client)
    row = await repo.start_run(
        agent_name=_ag("noscores"), optimizer_type="gepa", budget_preset="light", trainset_size=5
    )
    updated = await repo.complete_run(row["run_id"])

    assert updated["status"] == "completed"
    assert updated["baseline_score"] is None
    assert updated["optimized_score"] is None
    assert updated["improvement_percent"] is None


async def test_fail_run_records_error(client):
    """fail_run persists the error message + traceback with status 'failed'."""
    from src.repositories.prompt_optimization import PromptOptimizationRunRepository

    repo = PromptOptimizationRunRepository(client)
    row = await repo.start_run(
        agent_name=_ag("fail"), optimizer_type="gepa", budget_preset="light", trainset_size=5
    )
    updated = await repo.fail_run(row["run_id"], "optimizer returned no module", "Traceback: ...")

    assert updated["status"] == "failed"
    assert updated["error_message"] == "optimizer returned no module"
    assert updated["error_traceback"] == "Traceback: ..."
    assert updated["completed_at"] is not None


# ---------------------------------------------------------------------------
# optimized_instructions
# ---------------------------------------------------------------------------


async def _mk_run(client, agent_name):
    from src.repositories.prompt_optimization import PromptOptimizationRunRepository

    repo = PromptOptimizationRunRepository(client)
    return await repo.start_run(
        agent_name=agent_name, optimizer_type="gepa", budget_preset="light", trainset_size=5
    )


async def test_record_instructions_rows_land(client):
    """record_instructions persists one row per predictor with the version and
    a hash matching the artifact saver's (versioning.compute_instruction_hash)."""
    from src.optimization.gepa.versioning import compute_instruction_hash
    from src.repositories.prompt_optimization import OptimizedInstructionRepository

    agent = _ag("instr")
    run = await _mk_run(client, agent)
    repo = OptimizedInstructionRepository(client)

    rows = await repo.record_instructions(
        run_id=run["run_id"],
        agent_name=agent,
        version="gepa_v1_itest_20260810_000000",
        entries=[("predict", "Answer concisely."), ("classify", "Pick exactly one label.")],
        val_score=0.55,
        candidate_idx=3,
        parent_indices=[0, 1],
        discovery_eval_count=12,
    )

    assert len(rows) == 2
    by_pred = {r["predictor_name"]: r for r in rows}
    assert by_pred["predict"]["instruction_text"] == "Answer concisely."
    assert by_pred["predict"]["instruction_hash"] == compute_instruction_hash("Answer concisely.")
    assert by_pred["classify"]["version"] == "gepa_v1_itest_20260810_000000"
    assert float(by_pred["classify"]["val_score"]) == pytest.approx(0.55)
    assert by_pred["classify"]["candidate_idx"] == 3
    assert by_pred["classify"]["parent_indices"] == [0, 1]


async def test_record_instructions_dedups_same_text(client):
    """Re-recording an identical instruction for the same (agent, predictor)
    must not raise and must not create a second row (hash dedup)."""
    from src.repositories.prompt_optimization import OptimizedInstructionRepository

    agent = _ag("dedup")
    run1 = await _mk_run(client, agent)
    run2 = await _mk_run(client, agent)
    repo = OptimizedInstructionRepository(client)

    await repo.record_instructions(
        run_id=run1["run_id"], agent_name=agent, version="gepa_v1_a", entries=[("p", "Same text.")]
    )
    await repo.record_instructions(
        run_id=run2["run_id"], agent_name=agent, version="gepa_v2_b", entries=[("p", "Same text.")]
    )

    result = (
        await client.table("optimized_instructions").select("*").eq("agent_name", agent).execute()
    )
    assert len(result.data) == 1


async def test_multiple_historical_versions_coexist(client):
    """Three successive optimizations of one predictor must yield three
    coexisting history rows. RED against the pre-035 schema: the
    unique_active_instruction constraint allows at most ONE inactive row per
    (agent, predictor), so the second insert raises 23505."""
    from src.repositories.prompt_optimization import OptimizedInstructionRepository

    agent = _ag("history")
    repo = OptimizedInstructionRepository(client)
    for n, text in enumerate(["First try.", "Second try.", "Third try."], start=1):
        run = await _mk_run(client, agent)
        await repo.record_instructions(
            run_id=run["run_id"], agent_name=agent, version=f"gepa_v{n}_h", entries=[("p", text)]
        )

    result = (
        await client.table("optimized_instructions").select("*").eq("agent_name", agent).execute()
    )
    assert len(result.data) == 3


async def test_same_text_across_agents_coexists(client):
    """Two agents producing identical instruction text (dspy's default
    signature instruction is shared boilerplate) must both persist. RED
    against the pre-035 schema: idx_opt_instructions_hash is globally unique."""
    from src.repositories.prompt_optimization import OptimizedInstructionRepository

    shared = "Given the fields `question`, produce the fields `answer`."
    repo = OptimizedInstructionRepository(client)
    for suffix in ("agent_a", "agent_b"):
        agent = _ag(suffix)
        run = await _mk_run(client, agent)
        await repo.record_instructions(
            run_id=run["run_id"], agent_name=agent, version="gepa_v1_x", entries=[("p", shared)]
        )

    result = (
        await client.table("optimized_instructions")
        .select("agent_name")
        .like("agent_name", f"{PREFIX}agent_%")
        .execute()
    )
    assert len(result.data) == 2


async def test_long_realistic_version_id_accepted(client):
    """Real version ids exceed 50 chars (gepa_v1_feedback_learner_recommendation_
    YYYYMMDD_HHMMSS is 55). RED against the pre-035 schema: version VARCHAR(50)."""
    from src.repositories.prompt_optimization import OptimizedInstructionRepository

    agent = _ag("longver")
    run = await _mk_run(client, agent)
    version = "gepa_v1_feedback_learner_recommendation_20260810_133055"
    assert len(version) > 50

    rows = await OptimizedInstructionRepository(client).record_instructions(
        run_id=run["run_id"], agent_name=agent, version=version, entries=[("p", "Long version.")]
    )
    assert rows[0]["version"] == version


async def test_activate_enforces_single_active(client):
    """Activating a new version deactivates the previous one: exactly one
    active row per (agent, predictor), the old row keeps deactivated_at."""
    from src.repositories.prompt_optimization import OptimizedInstructionRepository

    agent = _ag("activate")
    repo = OptimizedInstructionRepository(client)
    run1 = await _mk_run(client, agent)
    run2 = await _mk_run(client, agent)
    [v1] = await repo.record_instructions(
        run_id=run1["run_id"], agent_name=agent, version="gepa_v1_act", entries=[("p", "One.")]
    )
    [v2] = await repo.record_instructions(
        run_id=run2["run_id"], agent_name=agent, version="gepa_v2_act", entries=[("p", "Two.")]
    )

    await repo.activate(v1["instruction_id"])
    await repo.activate(v2["instruction_id"])

    active = await repo.get_active(agent)
    assert len(active) == 1
    assert active[0]["instruction_id"] == v2["instruction_id"]
    assert active[0]["activated_at"] is not None

    old = (
        await client.table("optimized_instructions")
        .select("*")
        .eq("instruction_id", v1["instruction_id"])
        .execute()
    )
    assert old.data[0]["is_active"] is False
    assert old.data[0]["deactivated_at"] is not None


async def test_activate_failure_restores_previous_active(client, monkeypatch):
    """If the activate-target statement fails after the deactivate statement
    succeeded, the previously-active version must be restored (best-effort
    compensation): an exception mid-activate must not leave the
    (agent, predictor) pair with NO active version. (A hard process crash
    still can — postgrest has no transactions — which stays the documented
    fail-safe direction.)"""
    import src.repositories.prompt_optimization as po

    agent = _ag("actrestore")
    repo = po.OptimizedInstructionRepository(client)
    run1 = await _mk_run(client, agent)
    run2 = await _mk_run(client, agent)
    [v1] = await repo.record_instructions(
        run_id=run1["run_id"], agent_name=agent, version="gepa_v1_ar", entries=[("p", "One.")]
    )
    [v2] = await repo.record_instructions(
        run_id=run2["run_id"], agent_name=agent, version="gepa_v2_ar", entries=[("p", "Two.")]
    )
    await repo.activate(v1["instruction_id"])

    real_exec = po._exec
    calls = {"n": 0}

    async def failing_exec(query):
        # activate() statement order: 1 select target, 2 deactivate previous,
        # 3 activate target. Fail exactly the third; the restore statement
        # (4th) passes through to the real DB.
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected activate failure")
        return await real_exec(query)

    monkeypatch.setattr(po, "_exec", failing_exec)
    with pytest.raises(RuntimeError, match="injected activate failure"):
        await repo.activate(v2["instruction_id"])
    monkeypatch.setattr(po, "_exec", real_exec)

    active = await repo.get_active(agent)
    assert len(active) == 1
    assert active[0]["instruction_id"] == v1["instruction_id"]


# ---------------------------------------------------------------------------
# optimized_tool_descriptions
# ---------------------------------------------------------------------------


async def test_tool_description_roundtrip(client):
    """Tool descriptions persist, activate, and read back. The runtime
    producer waits on dspy GEPA tool-optimization support; the persistence
    layer is complete now."""
    from src.repositories.prompt_optimization import OptimizedToolDescriptionRepository

    agent = _ag("tools")
    run = await _mk_run(client, agent)
    repo = OptimizedToolDescriptionRepository(client)

    row = await repo.record_tool_description(
        run_id=run["run_id"],
        agent_name=agent,
        tool_name="causal_forest",
        version="gepa_v1_feedback_learner_recommendation_20260810_133055",
        description_text="Estimates heterogeneous treatment effects.",
        argument_descriptions={"n_estimators": "Number of trees."},
        original_description="Causal forest tool.",
    )
    assert row["tool_description_id"]

    await repo.activate(row["tool_description_id"])
    active = await repo.get_active(agent)
    assert len(active) == 1
    assert active[0]["tool_name"] == "causal_forest"
    assert active[0]["description_text"] == "Estimates heterogeneous treatment effects."


async def test_tool_description_history_coexists(client):
    """Two inactive versions of one tool's description must coexist. RED
    against the pre-035 unique_active_tool_desc constraint (same defect as
    unique_active_instruction)."""
    from src.repositories.prompt_optimization import OptimizedToolDescriptionRepository

    agent = _ag("toolhist")
    repo = OptimizedToolDescriptionRepository(client)
    for n, text in enumerate(["Old description.", "New description."], start=1):
        run = await _mk_run(client, agent)
        await repo.record_tool_description(
            run_id=run["run_id"],
            agent_name=agent,
            tool_name="linear_dml",
            version=f"gepa_v{n}_t",
            description_text=text,
        )

    result = (
        await client.table("optimized_tool_descriptions")
        .select("*")
        .eq("agent_name", agent)
        .execute()
    )
    assert len(result.data) == 2


# ---------------------------------------------------------------------------
# prompt_ab_tests + prompt_ab_test_observations
# ---------------------------------------------------------------------------


async def test_ab_test_persist_and_finalize_roundtrip(client):
    """GEPAABTest lifecycle persists end-to-end: save -> bulk observations ->
    finalize with the REAL analyze() output (t-test aggregates, winner)."""
    from src.optimization.gepa.ab_test import GEPAABTest
    from src.repositories.prompt_optimization import PromptABTestRepository

    ab = GEPAABTest(
        test_name=f"{PREFIX}ab_roundtrip",
        agent_name=_ag("ab"),
        traffic_split=0.5,
        target_sample_size=60,
    )
    ab.start()
    repo = PromptABTestRepository(client)
    saved = await repo.save_test(ab, created_by="itest")
    assert saved["test_id"] == ab.test_id
    assert saved["status"] == "running"

    for i in range(70):
        variant = "gepa" if i % 2 == 0 else "baseline"
        ab.record_observation(
            request_id=f"req-{i}",
            variant=variant,
            score=0.9 if variant == "gepa" else 0.3,
            latency_ms=100 + i,
            success=True,
            user_id=f"{PREFIX}user",
        )
    inserted = await repo.record_observations(ab.observations)
    assert inserted == 70

    results = ab.analyze()
    assert results.is_significant  # separated distributions, n=35 per arm
    final = await repo.finalize_test(ab, results)

    assert final["status"] == "completed"
    assert final["baseline_requests"] == 35
    assert final["treatment_requests"] == 35
    assert float(final["baseline_score_avg"]) == pytest.approx(0.3)
    assert float(final["treatment_score_avg"]) == pytest.approx(0.9)
    assert final["is_significant"] is True
    assert final["winner"] == "gepa"

    count = (
        await client.table("prompt_ab_test_observations")
        .select("observation_id", count="exact")
        .eq("test_id", ab.test_id)
        .execute()
    )
    assert count.count == 70


async def test_ab_test_load_restores_state(client):
    """load_test reconstructs a GEPAABTest (config + status + observations)
    from the DB so an A/B test survives process restart."""
    from src.optimization.gepa.ab_test import GEPAABTest
    from src.repositories.prompt_optimization import PromptABTestRepository

    ab = GEPAABTest(
        test_name=f"{PREFIX}ab_reload",
        agent_name=_ag("ab_reload"),
        traffic_split=0.25,
        target_sample_size=10,
    )
    ab.start()
    repo = PromptABTestRepository(client)
    await repo.save_test(ab)
    for i in range(6):
        ab.record_observation(
            request_id=f"req-{i}", variant="gepa" if i < 3 else "baseline", score=0.5
        )
    await repo.record_observations(ab.observations)

    loaded = await repo.load_test(ab.test_id)

    assert loaded is not None
    assert loaded.test_id == ab.test_id
    assert loaded.test_name == f"{PREFIX}ab_reload"
    assert loaded.agent_name == _ag("ab_reload")
    assert loaded.traffic_split == pytest.approx(0.25)
    assert loaded.status == "running"
    assert len(loaded.observations) == 6
    assert {o.variant for o in loaded.observations} == {"gepa", "baseline"}


# ---------------------------------------------------------------------------
# Recorder seam (what the optimization runner / celery legs call)
# ---------------------------------------------------------------------------


async def test_recorder_never_raises_without_usable_client():
    """The recorder is best-effort: an unusable client yields None/False,
    never an exception — a DB outage must not fail an optimization run."""
    from src.repositories.prompt_optimization import (
        record_run_completed,
        record_run_failed,
        record_run_started,
    )

    unusable = object()
    run_id = await record_run_started(
        agent_name=_ag("nodb"),
        optimizer_type="gepa",
        budget_preset="light",
        trainset_size=3,
        client=unusable,
    )
    assert run_id is None

    ok = await record_run_completed(str(uuid.uuid4()), client=unusable)
    assert ok is False
    ok = await record_run_failed(str(uuid.uuid4()), "boom", client=unusable)
    assert ok is False


async def test_record_run_discarded_removes_provisional_row(client):
    """A pre-compile optimizer skip spends no budget, so the provisional
    'running' row must be DELETED — the table's contract is that a row exists
    exactly when real metric/LLM calls were made (or died trying)."""
    from src.repositories.prompt_optimization import (
        record_run_discarded,
        record_run_started,
    )

    run_id = await record_run_started(
        agent_name=_ag("discard"),
        optimizer_type="gepa",
        budget_preset="light",
        trainset_size=6,
        client=client,
    )
    assert run_id is not None

    ok = await record_run_discarded(run_id, client=client)
    assert ok is True

    left = (
        await client.table("prompt_optimization_runs")
        .select("run_id")
        .eq("run_id", run_id)
        .execute()
    )
    assert left.data == []


async def test_record_run_discarded_never_raises():
    """Same never-raise contract as the other recorder functions."""
    from src.repositories.prompt_optimization import record_run_discarded

    assert await record_run_discarded(None) is False
    assert await record_run_discarded(str(uuid.uuid4()), client=object()) is False


async def test_recorder_records_real_dspy_module_artifact(client, tmp_path):
    """End-to-end over a REAL dspy module: save the artifact exactly as the
    production runner does, then record the run + its instruction rows, linked
    by the artifact's version_id."""
    import dspy

    from src.optimization.gepa.versioning import save_optimized_module
    from src.repositories.prompt_optimization import (
        OptimizedInstructionRepository,
        record_run_completed,
        record_run_started,
    )

    agent = _ag("realmodule")
    module = dspy.Predict("question -> answer")
    info = save_optimized_module(module, agent_name=agent, output_dir=str(tmp_path))

    run_id = await record_run_started(
        agent_name=agent,
        optimizer_type="gepa",
        budget_preset="light",
        trainset_size=4,
        client=client,
    )
    assert run_id is not None

    ok = await record_run_completed(run_id, module=module, artifact_info=info, client=client)
    assert ok is True

    run = await client.table("prompt_optimization_runs").select("*").eq("run_id", run_id).execute()
    assert run.data[0]["status"] == "completed"
    assert run.data[0]["log_dir"] == info["path"]

    rows = await OptimizedInstructionRepository(client).get_for_run(run_id)
    assert len(rows) >= 1
    assert rows[0]["version"] == info["version_id"]
    assert rows[0]["instruction_text"]  # dspy's real default signature instruction


async def test_runner_skip_path_writes_no_rows(client):
    """run_feedback_learner_optimization with an unmeetable reward floor skips
    before any optimizer is built — and must create NO run rows."""
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    before = (
        await client.table("prompt_optimization_runs")
        .select("run_id", count="exact")
        .like("agent_name", "feedback_learner%")
        .execute()
    )

    # client is NOT passed: the runner resolves the process-wide sync client
    # exactly as the celery beat does (SignalCollectorAdapter is sync-client
    # only — an async client would leak an un-awaited execute() coroutine).
    # min_reward=999.0 cannot be met by any real signal (rewards are 0-1), so
    # the skip is deterministic against live data.
    result = await run_feedback_learner_optimization(min_reward=999.0)
    assert result["status"] == "skipped_insufficient_signals"

    after = (
        await client.table("prompt_optimization_runs")
        .select("run_id", count="exact")
        .like("agent_name", "feedback_learner%")
        .execute()
    )
    assert after.count == before.count
