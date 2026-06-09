"""Faithful (real-DB, NO mocks) regression tests for issue #821 — the latent
``await get_supabase_client()`` (await-on-sync) bug across task/mlops sites and
the ``*Repository(client=...)`` wrong-kwarg bug in the copilotkit agents path
and the observability connector. Same root cause as PR #820 (experiment
monitor H3b).

WHY (both patterns swallow a TypeError and mask a crash as empty/fallback):
  * ``get_supabase_client`` is SYNC; ``await``-ing it raises
    ``TypeError: object Client can't be used in 'await' expression``. The
    surrounding ``except`` returns ``{"status": "failed", ...}`` / a skipped
    result while real rows exist (621 running ml_experiments).
  * ``BaseRepository.__init__`` takes ``supabase_client=``; passing ``client=``
    raises ``TypeError`` → the helper's ``except`` returns ``None`` → silent
    fallback to sample agents / empty spans (5313 real ml_observability_spans
    exist; 13 real agent_registry rows).

Opt-in (real docker supabase-db required), skipped in CI by default. The
static AST guard in tests/unit/test_supabase_client_misuse_guard.py pins BOTH
patterns to zero in CI without a DB.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_async_supabase_client_realdb.py -p no:cacheprovider

COVERAGE of the 8 await-on-sync sites:
  * ab_testing_tasks.py:356  -> enrollment_health_check (behavioral, read-only)
  * ab_testing_tasks.py:1033 -> cleanup_old_ab_results (behavioral, no-op delete)
  * ab_testing_tasks.py:534  -> srm_detection_sweep   (AST guard only — runs SRM
        over all 621 running experiments; expensive, not destructive)
  * ab_testing_tasks.py:946  -> check_all_active_experiments (AST guard only —
        enqueues a scheduled_interim_analysis task PER running experiment; would
        flood the broker with 621 real tasks)
  * drift_monitoring_tasks.py:404 -> cleanup_old_drift_history (behavioral, no-op)
  * feedback_loop_tasks.py:148 -> _execute_feedback_loop (behavioral, empty
        prediction_types => acquires client, skips the labeling RPC)
  * feedback_loop_tasks.py:418 -> analyze_concept_drift_from_truth (behavioral, read)
  * mlops/optuna_optimizer.py:762 -> OptunaOptimizer.save_to_database
        (behavioral, writes one ml_hpo_studies row then deletes it)
The two AST-guard-only sites live in the same modules as behaviorally-proven
sites and share the identical one-line fix.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

_AWAIT_ERR_TOKENS = ("await", "can't be used in 'await'", "coroutine")


def _no_await_typeerror(blob: object) -> bool:
    """True if the (stringified) result contains no await-on-sync TypeError."""
    s = str(blob).lower()
    return not any(tok in s for tok in _AWAIT_ERR_TOKENS)


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the cached async client so each test builds a fresh one on its OWN
    event loop (the global cache binds httpx.AsyncClient to the creating loop;
    pytest's per-test loops would otherwise reuse a client from a closed loop).
    Also reset the observability span-repository singletons so the kwarg fix is
    exercised fresh each test. Test-only isolation; prod has one long-lived loop.
    """
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    try:
        import src.agents.ml_foundation.observability_connector.nodes.metrics_aggregator as ma

        ma._span_repository = None
    except Exception:  # pragma: no cover - defensive
        pass
    yield
    factories._async_supabase_client = None


# ---------------------------------------------------------------------------
# A/B testing tasks
# ---------------------------------------------------------------------------


def test_enrollment_health_check_acquires_async_client():
    """ab_testing_tasks.py:356 — #821 fix: the task now acquires the ASYNC
    client and RUNS its query (the await-on-sync TypeError is gone).

    This surfaced a SEPARATE pre-existing schema-drift bug that the await
    crash had been masking: ``ml_experiments`` has no ``name``/``config``
    columns (real: ``experiment_name`` + structured fields). That is NOT
    #821's client-wiring root cause — see docs follow-up. Here we assert ONLY
    the #821 fix: no await/coroutine TypeError, and any failure is a real
    schema error proving the query actually executed against the live DB."""
    from src.tasks.ab_testing_tasks import enrollment_health_check

    result = enrollment_health_check.run()

    assert _no_await_typeerror(result), f"await-on-sync TypeError leaked: {result}"
    if result.get("status") == "failed":
        assert "does not exist" in str(result.get("error", "")), (
            f"failure is not the expected exposed schema-drift bug: {result}"
        )


def test_cleanup_old_ab_results_noop_completes():
    """ab_testing_tasks.py:1033 — huge retention => deletes nothing, but the
    task must still acquire the async client and complete (not fail on await)."""
    from src.tasks.ab_testing_tasks import cleanup_old_ab_results

    result = cleanup_old_ab_results.run(retention_days=100000)

    assert _no_await_typeerror(result), f"await-on-sync TypeError leaked: {result}"
    assert result.get("status") == "completed", f"expected completed, got {result}"


# ---------------------------------------------------------------------------
# Drift monitoring tasks
# ---------------------------------------------------------------------------


def test_cleanup_old_drift_history_acquires_async_client():
    """drift_monitoring_tasks.py:404 — #821 fix: the task now acquires the
    ASYNC client and RUNS its query (the await-on-sync TypeError is gone).

    This surfaced a SEPARATE pre-existing schema-drift bug: ``ml_drift_history``
    has no ``detected_at`` column (real timestamp column: ``created_at``). That
    is NOT #821's client-wiring root cause — see docs follow-up. We assert ONLY
    the #821 fix here."""
    from src.tasks.drift_monitoring_tasks import cleanup_old_drift_history

    result = cleanup_old_drift_history.run(retention_days=100000)

    assert _no_await_typeerror(result), f"await-on-sync TypeError leaked: {result}"
    if result.get("status") == "failed":
        assert "does not exist" in str(result.get("error", "")), (
            f"failure is not the expected exposed schema-drift bug: {result}"
        )


# ---------------------------------------------------------------------------
# Feedback loop tasks
# ---------------------------------------------------------------------------


async def test_execute_feedback_loop_acquires_async_client_without_rpc():
    """feedback_loop_tasks.py:148 — passing an empty prediction_types list
    exercises the client acquisition (line 148) but skips the labeling RPC, so
    this is a safe faithful probe. Buggy code fails closed with an await error."""
    from src.tasks.feedback_loop_tasks import _execute_feedback_loop

    result = await _execute_feedback_loop(
        prediction_types=[],
        task_id="itest-821",
        window_name="short",
    )

    assert _no_await_typeerror(result), f"await-on-sync TypeError leaked: {result}"
    assert result.get("status") == "completed", f"expected completed, got {result}"
    assert result.get("total_labeled") == 0


def test_analyze_concept_drift_no_await_error():
    """feedback_loop_tasks.py:418 — read-only concept-drift analysis must
    acquire the async client; it must not fail closed with an await error."""
    from src.tasks.feedback_loop_tasks import analyze_concept_drift_from_truth

    result = analyze_concept_drift_from_truth.run()

    assert _no_await_typeerror(result), f"await-on-sync TypeError leaked: {result}"
    assert result.get("status") in {"completed", "no_data", "partial"}, (
        f"unexpected status: {result}"
    )


# ---------------------------------------------------------------------------
# Optuna optimizer (mlops)
# ---------------------------------------------------------------------------


async def test_optuna_save_to_database_writes_real_row_then_cleanup():
    """mlops/optuna_optimizer.py:762 — save_to_database must acquire the async
    client and persist a real ml_hpo_studies row. Buggy code returns
    {"success": False, "error": "...await..."}."""
    import optuna

    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.optuna_optimizer import OptunaOptimizer

    # experiment_id is a nullable FK to ml_experiments -> None keeps the insert
    # FK-clean without touching real experiment rows.
    optimizer = OptunaOptimizer(experiment_id=None, use_config=False)

    study = optuna.create_study(direction="maximize", study_name="itest_821_async")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=2)

    optimization_results = {
        "n_trials": 2,
        "n_completed": 2,
        "n_pruned": 0,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "duration_seconds": 0.01,
    }

    result = await optimizer.save_to_database(
        study=study,
        optimization_results=optimization_results,
        algorithm_name="itest",
        problem_type="binary_classification",
        metric="roc_auc",
    )

    study_id = result.get("study_id")
    try:
        assert _no_await_typeerror(result), f"await-on-sync TypeError leaked: {result}"
        assert result.get("success") is True, f"expected real save success, got {result}"
        assert study_id, "expected a study_id from a real insert"
    finally:
        if study_id:
            client = await get_async_supabase_client()
            await client.table("ml_hpo_trials").delete().eq("study_id", study_id).execute()
            await client.table("ml_hpo_studies").delete().eq("id", study_id).execute()


# ---------------------------------------------------------------------------
# copilotkit agents path (compound: wrong kwarg + needs async client)
# ---------------------------------------------------------------------------


async def test_agent_registry_repository_reaches_real_agents():
    """copilotkit.py:908 — #821 fix: AgentRegistryRepository is now built with
    ``supabase_client=<async client>`` (was ``client=`` -> TypeError -> None ->
    sample fallback). Decisive proof: the async repo reaches the 13 real
    ``agent_registry`` rows via the schema-clean ``get_active_agents`` path."""
    from src.api.routes.copilotkit import _get_agent_registry_repository

    repo = await _get_agent_registry_repository()

    assert repo is not None, "repo is None — async/kwarg wiring still broken"
    assert repo.client is not None, "repo has no client — async wiring broken"

    agents = await repo.get_active_agents()
    assert len(agents) > 0, f"expected real active agents, got {agents}"


@pytest.mark.xfail(
    reason=(
        "Exposed pre-existing schema drift (NOT #821): agent_registry has no "
        "'tier' column (real: 'agent_tier' text categories), but "
        "AgentRegistryRepository.get_by_tier filters {'tier': int} and "
        "_fetch_agents_from_db loops get_by_tier(range(1,6)). The #821 "
        "client-wiring fix is proven by "
        "test_agent_registry_repository_reaches_real_agents; this end-to-end "
        "path will xpass once the schema-drift follow-up (#825) lands."
    ),
    strict=False,
)
async def test_fetch_agents_from_db_end_to_end_blocked_by_tier_schema_drift():
    """Documents that the full ``_fetch_agents_from_db`` path remains blocked
    by the ``agent_registry.tier`` schema-drift bug AFTER the #821 client-wiring
    fix — surfaced loudly so it is tracked, not silently fallen-back."""
    from src.api.routes.copilotkit import _fetch_agents_from_db

    agents = await _fetch_agents_from_db()
    assert agents is not None and len(agents) > 0


# ---------------------------------------------------------------------------
# observability connector (kwarg-only; sync client + sync execute)
# ---------------------------------------------------------------------------


def test_observability_agent_span_repository_constructed_and_reads_real_spans():
    """observability_connector/agent.py:91 — span_repository must build (the
    client= kwarg raised TypeError -> None -> is_db_enabled False -> mock)."""
    from src.agents.ml_foundation.observability_connector.agent import (
        ObservabilityConnectorAgent,
    )

    agent = ObservabilityConnectorAgent()

    repo = agent.span_repository
    assert repo is not None, "span_repository is None — client= kwarg TypeError"
    assert agent.is_db_enabled is True

    # Faithful: the repo's real (sync) client reaches the real spans table.
    result = repo.client.table("ml_observability_spans").select("span_id").limit(1).execute()
    assert result.data, "expected to read at least one of the 5313 real spans"


def test_metrics_aggregator_span_repository_constructed():
    """metrics_aggregator.py:30 — _get_span_repository must build the repo (the
    client= kwarg raised TypeError -> None -> _get_spans_from_repository [] )."""
    from src.agents.ml_foundation.observability_connector.nodes import (
        metrics_aggregator,
    )

    repo = metrics_aggregator._get_span_repository()
    assert repo is not None, "_get_span_repository is None — client= kwarg TypeError"
    result = repo.client.table("ml_observability_spans").select("span_id").limit(1).execute()
    assert result.data, "expected to read at least one of the 5313 real spans"
