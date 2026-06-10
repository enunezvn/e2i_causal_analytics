"""Faithful (real-DB, NO mocks) regression tests for issue #825 — the code-vs-DB
column drift unmasked once #821 fixed the await-on-sync client wiring (the
swallowed ``TypeError`` had been firing first and hiding these 42703 errors).

Three independent drifts, all verified against the live docker ``supabase-db``:

  * ``ml_experiments`` has no ``name``/``config`` columns. Real: ``experiment_name``
    plus structured fields; allocation ratios are NOT persisted (request-time only),
    so SRM's expected ratio must be derived uniform over the observed variants.
    Sites: enrollment_health_check, srm_detection_sweep, check_all_active_experiments.
  * ``ml_drift_history`` has no ``detected_at`` column. Real timestamp: ``created_at``.
    Site: cleanup_old_drift_history.
  * ``agent_registry`` has no int ``tier`` column. Real: ``agent_tier`` text-category
    enum (coordination/causal_analytics/monitoring/ml_predictions/self_improvement).
    Sites: AgentRegistryRepository.get_by_tier, copilotkit._fetch_agents_from_db.

Opt-in (real docker supabase-db required), skipped in CI by default; run -n0.

    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_issue_825_schema_drift_realdb.py -p no:cacheprovider -n0
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)


def _no_schema_drift(blob: object) -> bool:
    """True if the (stringified) result contains no Postgres 42703 column error."""
    s = str(blob).lower()
    return "does not exist" not in s and "42703" not in s


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the cached async client so each test builds a fresh one on its OWN
    event loop (the global cache binds httpx.AsyncClient to the creating loop)."""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


# ---------------------------------------------------------------------------
# ml_experiments column drift (ab_testing_tasks.py)
# ---------------------------------------------------------------------------


def test_enrollment_health_check_completes_without_schema_drift():
    """enrollment_health_check selects experiment_name (not name/config) and the
    query executes cleanly against the 621 real running ml_experiments."""
    from src.tasks.ab_testing_tasks import enrollment_health_check

    result = enrollment_health_check.run()

    assert _no_schema_drift(result), f"ml_experiments schema drift leaked: {result}"
    assert result.get("status") == "completed", f"expected completed, got {result}"
    assert result.get("experiments_checked", 0) > 0, (
        f"expected to reach real running experiments, got {result}"
    )


def test_srm_detection_sweep_completes_without_schema_drift():
    """srm_detection_sweep selects experiment_name (not name/config). With 0 real
    assignments every experiment is insufficient_data, so this is a read-only sweep
    over the 621 running experiments and must complete with no 42703 error."""
    from src.tasks.ab_testing_tasks import srm_detection_sweep

    result = srm_detection_sweep.run()

    assert _no_schema_drift(result), f"ml_experiments schema drift leaked: {result}"
    assert result.get("status") == "completed", f"expected completed, got {result}"
    assert result.get("experiments_checked", 0) > 0, (
        f"expected to reach real running experiments, got {result}"
    )


def test_check_all_active_experiments_reads_real_experiment_name(monkeypatch):
    """check_all_active_experiments selects experiment_name (not name). We isolate
    the downstream Celery enqueue (``scheduled_interim_analysis.delay``) — NOT the
    system under test — so the test does not flood the broker with 621 real tasks;
    the schema-drift query against the live DB is exercised for real."""
    import src.tasks.ab_testing_tasks as abt

    captured = []

    class _FakeAsyncResult:
        id = "itest-825"

    def _capture_delay(*args, **kwargs):
        captured.append(kwargs.get("experiment_id"))
        return _FakeAsyncResult()

    monkeypatch.setattr(abt.scheduled_interim_analysis, "delay", _capture_delay)

    result = abt.check_all_active_experiments.run()

    assert _no_schema_drift(result), f"ml_experiments schema drift leaked: {result}"
    assert result.get("status") == "completed", f"expected completed, got {result}"
    assert result.get("experiments_found", 0) > 0, (
        f"expected to reach real running experiments, got {result}"
    )
    # The display name must come from the real experiment_name column, not the
    # phantom 'name' column (which would silently yield "Unknown" for every row).
    queued = result.get("queued_tasks", [])
    assert queued, f"expected queued tasks for real experiments, got {result}"
    assert any(t.get("name") and t["name"] != "Unknown" for t in queued), (
        f"expected real experiment_name values, got {queued[:3]}"
    )


# ---------------------------------------------------------------------------
# ml_drift_history column drift (drift_monitoring_tasks.py)
# ---------------------------------------------------------------------------


def test_cleanup_old_drift_history_completes_without_schema_drift():
    """cleanup_old_drift_history deletes on created_at (not detected_at). A huge
    retention deletes nothing but the DELETE must execute cleanly (no 42703)."""
    from src.tasks.drift_monitoring_tasks import cleanup_old_drift_history

    result = cleanup_old_drift_history.run(retention_days=100000)

    assert _no_schema_drift(result), f"ml_drift_history schema drift leaked: {result}"
    assert result.get("status") == "completed", f"expected completed, got {result}"


# ---------------------------------------------------------------------------
# agent_registry tier drift (repository + copilotkit)
# ---------------------------------------------------------------------------


async def test_fetch_agents_from_db_returns_all_real_agents():
    """copilotkit._fetch_agents_from_db reaches the real agent_registry roster via
    the agent_tier text-category schema (was blocked by the phantom int 'tier'
    column). Decisive: all 13 real active agents come back with integer tiers."""
    from src.api.routes.copilotkit import _fetch_agents_from_db

    agents = await _fetch_agents_from_db()

    assert agents is not None and len(agents) == 13, (
        f"expected all 13 real agent_registry rows, got {agents}"
    )
    tiers = {a["tier"] for a in agents}
    assert tiers == {1, 2, 3, 4, 5}, f"expected tiers 1-5 from agent_tier categories, got {tiers}"
    # No silent fallback: every agent name is a real roster name, not a sample.
    names = {a["id"] for a in agents}
    assert "orchestrator" in names and "causal_impact" in names, f"missing real agents: {names}"


async def test_get_by_tier_returns_real_category_agents():
    """AgentRegistryRepository.get_by_tier(int) maps the int to the real agent_tier
    text category and returns the live rows (was filtering a phantom 'tier' int
    column -> 42703 -> empty/fallback). Tier 2 = causal_analytics = 3 agents."""
    from src.api.routes.copilotkit import _get_agent_registry_repository

    repo = await _get_agent_registry_repository()
    assert repo is not None and repo.client is not None

    tier2 = await repo.get_by_tier(2)
    names = {a.get("agent_name") for a in tier2}
    assert names == {"causal_impact", "gap_analyzer", "heterogeneous_optimizer"}, (
        f"expected the 3 real causal_analytics agents, got {names}"
    )
