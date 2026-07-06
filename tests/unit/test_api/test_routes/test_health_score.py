"""Unit tests for Health Score API route handlers.

Tests all endpoints and helper functions in src/api/routes/health_score.py.
Mocks all external dependencies to ensure unit test isolation.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# Import route functions and models
from src.api.routes.health_score import (
    # Enums
    AgentHealth,
    CheckScope,
    ComponentHealth,
    ComponentStatus,
    DataProvenance,
    HealthGrade,
    ModelHealth,
    ModelStatus,
    PipelineHealth,
    PipelineStatus,
    # Helper functions
    _execute_health_check,
    _fetch_agent_health,
    _fetch_model_health,
    _fetch_pipeline_health,
    _generate_mock_health_response,
    _generate_recommendations,
    _get_mock_agent_health,
    _get_mock_component_health,
    _get_mock_model_health,
    _get_mock_pipeline_health,
    # Module-level storage
    _health_history,
    _record_full_check,
    _trend_from_scores,
    full_health_check,
    get_agent_health,
    get_component_health,
    get_health_history,
    get_model_health,
    get_pipeline_health,
    get_service_status,
    quick_health_check,
    # Endpoints
    run_health_check,
    run_scheduled_full_check,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_health_history():
    """Clear health history before each test."""
    _health_history.clear()
    yield
    _health_history.clear()


@pytest.fixture(autouse=True)
def _no_live_health_sources(monkeypatch):
    """Unit tests have no live Supabase. Default the health-source client to None
    so the dimension fetchers report 'backend unavailable' and the /models,
    /pipelines, /agents endpoints take the deterministic mock-fallback path.

    Also default to dev mock-fallback ALLOWED (E2I_REQUIRE_AGENT_IMPORT=0) so the
    backend-down guard serves clearly-tagged placeholder instead of 503 — unit
    tests are not production. Fail-closed tests override with "1".

    Tests exercising the real-data path patch _health_source_client (fetcher
    tests) or the fetcher itself (endpoint tests) — the inner patch wins."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch("src.api.routes.health_score._health_source_client", return_value=None):
        yield


@pytest.fixture
def mock_agent_result():
    """Mock Health Score agent result."""
    return MagicMock(
        overall_health_score=85.5,
        health_grade="B",
        component_health_score=0.9,
        model_health_score=0.85,
        pipeline_health_score=0.88,
        agent_health_score=0.95,
        critical_issues=[],
        warnings=["Some warning"],
        health_summary="System health is good",
        total_latency_ms=1250,
        check_latency_ms=1250,
        timestamp=datetime.now(timezone.utc).isoformat(),
        # F1: the route now propagates the agent's real provenance into the
        # response; must be a concrete string (not an auto-MagicMock attr).
        data_provenance="partial",
    )


# =============================================================================
# ENDPOINT TESTS - run_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_run_health_check_full_scope():
    """Test run_health_check with FULL scope."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        result = await run_health_check(scope=CheckScope.FULL)

        assert result.overall_health_score == 85.0
        assert result.health_grade == HealthGrade.B
        assert result.check_id.startswith("hs_")
        # Function sets check_latency_ms, so it should be > 0
        assert result.check_latency_ms >= 0
        mock_execute.assert_called_once_with(CheckScope.FULL)


@pytest.mark.asyncio
async def test_run_health_check_quick_scope():
    """Test run_health_check with QUICK scope."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.QUICK,
            overall_health_score=90.0,
            health_grade=HealthGrade.A,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        result = await run_health_check(scope=CheckScope.QUICK)

        assert result.overall_health_score == 90.0
        assert result.check_scope == CheckScope.QUICK
        mock_execute.assert_called_once_with(CheckScope.QUICK)


@pytest.mark.asyncio
async def test_run_health_check_stores_history():
    """Test that health check results are stored in history."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 0
        mock_result.data_provenance = "measured"
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)

        assert len(_health_history) == 1
        assert _health_history[0].overall_health_score == 85.0


@pytest.mark.asyncio
async def test_run_health_check_limits_history_to_100():
    """Test that health history is limited to last 100 entries."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 0
        mock_result.data_provenance = "measured"
        mock_execute.return_value = mock_result

        # Add 105 entries
        for _ in range(105):
            await run_health_check(scope=CheckScope.FULL)

        assert len(_health_history) == 100


@pytest.mark.asyncio
async def test_run_health_check_exception_handling():
    """Test that exceptions are properly handled."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_execute.side_effect = RuntimeError("Test error")

        with pytest.raises(Exception) as exc_info:
            await run_health_check(scope=CheckScope.FULL)

        assert "Health check failed" in str(exc_info.value)


# =============================================================================
# ENDPOINT TESTS - quick_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_quick_health_check_calls_run_with_quick_scope():
    """Test quick_health_check calls run_health_check with QUICK scope."""
    with patch("src.api.routes.health_score.run_health_check") as mock_run:
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = await quick_health_check()

        assert result == mock_result
        mock_run.assert_called_once_with(scope=CheckScope.QUICK)


# =============================================================================
# ENDPOINT TESTS - full_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_full_health_check_calls_run_with_full_scope():
    """Test full_health_check calls run_health_check with FULL scope."""
    with patch("src.api.routes.health_score.run_health_check") as mock_run:
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = await full_health_check()

        assert result == mock_result
        mock_run.assert_called_once_with(scope=CheckScope.FULL)


# -----------------------------------------------------------------------------
# F1: /check must surface the AGENT'S real provenance, not a hardcoded
# 'measured'. With no real stores wired, the agent fails closed to 'unknown';
# the route must propagate that, not fabricate health.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_health_check_propagates_agent_provenance(mock_agent_result):
    """_execute_health_check must copy the agent's data_provenance into the
    response. A real default agent (no model/pipeline/agent stores) reports
    'partial' or 'unknown' — never a hardcoded 'measured'."""
    mock_agent_result.data_provenance = "unknown"
    mock_agent = MagicMock()
    mock_agent.check_health = AsyncMock(return_value=mock_agent_result)
    # _execute_health_check imports HealthScoreAgent from src.agents.health_score,
    # so the patch target is the source module (mirrors the other route tests).
    with patch("src.agents.health_score.HealthScoreAgent", return_value=mock_agent):
        result = await _execute_health_check(CheckScope.FULL)
    assert result.data_provenance == "unknown"


@pytest.mark.asyncio
async def test_execute_health_check_wires_real_health_client():
    """_execute_health_check must construct the agent with the REAL
    SupabaseHealthClient so component health is genuinely measured (not the
    fail-open mock path). SupabaseHealthClient is left UNPATCHED so we assert a
    real instance is threaded through."""
    from src.agents.health_score import SupabaseHealthClient

    captured = {}

    def _capture(*args, **kwargs):
        captured["kwargs"] = kwargs
        agent = MagicMock()
        agent.check_health = AsyncMock(
            return_value=MagicMock(
                overall_health_score=25.0,
                health_grade="F",
                component_health_score=1.0,
                model_health_score=0.0,
                pipeline_health_score=0.0,
                agent_health_score=0.0,
                critical_issues=[],
                warnings=[],
                health_summary="partial",
                total_latency_ms=10,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data_provenance="partial",
            )
        )
        return agent

    with patch("src.agents.health_score.HealthScoreAgent", side_effect=_capture):
        await _execute_health_check(CheckScope.FULL)

    assert "health_client" in captured["kwargs"]
    assert isinstance(captured["kwargs"]["health_client"], SupabaseHealthClient)


# -----------------------------------------------------------------------------
# UNWIRED-BACKEND FIX: /health-score/full must compute model/pipeline/agent
# health from the REAL tables (ml_model_health_dashboard / etl_pipeline_metrics /
# agent_registry+audit_chain_entries) — not fail-closed to null. The route must
# construct the agent with the three real store adapters (DRY-reusing _fetch_*),
# so the full graph's model/pipeline/agent nodes actually run.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_health_check_wires_real_stores_for_full_scope():
    """REGRESSION: full scope must thread metrics_store/pipeline_store/
    agent_registry into the agent so the model/pipeline/agent dimensions are
    computed from the real tables — before this fix only health_client was wired,
    leaving the three dimensions null."""
    from src.api.routes.health_score import (
        _AgentRegistryAdapter,
        _ModelMetricsStoreAdapter,
        _PipelineStoreAdapter,
    )

    captured = {}

    def _capture(*args, **kwargs):
        captured["kwargs"] = kwargs
        agent = MagicMock()
        agent.check_health = AsyncMock(
            return_value=MagicMock(
                overall_health_score=95.0,
                health_grade="A",
                component_health_score=1.0,
                model_health_score=1.0,
                pipeline_health_score=1.0,
                agent_health_score=1.0,
                critical_issues=[],
                warnings=[],
                health_summary="measured",
                total_latency_ms=10,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data_provenance="measured",
            )
        )
        return agent

    with patch("src.agents.health_score.HealthScoreAgent", side_effect=_capture):
        await _execute_health_check(CheckScope.FULL)

    kw = captured["kwargs"]
    assert isinstance(kw.get("metrics_store"), _ModelMetricsStoreAdapter)
    assert isinstance(kw.get("pipeline_store"), _PipelineStoreAdapter)
    assert isinstance(kw.get("agent_registry"), _AgentRegistryAdapter)


@pytest.mark.asyncio
async def test_model_store_adapter_reuses_fetch_and_carries_status():
    """The model store adapter must REUSE _fetch_model_health (single source of
    truth) and carry its authoritative status — the agent's model node then
    scores identically to the /models endpoint, never re-deriving a divergent
    status."""
    from src.api.routes.health_score import _ModelMetricsStoreAdapter

    real_models = [
        ModelHealth(model_id="m1", model_name="alpha", status=ModelStatus.HEALTHY, auc_roc=0.83),
        ModelHealth(model_id="m2", model_name="beta", status=ModelStatus.DEGRADED),
    ]
    with patch(
        "src.api.routes.health_score._fetch_model_health",
        return_value=(real_models, DataProvenance.PARTIAL),
    ):
        adapter = _ModelMetricsStoreAdapter()
        ids = await adapter.get_active_models()
        assert ids == ["m1", "m2"]
        m1 = await adapter.get_model_metrics("m1", "24h")
        m2 = await adapter.get_model_metrics("m2", "24h")
    # Status is the reader's authoritative value, NOT re-derived.
    assert m1["status"] == "healthy"
    assert m2["status"] == "degraded"
    assert m1["auc_roc"] == 0.83


@pytest.mark.asyncio
async def test_model_store_adapter_fails_closed_when_backend_down():
    """When _fetch_model_health signals the backend is unreachable (provenance
    None), the adapter must raise HealthSourceUnavailable so the node fails closed
    to an honest null — never a fabricated healthy score."""
    from src.api.routes.health_score import HealthSourceUnavailable, _ModelMetricsStoreAdapter

    with patch(
        "src.api.routes.health_score._fetch_model_health",
        return_value=([], None),
    ):
        adapter = _ModelMetricsStoreAdapter()
        with pytest.raises(HealthSourceUnavailable):
            await adapter.get_active_models()


@pytest.mark.asyncio
async def test_pipeline_store_adapter_carries_status_and_fails_closed():
    """Pipeline adapter reuses _fetch_pipeline_health, carries its status, and
    fails closed when the backend is down."""
    from src.api.routes.health_score import HealthSourceUnavailable, _PipelineStoreAdapter

    real = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=12.0,
            status=PipelineStatus.HEALTHY,
        )
    ]
    with patch(
        "src.api.routes.health_score._fetch_pipeline_health",
        return_value=(real, DataProvenance.MEASURED),
    ):
        adapter = _PipelineStoreAdapter()
        names = await adapter.get_all_pipelines()
        st = await adapter.get_pipeline_status("rwd_ingest")
    assert names == ["rwd_ingest"]
    assert st["status"] == "healthy"
    assert st["failed"] is False

    with patch(
        "src.api.routes.health_score._fetch_pipeline_health",
        return_value=([], None),
    ):
        adapter = _PipelineStoreAdapter()
        with pytest.raises(HealthSourceUnavailable):
            await adapter.get_all_pipelines()


@pytest.mark.asyncio
async def test_pipeline_store_adapter_carries_freshness_hours():
    """codex r3 MEDIUM: the adapter must pass _fetch_pipeline_health's already-
    computed freshness_hours through so the node does NOT recompute it and emit a
    -1 sentinel (formatted as '(-1.0 hours)') when a row lacks a usable run_end."""
    from src.api.routes.health_score import _PipelineStoreAdapter

    real = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=43.99,
            status=PipelineStatus.HEALTHY,
        )
    ]
    with patch(
        "src.api.routes.health_score._fetch_pipeline_health",
        return_value=(real, DataProvenance.MEASURED),
    ):
        adapter = _PipelineStoreAdapter()
        st = await adapter.get_pipeline_status("rwd_ingest")
    assert st["freshness_hours"] == 43.99


@pytest.mark.asyncio
async def test_agent_registry_adapter_passes_null_telemetry_honestly():
    """Agent registry adapter reuses _fetch_agent_health. A registered, available
    agent with no recent telemetry must keep success_rate/avg_latency_ms NULL
    (unmeasured) — never fabricated to 1.0/0. The agent NODE separately treats a
    None success_rate as 'available => not penalized' so the score still matches
    the /agents endpoint without inventing a measurement."""
    from src.api.routes.health_score import HealthSourceUnavailable, _AgentRegistryAdapter

    roster = [
        AgentHealth(
            agent_name="gap_analyzer",
            tier=2,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        )
    ]
    with patch(
        "src.api.routes.health_score._fetch_agent_health",
        return_value=(roster, DataProvenance.PARTIAL),
    ):
        adapter = _AgentRegistryAdapter()
        agents = await adapter.get_all_agents()
        metrics = await adapter.get_agent_metrics("gap_analyzer")
    assert agents == [{"name": "gap_analyzer", "tier": 2}]
    assert metrics["available"] is True
    # null telemetry stays NULL — honest, not fabricated.
    assert metrics["success_rate"] is None
    assert metrics["avg_latency_ms"] is None

    with patch(
        "src.api.routes.health_score._fetch_agent_health",
        return_value=([], None),
    ):
        adapter = _AgentRegistryAdapter()
        with pytest.raises(HealthSourceUnavailable):
            await adapter.get_all_agents()


@pytest.mark.asyncio
async def test_reconcile_full_provenance_downgrades_on_partial_source():
    """_reconcile_full_provenance must downgrade a 'measured' composite to
    'partial' when ANY wired adapter saw a PARTIAL source, and leave it 'measured'
    only when every adapter source was MEASURED. Non-'measured' composites pass
    through untouched."""
    from src.api.routes.health_score import _reconcile_full_provenance

    partial_adapter = MagicMock(provenance=DataProvenance.PARTIAL)
    measured_adapter = MagicMock(provenance=DataProvenance.MEASURED)
    unloaded_adapter = MagicMock(provenance=None)

    # measured composite + a partial source -> partial
    assert (
        _reconcile_full_provenance("measured", measured_adapter, partial_adapter, measured_adapter)
        == "partial"
    )
    # measured composite + all measured sources -> stays measured
    assert (
        _reconcile_full_provenance("measured", measured_adapter, measured_adapter, measured_adapter)
        == "measured"
    )
    # unloaded adapters (QUICK scope never touched them) don't force a downgrade
    assert _reconcile_full_provenance("measured", unloaded_adapter, unloaded_adapter) == "measured"
    # non-measured composites pass through unchanged
    assert _reconcile_full_provenance("partial", measured_adapter) == "partial"
    assert _reconcile_full_provenance("unknown", partial_adapter) == "unknown"


# =============================================================================
# ENDPOINT TESTS - get_component_health
# =============================================================================


def _measured_components() -> list:
    now = datetime.now(timezone.utc).isoformat()
    return [
        ComponentHealth(
            component_name="postgresql",
            status=ComponentStatus.HEALTHY,
            latency_ms=11,
            last_check=now,
        ),
        ComponentHealth(
            component_name="redis", status=ComponentStatus.HEALTHY, latency_ms=2, last_check=now
        ),
        ComponentHealth(
            component_name="falkordb",
            status=ComponentStatus.DEGRADED,
            latency_ms=210,
            last_check=now,
        ),
    ]


def _patch_measured_component_check():
    """Patch _fetch_component_health to return MEASURED component statuses
    (mirrors the live SupabaseHealthClient direct-probe path)."""
    return patch(
        "src.api.routes.health_score._fetch_component_health",
        new=AsyncMock(return_value=(_measured_components(), DataProvenance.MEASURED)),
    )


@pytest.mark.asyncio
async def test_get_component_health_success():
    """/components returns MEASURED component details from the real agent path."""
    with _patch_measured_component_check():
        result = await get_component_health()

    assert result.total_components == 3
    assert 0.0 <= result.component_health_score <= 1.0
    assert len(result.components) == result.total_components
    assert result.check_latency_ms >= 0
    assert result.data_provenance == DataProvenance.MEASURED


@pytest.mark.asyncio
async def test_get_component_health_score_calculation():
    """Component health score = (healthy + 0.5*degraded) / total over real data."""
    with _patch_measured_component_check():
        result = await get_component_health()

    expected_score = (
        result.healthy_count * 1.0 + result.degraded_count * 0.5
    ) / result.total_components
    assert abs(result.component_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_component_health_counts():
    """Test component health counts are correct."""
    with _patch_measured_component_check():
        result = await get_component_health()

    total = result.healthy_count + result.degraded_count + result.unhealthy_count
    assert total == result.total_components


@pytest.mark.asyncio
async def test_get_component_health_includes_real_components():
    """Components come from the real check, not the hardcoded mock list."""
    with _patch_measured_component_check():
        result = await get_component_health()

    component_names = [c.component_name for c in result.components]
    assert "postgresql" in component_names
    assert "redis" in component_names
    # The fabricated mock-only component 'opik' (latency 250) must NOT appear.
    assert not any(c.latency_ms == 250 for c in result.components)


# -----------------------------------------------------------------------------
# Silent-mock fix (F-010-backend follow-up): /components and /models previously
# called _get_mock_*() UNCONDITIONALLY, presenting fabricated component/model
# health in the dashboard as real with no flag. They must now (a) fail-closed
# (503) on agent ImportError in production, and (b) DISCLOSE placeholder
# provenance when mock-fallback is explicitly allowed (dev), mirroring #429.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_component_health_fails_closed_in_production(monkeypatch):
    """In a fail-closed environment, /components must raise 503 when the live
    component probe is unreachable, rather than return fabricated data."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with patch(
        "src.api.routes.health_score._fetch_component_health",
        new=AsyncMock(return_value=([], None)),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_component_health()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "health_source_unavailable"


@pytest.mark.asyncio
async def test_get_component_health_discloses_placeholder_provenance(monkeypatch):
    """When mock-fallback is explicitly allowed (dev) and the probe is
    unreachable, /components must DISCLOSE placeholder provenance rather than
    presenting fabricated data as real."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch(
        "src.api.routes.health_score._fetch_component_health",
        new=AsyncMock(return_value=([], None)),
    ):
        result = await get_component_health()
    assert result.data_provenance == "placeholder"
    assert result.total_components > 0


@pytest.mark.asyncio
async def test_get_component_health_measured_via_direct_probe():
    """REWIRED: /components now probes SupabaseHealthClient DIRECTLY (not the
    composite agent, whose output never surfaced per-component statuses) and tags
    genuinely measured component data 'measured', never 'placeholder'."""
    with _patch_measured_component_check():
        result = await get_component_health()
    assert result.data_provenance == DataProvenance.MEASURED
    # No fabricated 'opik @ 250ms' mock component leaks through.
    assert not any(c.latency_ms == 250 for c in result.components)


# =============================================================================
# C: health TREND must record only FULL-scope checks (no quick-100 pollution)
# =============================================================================


@pytest.mark.asyncio
async def test_run_health_check_quick_scope_not_recorded_in_history():
    """A QUICK (component-only) check must NOT pollute the health trend: its
    overall score is component-only (e.g. 100/A) and would render a misleadingly
    flat line. Only FULL all-dimension checks are faithful overall data points."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.QUICK,
            overall_health_score=100.0,
            health_grade=HealthGrade.A,
        )
        mock_result.check_latency_ms = 0
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.QUICK)

    assert len(_health_history) == 0


@pytest.mark.asyncio
async def test_run_health_check_untrusted_provenance_not_recorded_in_history():
    """A placeholder (dev mock fallback) or unknown (fail-closed default) full
    check must NOT be stored: recording it would replot as historical truth the
    very fabricated score the live dashboard refuses to render."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for provenance in ("placeholder", "unknown"):
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=87.6,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 0
            mock_result.data_provenance = provenance
            mock_execute.return_value = mock_result

            await run_health_check(scope=CheckScope.FULL)

    assert len(_health_history) == 0


# =============================================================================
# _fetch_component_health (direct SupabaseHealthClient probe — B)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_component_health_maps_probe_results(monkeypatch):
    """Each component is probed; ok -> healthy, degraded -> degraded, and the
    dimension is tagged MEASURED (real live statuses)."""
    from src.api.routes import health_score as hs

    class _FakeClient:
        async def check(self, endpoint: str):
            return {
                "/health/db": {"ok": True, "latency_ms": 10},
                "/health/cache": {"ok": True, "latency_ms": 2},
                "/health/vectors": {"ok": True, "latency_ms": 5},
                "/health/api": {"ok": True, "latency_ms": 3},
                "/health/queue": {"ok": False, "degraded": True, "latency_ms": 0},
            }[endpoint]

        async def close(self):
            return None

    monkeypatch.setattr(
        "src.agents.health_score.health_client.SupabaseHealthClient",
        lambda *a, **k: _FakeClient(),
    )
    components, provenance = await hs._fetch_component_health()
    assert provenance == DataProvenance.MEASURED
    assert len(components) == 5
    by_name = {c.component_name: c.status for c in components}
    assert by_name["database"] == ComponentStatus.HEALTHY
    assert by_name["message_queue"] == ComponentStatus.DEGRADED


@pytest.mark.asyncio
async def test_fetch_component_health_all_fail_returns_none(monkeypatch):
    """If EVERY probe raises, the backend is genuinely unreachable -> return None
    provenance so the caller fails closed (never fabricated)."""
    from src.api.routes import health_score as hs

    class _BoomClient:
        async def check(self, endpoint: str):
            raise RuntimeError("backend down")

        async def close(self):
            return None

    monkeypatch.setattr(
        "src.agents.health_score.health_client.SupabaseHealthClient",
        lambda *a, **k: _BoomClient(),
    )
    components, provenance = await hs._fetch_component_health()
    assert provenance is None
    assert components == []


@pytest.mark.asyncio
async def test_fetch_component_health_mixed_results_are_measured(monkeypatch):
    """The likely production case: a MIX of healthy / degraded / unhealthy probe
    dicts is tagged MEASURED with the correct per-component status mapping."""
    from src.api.routes import health_score as hs

    class _MixedClient:
        async def check(self, endpoint: str):
            return {
                "/health/db": {"ok": True, "latency_ms": 10},  # healthy
                "/health/cache": {"ok": True, "latency_ms": 2},  # healthy
                "/health/vectors": {"ok": False, "degraded": True},  # degraded
                "/health/api": {"ok": True, "latency_ms": 3},  # healthy
                "/health/queue": {"ok": False, "error": "broker down"},  # unhealthy
            }[endpoint]

        async def close(self):
            return None

    monkeypatch.setattr(
        "src.agents.health_score.health_client.SupabaseHealthClient",
        lambda *a, **k: _MixedClient(),
    )
    components, provenance = await hs._fetch_component_health()
    assert provenance == DataProvenance.MEASURED
    by_name = {c.component_name: c.status for c in components}
    assert by_name["database"] == ComponentStatus.HEALTHY
    assert by_name["vector_store"] == ComponentStatus.DEGRADED
    assert by_name["message_queue"] == ComponentStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_get_model_health_fails_closed_in_production(monkeypatch):
    """In a fail-closed environment, /models must raise 503 when the real data
    source is unavailable, rather than return fabricated model accuracy/metrics.
    (_health_source_client is None via the autouse fixture.)"""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with pytest.raises(HTTPException) as exc_info:
        await get_model_health()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "health_source_unavailable"


@pytest.mark.asyncio
async def test_get_model_health_discloses_placeholder_provenance(monkeypatch):
    """When mock-fallback is explicitly allowed (dev), /models must DISCLOSE
    placeholder provenance."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        result = await get_model_health()
    assert result.data_provenance == "placeholder"
    assert result.total_models > 0


@pytest.mark.asyncio
async def test_get_model_health_measured_from_dashboard():
    """REWIRED: /models now reads real rows from ml_model_health_dashboard and
    tags them measured/partial — NEVER the old hardcoded accuracy=0.89/auc=0.92."""
    real_models = [
        ModelHealth(
            model_id="d765b451", model_name="csu_lr_balanced_v1", status=ModelStatus.HEALTHY
        ),
        ModelHealth(model_id="5fd7826b", model_name="csu_lr_full_v1", status=ModelStatus.DEGRADED),
    ]
    with patch(
        "src.api.routes.health_score._fetch_model_health",
        return_value=(real_models, DataProvenance.PARTIAL),
    ):
        result = await get_model_health()
    assert result.data_provenance == DataProvenance.PARTIAL
    assert result.total_models == 2
    assert not any(m.model_id == "churn_predictor_v2" for m in result.models)
    assert not any(m.accuracy == 0.89 for m in result.models)
    assert not any(m.auc_roc == 0.92 for m in result.models)


# -----------------------------------------------------------------------------
# Silent-mock fix (C8 / HIGH#5 completion): /pipelines and /agents ALSO called
# _get_mock_*() UNCONDITIONALLY (PR #666 only fixed /components and /models),
# presenting fabricated pipeline freshness/row-counts and agent success-rate in
# the SystemHealth dashboard as real with no flag. They must now mirror
# /components+/models: (a) fail-closed (503) on agent ImportError in production,
# and (b) DISCLOSE placeholder provenance when mock-fallback is explicitly
# allowed (dev).
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pipeline_health_fails_closed_in_production(monkeypatch):
    """In a fail-closed environment, /pipelines must raise 503 when the real data
    source is unavailable, rather than return fabricated freshness/row-counts."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with pytest.raises(HTTPException) as exc_info:
        await get_pipeline_health()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "health_source_unavailable"


@pytest.mark.asyncio
async def test_get_pipeline_health_discloses_placeholder_provenance(monkeypatch):
    """When mock-fallback is explicitly allowed (dev), /pipelines must DISCLOSE
    placeholder provenance rather than presenting fabricated data as real."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        result = await get_pipeline_health()
    assert result.data_provenance == "placeholder"
    assert result.total_pipelines > 0


@pytest.mark.asyncio
async def test_get_pipeline_health_measured_from_etl():
    """REWIRED: /pipelines now reads real latest-run-per-pipeline from
    etl_pipeline_metrics, tagged measured — never the hardcoded mock rows."""
    real_pipelines = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=12.0,
            status=PipelineStatus.HEALTHY,
        ),
    ]
    with patch(
        "src.api.routes.health_score._fetch_pipeline_health",
        return_value=(real_pipelines, DataProvenance.MEASURED),
    ):
        result = await get_pipeline_health()
    assert result.data_provenance == DataProvenance.MEASURED
    assert result.total_pipelines == 1
    # The fabricated mock pipelines must not appear.
    assert not any(p.pipeline_name == "hcp_data_ingestion" for p in result.pipelines)


@pytest.mark.asyncio
async def test_get_agent_health_fails_closed_in_production(monkeypatch):
    """In a fail-closed environment, /agents must raise 503 when the real data
    source is unavailable, rather than return fabricated success-rate/latency."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with pytest.raises(HTTPException) as exc_info:
        await get_agent_health()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "health_source_unavailable"


@pytest.mark.asyncio
async def test_get_agent_health_discloses_placeholder_provenance(monkeypatch):
    """When mock-fallback is explicitly allowed (dev), /agents must DISCLOSE
    placeholder provenance."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        result = await get_agent_health()
    assert result.data_provenance == "placeholder"
    assert result.total_agents > 0


@pytest.mark.asyncio
async def test_get_agent_health_partial_when_no_recent_telemetry():
    """REWIRED: /agents now reads the real agent_registry roster; with no recent
    telemetry it reports PARTIAL with NULL runtime metrics (never a fabricated
    success_rate=0.98 / latency=150)."""
    roster_agents = [
        AgentHealth(
            agent_name="gap_analyzer",
            tier=2,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        ),
        AgentHealth(
            agent_name="causal_impact",
            tier=3,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        ),
    ]
    with patch(
        "src.api.routes.health_score._fetch_agent_health",
        return_value=(roster_agents, DataProvenance.PARTIAL),
    ):
        result = await get_agent_health()
    assert result.data_provenance == DataProvenance.PARTIAL
    assert result.total_agents == 2
    # Runtime metrics are honestly null, not fabricated.
    assert all(a.success_rate is None for a in result.agents)
    assert all(a.avg_latency_ms is None for a in result.agents)


# =============================================================================
# ENDPOINT TESTS - get_model_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_model_health_success():
    """Test get_model_health returns model details."""
    result = await get_model_health()

    assert result.total_models > 0
    assert result.model_health_score >= 0.0
    assert result.model_health_score <= 1.0
    assert len(result.models) == result.total_models
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_model_health_score_calculation():
    """Test model health score is calculated correctly."""
    result = await get_model_health()

    expected_score = (
        result.healthy_count * 1.0 + result.degraded_count * 0.5
    ) / result.total_models

    assert abs(result.model_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_model_health_empty_models():
    """Test model health handles empty model list."""
    with patch("src.api.routes.health_score._get_mock_model_health", return_value=[]):
        result = await get_model_health()

        assert result.total_models == 0
        assert result.model_health_score == 1.0  # Default for empty list


@pytest.mark.asyncio
async def test_get_model_health_includes_metrics():
    """Test models include performance metrics."""
    result = await get_model_health()

    for model in result.models:
        assert hasattr(model, "model_id")
        assert hasattr(model, "status")
        assert model.error_rate >= 0.0


# =============================================================================
# ENDPOINT TESTS - get_pipeline_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_pipeline_health_success():
    """Test get_pipeline_health returns pipeline details."""
    result = await get_pipeline_health()

    assert result.total_pipelines > 0
    assert result.pipeline_health_score >= 0.0
    assert result.pipeline_health_score <= 1.0
    assert len(result.pipelines) == result.total_pipelines
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_pipeline_health_score_calculation():
    """Test pipeline health score is calculated correctly."""
    result = await get_pipeline_health()

    expected_score = (
        result.healthy_count * 1.0 + result.stale_count * 0.5
    ) / result.total_pipelines

    assert abs(result.pipeline_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_pipeline_health_counts():
    """Test pipeline health counts are correct."""
    result = await get_pipeline_health()

    total = result.healthy_count + result.stale_count + result.failed_count
    assert total == result.total_pipelines


@pytest.mark.asyncio
async def test_get_pipeline_health_empty_pipelines():
    """Test pipeline health handles empty pipeline list."""
    with patch("src.api.routes.health_score._get_mock_pipeline_health", return_value=[]):
        result = await get_pipeline_health()

        assert result.total_pipelines == 0
        assert result.pipeline_health_score == 1.0


# =============================================================================
# ENDPOINT TESTS - get_agent_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_agent_health_success():
    """Test get_agent_health returns agent details."""
    result = await get_agent_health()

    assert result.total_agents > 0
    assert result.agent_health_score >= 0.0
    assert result.agent_health_score <= 1.0
    assert len(result.agents) == result.total_agents
    assert result.check_latency_ms >= 0


@pytest.mark.asyncio
async def test_get_agent_health_score_calculation():
    """Test agent health score is calculated correctly."""
    result = await get_agent_health()

    expected_score = result.available_count / result.total_agents
    assert abs(result.agent_health_score - expected_score) < 0.01


@pytest.mark.asyncio
async def test_get_agent_health_by_tier():
    """Test agent health groups agents by tier."""
    result = await get_agent_health()

    assert isinstance(result.by_tier, dict)
    assert len(result.by_tier) > 0

    # Verify tier counts
    total_in_tiers = sum(result.by_tier.values())
    assert total_in_tiers == result.total_agents


@pytest.mark.asyncio
async def test_get_agent_health_empty_agents():
    """Test agent health handles empty agent list."""
    with patch("src.api.routes.health_score._get_mock_agent_health", return_value=[]):
        result = await get_agent_health()

        assert result.total_agents == 0
        assert result.agent_health_score == 1.0


# =============================================================================
# ENDPOINT TESTS - get_health_history
# =============================================================================


@pytest.mark.asyncio
async def test_get_health_history_empty():
    """Empty history must report avg=None and trend='unknown' (NOT a fabricated
    0.0 average / 'stable' trend the dashboard would render as a real metric)."""
    result = await get_health_history(limit=20)

    assert result.total_checks == 0
    assert len(result.checks) == 0
    assert result.avg_health_score is None
    assert result.trend == "unknown"


@pytest.mark.asyncio
async def test_get_health_history_with_data():
    """Test get_health_history returns historical data."""
    # Add some history
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_result.data_provenance = "measured"
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)
        await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.total_checks == 2
    assert len(result.checks) == 2
    assert result.avg_health_score == 85.0


@pytest.mark.asyncio
async def test_get_health_history_items_carry_provenance():
    """History items must carry the recorded check's provenance on the wire so
    the frontend can apply the same fail-closed trust rule to historical rows."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_result.data_provenance = "partial"
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.checks[0].data_provenance == "partial"


@pytest.mark.asyncio
async def test_get_health_history_respects_limit():
    """Test get_health_history respects limit parameter."""
    # Add 10 entries
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_result.data_provenance = "measured"
        mock_execute.return_value = mock_result

        for _ in range(10):
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=5)

    assert len(result.checks) == 5


@pytest.mark.asyncio
async def test_get_health_history_trend_improving():
    """Test trend calculation for improving health."""
    # Add entries with improving scores
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for score in [70.0, 75.0, 80.0, 85.0, 90.0]:
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=score,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 100
            mock_result.critical_issues = []
            mock_result.timestamp = datetime.now(timezone.utc).isoformat()
            mock_result.data_provenance = "measured"
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "improving"


@pytest.mark.asyncio
async def test_get_health_history_trend_declining():
    """Test trend calculation for declining health."""
    # Add entries with declining scores
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for score in [90.0, 85.0, 80.0, 75.0, 70.0]:
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=score,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 100
            mock_result.critical_issues = []
            mock_result.timestamp = datetime.now(timezone.utc).isoformat()
            mock_result.data_provenance = "measured"
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "declining"


@pytest.mark.asyncio
async def test_get_health_history_trend_stable():
    """Test trend calculation for stable health."""
    # Add entries with stable scores
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        for _ in range(5):
            mock_result = MagicMock(
                check_id="",
                check_scope=CheckScope.FULL,
                overall_health_score=85.0,
                health_grade=HealthGrade.B,
            )
            mock_result.check_latency_ms = 100
            mock_result.critical_issues = []
            mock_result.timestamp = datetime.now(timezone.utc).isoformat()
            mock_result.data_provenance = "measured"
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "stable"


# =============================================================================
# DURABLE HISTORY (migration 096) — write-through, windowed read, heartbeat
# =============================================================================
# The autouse _no_live_health_sources fixture keeps _health_source_client None,
# so every test ABOVE exercises the in-memory fallback path unchanged. The
# tests below patch in a fake Supabase-style client to pin the durable path.


class _FakeHistoryQuery:
    """Chained-builder fake for the two history relations."""

    def __init__(self, db: "_FakeHistoryDB", table: str) -> None:
        self._db = db
        self._table = table
        self._op = "select"
        self._cols = ""
        self._payload: dict | None = None

    def select(self, cols: str):
        self._op = "select"
        self._cols = cols
        return self

    def insert(self, payload: dict):
        self._op = "insert"
        self._payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    def gte(self, col, value):
        self._db.gte_args.append((self._table, col, value))
        return self

    def lt(self, _col, value):
        self._payload = {"lt": value}
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        self._db.limit_args.append((self._table, n))
        return self

    def execute(self):
        if self._db.fail:
            raise RuntimeError("db down")
        if self._op == "insert":
            self._db.inserted.append((self._table, self._payload))
            return MagicMock(data=[self._payload])
        if self._op == "delete":
            self._db.deleted.append((self._table, self._payload))
            return MagicMock(data=[])
        if self._table == "health_check_history_daily":
            return MagicMock(data=self._db.daily_rows)
        # health_check_history: the rate-limit probe selects only checked_at;
        # the windowed read selects the full column list.
        if self._cols.strip() == "checked_at":
            return MagicMock(data=self._db.last_rows)
        return MagicMock(data=self._db.raw_rows)


class _FakeHistoryDB:
    """Records inserts/deletes; serves canned rows for the two reads."""

    def __init__(self, last_rows=None, raw_rows=None, daily_rows=None, fail=False) -> None:
        self.last_rows = last_rows or []
        self.raw_rows = raw_rows or []
        self.daily_rows = daily_rows or []
        self.fail = fail
        self.inserted: list = []
        self.deleted: list = []
        self.gte_args: list = []
        self.limit_args: list = []

    def table(self, name: str) -> _FakeHistoryQuery:
        return _FakeHistoryQuery(self, name)


def _full_check_result(score: float = 85.0, provenance: str = "partial") -> MagicMock:
    result = MagicMock(
        check_id="hs_test123",
        check_scope=CheckScope.FULL,
        overall_health_score=score,
        health_grade=HealthGrade.B,
        component_health_score=0.9,
        model_health_score=0.8,
        pipeline_health_score=0.85,
        agent_health_score=0.95,
    )
    result.critical_issues = []
    result.warnings = []
    result.timestamp = datetime.now(timezone.utc).isoformat()
    result.data_provenance = provenance
    return result


@pytest.mark.asyncio
async def test_get_health_history_serves_durable_window():
    """With a reachable DB, /history serves the durable table: raw checks come
    back ASCENDING (chart contract), daily buckets ride along, the average is
    weighted over the WHOLE window, and window_days flags the durable path."""
    raw_desc = [
        {
            "check_id": f"hs_{d}",
            "checked_at": f"2026-07-0{d}T12:00:00+00:00",
            "overall_health_score": 80.0 + d,
            "health_grade": "B",
            "critical_issues_count": 0,
            "data_provenance": "partial",
        }
        for d in (6, 5, 4)  # DB returns newest-first
    ]
    daily = [
        {
            "day": "2026-07-04",
            "avg_score": 84.0,
            "min_score": 83.0,
            "max_score": 85.0,
            "checks_count": 4,
            "data_provenance": "partial",
        },
        {
            "day": "2026-07-05",
            "avg_score": 85.0,
            "min_score": 84.0,
            "max_score": 86.0,
            "checks_count": 4,
            "data_provenance": "partial",
        },
        {
            "day": "2026-07-06",
            "avg_score": 86.0,
            "min_score": 86.0,
            "max_score": 86.0,
            "checks_count": 2,
            "data_provenance": "measured",
        },
    ]
    db = _FakeHistoryDB(raw_rows=raw_desc, daily_rows=daily)
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        result = await get_health_history(limit=20, days=30)

    assert result.window_days == 30
    assert [c.check_id for c in result.checks] == ["hs_4", "hs_5", "hs_6"]
    assert [p.date for p in result.daily] == ["2026-07-04", "2026-07-05", "2026-07-06"]
    assert result.total_checks == 10
    expected_avg = (84.0 * 4 + 85.0 * 4 + 86.0 * 2) / 10
    assert result.avg_health_score == pytest.approx(expected_avg)
    # Exactly 3 daily points -> first-3 and last-3 coincide -> stable.
    assert result.trend == "stable"


@pytest.mark.asyncio
async def test_get_health_history_durable_trend_from_daily_averages():
    """Trend comes from daily averages once >=3 days exist (not raw checks)."""
    daily = [
        {
            "day": f"2026-07-0{i}",
            "avg_score": s,
            "min_score": s,
            "max_score": s,
            "checks_count": 1,
            "data_provenance": "partial",
        }
        for i, s in enumerate([70.0, 70.0, 70.0, 90.0, 90.0, 90.0], start=1)
    ]
    db = _FakeHistoryDB(raw_rows=[], daily_rows=daily)
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        result = await get_health_history(limit=20, days=30)
    assert result.trend == "improving"


@pytest.mark.asyncio
async def test_get_health_history_daily_window_is_exactly_days_dates():
    """days=N must never admit N+1 dates: the inclusive gte cutoff starts at
    today-(N-1) UTC (today's partial bucket is day 1) and the daily read is
    capped at N rows (codex round-2 LOW: days=30 previously spanned 31)."""
    db = _FakeHistoryDB(raw_rows=[], daily_rows=[])
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        before = datetime.now(timezone.utc)
        await get_health_history(limit=20, days=30)
        after = datetime.now(timezone.utc)

    daily_gtes = [(c, v) for (t, c, v) in db.gte_args if t == "health_check_history_daily"]
    assert len(daily_gtes) == 1
    col, cutoff = daily_gtes[0]
    assert col == "day"
    # Two candidates only in case the call straddled a UTC midnight.
    expected = {
        (before - timedelta(days=29)).date().isoformat(),
        (after - timedelta(days=29)).date().isoformat(),
    }
    assert cutoff in expected
    assert [n for (t, n) in db.limit_args if t == "health_check_history_daily"] == [30]


@pytest.mark.asyncio
async def test_get_health_history_durable_read_failure_falls_back_to_memory():
    """A failing durable read must fall back to the in-memory list (old
    behavior), with daily EMPTY and window_days null — repackaging
    minutes-scale process history as day buckets would fabricate a month."""
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_execute.return_value = _full_check_result(score=88.0)
        await run_health_check(scope=CheckScope.FULL)

    db = _FakeHistoryDB(fail=True)
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        result = await get_health_history(limit=20, days=30)

    assert result.window_days is None
    assert result.daily == []
    assert len(result.checks) == 1
    assert result.checks[0].overall_health_score == 88.0


@pytest.mark.asyncio
async def test_get_health_history_durable_empty_is_honest_empty():
    """A reachable-but-empty table is an honest zero — no in-memory bleed-in,
    no fabricated average/trend."""
    db = _FakeHistoryDB(raw_rows=[], daily_rows=[])
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        result = await get_health_history(limit=20, days=30)
    assert result.total_checks == 0
    assert result.checks == []
    assert result.daily == []
    assert result.avg_health_score is None
    assert result.trend == "unknown"


def test_record_full_check_writes_durable_row_and_prunes():
    """A trusted FULL check writes one durable row (faithful payload) and
    piggybacks the 90-day retention sweep on the same path."""
    db = _FakeHistoryDB(last_rows=[])
    result = _full_check_result(score=85.5, provenance="partial")
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        _record_full_check(result)

    assert len(_health_history) == 1  # in-memory cache still fed
    assert [t for t, _ in db.inserted] == ["health_check_history"]
    payload = db.inserted[0][1]
    assert payload["overall_health_score"] == 85.5
    assert payload["health_grade"] == "B"
    assert payload["data_provenance"] == "partial"
    assert payload["check_scope"] == "full"
    assert db.deleted and db.deleted[0][0] == "health_check_history"


def test_record_full_check_rate_limited_by_fresh_row():
    """A durable row younger than the write interval suppresses the insert
    (the dashboard polls /full every 60s — per-minute rows add no trend
    resolution). The in-memory cache still records."""
    fresh = datetime.now(timezone.utc).isoformat()
    db = _FakeHistoryDB(last_rows=[{"checked_at": fresh}])
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        _record_full_check(_full_check_result())

    assert db.inserted == []
    assert len(_health_history) == 1


def test_record_full_check_stale_last_row_writes():
    """A durable row older than the write interval does NOT suppress."""
    stale = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    db = _FakeHistoryDB(last_rows=[{"checked_at": stale}])
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        _record_full_check(_full_check_result())
    assert len(db.inserted) == 1


def test_record_full_check_untrusted_never_reaches_durable():
    """placeholder/unknown provenance is rejected before any storage — the DB
    CHECK constraint is the backstop, this gate is the front door."""
    db = _FakeHistoryDB()
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        _record_full_check(_full_check_result(provenance="placeholder"))
    assert db.inserted == []
    assert len(_health_history) == 0


def test_record_full_check_durable_write_failure_never_raises():
    """Recording is best-effort: a DB failure must not fail the health check."""
    db = _FakeHistoryDB(fail=True)
    with patch("src.api.routes.health_score._health_source_client", return_value=db):
        _record_full_check(_full_check_result())  # must not raise
    assert len(_health_history) == 1


@pytest.mark.asyncio
async def test_run_scheduled_full_check_records_like_the_endpoint():
    """The lifespan heartbeat entry point runs a FULL check, assigns a check_id
    and records through the exact same gate as GET /check?scope=full."""
    db = _FakeHistoryDB(last_rows=[])
    with (
        patch("src.api.routes.health_score._execute_health_check") as mock_execute,
        patch("src.api.routes.health_score._health_source_client", return_value=db),
    ):
        mock_execute.return_value = _full_check_result(score=91.0)
        await run_scheduled_full_check()

    assert len(_health_history) == 1
    assert _health_history[0].check_id.startswith("hs_")
    assert len(db.inserted) == 1
    assert db.inserted[0][1]["overall_health_score"] == 91.0


def test_trend_from_scores_thresholds():
    """±5 dead band; 'unknown' below 3 points — never a fabricated 'stable'."""
    assert _trend_from_scores([]) == "unknown"
    assert _trend_from_scores([80.0, 90.0]) == "unknown"
    assert _trend_from_scores([70.0, 70.0, 70.0, 90.0, 90.0, 90.0]) == "improving"
    assert _trend_from_scores([90.0, 90.0, 90.0, 70.0, 70.0, 70.0]) == "declining"
    assert _trend_from_scores([85.0, 84.0, 86.0, 85.0]) == "stable"


# =============================================================================
# ENDPOINT TESTS - get_service_status
# =============================================================================


@pytest.mark.asyncio
async def test_get_service_status_agent_available():
    """Test get_service_status when agent is available."""
    with patch("src.agents.health_score.HealthScoreAgent"):
        result = await get_service_status()

        assert result.status == "healthy"
        assert result.agent_available is True


@pytest.mark.asyncio
async def test_get_service_status_agent_unavailable():
    """Test get_service_status when agent is unavailable."""
    # Patch the import to raise ImportError
    import sys

    with patch.dict(sys.modules, {"src.agents.health_score": None}):
        result = await get_service_status()

        assert result.status == "degraded"
        assert result.agent_available is False


@pytest.mark.asyncio
async def test_get_service_status_with_history():
    """Test get_service_status includes history metrics."""
    # Add some history
    with patch("src.api.routes.health_score._execute_health_check") as mock_execute:
        mock_result = MagicMock(
            check_id="",
            check_scope=CheckScope.FULL,
            overall_health_score=85.0,
            health_grade=HealthGrade.B,
        )
        mock_result.check_latency_ms = 100
        mock_result.critical_issues = []
        mock_result.timestamp = datetime.now(timezone.utc).isoformat()
        mock_result.data_provenance = "measured"
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)

    with patch("src.agents.health_score.HealthScoreAgent"):
        result = await get_service_status()

        assert result.last_check is not None
        assert result.checks_24h > 0
        assert result.avg_check_latency_ms >= 0


# =============================================================================
# HELPER FUNCTION TESTS - _execute_health_check
# =============================================================================


@pytest.mark.asyncio
async def test_execute_health_check_with_real_agent(mock_agent_result):
    """Test _execute_health_check uses real agent when available."""
    mock_agent = MagicMock()
    mock_agent.check_health = AsyncMock(return_value=mock_agent_result)

    with patch("src.agents.health_score.HealthScoreAgent", return_value=mock_agent):
        result = await _execute_health_check(CheckScope.FULL)

        assert result.overall_health_score == 85.5
        assert result.health_grade == HealthGrade.B
        mock_agent.check_health.assert_called_once_with(scope="full")


@pytest.mark.asyncio
async def test_execute_health_check_quick_mode(mock_agent_result):
    """Test _execute_health_check uses quick_check for QUICK scope."""
    mock_agent = MagicMock()
    mock_agent.quick_check = AsyncMock(return_value=mock_agent_result)

    with patch("src.agents.health_score.HealthScoreAgent", return_value=mock_agent):
        await _execute_health_check(CheckScope.QUICK)

        mock_agent.quick_check.assert_called_once()


@pytest.mark.asyncio
async def test_execute_health_check_passes_unmeasured_scores_as_none(mock_agent_result):
    """Codex F1.1: an unmeasured per-dimension score (None on the agent output)
    must pass through to HealthScoreResponse as None, NOT be coerced to 0.0/1.0
    and presented to the dashboard as a real measurement."""
    mock_agent_result.component_health_score = 0.9  # measured
    mock_agent_result.model_health_score = None  # unmeasured
    mock_agent_result.pipeline_health_score = None
    mock_agent_result.agent_health_score = None
    mock_agent_result.data_provenance = "partial"
    mock_agent = MagicMock()
    mock_agent.check_health = AsyncMock(return_value=mock_agent_result)

    with patch("src.agents.health_score.HealthScoreAgent", return_value=mock_agent):
        result = await _execute_health_check(CheckScope.FULL)

    assert result.component_health_score == 0.9
    assert result.model_health_score is None
    assert result.pipeline_health_score is None
    assert result.agent_health_score is None
    assert result.data_provenance == "partial"


@pytest.mark.asyncio
async def test_execute_health_check_falls_back_to_mock_when_explicitly_allowed(monkeypatch):
    """Mock-fallback is gated on E2I_REQUIRE_AGENT_IMPORT=0 or ENVIRONMENT in dev set.

    Closed-by-default policy means CI default (env unset) raises 503 on
    ImportError. This test pins the explicit dev opt-in path still works.
    """
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        result = await _execute_health_check(CheckScope.FULL)

        assert result.overall_health_score > 0
        assert result.warnings[0] == "Using mock data - Health Score agent not available"


@pytest.mark.asyncio
async def test_execute_health_check_raises_503_when_mock_disabled(monkeypatch):
    """Closed-by-default: ImportError must raise 503 when mock-fallback is disabled.

    Pin the new fail-closed behavior (codex iter-1 H1 fix) — unset/misspelled
    ENVIRONMENT must NOT silently enable fabricated data.
    """
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        with pytest.raises(HTTPException) as exc_info:
            await _execute_health_check(CheckScope.FULL)
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"] == "agent_unavailable"


@pytest.mark.asyncio
async def test_execute_health_check_agent_exception():
    """Test _execute_health_check handles agent exceptions."""
    mock_agent = MagicMock()
    mock_agent.check_health = AsyncMock(side_effect=RuntimeError("Agent error"))

    with patch("src.agents.health_score.HealthScoreAgent", return_value=mock_agent):
        with pytest.raises(RuntimeError):
            await _execute_health_check(CheckScope.FULL)


# =============================================================================
# HELPER FUNCTION TESTS - _generate_mock_health_response
# =============================================================================


def test_generate_mock_health_response_full_scope():
    """Test mock response generation for FULL scope."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.FULL, start_time)

    assert result.check_scope == CheckScope.FULL
    assert result.component_statuses is not None
    assert result.model_metrics is not None
    assert result.pipeline_statuses is not None
    assert result.agent_statuses is not None


def test_generate_mock_health_response_quick_scope():
    """Test mock response generation for QUICK scope."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.QUICK, start_time)

    assert result.check_scope == CheckScope.QUICK
    assert result.component_statuses is not None
    assert result.model_metrics is None


def test_generate_mock_health_response_models_scope():
    """Test mock response generation for MODELS scope."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.MODELS, start_time)

    assert result.check_scope == CheckScope.MODELS
    assert result.model_metrics is not None
    assert result.component_statuses is None


def test_generate_mock_health_response_grade_A():
    """Test grade A assignment for high scores."""
    import time

    start_time = time.time()

    with patch("src.api.routes.health_score._get_mock_component_health"):
        with patch("src.api.routes.health_score._get_mock_model_health"):
            with patch("src.api.routes.health_score._get_mock_pipeline_health"):
                with patch("src.api.routes.health_score._get_mock_agent_health"):
                    # Mock all scores to be 1.0
                    result = _generate_mock_health_response(CheckScope.FULL, start_time)

                    # Overall should be high
                    assert result.overall_health_score >= 80


def test_generate_mock_health_response_tagged_placeholder():
    """The dev-offline mock fallback MUST tag data_provenance='placeholder' (not
    the 'unknown' default) so its hardcoded sample scores are never mistaken for
    real measurements by consumers / the dashboard chart."""
    import time

    result = _generate_mock_health_response(CheckScope.FULL, time.time())
    assert result.data_provenance == "placeholder"


def test_generate_mock_health_response_includes_recommendations():
    """Test mock response includes recommendations."""
    import time

    start_time = time.time()
    result = _generate_mock_health_response(CheckScope.FULL, start_time)

    assert len(result.recommendations) > 0


# =============================================================================
# HELPER FUNCTION TESTS - _get_mock_* functions
# =============================================================================


def test_get_mock_component_health():
    """Test _get_mock_component_health returns expected components."""
    components = _get_mock_component_health()

    assert len(components) > 0
    component_names = [c.component_name for c in components]
    assert "postgresql" in component_names
    assert "redis" in component_names
    assert "falkordb" in component_names
    assert "mlflow" in component_names


def test_get_mock_model_health():
    """Test _get_mock_model_health returns expected models."""
    models = _get_mock_model_health()

    assert len(models) > 0
    for model in models:
        assert model.model_id is not None
        assert model.model_name is not None
        assert model.status in [ModelStatus.HEALTHY, ModelStatus.DEGRADED, ModelStatus.UNHEALTHY]


def test_get_mock_pipeline_health():
    """Test _get_mock_pipeline_health returns expected pipelines."""
    pipelines = _get_mock_pipeline_health()

    assert len(pipelines) > 0
    for pipeline in pipelines:
        assert pipeline.pipeline_name is not None
        assert pipeline.status in [
            PipelineStatus.HEALTHY,
            PipelineStatus.STALE,
            PipelineStatus.FAILED,
        ]
        assert pipeline.rows_processed >= 0


def test_get_mock_agent_health():
    """Test _get_mock_agent_health returns expected agents."""
    agents = _get_mock_agent_health()

    assert len(agents) > 0
    for agent in agents:
        assert agent.agent_name is not None
        assert agent.tier >= 0
        assert agent.tier <= 5
        assert isinstance(agent.available, bool)


# =============================================================================
# HELPER FUNCTION TESTS - _generate_recommendations
# =============================================================================


def test_generate_recommendations_all_healthy():
    """Test recommendations when all scores are healthy."""
    recommendations = _generate_recommendations(0.9, 0.9, 0.9, 0.9)

    assert len(recommendations) == 1
    assert "Continue monitoring" in recommendations[0]


def test_generate_recommendations_low_component_score():
    """Test recommendations when component score is low."""
    recommendations = _generate_recommendations(0.7, 0.9, 0.9, 0.9)

    assert any("component" in r.lower() for r in recommendations)


def test_generate_recommendations_low_model_score():
    """Test recommendations when model score is low."""
    recommendations = _generate_recommendations(0.9, 0.7, 0.9, 0.9)

    assert any("model" in r.lower() for r in recommendations)


def test_generate_recommendations_low_pipeline_score():
    """Test recommendations when pipeline score is low."""
    recommendations = _generate_recommendations(0.9, 0.9, 0.7, 0.9)

    assert any("pipeline" in r.lower() for r in recommendations)


def test_generate_recommendations_low_agent_score():
    """Test recommendations when agent score is low."""
    recommendations = _generate_recommendations(0.9, 0.9, 0.9, 0.7)

    assert any("agent" in r.lower() for r in recommendations)


def test_generate_recommendations_multiple_issues():
    """Test recommendations when multiple scores are low."""
    recommendations = _generate_recommendations(0.7, 0.7, 0.7, 0.7)

    assert len(recommendations) == 4


# =============================================================================
# REAL-DATA FETCHERS - _fetch_model_health / _fetch_pipeline_health /
# _fetch_agent_health (the rewire that replaces the hardcoded placeholders)
# =============================================================================


class _FakeQuery:
    """Minimal stub of the supabase-py query builder; ignores filters and
    returns the canned rows for the table on execute()."""

    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        # Honor ``.eq(col, val)`` so the is_synthetic=False structural guard is
        # actually exercised. Default-to-val on a missing key keeps rows that
        # omit the column (backward compatible with fixtures predating it).
        self._eq_filters = getattr(self, "_eq_filters", [])
        self._eq_filters.append((col, val))
        return self

    def gte(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    @property
    def not_(self):
        self._negate = True
        return self

    def is_(self, col, val):
        # Honor ``query.not_.is_(col, "null")`` so the model-health metric-presence
        # filter is actually exercised by the fake (excludes rows where col IS NULL).
        if getattr(self, "_negate", False) and str(val).lower() == "null":
            self._exclude_null_cols = getattr(self, "_exclude_null_cols", [])
            self._exclude_null_cols.append(col)
        self._negate = False
        return self

    def execute(self):
        rows = list(self._rows)
        for col, val in getattr(self, "_eq_filters", []):
            rows = [r for r in rows if r.get(col, val) == val]
        for col in getattr(self, "_exclude_null_cols", []):
            rows = [r for r in rows if r.get(col) is not None]
        return MagicMock(data=rows)


class _FakeDB:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _FakeQuery(self._tables.get(name, []))


def _patch_health_client(db):
    return patch("src.api.routes.health_score._health_source_client", return_value=db)


def test_fetch_model_health_none_when_no_client():
    """No client -> (empty, None) so the endpoint fails closed, not 500."""
    with _patch_health_client(None):
        models, prov = _fetch_model_health()
    assert models == []
    assert prov is None


def test_fetch_model_health_excludes_metricless_rows():
    """WS-BACKEND: rows with NO performance metric (latest_metric_value NULL) — the
    362 synthetic experiment artifacts that polluted the card as "362/362" blank —
    are EXCLUDED. The Model-Health card surfaces only models carrying a real metric,
    so an all-metric-less dashboard reads UNKNOWN (honest empty), never a wall of
    blank rows. (A model WITH a metric still shows its real value, never a hardcoded
    0.89 — see test_fetch_model_health_partial_even_with_auc_metric.)"""
    db = _FakeDB(
        {
            "ml_model_health_dashboard": [
                {
                    "model_id": "d765b451",
                    "model_name": "csu_lr_balanced_v1",
                    "model_stage": "production",
                    "health_status": "healthy",
                    "latest_metric_value": None,
                    "primary_metric": "auc",
                },
                {
                    "model_id": "5fd7826b",
                    "model_name": "csu_lr_full_v1",
                    "model_stage": "production",
                    "health_status": "critical",
                    "latest_metric_value": None,
                    "primary_metric": "auc",
                },
            ]
        }
    )
    with _patch_health_client(db):
        models, prov = _fetch_model_health()
    assert models == []
    assert prov == DataProvenance.UNKNOWN


def test_fetch_model_health_excludes_synthetic_even_with_metric():
    """WS-BACKEND (codex r2 RESOLUTION-4): a synthetic experiment artifact that DOES
    carry a performance metric must STILL be excluded — structurally, via the
    is_synthetic=False guard (migration 031 exposes is_synthetic on the view) — not
    merely because synthetic rows happen to be metric-less today. The live registry
    holds 720 is_synthetic=true rows under stage IN (production, staging); only the
    14 real models (all is_synthetic=false) may reach the card. Here the synthetic
    row has a real-looking 0.91 auc and must NOT survive; only the gold-standard
    model does."""
    db = _FakeDB(
        {
            "ml_model_health_dashboard": [
                {
                    "model_id": "synth1",
                    "model_name": "scenario_synth_v1",
                    "model_stage": "staging",
                    "health_status": "healthy",
                    "latest_metric_value": 0.91,
                    "primary_metric": "auc",
                    "is_synthetic": True,
                },
                {
                    "model_id": "real1",
                    "model_name": "initiation_kisqali_goldstd_lr_v1",
                    "model_stage": "staging",
                    "health_status": "healthy",
                    "latest_metric_value": 0.68,
                    "primary_metric": "auc",
                    "is_synthetic": False,
                },
            ]
        }
    )
    with _patch_health_client(db):
        models, prov = _fetch_model_health()
    assert [m.model_name for m in models] == ["initiation_kisqali_goldstd_lr_v1"]
    assert models[0].auc_roc == 0.68
    assert prov == DataProvenance.PARTIAL


def test_fetch_model_health_partial_even_with_auc_metric():
    """A real auc value populates auc_roc, but precision/recall/f1/latency/
    predictions/error_rate have no source (ml_performance_metrics empty), so the
    dimension is honestly PARTIAL — never MEASURED — and the unsourced count/rate
    fields are NULL, not a fabricated 0."""
    db = _FakeDB(
        {
            "ml_model_health_dashboard": [
                {
                    "model_id": "m1",
                    "model_name": "m1",
                    "model_stage": "production",
                    "health_status": "healthy",
                    "latest_metric_value": 0.83,
                    "primary_metric": "auc",
                },
            ]
        }
    )
    with _patch_health_client(db):
        models, prov = _fetch_model_health()
    assert prov == DataProvenance.PARTIAL
    assert models[0].auc_roc == 0.83
    assert models[0].predictions_last_24h is None
    assert models[0].error_rate is None


def test_fetch_model_health_unknown_when_empty():
    """Client present but zero production rows -> UNKNOWN (honest empty)."""
    with _patch_health_client(_FakeDB({"ml_model_health_dashboard": []})):
        models, prov = _fetch_model_health()
    assert models == []
    assert prov == DataProvenance.UNKNOWN


def test_fetch_pipeline_health_latest_per_pipeline_and_status():
    """Latest run per pipeline; failed status maps FAILED; stale run maps STALE."""
    db = _FakeDB(
        {
            "etl_pipeline_metrics": [
                {
                    "pipeline_name": "rwd_ingest",
                    "run_end": "2019-01-01T00:00:00+00:00",
                    "run_start": "2019-01-01T00:00:00+00:00",
                    "records_processed": 5,
                    "status": "completed",
                },
                {
                    "pipeline_name": "broken",
                    "run_end": "2026-06-12T00:00:00+00:00",
                    "run_start": "2026-06-12T00:00:00+00:00",
                    "records_processed": 0,
                    "status": "failed",
                },
            ]
        }
    )
    with _patch_health_client(db):
        pipelines, prov = _fetch_pipeline_health()
    assert prov == DataProvenance.MEASURED
    by_name = {p.pipeline_name: p for p in pipelines}
    assert by_name["rwd_ingest"].status == PipelineStatus.STALE  # 2019 run -> very stale
    assert by_name["broken"].status == PipelineStatus.FAILED


def test_fetch_agent_health_partial_null_metrics_without_telemetry():
    """Roster present, no telemetry rows -> PARTIAL with NULL runtime metrics,
    NOT a fabricated success_rate."""
    db = _FakeDB(
        {
            "agent_registry": [
                {"agent_name": "gap_analyzer", "agent_tier": 2, "is_active": True},
                {"agent_name": "explainer", "agent_tier": 5, "is_active": False},
            ],
            "audit_chain_entries": [],
        }
    )
    with _patch_health_client(db):
        agents, prov = _fetch_agent_health()
    assert prov == DataProvenance.PARTIAL
    assert len(agents) == 2
    assert all(a.success_rate is None and a.avg_latency_ms is None for a in agents)
    assert agents[0].available is True
    assert agents[1].available is False


def test_fetch_agent_health_measured_with_telemetry():
    """Telemetry with valid latencies -> MEASURED. success_rate/latency aggregate
    the wider window; invocations_24h counts ONLY the last 24h (its name)."""
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=2)).isoformat()  # within 24h
    older = (now - timedelta(days=10)).isoformat()  # within 30d window, not 24h
    db = _FakeDB(
        {
            "agent_registry": [
                {"agent_name": "gap_analyzer", "agent_tier": 2, "is_active": True},
            ],
            "audit_chain_entries": [
                {
                    "agent_name": "gap_analyzer",
                    "duration_ms": 100,
                    "validation_passed": True,
                    "created_at": recent,
                },
                {
                    "agent_name": "gap_analyzer",
                    "duration_ms": 300,
                    "validation_passed": False,
                    "created_at": older,
                },
            ],
        }
    )
    with _patch_health_client(db):
        agents, prov = _fetch_agent_health()
    assert prov == DataProvenance.MEASURED
    a = agents[0]
    assert a.invocations_24h == 1  # only the recent (<24h) entry
    assert a.success_rate == 0.5  # 1 ok / 2 total over the wider window
    assert a.avg_latency_ms == 200  # mean(100, 300)


def test_fetch_agent_health_partial_when_telemetry_lacks_latency():
    """Telemetry rows but NO valid duration -> avg_latency null -> the agent is
    not fully sourced -> PARTIAL (not MEASURED with a null latency)."""
    now = datetime.now(timezone.utc)
    db = _FakeDB(
        {
            "agent_registry": [
                {"agent_name": "gap_analyzer", "agent_tier": 2, "is_active": True},
            ],
            "audit_chain_entries": [
                {
                    "agent_name": "gap_analyzer",
                    "duration_ms": 0,  # non-positive -> not a measured latency
                    "validation_passed": True,
                    "created_at": (now - timedelta(hours=1)).isoformat(),
                },
            ],
        }
    )
    with _patch_health_client(db):
        agents, prov = _fetch_agent_health()
    assert prov == DataProvenance.PARTIAL
    assert agents[0].avg_latency_ms is None
    assert agents[0].success_rate == 1.0  # rate is still measured


def test_fetch_agent_health_none_when_no_client():
    with _patch_health_client(None):
        agents, prov = _fetch_agent_health()
    assert agents == []
    assert prov is None
