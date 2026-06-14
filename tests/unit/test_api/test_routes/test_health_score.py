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
async def test_full_health_score_degraded_model_with_null_error_rate_no_crash():
    """REGRESSION (codex r2 HIGH): a degraded/unhealthy model whose error_rate is
    UNMEASURED (None, as the real ml_model_health_dashboard adapter passes through)
    must NOT crash the score composer's diagnosis (None > 0.1 TypeError). The
    composite must still complete with a real, non-null model score (NOT collapse
    to a failed/unknown composite)."""
    real_models = [
        ModelHealth(model_id="m1", model_name="alpha", status=ModelStatus.DEGRADED),
        ModelHealth(model_id="m2", model_name="beta", status=ModelStatus.UNHEALTHY),
    ]
    real_pipelines = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=12.0,
            status=PipelineStatus.HEALTHY,
        )
    ]
    real_agents = [
        AgentHealth(
            agent_name="orchestrator",
            tier=1,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        )
    ]

    class _StubHealthClient:
        async def check(self, endpoint: str):
            return {"ok": True}

    with (
        patch(
            "src.api.routes.health_score._fetch_model_health",
            return_value=(real_models, DataProvenance.PARTIAL),
        ),
        patch(
            "src.api.routes.health_score._fetch_pipeline_health",
            return_value=(real_pipelines, DataProvenance.MEASURED),
        ),
        patch(
            "src.api.routes.health_score._fetch_agent_health",
            return_value=(real_agents, DataProvenance.PARTIAL),
        ),
        patch("src.agents.health_score.SupabaseHealthClient", _StubHealthClient),
    ):
        result = await _execute_health_check(CheckScope.FULL)

    # Diagnosis did not crash: model dimension is a REAL measurement (not None),
    # and the composite did not collapse to a failed/unknown 'F'.
    assert result.model_health_score is not None
    # 1 degraded (0.5) + 1 unhealthy (0.0) over 2 -> 0.25
    assert result.model_health_score == 0.25
    assert result.data_provenance == "partial"


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


@pytest.mark.asyncio
async def test_full_health_score_computes_real_dimensions_end_to_end():
    """END-TO-END (faithful, no agent mock): with the real store adapters wired
    over patched _fetch_* readers returning real-shaped rows, the full health
    SCORE must compute non-null model/pipeline/agent dimensions — proving the
    defect (three nulls) is fixed through the real agent graph, not a stubbed
    agent. Provenance is reconciled to 'partial' (not 'measured') because the
    model + agent readers are PARTIAL: the dimension SCORES are real, but some
    sub-fields (model accuracy/latency, agent runtime telemetry) are unsourced —
    so /full must not overclaim relative to /models and /agents."""
    real_models = [
        ModelHealth(model_id="m1", model_name="alpha", status=ModelStatus.HEALTHY),
        ModelHealth(model_id="m2", model_name="beta", status=ModelStatus.HEALTHY),
    ]
    real_pipelines = [
        PipelineHealth(
            pipeline_name="rwd_ingest",
            last_run="2026-06-12T00:00:00+00:00",
            last_success="2026-06-12T00:00:00+00:00",
            rows_processed=10000,
            freshness_hours=12.0,
            status=PipelineStatus.HEALTHY,
        )
    ]
    real_agents = [
        AgentHealth(
            agent_name="orchestrator",
            tier=1,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        ),
        AgentHealth(
            agent_name="causal_impact",
            tier=2,
            available=True,
            avg_latency_ms=None,
            success_rate=None,
            last_invocation=None,
            invocations_24h=0,
        ),
    ]

    # A component health_client implementing the node's HealthClient Protocol
    # (check(endpoint) -> {"ok": ...}) so every component reports healthy and the
    # component dimension is measured (so provenance can reach 'measured').
    class _StubHealthClient:
        async def check(self, endpoint: str):
            return {"ok": True}

    with (
        patch(
            "src.api.routes.health_score._fetch_model_health",
            return_value=(real_models, DataProvenance.PARTIAL),
        ),
        patch(
            "src.api.routes.health_score._fetch_pipeline_health",
            return_value=(real_pipelines, DataProvenance.MEASURED),
        ),
        patch(
            "src.api.routes.health_score._fetch_agent_health",
            return_value=(real_agents, DataProvenance.PARTIAL),
        ),
        patch("src.agents.health_score.SupabaseHealthClient", _StubHealthClient),
    ):
        result = await _execute_health_check(CheckScope.FULL)

    # The three previously-null dimensions are now REAL, computed from the tables.
    assert result.model_health_score is not None
    assert result.pipeline_health_score is not None
    assert result.agent_health_score is not None
    # All measured-and-healthy -> scores 1.0 (matches the per-dimension endpoints).
    assert result.model_health_score == 1.0
    assert result.pipeline_health_score == 1.0
    assert result.agent_health_score == 1.0
    # Honest provenance: scores are real, but model/agent sub-fields are unsourced
    # (readers PARTIAL) -> reconciled to 'partial', never overclaiming 'measured'.
    assert result.data_provenance == "partial"


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
    """Patch _execute_health_check to return a MEASURED result with real
    component_statuses (mirrors the live agent path)."""
    result = MagicMock(
        data_provenance="measured",
        component_statuses=_measured_components(),
    )
    return patch(
        "src.api.routes.health_score._execute_health_check",
        new=AsyncMock(return_value=result),
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
    """In a fail-closed environment, /components must raise 503 rather than
    return fabricated component health data."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        with pytest.raises(HTTPException) as exc_info:
            await get_component_health()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "agent_unavailable"


@pytest.mark.asyncio
async def test_get_component_health_discloses_placeholder_provenance(monkeypatch):
    """When mock-fallback is explicitly allowed (dev), /components must DISCLOSE
    that the data is placeholder/fabricated rather than presenting it as real."""
    monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
    with patch("src.agents.health_score.HealthScoreAgent", side_effect=ImportError):
        result = await get_component_health()
    assert result.data_provenance == "placeholder"
    assert result.total_components > 0


@pytest.mark.asyncio
async def test_get_component_health_measured_when_agent_runs():
    """REWIRED (was 'serves_placeholder_even_when_agent_available'): /components
    now INVOKES the real agent and tags genuinely measured component data
    'measured', never 'placeholder'."""
    with _patch_measured_component_check():
        result = await get_component_health()
    assert result.data_provenance == DataProvenance.MEASURED
    # No fabricated 'opik @ 250ms' mock component leaks through.
    assert not any(c.latency_ms == 250 for c in result.components)


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
        mock_execute.return_value = mock_result

        await run_health_check(scope=CheckScope.FULL)
        await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.total_checks == 2
    assert len(result.checks) == 2
    assert result.avg_health_score == 85.0


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
            mock_execute.return_value = mock_result
            await run_health_check(scope=CheckScope.FULL)

    result = await get_health_history(limit=20)

    assert result.trend == "stable"


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

    def eq(self, *a, **k):
        return self

    def gte(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return MagicMock(data=list(self._rows))


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


def test_fetch_model_health_partial_when_no_metric_values():
    """Real rows but empty ml_performance_metrics (latest_metric_value NULL):
    perf fields stay null, provenance is PARTIAL, status is mapped — NEVER the
    hardcoded accuracy=0.89."""
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
    assert prov == DataProvenance.PARTIAL
    assert len(models) == 2
    assert models[0].status == ModelStatus.HEALTHY
    assert models[1].status == ModelStatus.UNHEALTHY
    assert all(m.accuracy is None and m.auc_roc is None for m in models)
    assert all(m.accuracy != 0.89 for m in models)


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
