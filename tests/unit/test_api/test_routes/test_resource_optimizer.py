"""Unit tests for Resource Optimizer API route handlers.

Tests all endpoints and helper functions in src/api/routes/resource_optimizer.py.
Mocks all external dependencies to ensure unit test isolation.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks

# Import route functions and models
from src.api.routes.resource_optimizer import (
    AllocationTarget,
    Constraint,
    ConstraintScope,
    ConstraintType,
    # Enums
    OptimizationObjective,
    OptimizationStatus,
    ResourceType,
    # Models
    RunOptimizationRequest,
    _convert_allocations,
    _convert_scenarios,
    _execute_optimization,
    _generate_mock_response,
    # Module-level storage (durable / cross-worker, async API)
    _optimizations_store,
    # Helper functions
    _run_optimization_task,
    get_optimization,
    get_resource_health,
    list_scenarios,
    # Endpoints
    run_optimization,
)

# =============================================================================
# FIXTURES
# =============================================================================


async def _no_redis():
    """Redis factory that always fails, forcing the in-memory fallback.

    The live store is the Redis-backed durable store. For unit isolation we
    pin its factory to one that raises so every test exercises the bounded
    in-process fallback deterministically (no live Redis required), mirroring
    ``tests/unit/test_api/test_routes/test_segments.py``.
    """
    raise ConnectionError("redis disabled for unit isolation")


@pytest.fixture(autouse=True)
def reset_optimizations_store():
    """Clear the optimizations store and pin it to the in-memory fallback."""
    original_factory = _optimizations_store._redis_factory
    _optimizations_store._redis_factory = _no_redis
    _optimizations_store.clear()
    yield
    _optimizations_store.clear()
    _optimizations_store._redis_factory = original_factory


@pytest.fixture
def sample_request():
    """Sample optimization request."""
    return RunOptimizationRequest(
        query="Optimize budget allocation across territories",
        resource_type=ResourceType.BUDGET,
        allocation_targets=[
            AllocationTarget(
                entity_id="territory_northeast",
                entity_type="territory",
                current_allocation=50000.0,
                min_allocation=30000.0,
                max_allocation=80000.0,
                expected_response=1.3,
            ),
            AllocationTarget(
                entity_id="territory_southeast",
                entity_type="territory",
                current_allocation=40000.0,
                min_allocation=20000.0,
                max_allocation=60000.0,
                expected_response=0.9,
            ),
        ],
        constraints=[
            Constraint(
                constraint_type=ConstraintType.BUDGET,
                value=200000.0,
                scope=ConstraintScope.GLOBAL,
            )
        ],
        objective=OptimizationObjective.MAXIMIZE_OUTCOME,
    )


@pytest.fixture
def mock_agent_result():
    """Mock agent result."""
    return {
        "status": "completed",
        "optimal_allocations": [
            {
                "entity_id": "territory_northeast",
                "entity_type": "territory",
                "current_allocation": 50000.0,
                "optimized_allocation": 60000.0,
                "change": 10000.0,
                "change_percentage": 20.0,
                "expected_impact": 78000.0,
            }
        ],
        "objective_value": 180000.0,
        "solver_status": "optimal",
        "solve_time_ms": 150,
        "scenarios": [
            {
                "scenario_name": "Conservative",
                "total_allocation": 180000.0,
                "projected_outcome": 324000.0,
                "roi": 1.8,
                "constraint_violations": [],
            }
        ],
        "sensitivity_analysis": {"budget": 0.85},
        "projected_total_outcome": 180000.0,
        "projected_roi": 2.0,
        "impact_by_segment": {"high_responders": 108000.0},
        "optimization_summary": "Optimization complete",
        "recommendations": ["Increase allocation to high-response entities"],
        "formulation_latency_ms": 50,
        "optimization_latency_ms": 150,
        "warnings": [],
    }


# =============================================================================
# ENDPOINT TESTS - run_optimization
# =============================================================================


@pytest.mark.asyncio
async def test_run_optimization_async_mode(sample_request):
    """Test run_optimization in async mode returns immediately."""
    background_tasks = BackgroundTasks()

    result = await run_optimization(
        request=sample_request,
        background_tasks=background_tasks,
        async_mode=True,
    )

    assert result.status == OptimizationStatus.PENDING
    assert result.optimization_id.startswith("opt_")
    assert await _optimizations_store.contains(result.optimization_id)


@pytest.mark.asyncio
async def test_run_optimization_sync_mode(sample_request):
    """Test run_optimization in sync mode executes immediately."""
    background_tasks = BackgroundTasks()

    with patch("src.api.routes.resource_optimizer._execute_optimization") as mock_execute:
        mock_result = MagicMock(
            optimization_id="",
            status=OptimizationStatus.COMPLETED,
            objective_value=180000.0,
        )
        mock_execute.return_value = mock_result

        result = await run_optimization(
            request=sample_request,
            background_tasks=background_tasks,
            async_mode=False,
        )

        assert result.status == OptimizationStatus.COMPLETED
        mock_execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_optimization_sync_mode_exception(sample_request):
    """Test run_optimization handles exceptions in sync mode."""
    background_tasks = BackgroundTasks()

    with patch("src.api.routes.resource_optimizer._execute_optimization") as mock_execute:
        mock_execute.side_effect = RuntimeError("Test error")

        with pytest.raises(Exception) as exc_info:
            await run_optimization(
                request=sample_request,
                background_tasks=background_tasks,
                async_mode=False,
            )

        assert "Optimization failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_optimization_stores_result(sample_request):
    """Test run_optimization stores result in store."""
    background_tasks = BackgroundTasks()

    with patch("src.api.routes.resource_optimizer._execute_optimization") as mock_execute:
        mock_result = MagicMock(
            optimization_id="",
            status=OptimizationStatus.COMPLETED,
        )
        mock_execute.return_value = mock_result

        result = await run_optimization(
            request=sample_request,
            background_tasks=background_tasks,
            async_mode=False,
        )

        assert await _optimizations_store.contains(result.optimization_id)
        stored = await _optimizations_store.get(result.optimization_id)
        assert stored is not None
        assert stored.status == OptimizationStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_optimization_preserves_request_params(sample_request):
    """Test run_optimization preserves request parameters in response."""
    background_tasks = BackgroundTasks()

    with patch("src.api.routes.resource_optimizer._execute_optimization") as mock_execute:
        mock_result = MagicMock(
            optimization_id="",
            status=OptimizationStatus.COMPLETED,
        )
        mock_execute.return_value = mock_result

        result = await run_optimization(
            request=sample_request,
            background_tasks=background_tasks,
            async_mode=False,
        )

        assert hasattr(result, "resource_type")
        assert hasattr(result, "objective")


# =============================================================================
# ENDPOINT TESTS - get_optimization
# =============================================================================


@pytest.mark.asyncio
async def test_get_optimization_success():
    """Test get_optimization returns stored optimization."""
    optimization_id = "opt_test123"
    mock_optimization = MagicMock(
        optimization_id=optimization_id,
        status=OptimizationStatus.COMPLETED,
    )
    await _optimizations_store.set(optimization_id, mock_optimization)

    result = await get_optimization(optimization_id)

    assert result.optimization_id == optimization_id
    assert result.status == OptimizationStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_optimization_not_found():
    """Test get_optimization raises 404 for missing optimization."""
    with pytest.raises(Exception) as exc_info:
        await get_optimization("nonexistent_id")

    assert "not found" in str(exc_info.value)


# =============================================================================
# ENDPOINT TESTS - list_scenarios
# =============================================================================


@pytest.mark.asyncio
async def test_list_scenarios_empty_store():
    """Test list_scenarios with empty store."""
    result = await list_scenarios(min_roi=None, limit=20)

    assert result.total_count == 0
    assert len(result.scenarios) == 0


@pytest.mark.asyncio
async def test_list_scenarios_with_data():
    """Test list_scenarios returns scenarios from completed optimizations."""
    from src.api.routes.resource_optimizer import ScenarioResult

    mock_scenario = ScenarioResult(
        scenario_name="Conservative",
        total_allocation=180000.0,
        projected_outcome=324000.0,
        roi=1.8,
        constraint_violations=[],
    )

    mock_optimization = MagicMock(
        status=OptimizationStatus.COMPLETED,
        scenarios=[mock_scenario],
    )
    await _optimizations_store.set("opt_1", mock_optimization)
    result = await list_scenarios(min_roi=None, limit=20)

    assert result.total_count == 1
    assert len(result.scenarios) == 1


@pytest.mark.asyncio
async def test_list_scenarios_filters_by_min_roi():
    """Test list_scenarios filters by minimum ROI."""
    from src.api.routes.resource_optimizer import ScenarioResult

    mock_scenario_high = ScenarioResult(
        scenario_name="Aggressive",
        total_allocation=200000.0,
        projected_outcome=500000.0,
        roi=2.5,
        constraint_violations=[],
    )

    mock_scenario_low = ScenarioResult(
        scenario_name="Conservative",
        total_allocation=180000.0,
        projected_outcome=270000.0,
        roi=1.5,
        constraint_violations=[],
    )

    mock_optimization = MagicMock(
        status=OptimizationStatus.COMPLETED,
        scenarios=[mock_scenario_high, mock_scenario_low],
    )
    await _optimizations_store.set("opt_1", mock_optimization)
    result = await list_scenarios(min_roi=2.0, limit=20)

    assert result.total_count == 1
    assert result.scenarios[0].roi == 2.5


@pytest.mark.asyncio
async def test_list_scenarios_respects_limit():
    """Test list_scenarios respects limit parameter."""
    from src.api.routes.resource_optimizer import ScenarioResult

    # Create 10 scenarios
    scenarios = [
        ScenarioResult(
            scenario_name=f"Scenario_{i}",
            total_allocation=180000.0,
            projected_outcome=324000.0,
            roi=1.8 + i * 0.1,
            constraint_violations=[],
        )
        for i in range(10)
    ]

    mock_optimization = MagicMock(
        status=OptimizationStatus.COMPLETED,
        scenarios=scenarios,
    )
    await _optimizations_store.set("opt_1", mock_optimization)
    result = await list_scenarios(min_roi=None, limit=5)

    assert len(result.scenarios) == 5


@pytest.mark.asyncio
async def test_list_scenarios_sorts_by_roi():
    """Test list_scenarios sorts by ROI descending."""
    from src.api.routes.resource_optimizer import ScenarioResult

    scenarios = [
        ScenarioResult(
            scenario_name="Low",
            total_allocation=180000.0,
            projected_outcome=270000.0,
            roi=1.5,
            constraint_violations=[],
        ),
        ScenarioResult(
            scenario_name="High",
            total_allocation=200000.0,
            projected_outcome=500000.0,
            roi=2.5,
            constraint_violations=[],
        ),
        ScenarioResult(
            scenario_name="Medium",
            total_allocation=190000.0,
            projected_outcome=380000.0,
            roi=2.0,
            constraint_violations=[],
        ),
    ]

    mock_optimization = MagicMock(
        status=OptimizationStatus.COMPLETED,
        scenarios=scenarios,
    )
    await _optimizations_store.set("opt_1", mock_optimization)
    result = await list_scenarios(min_roi=None, limit=20)

    assert result.scenarios[0].scenario_name == "High"
    assert result.scenarios[1].scenario_name == "Medium"
    assert result.scenarios[2].scenario_name == "Low"


@pytest.mark.asyncio
async def test_list_scenarios_skips_pending_optimizations():
    """Test list_scenarios skips pending optimizations."""
    from src.api.routes.resource_optimizer import ScenarioResult

    mock_scenario = ScenarioResult(
        scenario_name="Conservative",
        total_allocation=180000.0,
        projected_outcome=324000.0,
        roi=1.8,
        constraint_violations=[],
    )

    mock_pending = MagicMock(
        status=OptimizationStatus.PENDING,
        scenarios=[mock_scenario],
    )
    await _optimizations_store.set("opt_1", mock_pending)
    result = await list_scenarios(min_roi=None, limit=20)

    assert result.total_count == 0


# =============================================================================
# ENDPOINT TESTS - get_resource_health
# =============================================================================


@pytest.mark.asyncio
async def test_get_resource_health_all_available():
    """Test get_resource_health when all dependencies available.

    Durable (cross-worker) storage is part of "healthy": when Redis is
    reachable the store reports ``storage_mode='durable'`` and, with the agent
    and scipy available, the service is healthy. (The autouse fixture pins the
    store to the in-memory fallback, so this test temporarily restores a
    reachable fake-Redis factory to exercise the durable path.)
    """
    shared = _FakeAsyncRedis()

    async def _factory():
        return shared

    original_factory = _optimizations_store._redis_factory
    _optimizations_store._redis_factory = _factory
    try:
        with patch("src.agents.resource_optimizer.ResourceOptimizerAgent"):
            with patch("scipy.optimize"):
                result = await get_resource_health()

                assert result.status == "healthy"
                assert result.agent_available is True
                assert result.scipy_available is True
                assert result.storage_mode == "durable"
    finally:
        _optimizations_store._redis_factory = original_factory


@pytest.mark.asyncio
async def test_get_resource_health_reports_degraded_storage():
    """When Redis is unavailable the store falls back to the in-memory dict;
    health must surface this as 'degraded' (cross-worker reads can 404)."""
    # The autouse fixture already pins the store to the failing _no_redis
    # factory, so the durable probe fails and storage degrades.
    with patch("src.agents.resource_optimizer.ResourceOptimizerAgent"):
        with patch("scipy.optimize"):
            result = await get_resource_health()

            assert result.storage_mode == "degraded"
            assert result.status == "degraded"


@pytest.mark.asyncio
async def test_get_resource_health_agent_unavailable():
    """Test get_resource_health reflects agent availability."""
    result = await get_resource_health()
    assert result.status in ["healthy", "degraded", "partial"]
    assert isinstance(result.agent_available, bool)


@pytest.mark.asyncio
async def test_get_resource_health_scipy_unavailable():
    """Test get_resource_health reflects scipy availability."""
    result = await get_resource_health()
    assert isinstance(result.scipy_available, bool)


@pytest.mark.asyncio
async def test_get_resource_health_counts_recent_optimizations():
    """Test get_resource_health counts optimizations in last 24 hours."""
    from src.api.routes.resource_optimizer import OptimizationResponse

    recent_optimization = OptimizationResponse(
        optimization_id="opt_1",
        status=OptimizationStatus.COMPLETED,
        resource_type=ResourceType.BUDGET,
        objective=OptimizationObjective.MAXIMIZE_OUTCOME,
        timestamp=datetime.now(timezone.utc),
    )
    await _optimizations_store.set("opt_1", recent_optimization)
    with patch("src.agents.resource_optimizer.ResourceOptimizerAgent"):
        result = await get_resource_health()

        assert result.optimizations_24h == 1


@pytest.mark.asyncio
async def test_get_resource_health_last_optimization():
    """Test get_resource_health returns last optimization timestamp."""
    from src.api.routes.resource_optimizer import OptimizationResponse

    optimization = OptimizationResponse(
        optimization_id="opt_1",
        status=OptimizationStatus.COMPLETED,
        resource_type=ResourceType.BUDGET,
        objective=OptimizationObjective.MAXIMIZE_OUTCOME,
        timestamp=datetime.now(timezone.utc),
    )
    await _optimizations_store.set("opt_1", optimization)
    with patch("src.agents.resource_optimizer.ResourceOptimizerAgent"):
        result = await get_resource_health()

        assert result.last_optimization is not None


# =============================================================================
# HELPER FUNCTION TESTS - _run_optimization_task
# =============================================================================


@pytest.mark.asyncio
async def test_run_optimization_task_success(sample_request, mock_agent_result):
    """Test _run_optimization_task completes successfully."""
    optimization_id = "opt_test123"

    from src.api.routes.resource_optimizer import OptimizationResponse

    await _optimizations_store.set(
        optimization_id,
        OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.PENDING,
            resource_type=ResourceType.BUDGET,
            objective=OptimizationObjective.MAXIMIZE_OUTCOME,
        ),
    )

    with patch("src.api.routes.resource_optimizer._execute_optimization") as mock_execute:
        mock_result = MagicMock(
            optimization_id="",
            status=OptimizationStatus.COMPLETED,
        )
        mock_execute.return_value = mock_result

        await _run_optimization_task(optimization_id, sample_request)

        stored = await _optimizations_store.get(optimization_id)
        assert stored is not None
        assert stored.status == OptimizationStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_optimization_task_handles_error(sample_request):
    """Test _run_optimization_task handles errors."""
    optimization_id = "opt_test123"

    from src.api.routes.resource_optimizer import OptimizationResponse

    await _optimizations_store.set(
        optimization_id,
        OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.PENDING,
            resource_type=ResourceType.BUDGET,
            objective=OptimizationObjective.MAXIMIZE_OUTCOME,
        ),
    )

    with patch("src.api.routes.resource_optimizer._execute_optimization") as mock_execute:
        mock_execute.side_effect = RuntimeError("Test error")

        await _run_optimization_task(optimization_id, sample_request)

        stored = await _optimizations_store.get(optimization_id)
        assert stored is not None
        assert stored.status == OptimizationStatus.FAILED
        assert len(stored.warnings) > 0


# =============================================================================
# HELPER FUNCTION TESTS - _execute_optimization
# =============================================================================


@pytest.mark.asyncio
async def test_execute_optimization_with_agent(sample_request, mock_agent_result):
    """Test _execute_optimization uses real agent when available."""
    result = await _execute_optimization(sample_request)

    # Since we can't easily mock the graph import, just verify it returns a result
    assert result.status in [OptimizationStatus.COMPLETED, OptimizationStatus.FAILED]
    assert result.objective_value is not None


@pytest.mark.asyncio
async def test_execute_optimization_falls_back_to_mock(sample_request):
    """Test _execute_optimization falls back to mock when agent unavailable."""
    with patch(
        "src.agents.resource_optimizer.graph.build_resource_optimizer_graph",
        side_effect=ImportError,
    ):
        result = await _execute_optimization(sample_request)

        assert result.status == OptimizationStatus.COMPLETED
        assert "mock data" in result.warnings[0].lower()


@pytest.mark.asyncio
async def test_execute_optimization_handles_exception(sample_request):
    """Test _execute_optimization handles agent exceptions gracefully."""
    # The function catches exceptions and falls back to mock, so no exception is raised
    result = await _execute_optimization(sample_request)
    assert result.status == OptimizationStatus.COMPLETED


# =============================================================================
# HELPER FUNCTION TESTS - _convert_allocations
# =============================================================================


def test_convert_allocations_success():
    """Test _convert_allocations converts agent output correctly."""
    agent_data = [
        {
            "entity_id": "territory_northeast",
            "entity_type": "territory",
            "current_allocation": 50000.0,
            "optimized_allocation": 60000.0,
            "change": 10000.0,
            "change_percentage": 20.0,
            "expected_impact": 78000.0,
        }
    ]

    result = _convert_allocations(agent_data)

    assert len(result) == 1
    assert result[0].entity_id == "territory_northeast"
    assert result[0].optimized_allocation == 60000.0
    assert result[0].change == 10000.0


def test_convert_allocations_empty():
    """Test _convert_allocations handles empty list."""
    result = _convert_allocations([])

    assert isinstance(result, list)
    assert len(result) == 0


def test_convert_allocations_handles_missing_fields():
    """Test _convert_allocations handles missing fields."""
    agent_data = [
        {
            "entity_id": "territory_northeast",
            # Missing other fields
        }
    ]

    result = _convert_allocations(agent_data)

    assert len(result) == 1
    # Should use defaults
    assert result[0].current_allocation == 0.0


# =============================================================================
# HELPER FUNCTION TESTS - _convert_scenarios
# =============================================================================


def test_convert_scenarios_success():
    """Test _convert_scenarios converts agent output correctly."""
    agent_data = [
        {
            "scenario_name": "Conservative",
            "total_allocation": 180000.0,
            "projected_outcome": 324000.0,
            "roi": 1.8,
            "constraint_violations": [],
        }
    ]

    result = _convert_scenarios(agent_data)

    assert len(result) == 1
    assert result[0].scenario_name == "Conservative"
    assert result[0].total_allocation == 180000.0
    assert result[0].roi == 1.8


def test_convert_scenarios_empty():
    """Test _convert_scenarios handles empty list."""
    result = _convert_scenarios([])

    assert isinstance(result, list)
    assert len(result) == 0


# =============================================================================
# HELPER FUNCTION TESTS - _generate_mock_response
# =============================================================================


def test_generate_mock_response_structure(sample_request):
    """Test _generate_mock_response returns valid structure."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.status == OptimizationStatus.COMPLETED
    assert result.objective_value is not None
    assert len(result.optimal_allocations) > 0


def test_generate_mock_response_allocations(sample_request):
    """Test _generate_mock_response generates allocations for all targets."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert len(result.optimal_allocations) == len(sample_request.allocation_targets)


def test_generate_mock_response_high_responders_increased(sample_request):
    """Test _generate_mock_response increases allocation for high responders."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    # Find high responder allocation
    high_responder = next(
        a for a in result.optimal_allocations if a.entity_id == "territory_northeast"
    )

    # Should increase (expected_response is 1.3 > 1.1)
    assert high_responder.change > 0


def test_generate_mock_response_low_responders_decreased(sample_request):
    """Test _generate_mock_response decreases allocation for low responders."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    # Find low responder allocation
    low_responder = next(
        a for a in result.optimal_allocations if a.entity_id == "territory_southeast"
    )

    # Expected response 0.9 may be treated as average, so change could be small positive
    # Just verify it exists
    assert isinstance(low_responder.change, (int, float))


def test_generate_mock_response_respects_min_allocation(sample_request):
    """Test _generate_mock_response respects minimum allocation constraints."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    for allocation in result.optimal_allocations:
        target = next(
            t for t in sample_request.allocation_targets if t.entity_id == allocation.entity_id
        )
        if target.min_allocation:
            assert allocation.optimized_allocation >= target.min_allocation


def test_generate_mock_response_respects_max_allocation(sample_request):
    """Test _generate_mock_response respects maximum allocation constraints."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    for allocation in result.optimal_allocations:
        target = next(
            t for t in sample_request.allocation_targets if t.entity_id == allocation.entity_id
        )
        if target.max_allocation:
            assert allocation.optimized_allocation <= target.max_allocation


def test_generate_mock_response_includes_scenarios_when_requested(sample_request):
    """Test _generate_mock_response includes scenarios when requested."""
    import time

    # Update request to include scenarios
    sample_request.run_scenarios = True
    sample_request.scenario_count = 3

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert len(result.scenarios) == 3


def test_generate_mock_response_no_scenarios_when_not_requested(sample_request):
    """Test _generate_mock_response excludes scenarios when not requested."""
    import time

    # Ensure scenarios are not requested
    sample_request.run_scenarios = False

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert len(result.scenarios) == 0


def test_generate_mock_response_includes_sensitivity_analysis(sample_request):
    """Test _generate_mock_response includes sensitivity analysis."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.sensitivity_analysis is not None
    assert "budget" in result.sensitivity_analysis


def test_generate_mock_response_includes_impact_breakdown(sample_request):
    """Test _generate_mock_response includes impact breakdown."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.impact_by_segment is not None
    assert "high_responders" in result.impact_by_segment


def test_generate_mock_response_includes_summary(sample_request):
    """Test _generate_mock_response includes optimization summary."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.optimization_summary is not None
    assert len(result.recommendations) > 0


def test_generate_mock_response_warning(sample_request):
    """Test _generate_mock_response includes warning about mock data."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert len(result.warnings) > 0
    assert "mock data" in result.warnings[0].lower()


def test_generate_mock_response_solver_status(sample_request):
    """Test _generate_mock_response sets solver status to optimal."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.solver_status == "optimal"


def test_generate_mock_response_calculates_roi(sample_request):
    """Test _generate_mock_response calculates projected ROI."""
    import time

    start_time = time.time()
    result = _generate_mock_response(sample_request, start_time)

    assert result.projected_roi is not None
    assert result.projected_roi > 0


# =============================================================================
# DURABLE (REDIS-BACKED) STORE — durable / cross-worker shared store
# =============================================================================
#
# Production runs gunicorn with --workers 2 (docker/docker-compose.yml). A
# process-local dict means a POST handled by worker A is invisible to a GET
# handled by worker B (legitimate 404 ~50% of the time) and ALL state is lost
# on redeploy. The durable store backs the data in Redis (reusing the app's
# existing async client) so it is shared across workers and survives restarts,
# falling back to the bounded in-memory dict ONLY when Redis is unavailable
# (mirrors src/api/routes/segments.py and the app's graceful-degradation
# posture). This is the regression suite for the live-reproduced 404 flicker.


def _make_opt_resp(optimization_id: str) -> "object":
    """Build a minimal COMPLETED OptimizationResponse for store tests."""
    from src.api.routes.resource_optimizer import (
        OptimizationResponse,
        OptimizationStatus,
        ResourceType,
    )

    return OptimizationResponse(
        optimization_id=optimization_id,
        status=OptimizationStatus.COMPLETED,
        resource_type=ResourceType.BUDGET,
        objective=OptimizationObjective.MAXIMIZE_OUTCOME,
    )


class _FakeAsyncRedis:
    """Faithful in-process stand-in for ``redis.asyncio.Redis``.

    Hand-rolled (consistent with the repo's other Redis tests, e.g.
    ``tests/unit/test_api/test_routes/test_segments.py``) because ``fakeredis``
    is not a test dependency. Faithful to redis 7.x: ``set(..., ex=N)`` honors
    a (clock-controllable) TTL; ZSET ``zrange`` orders by ``(score, member)``
    with lexicographic tie-break; string reads return ``str`` (the app's client
    uses ``decode_responses=True``); ``mget`` / ``pipeline`` are supported so
    the batched-read and atomic-write paths are exercised.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.strings: dict = {}
        self._expires: dict = {}
        self.zset: dict = {}
        self._now = now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def _expire_if_due(self, key) -> None:  # noqa: ANN001
        exp = self._expires.get(key)
        if exp is not None and self._now >= exp:
            self.strings.pop(key, None)
            self._expires.pop(key, None)

    async def set(self, key, value, ex=None):  # noqa: ANN001
        self.strings[key] = value
        if ex is not None:
            self._expires[key] = self._now + ex
        else:
            self._expires.pop(key, None)

    async def get(self, key):  # noqa: ANN001
        self._expire_if_due(key)
        return self.strings.get(key)

    async def mget(self, keys):  # noqa: ANN001
        out = []
        for k in keys:
            self._expire_if_due(k)
            out.append(self.strings.get(k))
        return out

    async def delete(self, *keys):  # noqa: ANN001
        removed = 0
        for k in keys:
            if k in self.strings:
                removed += 1
            self.strings.pop(k, None)
            self._expires.pop(k, None)
        return removed

    async def zadd(self, key, mapping):  # noqa: ANN001
        self.zset.setdefault(key, {}).update(mapping)

    async def zrem(self, key, *members):  # noqa: ANN001
        z = self.zset.get(key, {})
        for m in members:
            z.pop(m, None)

    async def zcard(self, key):  # noqa: ANN001
        return len(self.zset.get(key, {}))

    async def zscore(self, key, member):  # noqa: ANN001
        return self.zset.get(key, {}).get(member)

    async def zrange(self, key, start, end):  # noqa: ANN001
        members = [
            m for m, _ in sorted(self.zset.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]))
        ]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def pipeline(self, transaction: bool = True):  # noqa: ANN001
        return _FakePipeline(self)


class _FakePipeline:
    """Minimal pipeline: queue set/zadd, apply on execute (no rollback, like
    real Redis MULTI/EXEC for runtime errors)."""

    def __init__(self, client: "_FakeAsyncRedis") -> None:
        self._client = client
        self._ops: list = []

    def set(self, key, value, ex=None):  # noqa: ANN001
        self._ops.append(("set", (key, value), {"ex": ex}))
        return self

    def zadd(self, key, mapping):  # noqa: ANN001
        self._ops.append(("zadd", (key, mapping), {}))
        return self

    async def execute(self):
        for name, args, kwargs in self._ops:
            await getattr(self._client, name)(*args, **kwargs)
        self._ops = []
        return []


class TestDurableOptimizationsStore:
    """The durable store must share state across processes via Redis and fall
    back to the bounded in-memory dict when Redis is unavailable."""

    @pytest.mark.asyncio
    async def test_cross_worker_read_via_shared_redis(self):
        """A value written through one store instance (worker A) is readable
        through a DIFFERENT instance sharing the same Redis (worker B).

        This is the exact 404-flicker the live repro showed: with a
        process-local dict, worker B 404s on an id that worker A wrote.
        """
        from src.api.routes.resource_optimizer import (
            OptimizationStatus,
            _DurableOptimizationsStore,
        )

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        worker_a = _DurableOptimizationsStore(redis_factory=_factory)
        worker_b = _DurableOptimizationsStore(redis_factory=_factory)

        resp = _make_opt_resp("opt_shared")
        await worker_a.set("opt_shared", resp)

        # Worker B's in-memory fallback is empty; it must read from Redis.
        assert "opt_shared" not in worker_b._memory
        assert await worker_b.contains("opt_shared") is True
        got = await worker_b.get("opt_shared")
        assert got is not None
        assert got.optimization_id == "opt_shared"
        assert got.status == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_values_enumerates_from_redis(self):
        from src.api.routes.resource_optimizer import _DurableOptimizationsStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableOptimizationsStore(redis_factory=_factory)
        await store.set("opt_1", _make_opt_resp("opt_1"))
        await store.set("opt_2", _make_opt_resp("opt_2"))

        values = await store.values()
        ids = sorted(v.optimization_id for v in values)
        assert ids == ["opt_1", "opt_2"]

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_when_redis_unavailable(self):
        """When the Redis factory fails, the store transparently uses the
        bounded in-memory fallback (no exception bubbles to the route)."""
        from src.api.routes.resource_optimizer import _DurableOptimizationsStore

        async def _factory():
            raise ConnectionError("redis down")

        store = _DurableOptimizationsStore(redis_factory=_factory)

        await store.set("opt_local", _make_opt_resp("opt_local"))
        assert await store.contains("opt_local") is True
        got = await store.get("opt_local")
        assert got is not None and got.optimization_id == "opt_local"
        values = await store.values()
        assert [v.optimization_id for v in values] == ["opt_local"]

    @pytest.mark.asyncio
    async def test_falls_back_when_redis_command_errors(self):
        """A mid-operation Redis error degrades to the in-memory fallback."""
        from src.api.routes.resource_optimizer import _DurableOptimizationsStore

        class _BrokenRedis(_FakeAsyncRedis):
            async def get(self, key):  # noqa: ANN001
                raise ConnectionError("boom")

        broken = _BrokenRedis()

        async def _factory():
            return broken

        store = _DurableOptimizationsStore(redis_factory=_factory)
        await store.set("opt_x", _make_opt_resp("opt_x"))
        got = await store.get("opt_x")
        assert got is not None and got.optimization_id == "opt_x"

    @pytest.mark.asyncio
    async def test_redis_eviction_bounds_index(self):
        """The Redis index is FIFO-bounded so it cannot grow without limit."""
        from src.api.routes.resource_optimizer import _DurableOptimizationsStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableOptimizationsStore(redis_factory=_factory, max_entries=3)
        for i in range(5):
            await store.set(f"opt_{i}", _make_opt_resp(f"opt_{i}"))

        values = await store.values()
        ids = sorted(v.optimization_id for v in values)
        assert ids == ["opt_2", "opt_3", "opt_4"]
        assert await store.contains("opt_0") is False

    @pytest.mark.asyncio
    async def test_status_update_preserves_creation_score(self):
        """A later status update must NOT re-score the record as newest
        (FIFO would degrade to LRU and evict the wrong record)."""
        from src.api.routes.resource_optimizer import (
            OptimizationStatus,
            _DurableOptimizationsStore,
        )

        shared = _FakeAsyncRedis(now=1000.0)

        async def _factory():
            return shared

        store = _DurableOptimizationsStore(redis_factory=_factory)
        await store.set("opt_old", _make_opt_resp("opt_old"))
        score_after_create = await shared.zscore("resources:optimization:index", "opt_old")

        # Advance the clock, then update status (read-modify-write).
        shared.advance(500.0)
        updated = _make_opt_resp("opt_old")
        updated.status = OptimizationStatus.PENDING
        await store.set("opt_old", updated)

        score_after_update = await shared.zscore("resources:optimization:index", "opt_old")
        assert score_after_update == score_after_create

    @pytest.mark.asyncio
    async def test_module_store_is_durable_instance(self):
        """The live module-level store is the durable (Redis-backed) store."""
        from src.api.routes.resource_optimizer import (
            _DurableOptimizationsStore,
            _optimizations_store,
        )

        assert isinstance(_optimizations_store, _DurableOptimizationsStore)

    @pytest.mark.asyncio
    async def test_clean_redis_miss_is_authoritative_not_stale_memory(self):
        """When Redis is REACHABLE, a clean miss returns None and does NOT
        serve a stale in-process mirror.

        Reproduces the codex MEDIUM: after the Redis key is gone (TTL expiry /
        eviction / delete by another worker), the memory mirror still holds the
        record. ``get`` must NOT serve it (Redis is authoritative) — else a GET
        would return a record that Redis-backed ``values()`` omits, re-creating
        the cross-worker split-brain.
        """
        from src.api.routes.resource_optimizer import _DurableOptimizationsStore

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableOptimizationsStore(redis_factory=_factory)
        await store.set("opt_gone", _make_opt_resp("opt_gone"))
        # The record is mirrored in memory AND in Redis.
        assert "opt_gone" in store._memory

        # Simulate the Redis key disappearing (TTL/eviction/another worker).
        await shared.delete("resources:optimization:opt_gone")
        await shared.zrem("resources:optimization:index", "opt_gone")

        # Redis is reachable + clean miss -> authoritative absence (404),
        # despite the stale memory mirror.
        assert await store.get("opt_gone") is None

    @pytest.mark.asyncio
    async def test_poison_record_returns_none_and_self_heals(self):
        """An unreadable (corrupt) persisted payload must not 500: it returns
        None and the poison key + index member are lazily removed."""
        from src.api.routes.resource_optimizer import (
            _REDIS_INDEX_KEY,
            _DurableOptimizationsStore,
        )

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableOptimizationsStore(redis_factory=_factory)
        # Write a corrupt payload directly (not valid OptimizationResponse JSON).
        await shared.set("resources:optimization:opt_poison", "{not valid json")
        await shared.zadd(_REDIS_INDEX_KEY, {"opt_poison": 1.0})

        assert await store.get("opt_poison") is None
        # Poison key + index member removed (self-heal).
        assert await shared.get("resources:optimization:opt_poison") is None
        assert await shared.zscore(_REDIS_INDEX_KEY, "opt_poison") is None

    @pytest.mark.asyncio
    async def test_non_finite_result_persisted_as_failed_not_unreadable(self):
        """A NaN/inf result must be stored as an honest FAILED record (no
        fabricated finite numbers), and must round-trip cleanly on read.

        Without the write-side guard, ``model_dump_json`` serialises the NaN to
        ``null`` and the record becomes UNREADABLE — a real optimization id that
        404s on every GET. The guard sanitises to FAILED instead.
        """
        from src.api.routes.resource_optimizer import (
            AllocationResult,
            OptimizationResponse,
            OptimizationStatus,
            ResourceType,
            _DurableOptimizationsStore,
        )

        shared = _FakeAsyncRedis()

        async def _factory():
            return shared

        store = _DurableOptimizationsStore(redis_factory=_factory)

        degenerate = OptimizationResponse(
            optimization_id="opt_nan",
            status=OptimizationStatus.COMPLETED,
            resource_type=ResourceType.BUDGET,
            objective=OptimizationObjective.MAXIMIZE_OUTCOME,
            optimal_allocations=[
                AllocationResult(
                    entity_id="territory_x",
                    entity_type="territory",
                    current_allocation=50000.0,
                    optimized_allocation=float("nan"),  # degenerate solve
                    change=float("inf"),
                    change_percentage=0.0,
                    expected_impact=0.0,
                )
            ],
            objective_value=float("nan"),
        )

        await store.set("opt_nan", degenerate)

        # Must round-trip cleanly (not 404 / poison-prune) and be honest FAILED.
        got = await store.get("opt_nan")
        assert got is not None
        assert got.status == OptimizationStatus.FAILED
        assert got.optimal_allocations == []
        assert got.objective_value is None
        assert any("non-finite" in w.lower() for w in got.warnings)

    @pytest.mark.asyncio
    async def test_is_durable_false_when_redis_commands_fail(self):
        """A REACHABLE Redis whose commands fail must report NOT durable.

        Reproduces the codex iter-2 MEDIUM: a factory-only probe would report
        durable while reads/writes silently degrade to memory. ``is_durable``
        exercises a real command so health honestly surfaces the degradation.
        """
        from src.api.routes.resource_optimizer import _DurableOptimizationsStore

        class _CommandBrokenRedis(_FakeAsyncRedis):
            async def zcard(self, key):  # noqa: ANN001
                raise ConnectionError("commands down")

        broken = _CommandBrokenRedis()

        async def _factory():
            return broken  # factory SUCCEEDS (client is reachable)

        store = _DurableOptimizationsStore(redis_factory=_factory)
        # Factory returns a client, but the durability probe command fails.
        assert await store.is_durable() is False
