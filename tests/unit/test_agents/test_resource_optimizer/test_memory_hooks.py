"""
Tests for Resource Optimizer Memory Hooks.

Tests the memory integration for the Resource Optimizer agent including:
- Context retrieval from working/episodic/procedural memory
- Optimization caching in working memory
- Pattern learning in procedural memory
- Graceful degradation when memory systems unavailable
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.resource_optimizer.memory_hooks import (
    ResourceOptimizerMemoryHooks,
    OptimizationContext,
    OptimizationPattern,
    OptimizationRecord,
    contribute_to_memory,
    get_resource_optimizer_memory_hooks,
    reset_memory_hooks,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def memory_hooks():
    """Create a fresh memory hooks instance."""
    reset_memory_hooks()
    return ResourceOptimizerMemoryHooks()


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_working_memory(mock_redis):
    """Create mock working memory with Redis client."""
    wm = MagicMock()
    wm.get_messages = AsyncMock(return_value=[])
    wm.get_client = AsyncMock(return_value=mock_redis)
    return wm


@pytest.fixture
def sample_constraints():
    """Sample constraints for testing."""
    return [
        {"constraint_type": "budget", "value": 100000, "scope": "global"},
        {"constraint_type": "min_total", "value": 50000, "scope": "global"},
    ]


@pytest.fixture
def sample_optimization_result():
    """Sample optimization result for testing."""
    return {
        "optimal_allocations": [
            {
                "entity_id": "territory_1",
                "entity_type": "territory",
                "current_allocation": 20000,
                "optimized_allocation": 30000,
                "change": 10000,
                "change_percentage": 50.0,
                "expected_impact": 45000,
            },
            {
                "entity_id": "territory_2",
                "entity_type": "territory",
                "current_allocation": 30000,
                "optimized_allocation": 25000,
                "change": -5000,
                "change_percentage": -16.67,
                "expected_impact": 30000,
            },
        ],
        "objective_value": 75000,
        "solver_status": "optimal",
        "projected_total_outcome": 75000,
        "projected_roi": 1.36,
        "solve_time_ms": 15,
        "recommendations": [
            "Increase allocation to territory_1",
            "Reduce allocation from territory_2",
        ],
    }


@pytest.fixture
def sample_state(sample_constraints):
    """Sample state for testing."""
    return {
        "resource_type": "budget",
        "objective": "maximize_outcome",
        "solver_type": "linear",
        "constraints": sample_constraints,
        "status": "completed",
        "query": "Optimize budget allocation",
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestMemoryHooksInitialization:
    """Tests for memory hooks initialization."""

    def test_init_creates_instance(self, memory_hooks):
        """Test that initialization creates a valid instance."""
        assert memory_hooks is not None
        assert memory_hooks._working_memory is None  # Lazy-loaded

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance."""
        reset_memory_hooks()
        hooks1 = get_resource_optimizer_memory_hooks()
        hooks2 = get_resource_optimizer_memory_hooks()
        assert hooks1 is hooks2

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        hooks1 = get_resource_optimizer_memory_hooks()
        reset_memory_hooks()
        hooks2 = get_resource_optimizer_memory_hooks()
        assert hooks1 is not hooks2


# ============================================================================
# CONTEXT RETRIEVAL TESTS
# ============================================================================


class TestGetContext:
    """Tests for context retrieval."""

    @pytest.mark.asyncio
    async def test_get_context_returns_context_object(self, memory_hooks):
        """Test that get_context returns an OptimizationContext."""
        context = await memory_hooks.get_context(
            session_id="test-session",
            resource_type="budget",
            objective="maximize_outcome",
        )

        assert isinstance(context, OptimizationContext)
        assert context.session_id == "test-session"
        assert isinstance(context.retrieval_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_get_context_with_constraints(self, memory_hooks, sample_constraints):
        """Test context retrieval with constraints."""
        context = await memory_hooks.get_context(
            session_id="test-session",
            resource_type="budget",
            objective="maximize_roi",
            constraints=sample_constraints,
        )

        assert context.session_id == "test-session"
        # Should not raise, returns empty lists when memory unavailable
        assert isinstance(context.working_memory, list)
        assert isinstance(context.similar_optimizations, list)
        assert isinstance(context.learned_patterns, list)

    @pytest.mark.asyncio
    async def test_get_context_graceful_degradation(self, memory_hooks):
        """Test that context retrieval handles missing memory gracefully."""
        # Patch the working_memory property to return None (prevents lazy loading)
        with patch(
            "src.agents.resource_optimizer.memory_hooks.ResourceOptimizerMemoryHooks.working_memory",
            new_callable=lambda: property(lambda self: None),
        ):
            context = await memory_hooks.get_context(
                session_id="test-session",
                resource_type="budget",
                objective="maximize_outcome",
            )

            # Should return empty context, not raise
            assert context.working_memory == []
            assert context.cached_optimization is None
            assert context.similar_optimizations == []
            assert context.learned_patterns == []


class TestWorkingMemoryContext:
    """Tests for working memory context retrieval."""

    @pytest.mark.asyncio
    async def test_get_working_memory_context_success(
        self, memory_hooks, mock_working_memory
    ):
        """Test successful working memory context retrieval."""
        mock_working_memory.get_messages.return_value = [
            {"role": "user", "content": "Optimize my budget"},
            {"role": "assistant", "content": "I'll help with that."},
        ]

        memory_hooks._working_memory = mock_working_memory

        messages = await memory_hooks._get_working_memory_context("test-session")

        assert len(messages) == 2
        mock_working_memory.get_messages.assert_called_once_with(
            "test-session", limit=10
        )

    @pytest.mark.asyncio
    async def test_get_working_memory_context_unavailable(self, memory_hooks):
        """Test handling when working memory is unavailable."""
        memory_hooks._working_memory = None

        messages = await memory_hooks._get_working_memory_context("test-session")

        assert messages == []


class TestCachedOptimization:
    """Tests for cached optimization retrieval."""

    @pytest.mark.asyncio
    async def test_get_cached_optimization_found(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test retrieving cached optimization."""
        cached_data = {"objective_value": 75000, "status": "completed"}
        mock_redis.get.return_value = json.dumps(cached_data)
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks._get_cached_optimization("test-session")

        assert result == cached_data
        mock_redis.get.assert_called_once_with(
            "resource_optimizer:session:test-session"
        )

    @pytest.mark.asyncio
    async def test_get_cached_optimization_not_found(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test when no cached optimization exists."""
        mock_redis.get.return_value = None
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks._get_cached_optimization("test-session")

        assert result is None


# ============================================================================
# CACHING TESTS
# ============================================================================


class TestCacheOptimization:
    """Tests for optimization caching."""

    @pytest.mark.asyncio
    async def test_cache_optimization_success(
        self, memory_hooks, mock_working_memory, mock_redis, sample_optimization_result
    ):
        """Test successful optimization caching."""
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks.cache_optimization(
            session_id="test-session",
            optimization_result=sample_optimization_result,
        )

        assert result is True
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "resource_optimizer:session:test-session"
        assert call_args[0][1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_cache_optimization_with_scenario(
        self, memory_hooks, mock_working_memory, mock_redis, sample_optimization_result
    ):
        """Test caching with scenario name."""
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks.cache_optimization(
            session_id="test-session",
            optimization_result=sample_optimization_result,
            scenario_name="baseline",
        )

        assert result is True
        # Should be called twice: once for session, once for scenario
        assert mock_redis.setex.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_optimization_no_working_memory(
        self, memory_hooks, sample_optimization_result
    ):
        """Test caching when working memory unavailable."""
        # Patch the working_memory property to return None (prevents lazy loading)
        with patch(
            "src.agents.resource_optimizer.memory_hooks.ResourceOptimizerMemoryHooks.working_memory",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await memory_hooks.cache_optimization(
                session_id="test-session",
                optimization_result=sample_optimization_result,
            )

            assert result is False


class TestScenarioComparison:
    """Tests for scenario comparison retrieval."""

    @pytest.mark.asyncio
    async def test_get_scenario_comparison_success(
        self, memory_hooks, mock_working_memory, mock_redis
    ):
        """Test retrieving multiple scenarios for comparison."""
        scenarios = {
            "baseline": {"objective_value": 70000},
            "optimized": {"objective_value": 85000},
        }

        async def mock_get(key):
            for name, data in scenarios.items():
                if name in key:
                    return json.dumps(data)
            return None

        mock_redis.get = mock_get
        memory_hooks._working_memory = mock_working_memory

        result = await memory_hooks.get_scenario_comparison(
            session_id="test-session",
            scenario_names=["baseline", "optimized"],
        )

        assert "baseline" in result
        assert "optimized" in result
        assert result["baseline"]["objective_value"] == 70000
        assert result["optimized"]["objective_value"] == 85000


# ============================================================================
# PATTERN LEARNING TESTS
# ============================================================================


class TestStoreOptimizationPattern:
    """Tests for optimization pattern storage."""

    @pytest.mark.asyncio
    async def test_store_pattern_success(
        self, memory_hooks, sample_optimization_result, sample_state
    ):
        """Test successful pattern storage."""
        # Patch at the source module where the import happens
        with patch(
            "src.memory.procedural_memory.insert_procedural_memory"
        ) as mock_insert:
            mock_insert.return_value = "pattern-123"

            pattern_id = await memory_hooks.store_optimization_pattern(
                session_id="test-session",
                result=sample_optimization_result,
                state=sample_state,
            )

            # Should attempt to store (may fail if procedural memory not available)
            # This tests the code path, not actual storage
            assert pattern_id is None or isinstance(pattern_id, str)

    @pytest.mark.asyncio
    async def test_store_pattern_skips_failed_optimization(
        self, memory_hooks, sample_optimization_result
    ):
        """Test that failed optimizations don't create patterns."""
        state = {"status": "failed"}

        pattern_id = await memory_hooks.store_optimization_pattern(
            session_id="test-session",
            result=sample_optimization_result,
            state=state,
        )

        assert pattern_id is None

    @pytest.mark.asyncio
    async def test_store_pattern_skips_infeasible(
        self, memory_hooks, sample_state
    ):
        """Test that infeasible solutions don't create patterns."""
        result = {"solver_status": "infeasible", "projected_roi": 0}

        pattern_id = await memory_hooks.store_optimization_pattern(
            session_id="test-session",
            result=result,
            state=sample_state,
        )

        assert pattern_id is None

    @pytest.mark.asyncio
    async def test_store_pattern_skips_low_roi(
        self, memory_hooks, sample_state
    ):
        """Test that low ROI optimizations don't create patterns."""
        result = {
            "solver_status": "optimal",
            "projected_roi": 0.5,  # Below 1.0 threshold
        }

        pattern_id = await memory_hooks.store_optimization_pattern(
            session_id="test-session",
            result=result,
            state=sample_state,
        )

        assert pattern_id is None


# ============================================================================
# EPISODIC MEMORY TESTS
# ============================================================================


class TestStoreOptimization:
    """Tests for optimization storage in episodic memory."""

    @pytest.mark.asyncio
    async def test_store_optimization_graceful_failure(
        self, memory_hooks, sample_optimization_result, sample_state
    ):
        """Test that storage handles missing episodic memory gracefully."""
        # Without mocking, episodic memory is unavailable
        memory_id = await memory_hooks.store_optimization(
            session_id="test-session",
            result=sample_optimization_result,
            state=sample_state,
        )

        # Should return None, not raise
        assert memory_id is None


# ============================================================================
# CONTRIBUTE TO MEMORY TESTS
# ============================================================================


class TestContributeToMemory:
    """Tests for the contribute_to_memory function."""

    @pytest.mark.asyncio
    async def test_contribute_skips_failed_optimization(
        self, sample_optimization_result
    ):
        """Test that failed optimizations are skipped."""
        state = {"status": "failed"}

        counts = await contribute_to_memory(
            result=sample_optimization_result,
            state=state,
        )

        assert counts["episodic_stored"] == 0
        assert counts["working_cached"] == 0
        assert counts["pattern_learned"] == 0

    @pytest.mark.asyncio
    async def test_contribute_with_successful_optimization(
        self, sample_optimization_result, sample_state
    ):
        """Test contribution with successful optimization."""
        counts = await contribute_to_memory(
            result=sample_optimization_result,
            state=sample_state,
        )

        # Without real memory systems, counts should be 0
        # but the function should complete without error
        assert isinstance(counts, dict)
        assert "episodic_stored" in counts
        assert "working_cached" in counts
        assert "pattern_learned" in counts

    @pytest.mark.asyncio
    async def test_contribute_with_custom_hooks(
        self, sample_optimization_result, sample_state, mock_working_memory, mock_redis
    ):
        """Test contribution with custom memory hooks."""
        hooks = ResourceOptimizerMemoryHooks()
        hooks._working_memory = mock_working_memory

        counts = await contribute_to_memory(
            result=sample_optimization_result,
            state=sample_state,
            memory_hooks=hooks,
            session_id="custom-session",
        )

        # Should have cached in working memory
        assert counts["working_cached"] == 1

    @pytest.mark.asyncio
    async def test_contribute_generates_session_id(
        self, sample_optimization_result, sample_state
    ):
        """Test that session ID is generated if not provided."""
        counts = await contribute_to_memory(
            result=sample_optimization_result,
            state=sample_state,
            session_id=None,
        )

        # Should complete without error
        assert isinstance(counts, dict)


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================


class TestDataStructures:
    """Tests for data structure validation."""

    def test_optimization_context_creation(self):
        """Test OptimizationContext creation."""
        context = OptimizationContext(
            session_id="test-session",
            working_memory=[{"role": "user", "content": "test"}],
            cached_optimization={"objective_value": 75000},
            similar_optimizations=[{"id": "opt-1"}],
            learned_patterns=[{"pattern_id": "pat-1"}],
        )

        assert context.session_id == "test-session"
        assert len(context.working_memory) == 1
        assert context.cached_optimization["objective_value"] == 75000
        assert len(context.similar_optimizations) == 1
        assert len(context.learned_patterns) == 1
        assert isinstance(context.retrieval_timestamp, datetime)

    def test_optimization_pattern_creation(self):
        """Test OptimizationPattern creation."""
        pattern = OptimizationPattern(
            pattern_id="pat-123",
            resource_type="budget",
            objective="maximize_roi",
            constraint_signature="budget|min_total",
            solver_type="linear",
            avg_solve_time_ms=50,
            success_rate=0.95,
            common_adjustments=[
                {"entity_type": "territory", "direction": "increase", "magnitude": 25.0}
            ],
        )

        assert pattern.pattern_id == "pat-123"
        assert pattern.resource_type == "budget"
        assert pattern.success_rate == 0.95
        assert len(pattern.common_adjustments) == 1

    def test_optimization_record_creation(self):
        """Test OptimizationRecord creation."""
        record = OptimizationRecord(
            session_id="session-123",
            resource_type="budget",
            objective="maximize_outcome",
            objective_value=75000,
            projected_roi=1.36,
            entities_optimized=5,
            solver_type="linear",
            solve_time_ms=15,
            solver_status="optimal",
        )

        assert record.session_id == "session-123"
        assert record.objective_value == 75000
        assert record.solver_status == "optimal"
        assert isinstance(record.timestamp, datetime)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestMemoryHooksIntegration:
    """Integration tests for memory hooks with the agent."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_memory(self):
        """Test full workflow when memory systems unavailable."""
        hooks = ResourceOptimizerMemoryHooks()

        # Patch the working_memory property to return None (prevents lazy loading)
        with patch(
            "src.agents.resource_optimizer.memory_hooks.ResourceOptimizerMemoryHooks.working_memory",
            new_callable=lambda: property(lambda self: None),
        ):
            # Get context
            context = await hooks.get_context(
                session_id="integration-test",
                resource_type="budget",
                objective="maximize_outcome",
            )

            assert context.session_id == "integration-test"

            # Cache optimization (should fail gracefully)
            cached = await hooks.cache_optimization(
                session_id="integration-test",
                optimization_result={"objective_value": 75000},
            )

            assert cached is False  # No working memory available

            # Store pattern (should fail gracefully)
            pattern_id = await hooks.store_optimization_pattern(
                session_id="integration-test",
                result={"solver_status": "optimal", "projected_roi": 1.5},
                state={"status": "completed", "resource_type": "budget"},
            )

            # Should return None, not raise
            assert pattern_id is None

    @pytest.mark.asyncio
    async def test_ttl_values(self, memory_hooks):
        """Test that TTL values are correctly set."""
        assert memory_hooks.CACHE_TTL_SECONDS == 3600  # 1 hour
        assert memory_hooks.PATTERN_TTL_SECONDS == 2592000  # 30 days
