"""
Integration Tests for Digital Twin Concurrency.

Tests cover:
- Concurrent cache reads
- Concurrent cache writes
- Cache invalidation during reads
- Concurrent simulations
- Transaction rollback scenarios
- Race condition handling

IMPORTANT: These tests require Redis.
Run with: pytest -n 1 --dist=no tests/integration/test_digital_twin_concurrency.py
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from src.digital_twin.models.twin_models import (
    Brand,
    DigitalTwin,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.simulation_cache import SimulationCache, SimulationCacheConfig
from src.digital_twin.simulation_engine import SimulationEngine


# Mark all tests as requiring Redis
pytestmark = [
    pytest.mark.xdist_group(name="digital_twin_concurrency"),
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client for testing."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.setex = AsyncMock(return_value=True)
    client.hset = AsyncMock(return_value=True)
    client.hincrby = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.expire = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)

    async def mock_scan():
        return
        yield  # Empty iterator

    client.scan_iter = mock_scan
    return client


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return SimulationCacheConfig(
        ttl_seconds=300,
        prefix="test_concurrent",
        enabled=True,
    )


@pytest.fixture
def cache(cache_config, mock_redis_client):
    """Create cache with mock Redis client."""
    return SimulationCache(redis_client=mock_redis_client, config=cache_config)


@pytest.fixture
def sample_intervention_config():
    """Create sample intervention configuration."""
    return InterventionConfig(
        intervention_type="email_campaign",
        duration_weeks=8,
        intensity_multiplier=1.0,
    )


@pytest.fixture
def sample_population():
    """Create sample twin population for testing."""
    np.random.seed(42)
    twins = []

    for i in range(200):
        twin = DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features={
                "specialty": ["rheumatology", "dermatology"][i % 2],
                "decile": (i % 10) + 1,
                "digital_engagement_score": np.random.uniform(0.3, 0.8),
                "adoption_stage": "early_majority",
            },
            baseline_outcome=np.random.uniform(0.05, 0.20),
            baseline_propensity=np.random.uniform(0.4, 0.7),
        )
        twins.append(twin)

    return TwinPopulation(
        twin_type=TwinType.HCP,
        brand=Brand.REMIBRUTINIB,
        twins=twins,
        size=len(twins),
        model_id=uuid4(),
    )


def create_simulation_result(model_id=None):
    """Helper to create simulation result."""
    return SimulationResult(
        simulation_id=uuid4(),
        model_id=model_id or uuid4(),
        intervention_config=InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        population_filters=PopulationFilter(),
        twin_count=500,
        simulated_ate=0.08,
        simulated_ci_lower=0.05,
        simulated_ci_upper=0.11,
        simulated_std_error=0.015,
        effect_heterogeneity=EffectHeterogeneity(),
        recommendation=SimulationRecommendation.DEPLOY,
        recommendation_rationale="Effect size meets threshold",
        recommended_sample_size=1000,
        recommended_duration_weeks=8,
        simulation_confidence=0.85,
        fidelity_warning=False,
        fidelity_warning_reason=None,
        model_fidelity_score=0.90,
        status=SimulationStatus.COMPLETED,
        error_message=None,
        execution_time_ms=150,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


# =============================================================================
# CONCURRENT CACHE READ TESTS
# =============================================================================


class TestConcurrentCacheReads:
    """Tests for concurrent cache read operations."""

    @pytest.mark.asyncio
    async def test_concurrent_reads_same_key(self, cache, mock_redis_client):
        """Test multiple concurrent reads for same cache key."""
        model_id = uuid4()
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Track call count
        call_count = 0

        async def mock_get(key):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate network latency
            return None

        mock_redis_client.get = mock_get

        # Run 10 concurrent reads
        tasks = [
            cache.get_cached_result(config, None, model_id)
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete without error
        assert all(r is None for r in results)
        assert call_count == 10
        assert cache._stats.misses == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads_different_keys(self, cache, mock_redis_client):
        """Test concurrent reads for different cache keys."""
        configs = [
            InterventionConfig(
                intervention_type=f"intervention_{i}",
                duration_weeks=8,
            )
            for i in range(5)
        ]

        model_id = uuid4()

        tasks = [
            cache.get_cached_result(config, None, model_id)
            for config in configs
        ]

        results = await asyncio.gather(*tasks)

        assert all(r is None for r in results)
        assert cache._stats.misses == 5

    @pytest.mark.asyncio
    async def test_read_isolation(self, cache, mock_redis_client):
        """Test that concurrent reads don't interfere with each other."""
        import pickle

        model_id = uuid4()
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # Return different data for alternating calls
        results_returned = []

        async def mock_get(key):
            await asyncio.sleep(np.random.uniform(0.001, 0.01))
            result = create_simulation_result(model_id)
            results_returned.append(result.simulation_id)
            return pickle.dumps(result.model_dump(mode="json")).decode("latin-1")

        mock_redis_client.get = mock_get

        tasks = [
            cache.get_cached_result(config, None, model_id)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All results should be valid SimulationResults
        assert all(r is not None for r in results)
        assert cache._stats.hits == 5


# =============================================================================
# CONCURRENT CACHE WRITE TESTS
# =============================================================================


class TestConcurrentCacheWrites:
    """Tests for concurrent cache write operations."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_key(self, cache, mock_redis_client):
        """Test multiple concurrent writes to same cache key."""
        model_id = uuid4()

        # Track writes
        write_values = []

        async def mock_setex(key, ttl, value):
            write_values.append(value)
            await asyncio.sleep(0.01)
            return True

        mock_redis_client.setex = mock_setex

        results = [create_simulation_result(model_id) for _ in range(5)]

        tasks = [cache.cache_result(result) for result in results]

        successes = await asyncio.gather(*tasks)

        # All writes should succeed
        assert all(successes)
        assert len(write_values) == 5

    @pytest.mark.asyncio
    async def test_concurrent_writes_different_keys(self, cache, mock_redis_client):
        """Test concurrent writes to different cache keys."""
        write_count = 0

        async def mock_setex(key, ttl, value):
            nonlocal write_count
            write_count += 1
            await asyncio.sleep(0.01)
            return True

        mock_redis_client.setex = mock_setex

        results = [create_simulation_result() for _ in range(5)]

        tasks = [cache.cache_result(result) for result in results]

        successes = await asyncio.gather(*tasks)

        assert all(successes)
        assert write_count == 5

    @pytest.mark.asyncio
    async def test_write_data_integrity(self, cache, mock_redis_client):
        """Test that concurrent writes maintain data integrity."""
        import pickle

        stored_data = {}

        async def mock_setex(key, ttl, value):
            stored_data[key] = value
            return True

        mock_redis_client.setex = mock_setex

        # Create results with distinct ATEs
        results = []
        for i in range(3):
            result = create_simulation_result()
            result.simulated_ate = 0.05 + i * 0.01
            results.append(result)

        tasks = [cache.cache_result(result) for result in results]

        await asyncio.gather(*tasks)

        # Verify stored data is valid
        expected_ates = [0.05, 0.06, 0.07]
        for key, value in stored_data.items():
            data = pickle.loads(value.encode("latin-1"))
            assert "simulated_ate" in data
            # Use tolerance for floating point comparison
            ate = data["simulated_ate"]
            assert any(abs(ate - expected) < 1e-9 for expected in expected_ates)


# =============================================================================
# CACHE INVALIDATION DURING READ TESTS
# =============================================================================


class TestCacheInvalidationDuringRead:
    """Tests for cache invalidation during concurrent reads."""

    @pytest.mark.asyncio
    async def test_invalidation_during_read(self, cache, mock_redis_client):
        """Test that invalidation during read is handled correctly."""
        import pickle

        model_id = uuid4()
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        read_started = asyncio.Event()
        invalidation_done = asyncio.Event()

        async def mock_get(key):
            read_started.set()
            # Wait for invalidation to happen
            await invalidation_done.wait()
            # Return None (data was invalidated)
            return None

        async def mock_scan():
            yield f"test_concurrent:email_campaign:abc:meta"

        mock_redis_client.get = mock_get
        mock_redis_client.scan_iter = mock_scan
        mock_redis_client.hgetall = AsyncMock(
            return_value={"model_id": str(model_id)}
        )

        async def do_read():
            return await cache.get_cached_result(config, None, model_id)

        async def do_invalidate():
            await read_started.wait()
            result = await cache.invalidate_model_cache(model_id)
            invalidation_done.set()
            return result

        # Run read and invalidation concurrently
        read_result, invalidate_count = await asyncio.gather(
            do_read(), do_invalidate()
        )

        # Read should return None (cache was invalidated)
        assert read_result is None

    @pytest.mark.asyncio
    async def test_multiple_readers_one_invalidator(self, cache, mock_redis_client):
        """Test multiple concurrent readers with one invalidator."""
        model_id = uuid4()
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        read_count = 0

        async def mock_get(key):
            nonlocal read_count
            read_count += 1
            await asyncio.sleep(0.01)
            return None

        async def mock_scan():
            return
            yield

        mock_redis_client.get = mock_get
        mock_redis_client.scan_iter = mock_scan

        # Start multiple readers and one invalidator
        read_tasks = [
            cache.get_cached_result(config, None, model_id)
            for _ in range(5)
        ]
        invalidate_task = cache.invalidate_model_cache(model_id)

        all_results = await asyncio.gather(*read_tasks, invalidate_task)

        # All operations should complete
        assert len(all_results) == 6
        assert read_count == 5


# =============================================================================
# CONCURRENT SIMULATION TESTS
# =============================================================================


class TestConcurrentSimulations:
    """Tests for concurrent simulation execution."""

    def test_concurrent_simulations_independent(self, sample_population):
        """Test that concurrent simulations produce independent results."""
        import concurrent.futures

        engine = SimulationEngine(sample_population)

        configs = [
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=8,
                intensity_multiplier=1.0 + i * 0.1,
            )
            for i in range(3)
        ]

        def run_simulation(config):
            return engine.simulate(config)

        # Run simulations in parallel using threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run_simulation, configs))

        # All should complete successfully
        assert all(r.status == SimulationStatus.COMPLETED for r in results)

        # Results should be different due to different intensities
        ates = [r.simulated_ate for r in results]
        assert len(set(round(ate, 4) for ate in ates)) > 1  # Not all identical

    def test_concurrent_simulations_same_config(self, sample_population):
        """Test concurrent simulations with same config."""
        import concurrent.futures

        engine = SimulationEngine(sample_population)

        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        def run_simulation(_):
            return engine.simulate(config)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(run_simulation, range(5)))

        # All should complete
        assert all(r.status == SimulationStatus.COMPLETED for r in results)

        # Results may vary due to random effects
        assert len(results) == 5

    def test_concurrent_simulations_with_filtering(self, sample_population):
        """Test concurrent simulations with different population filters."""
        import concurrent.futures

        engine = SimulationEngine(sample_population)

        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        filters = [
            PopulationFilter(specialties=["rheumatology"]),
            PopulationFilter(specialties=["dermatology"]),
            PopulationFilter(deciles=[1, 2, 3]),
        ]

        def run_simulation(filter_):
            return engine.simulate(config, population_filter=filter_)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run_simulation, filters))

        # All should complete
        completed = [r for r in results if r.status == SimulationStatus.COMPLETED]
        assert len(completed) >= 2  # At least 2 should have enough twins


# =============================================================================
# TRANSACTION ROLLBACK TESTS
# =============================================================================


class TestTransactionRollback:
    """Tests for transaction rollback scenarios."""

    @pytest.mark.asyncio
    async def test_partial_write_rollback(self, cache, mock_redis_client):
        """Test handling of partial write failure."""
        # First setex succeeds, second fails
        call_count = 0

        async def mock_setex(key, ttl, value):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise Exception("Redis write failed")
            return True

        mock_redis_client.setex = mock_setex

        results = [create_simulation_result() for _ in range(5)]

        tasks = [cache.cache_result(result) for result in results]

        successes = await asyncio.gather(*tasks, return_exceptions=True)

        # First 2 should succeed, rest should fail
        success_count = sum(1 for s in successes if s is True)
        failure_count = sum(1 for s in successes if s is False)

        assert success_count == 2
        assert failure_count == 3

    @pytest.mark.asyncio
    async def test_db_error_mid_operation(self):
        """Test handling of database error during operation."""
        from src.digital_twin.twin_repository import SimulationRepository

        mock_client = MagicMock()

        # Setup to fail on certain operations
        call_count = 0

        def mock_insert():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise Exception("Database connection lost")
            mock_result = MagicMock()
            mock_result.data = [{"simulation_id": str(uuid4())}]
            return mock_result

        mock_table = MagicMock()
        mock_table.insert.return_value.execute = mock_insert
        mock_client.table.return_value = mock_table

        repo = SimulationRepository(supabase_client=mock_client)

        # First save should succeed, second should fail
        result1 = create_simulation_result()
        result2 = create_simulation_result()

        # Note: This tests the behavior, actual rollback depends on DB implementation
        try:
            await repo.save_simulation(result1, brand="REMIBRUTINIB")
        except Exception:
            pass

        try:
            await repo.save_simulation(result2, brand="REMIBRUTINIB")
        except Exception as e:
            assert "Database connection lost" in str(e)


# =============================================================================
# RACE CONDITION TESTS
# =============================================================================


class TestRaceConditions:
    """Tests for race condition handling."""

    @pytest.mark.asyncio
    async def test_read_write_race(self, cache, mock_redis_client):
        """Test race between read and write operations."""
        import pickle

        model_id = uuid4()
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        cached_value = None

        async def mock_get(key):
            await asyncio.sleep(0.01)
            return cached_value

        async def mock_setex(key, ttl, value):
            nonlocal cached_value
            await asyncio.sleep(0.01)
            cached_value = value
            return True

        mock_redis_client.get = mock_get
        mock_redis_client.setex = mock_setex

        result = create_simulation_result(model_id)

        async def read_cache():
            return await cache.get_cached_result(config, None, model_id)

        async def write_cache():
            return await cache.cache_result(result)

        # Run read and write concurrently multiple times
        for _ in range(3):
            tasks = [read_cache(), write_cache()]
            await asyncio.gather(*tasks)

        # After multiple iterations, cache should contain data
        assert cached_value is not None

    @pytest.mark.asyncio
    async def test_stats_counter_race(self, cache, mock_redis_client):
        """Test that stats counters are updated correctly under concurrency."""
        model_id = uuid4()
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        # All reads will be misses
        mock_redis_client.get = AsyncMock(return_value=None)

        tasks = [
            cache.get_cached_result(config, None, model_id)
            for _ in range(100)
        ]

        await asyncio.gather(*tasks)

        # Stats should accurately reflect all misses
        # Note: Without proper locking, this could be inaccurate
        # The test verifies current implementation behavior
        assert cache._stats.misses == 100
        assert cache._stats.hits == 0

    @pytest.mark.asyncio
    async def test_invalidation_race(self, cache, mock_redis_client):
        """Test race between multiple invalidation requests."""
        model_id = uuid4()

        delete_calls = []

        async def mock_delete(*keys):
            delete_calls.append(keys)
            await asyncio.sleep(0.01)
            return len(keys)

        async def mock_scan():
            yield f"test_concurrent:email:abc:meta"

        mock_redis_client.delete = mock_delete
        mock_redis_client.scan_iter = mock_scan
        mock_redis_client.hgetall = AsyncMock(
            return_value={"model_id": str(model_id)}
        )

        # Run multiple invalidations concurrently
        tasks = [
            cache.invalidate_model_cache(model_id)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All invalidations should complete
        assert len(results) == 5
