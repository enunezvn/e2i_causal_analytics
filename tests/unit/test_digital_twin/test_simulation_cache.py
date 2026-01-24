"""
Unit Tests for Digital Twin Simulation Cache.

Tests cover:
- Cache initialization and configuration
- Cache key generation
- Cache hit and miss scenarios
- TTL expiration
- Model cache invalidation
- Cache statistics tracking
- Error handling when Redis unavailable
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from src.digital_twin.simulation_cache import (
    CacheStats,
    SimulationCache,
    SimulationCacheConfig,
    get_simulation_cache,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return SimulationCacheConfig(
        ttl_seconds=1800,
        prefix="test_twin_sim",
        enabled=True,
        max_cached_results=100,
    )


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.setex = AsyncMock(return_value=True)
    client.hset = AsyncMock(return_value=True)
    client.hincrby = AsyncMock(return_value=1)
    client.hgetall = AsyncMock(return_value={})
    client.expire = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.scan_iter = AsyncMock(return_value=iter([]))
    return client


@pytest.fixture
def cache(cache_config, mock_redis_client):
    """Create cache with mock Redis client."""
    return SimulationCache(redis_client=mock_redis_client, config=cache_config)


@pytest.fixture
def sample_intervention_config():
    """Create sample intervention configuration."""
    return InterventionConfig(
        intervention_type="email_campaign",
        channel="email",
        frequency="weekly",
        duration_weeks=8,
        intensity_multiplier=1.0,
        target_deciles=[1, 2, 3],
        target_specialties=["rheumatology"],
        target_regions=["northeast"],
    )


@pytest.fixture
def sample_population_filter():
    """Create sample population filter."""
    return PopulationFilter(
        specialties=["rheumatology", "dermatology"],
        deciles=[1, 2, 3],
        regions=["northeast"],
        adoption_stages=["early_majority"],
    )


@pytest.fixture
def sample_simulation_result(sample_intervention_config, sample_population_filter):
    """Create sample simulation result."""
    return SimulationResult(
        simulation_id=uuid4(),
        model_id=uuid4(),
        intervention_config=sample_intervention_config,
        population_filters=sample_population_filter,
        twin_count=500,
        simulated_ate=0.08,
        simulated_ci_lower=0.05,
        simulated_ci_upper=0.11,
        simulated_std_error=0.015,
        effect_heterogeneity=EffectHeterogeneity(),
        recommendation=SimulationRecommendation.DEPLOY,
        recommendation_rationale="Effect size meets deployment threshold",
        recommended_sample_size=1000,
        recommended_duration_weeks=8,
        simulation_confidence=0.85,
        fidelity_warning=False,
        fidelity_warning_reason=None,
        model_fidelity_score=0.90,
        status=SimulationStatus.COMPLETED,
        error_message=None,
        execution_time_ms=150,
        memory_usage_mb=25.5,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestSimulationCacheInit:
    """Tests for SimulationCache initialization."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        cache = SimulationCache()

        assert cache.config.enabled is True
        assert cache.config.ttl_seconds == 1800
        assert cache.config.prefix == "twin_sim"
        assert cache._stats.hits == 0
        assert cache._stats.misses == 0

    def test_init_custom_config(self, cache_config, mock_redis_client):
        """Test initialization with custom configuration."""
        cache = SimulationCache(redis_client=mock_redis_client, config=cache_config)

        assert cache.config == cache_config
        assert cache.config.prefix == "test_twin_sim"

    def test_init_disabled(self, mock_redis_client):
        """Test initialization with cache disabled."""
        config = SimulationCacheConfig(enabled=False)
        cache = SimulationCache(redis_client=mock_redis_client, config=config)

        assert cache.config.enabled is False


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_requests(self):
        """Test hit rate when no requests made."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate when all requests are hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate when all requests are misses."""
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0


# =============================================================================
# CACHE KEY GENERATION TESTS
# =============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_key_includes_model_id(
        self, cache, sample_intervention_config, sample_population_filter
    ):
        """Test that cache key includes model ID."""
        model_id = uuid4()

        key1 = cache._generate_cache_key(
            sample_intervention_config, sample_population_filter, model_id
        )

        # Different model ID should produce different key
        different_model_id = uuid4()
        key2 = cache._generate_cache_key(
            sample_intervention_config, sample_population_filter, different_model_id
        )

        assert key1 != key2

    def test_key_uniqueness_by_intervention(self, cache, sample_population_filter):
        """Test that different interventions produce different keys."""
        model_id = uuid4()

        config1 = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        config2 = InterventionConfig(
            intervention_type="call_frequency_increase",
            duration_weeks=8,
        )

        key1 = cache._generate_cache_key(config1, sample_population_filter, model_id)
        key2 = cache._generate_cache_key(config2, sample_population_filter, model_id)

        assert key1 != key2

    def test_key_uniqueness_by_filter(self, cache, sample_intervention_config):
        """Test that different filters produce different keys."""
        model_id = uuid4()

        filter1 = PopulationFilter(specialties=["rheumatology"])
        filter2 = PopulationFilter(specialties=["dermatology"])

        key1 = cache._generate_cache_key(sample_intervention_config, filter1, model_id)
        key2 = cache._generate_cache_key(sample_intervention_config, filter2, model_id)

        assert key1 != key2

    def test_key_consistency(
        self, cache, sample_intervention_config, sample_population_filter
    ):
        """Test that same inputs produce same key."""
        model_id = uuid4()

        key1 = cache._generate_cache_key(
            sample_intervention_config, sample_population_filter, model_id
        )
        key2 = cache._generate_cache_key(
            sample_intervention_config, sample_population_filter, model_id
        )

        assert key1 == key2

    def test_key_with_no_filter(self, cache, sample_intervention_config):
        """Test key generation with no population filter."""
        model_id = uuid4()

        key = cache._generate_cache_key(sample_intervention_config, None, model_id)

        assert key is not None
        assert "email_campaign" in key

    def test_make_key_adds_prefix(self, cache):
        """Test that _make_key adds configured prefix."""
        key = cache._make_key("test_key")
        assert key == "test_twin_sim:test_key"


# =============================================================================
# CACHE HIT/MISS TESTS
# =============================================================================


class TestCacheHitMiss:
    """Tests for cache hit and miss scenarios."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(
        self, cache, sample_intervention_config, sample_population_filter
    ):
        """Test that cache miss returns None."""
        model_id = uuid4()

        result = await cache.get_cached_result(
            sample_intervention_config, sample_population_filter, model_id
        )

        assert result is None
        assert cache._stats.misses == 1
        assert cache._stats.hits == 0

    @pytest.mark.asyncio
    async def test_cache_hit_returns_result(
        self,
        cache,
        mock_redis_client,
        sample_intervention_config,
        sample_population_filter,
        sample_simulation_result,
    ):
        """Test that cache hit returns cached result."""
        import pickle

        # Setup mock to return cached data
        result_dict = sample_simulation_result.model_dump(mode="json")
        cached_data = pickle.dumps(result_dict).decode("latin-1")
        mock_redis_client.get = AsyncMock(return_value=cached_data)

        model_id = sample_simulation_result.model_id

        result = await cache.get_cached_result(
            sample_intervention_config, sample_population_filter, model_id
        )

        assert result is not None
        assert result.simulated_ate == sample_simulation_result.simulated_ate
        assert cache._stats.hits == 1
        assert cache._stats.misses == 0

    @pytest.mark.asyncio
    async def test_cache_disabled_returns_none(
        self, mock_redis_client, sample_intervention_config, sample_population_filter
    ):
        """Test that disabled cache always returns None."""
        config = SimulationCacheConfig(enabled=False)
        cache = SimulationCache(redis_client=mock_redis_client, config=config)

        model_id = uuid4()

        result = await cache.get_cached_result(
            sample_intervention_config, sample_population_filter, model_id
        )

        assert result is None
        # Should not increment stats when disabled
        assert cache._stats.misses == 0


# =============================================================================
# CACHE WRITE TESTS
# =============================================================================


class TestCacheWrite:
    """Tests for caching simulation results."""

    @pytest.mark.asyncio
    async def test_cache_result_success(
        self, cache, mock_redis_client, sample_simulation_result
    ):
        """Test successfully caching a result."""
        success = await cache.cache_result(sample_simulation_result)

        assert success is True
        mock_redis_client.setex.assert_called_once()
        mock_redis_client.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_result_custom_ttl(
        self, cache, mock_redis_client, sample_simulation_result
    ):
        """Test caching with custom TTL."""
        custom_ttl = 600

        await cache.cache_result(sample_simulation_result, ttl_seconds=custom_ttl)

        # Verify TTL was passed to setex
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == custom_ttl

    @pytest.mark.asyncio
    async def test_cache_disabled_no_write(
        self, mock_redis_client, sample_simulation_result
    ):
        """Test that disabled cache doesn't write."""
        config = SimulationCacheConfig(enabled=False)
        cache = SimulationCache(redis_client=mock_redis_client, config=config)

        success = await cache.cache_result(sample_simulation_result)

        assert success is False
        mock_redis_client.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_stores_metadata(
        self, cache, mock_redis_client, sample_simulation_result
    ):
        """Test that caching stores metadata."""
        await cache.cache_result(sample_simulation_result)

        # Verify hset was called with metadata
        call_args = mock_redis_client.hset.call_args
        metadata = call_args[1]["mapping"]

        assert "created_at" in metadata
        assert "expires_at" in metadata
        assert "intervention_type" in metadata
        assert "model_id" in metadata
        assert "twin_count" in metadata


# =============================================================================
# CACHE INVALIDATION TESTS
# =============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_model_cache(self, cache, mock_redis_client):
        """Test invalidating cache for specific model."""
        model_id = uuid4()

        # Setup mock to return keys for this model (accepts match kwarg)
        async def mock_scan(match=None):
            yield "test_twin_sim:email_campaign:abc123:meta"

        mock_redis_client.scan_iter = mock_scan
        mock_redis_client.hgetall = AsyncMock(
            return_value={"model_id": str(model_id)}
        )

        count = await cache.invalidate_model_cache(model_id)

        assert count == 1
        mock_redis_client.delete.assert_called_once()
        assert cache._stats.invalidations == 1

    @pytest.mark.asyncio
    async def test_invalidate_model_no_matches(self, cache, mock_redis_client):
        """Test invalidation when no keys match model."""
        model_id = uuid4()

        async def mock_scan(match=None):
            return
            yield  # Empty iterator

        mock_redis_client.scan_iter = mock_scan

        count = await cache.invalidate_model_cache(model_id)

        assert count == 0
        mock_redis_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache, mock_redis_client):
        """Test invalidating all cached results."""
        async def mock_scan(match=None):
            yield "test_twin_sim:key1"
            yield "test_twin_sim:key2"
            yield "test_twin_sim:key1:meta"
            yield "test_twin_sim:key2:meta"

        mock_redis_client.scan_iter = mock_scan

        count = await cache.invalidate_all()

        assert count == 2  # 2 data keys (excludes meta)
        mock_redis_client.delete.assert_called_once()


# =============================================================================
# CACHE STATISTICS TESTS
# =============================================================================


class TestCacheStatistics:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_basic(self, cache, mock_redis_client):
        """Test getting basic cache statistics."""
        # Simulate some hits and misses
        cache._stats.hits = 10
        cache._stats.misses = 5
        cache._stats.invalidations = 2

        async def mock_scan(match=None):
            return
            yield  # Empty

        mock_redis_client.scan_iter = mock_scan
        mock_redis_client.ping = AsyncMock(return_value=True)

        stats = await cache.get_stats()

        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 10 / 15
        assert stats["invalidations"] == 2
        assert stats["enabled"] is True
        assert stats["redis_status"] == "connected"

    @pytest.mark.asyncio
    async def test_get_stats_by_intervention(self, cache, mock_redis_client):
        """Test stats breakdown by intervention type."""
        async def mock_scan(match=None):
            yield "test_twin_sim:email:abc:meta"
            yield "test_twin_sim:call:def:meta"

        mock_redis_client.scan_iter = mock_scan
        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.hgetall = AsyncMock(
            side_effect=[
                {"intervention_type": "email_campaign", "hit_count": "5"},
                {"intervention_type": "call_frequency", "hit_count": "3"},
            ]
        )

        stats = await cache.get_stats()

        assert "by_intervention_type" in stats
        assert stats["cached_entries"] == 2

    def test_reset_stats(self, cache):
        """Test resetting cache statistics."""
        cache._stats.hits = 100
        cache._stats.misses = 50

        cache.reset_stats()

        assert cache._stats.hits == 0
        assert cache._stats.misses == 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestCacheErrorHandling:
    """Tests for cache error handling."""

    @pytest.mark.asyncio
    async def test_no_redis_client(self, sample_intervention_config):
        """Test cache behavior when Redis client unavailable."""
        # Create cache without Redis client
        cache = SimulationCache(redis_client=None)

        # Mock _get_client to return None
        with patch.object(cache, "_get_client", return_value=None):
            result = await cache.get_cached_result(
                sample_intervention_config, None, uuid4()
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_error_on_get(
        self,
        cache,
        mock_redis_client,
        sample_intervention_config,
    ):
        """Test handling Redis error on get."""
        mock_redis_client.get = AsyncMock(side_effect=Exception("Redis connection error"))

        result = await cache.get_cached_result(
            sample_intervention_config, None, uuid4()
        )

        assert result is None
        assert cache._stats.errors == 1

    @pytest.mark.asyncio
    async def test_redis_error_on_set(
        self,
        cache,
        mock_redis_client,
        sample_simulation_result,
    ):
        """Test handling Redis error on set."""
        mock_redis_client.setex = AsyncMock(
            side_effect=Exception("Redis connection error")
        )

        success = await cache.cache_result(sample_simulation_result)

        assert success is False
        assert cache._stats.errors == 1

    @pytest.mark.asyncio
    async def test_invalid_cached_data(
        self,
        cache,
        mock_redis_client,
        sample_intervention_config,
    ):
        """Test handling of corrupted cached data."""
        mock_redis_client.get = AsyncMock(return_value="invalid_pickle_data")

        result = await cache.get_cached_result(
            sample_intervention_config, None, uuid4()
        )

        assert result is None
        assert cache._stats.errors == 1


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCacheFactory:
    """Tests for cache factory function."""

    def test_get_simulation_cache_default(self):
        """Test factory with default configuration."""
        cache = get_simulation_cache()

        assert isinstance(cache, SimulationCache)
        assert cache.config.enabled is True

    def test_get_simulation_cache_custom(self, cache_config, mock_redis_client):
        """Test factory with custom configuration."""
        cache = get_simulation_cache(
            redis_client=mock_redis_client,
            config=cache_config,
        )

        assert cache.config == cache_config
        assert cache._client == mock_redis_client
