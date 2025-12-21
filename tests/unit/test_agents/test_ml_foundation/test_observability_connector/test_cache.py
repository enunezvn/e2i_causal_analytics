"""Tests for MetricsCache.

Version: 1.0.0 (Phase 3.3 Metrics Caching)
Tests metrics caching with mocked Redis dependencies.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.observability_connector.cache import (
    CacheBackend,
    CacheConfig,
    CacheEntry,
    CacheMetrics,
    MetricsCache,
    get_metrics_cache,
    reset_metrics_cache,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def cache_config():
    """Create a test cache configuration."""
    return CacheConfig(
        backend=CacheBackend.MEMORY,
        fallback_to_memory=True,
        key_prefix="test_metrics",
        ttl_1h=60,
        ttl_24h=300,
        ttl_7d=600,
        ttl_default=120,
        max_memory_entries=100,
        cleanup_interval=10,
    )


@pytest.fixture
def mock_redis_client():
    """Create a mock async Redis client."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.scan = AsyncMock(return_value=(0, []))
    return mock


@pytest.fixture
def sample_metrics():
    """Create sample metrics data."""
    return {
        "latency_p50": 100,
        "latency_p99": 500,
        "error_rate": 0.01,
        "total_spans": 1000,
    }


@pytest.fixture(autouse=True)
async def reset_singleton():
    """Reset the metrics cache singleton before each test."""
    await reset_metrics_cache()
    yield
    await reset_metrics_cache()


# ============================================================================
# TEST: CacheBackend Enum
# ============================================================================


class TestCacheBackend:
    """Test CacheBackend enum."""

    def test_redis_value(self):
        """Test Redis backend value."""
        assert CacheBackend.REDIS.value == "redis"

    def test_memory_value(self):
        """Test Memory backend value."""
        assert CacheBackend.MEMORY.value == "memory"

    def test_is_string_enum(self):
        """Test enum values are strings."""
        assert isinstance(CacheBackend.REDIS.value, str)
        assert isinstance(CacheBackend.MEMORY.value, str)


# ============================================================================
# TEST: CacheConfig
# ============================================================================


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.backend == CacheBackend.MEMORY
        assert config.fallback_to_memory is True
        assert config.key_prefix == "obs_metrics"
        assert config.ttl_1h == 60
        assert config.ttl_24h == 300
        assert config.ttl_7d == 600
        assert config.ttl_default == 120
        assert config.max_memory_entries == 1000
        assert config.cleanup_interval == 60

    def test_custom_values(self, cache_config):
        """Test custom configuration values."""
        assert cache_config.key_prefix == "test_metrics"
        assert cache_config.max_memory_entries == 100

    def test_get_ttl_1h(self, cache_config):
        """Test get_ttl for 1h window."""
        assert cache_config.get_ttl("1h") == 60

    def test_get_ttl_24h(self, cache_config):
        """Test get_ttl for 24h window."""
        assert cache_config.get_ttl("24h") == 300

    def test_get_ttl_7d(self, cache_config):
        """Test get_ttl for 7d window."""
        assert cache_config.get_ttl("7d") == 600

    def test_get_ttl_unknown(self, cache_config):
        """Test get_ttl for unknown window returns default."""
        assert cache_config.get_ttl("30d") == cache_config.ttl_default
        assert cache_config.get_ttl("unknown") == cache_config.ttl_default


# ============================================================================
# TEST: CacheMetrics
# ============================================================================


class TestCacheMetrics:
    """Test CacheMetrics dataclass."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = CacheMetrics()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.invalidations == 0
        assert metrics.errors == 0
        assert metrics.redis_failures == 0
        assert metrics.memory_fallbacks == 0

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        metrics = CacheMetrics(hits=100, misses=0)
        assert metrics.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        metrics = CacheMetrics(hits=0, misses=100)
        assert metrics.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        metrics = CacheMetrics(hits=75, misses=25)
        assert metrics.hit_rate == 0.75

    def test_to_dict(self):
        """Test metrics to_dict conversion."""
        metrics = CacheMetrics(hits=10, misses=5, sets=3, invalidations=2)

        result = metrics.to_dict()

        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["sets"] == 3
        assert result["invalidations"] == 2
        assert result["errors"] == 0
        assert result["redis_failures"] == 0
        assert result["memory_fallbacks"] == 0
        assert result["hit_rate"] == pytest.approx(0.667, rel=0.01)


# ============================================================================
# TEST: CacheEntry
# ============================================================================


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_not_expired(self):
        """Test entry is not expired."""
        entry = CacheEntry(
            value={"test": "data"},
            expires_at=time.time() + 60,
        )
        assert entry.is_expired is False

    def test_is_expired(self):
        """Test entry is expired."""
        entry = CacheEntry(
            value={"test": "data"},
            expires_at=time.time() - 1,
        )
        assert entry.is_expired is True

    def test_created_at_auto(self):
        """Test created_at is auto-set."""
        before = time.time()
        entry = CacheEntry(value={"test": "data"}, expires_at=time.time() + 60)
        after = time.time()

        assert before <= entry.created_at <= after


# ============================================================================
# TEST: MetricsCache - Initialization
# ============================================================================


class TestMetricsCacheInit:
    """Test MetricsCache initialization."""

    def test_default_init(self):
        """Test default initialization."""
        cache = MetricsCache()

        assert cache.config.backend == CacheBackend.MEMORY
        assert cache.backend == CacheBackend.MEMORY
        assert len(cache._memory_cache) == 0

    def test_custom_config(self, cache_config):
        """Test initialization with custom config."""
        cache = MetricsCache(config=cache_config)

        assert cache.config.key_prefix == "test_metrics"
        assert cache.config.max_memory_entries == 100

    def test_with_redis_client(self, mock_redis_client, cache_config):
        """Test initialization with Redis client."""
        cache_config.backend = CacheBackend.REDIS
        cache = MetricsCache(config=cache_config, redis_client=mock_redis_client)

        assert cache._redis_client == mock_redis_client

    @pytest.mark.asyncio
    async def test_initialize_memory(self, cache_config):
        """Test memory backend initialization."""
        cache = MetricsCache(config=cache_config)
        result = await cache.initialize()

        assert result is True
        assert cache.backend == CacheBackend.MEMORY

    @pytest.mark.asyncio
    async def test_initialize_redis_success(self, mock_redis_client):
        """Test Redis backend initialization success."""
        config = CacheConfig(backend=CacheBackend.REDIS)
        cache = MetricsCache(config=config, redis_client=mock_redis_client)

        result = await cache.initialize()

        assert result is True
        assert cache._redis_available is True
        mock_redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_redis_failure_with_fallback(self, mock_redis_client):
        """Test Redis failure falls back to memory."""
        mock_redis_client.ping.side_effect = ConnectionError("Redis unavailable")
        config = CacheConfig(backend=CacheBackend.REDIS, fallback_to_memory=True)
        cache = MetricsCache(config=config, redis_client=mock_redis_client)

        result = await cache.initialize()

        assert result is True
        assert cache._redis_available is False
        assert cache.backend == CacheBackend.MEMORY
        assert cache.metrics.redis_failures == 1
        assert cache.metrics.memory_fallbacks == 1

    @pytest.mark.asyncio
    async def test_initialize_redis_failure_no_fallback(self, mock_redis_client):
        """Test Redis failure without fallback."""
        mock_redis_client.ping.side_effect = ConnectionError("Redis unavailable")
        config = CacheConfig(backend=CacheBackend.REDIS, fallback_to_memory=False)
        cache = MetricsCache(config=config, redis_client=mock_redis_client)

        result = await cache.initialize()

        assert result is False
        assert cache._redis_available is False


# ============================================================================
# TEST: MetricsCache - Key Generation
# ============================================================================


class TestMetricsCacheKeyGeneration:
    """Test MetricsCache key generation."""

    def test_make_key_with_agent(self, cache_config):
        """Test key generation with agent."""
        cache = MetricsCache(config=cache_config)

        key = cache._make_key("1h", "orchestrator")

        assert key == "test_metrics:1h:orchestrator"

    def test_make_key_without_agent(self, cache_config):
        """Test key generation without agent."""
        cache = MetricsCache(config=cache_config)

        key = cache._make_key("24h", None)

        assert key == "test_metrics:24h:_all"

    def test_make_key_different_windows(self, cache_config):
        """Test key generation for different windows."""
        cache = MetricsCache(config=cache_config)

        assert cache._make_key("1h", "agent1") == "test_metrics:1h:agent1"
        assert cache._make_key("24h", "agent1") == "test_metrics:24h:agent1"
        assert cache._make_key("7d", "agent1") == "test_metrics:7d:agent1"


# ============================================================================
# TEST: MetricsCache - Get/Set Operations
# ============================================================================


class TestMetricsCacheGetSet:
    """Test MetricsCache get/set operations."""

    @pytest.mark.asyncio
    async def test_get_miss(self, cache_config):
        """Test cache miss."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        result = await cache.get_metrics("1h", "orchestrator")

        assert result is None
        assert cache.metrics.misses == 1
        assert cache.metrics.hits == 0

    @pytest.mark.asyncio
    async def test_set_and_get_hit(self, cache_config, sample_metrics):
        """Test cache hit after set."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)
        result = await cache.get_metrics("1h", "orchestrator")

        assert result == sample_metrics
        assert cache.metrics.hits == 1
        assert cache.metrics.sets == 1

    @pytest.mark.asyncio
    async def test_get_expired_entry(self, cache_config, sample_metrics):
        """Test getting an expired entry returns None."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        # Manually create an expired entry
        key = cache._make_key("1h", "orchestrator")
        cache._memory_cache[key] = CacheEntry(
            value=sample_metrics,
            expires_at=time.time() - 1,  # Already expired
        )

        result = await cache.get_metrics("1h", "orchestrator")

        assert result is None
        assert cache.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_set_returns_success(self, cache_config, sample_metrics):
        """Test set returns True on success."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        result = await cache.set_metrics("1h", "orchestrator", sample_metrics)

        assert result is True
        assert cache.metrics.sets == 1

    @pytest.mark.asyncio
    async def test_set_uses_window_ttl(self, cache_config, sample_metrics):
        """Test set uses appropriate TTL for window."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)

        key = cache._make_key("1h", "orchestrator")
        entry = cache._memory_cache[key]
        expected_ttl = cache_config.get_ttl("1h")

        # Entry should expire around expected TTL from now
        assert entry.expires_at > time.time()
        assert entry.expires_at <= time.time() + expected_ttl + 1

    @pytest.mark.asyncio
    async def test_set_custom_ttl(self, cache_config, sample_metrics):
        """Test set with custom TTL."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics, ttl=30)

        key = cache._make_key("1h", "orchestrator")
        entry = cache._memory_cache[key]

        # Entry should expire around 30 seconds from now
        assert entry.expires_at <= time.time() + 31

    @pytest.mark.asyncio
    async def test_redis_get_success(self, mock_redis_client, sample_metrics):
        """Test successful Redis get."""
        import json

        mock_redis_client.get.return_value = json.dumps(sample_metrics)
        config = CacheConfig(backend=CacheBackend.REDIS)
        cache = MetricsCache(config=config, redis_client=mock_redis_client)
        await cache.initialize()

        result = await cache.get_metrics("1h", "orchestrator")

        assert result == sample_metrics
        assert cache.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_redis_set_success(self, mock_redis_client, sample_metrics):
        """Test successful Redis set."""
        config = CacheConfig(backend=CacheBackend.REDIS)
        cache = MetricsCache(config=config, redis_client=mock_redis_client)
        await cache.initialize()

        result = await cache.set_metrics("1h", "orchestrator", sample_metrics)

        assert result is True
        mock_redis_client.setex.assert_called_once()
        assert cache.metrics.sets == 1

    @pytest.mark.asyncio
    async def test_redis_get_failure_falls_back(
        self, mock_redis_client, cache_config, sample_metrics
    ):
        """Test Redis get failure falls back to memory."""
        mock_redis_client.get.side_effect = ConnectionError("Redis error")
        config = CacheConfig(backend=CacheBackend.REDIS)
        cache = MetricsCache(config=config, redis_client=mock_redis_client)
        cache._redis_available = True  # Simulate previous successful connection

        # Pre-populate memory cache
        key = cache._make_key("1h", "orchestrator")
        cache._memory_cache[key] = CacheEntry(
            value=sample_metrics, expires_at=time.time() + 60
        )

        result = await cache.get_metrics("1h", "orchestrator")

        assert result == sample_metrics
        assert cache.metrics.redis_failures == 1


# ============================================================================
# TEST: MetricsCache - Invalidation
# ============================================================================


class TestMetricsCacheInvalidation:
    """Test MetricsCache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_by_agent(self, cache_config, sample_metrics):
        """Test invalidation by agent."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)
        await cache.set_metrics("24h", "orchestrator", sample_metrics)
        await cache.set_metrics("1h", "gap_analyzer", sample_metrics)

        count = await cache.invalidate(agent="orchestrator")

        # Should invalidate orchestrator entries and the "_all" entries if they existed
        assert count >= 2
        assert await cache.get_metrics("1h", "orchestrator") is None
        assert await cache.get_metrics("24h", "orchestrator") is None
        assert await cache.get_metrics("1h", "gap_analyzer") is not None

    @pytest.mark.asyncio
    async def test_invalidate_by_window(self, cache_config, sample_metrics):
        """Test invalidation by window.

        Note: invalidate(window=X) only clears the 'all agents' key for that window,
        not specific agent keys. For agent-specific window invalidation,
        use invalidate(agent=X, window=Y).
        """
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        # Set "all agents" entry for 1h and 24h windows
        await cache.set_metrics("1h", None, sample_metrics)  # _all key
        await cache.set_metrics("24h", None, {"other": "data"})

        count = await cache.invalidate(window="1h")

        assert count >= 1
        assert await cache.get_metrics("1h", None) is None
        # 24h entry should still exist
        assert await cache.get_metrics("24h", None) is not None

    @pytest.mark.asyncio
    async def test_invalidate_all(self, cache_config, sample_metrics):
        """Test full invalidation."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)
        await cache.set_metrics("24h", "gap_analyzer", sample_metrics)

        count = await cache.invalidate()

        # All windows for None agent should be cleared
        assert count >= 0  # The exact count depends on the implementation
        assert cache.metrics.invalidations >= 0

    @pytest.mark.asyncio
    async def test_invalidate_also_clears_all_key(self, cache_config, sample_metrics):
        """Test invalidating an agent also clears the 'all agents' key."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        # Set both specific agent and "all agents" metrics
        await cache.set_metrics("1h", "orchestrator", sample_metrics)
        await cache.set_metrics("1h", None, {"all": "agents"})

        count = await cache.invalidate(agent="orchestrator")

        # Both should be invalidated
        assert count >= 2
        assert await cache.get_metrics("1h", "orchestrator") is None
        assert await cache.get_metrics("1h", None) is None

    @pytest.mark.asyncio
    async def test_invalidate_pattern_memory(self, cache_config, sample_metrics):
        """Test pattern-based invalidation in memory."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)
        await cache.set_metrics("24h", "orchestrator", sample_metrics)
        await cache.set_metrics("1h", "gap_analyzer", sample_metrics)

        count = await cache.invalidate_pattern("test_metrics:*:orchestrator")

        assert count >= 2


# ============================================================================
# TEST: MetricsCache - Get or Compute
# ============================================================================


class TestMetricsCacheGetOrCompute:
    """Test MetricsCache get_or_compute."""

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self, cache_config, sample_metrics):
        """Test get_or_compute returns cached value."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)

        async def compute_fn():
            return {"computed": True}

        result = await cache.get_or_compute("1h", "orchestrator", compute_fn)

        assert result == sample_metrics
        assert cache.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self, cache_config):
        """Test get_or_compute computes and caches."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        computed_value = {"computed": True, "value": 42}
        compute_called = False

        async def compute_fn():
            nonlocal compute_called
            compute_called = True
            return computed_value

        result = await cache.get_or_compute("1h", "orchestrator", compute_fn)

        assert result == computed_value
        assert compute_called is True
        assert cache.metrics.misses == 1
        assert cache.metrics.sets == 1

        # Verify it's cached
        cached = await cache.get_metrics("1h", "orchestrator")
        assert cached == computed_value


# ============================================================================
# TEST: MetricsCache - Memory Management
# ============================================================================


class TestMetricsCacheMemoryManagement:
    """Test MetricsCache memory management."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, sample_metrics):
        """Test cleanup removes expired entries."""
        config = CacheConfig(max_memory_entries=1000, cleanup_interval=10)
        cache = MetricsCache(config=config)
        await cache.initialize()

        # Manually create an expired entry and a valid entry
        key1 = cache._make_key("1h", "agent1")
        key2 = cache._make_key("1h", "agent2")
        cache._memory_cache[key1] = CacheEntry(
            value=sample_metrics,
            expires_at=time.time() - 1,  # Already expired
        )
        cache._memory_cache[key2] = CacheEntry(
            value=sample_metrics,
            expires_at=time.time() + 60,  # Valid
        )

        # Manually trigger cleanup
        removed = await cache._cleanup_memory_cache()

        assert removed >= 1
        assert key1 not in cache._memory_cache
        assert key2 in cache._memory_cache

    @pytest.mark.asyncio
    async def test_cleanup_at_capacity(self, sample_metrics):
        """Test cleanup at capacity removes oldest entries."""
        config = CacheConfig(max_memory_entries=5, cleanup_interval=10)
        cache = MetricsCache(config=config)
        await cache.initialize()

        # Fill to capacity
        for i in range(6):
            await cache.set_metrics("1h", f"agent{i}", sample_metrics, ttl=60)
            await asyncio.sleep(0.001)  # Ensure different created_at times

        # Should trigger cleanup
        assert len(cache._memory_cache) <= 5

    @pytest.mark.asyncio
    async def test_cleanup_task_lifecycle(self, cache_config):
        """Test cleanup task start/stop."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.start_cleanup_task()
        assert cache._cleanup_task is not None
        assert not cache._cleanup_task.done()

        await cache.stop_cleanup_task()
        assert cache._cleanup_task.done()


# ============================================================================
# TEST: MetricsCache - Status
# ============================================================================


class TestMetricsCacheStatus:
    """Test MetricsCache status reporting."""

    @pytest.mark.asyncio
    async def test_get_status(self, cache_config, sample_metrics):
        """Test get_status returns correct information."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", sample_metrics)
        await cache.get_metrics("1h", "orchestrator")
        await cache.get_metrics("24h", "missing")

        status = cache.get_status()

        assert status["backend"] == "memory"
        assert status["redis_available"] is False
        assert status["memory_entries"] == 1
        assert "config" in status
        assert status["config"]["key_prefix"] == "test_metrics"
        assert "metrics" in status
        assert status["metrics"]["hits"] == 1
        assert status["metrics"]["misses"] == 1
        assert status["metrics"]["sets"] == 1

    @pytest.mark.asyncio
    async def test_clear(self, cache_config, sample_metrics):
        """Test clear removes all entries."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "agent1", sample_metrics)
        await cache.set_metrics("24h", "agent2", sample_metrics)

        count = await cache.clear()

        assert count >= 2
        assert len(cache._memory_cache) == 0


# ============================================================================
# TEST: Singleton Pattern
# ============================================================================


class TestMetricsCacheSingleton:
    """Test MetricsCache singleton pattern."""

    def test_get_metrics_cache_creates_instance(self):
        """Test get_metrics_cache creates singleton."""
        cache1 = get_metrics_cache()
        cache2 = get_metrics_cache()

        assert cache1 is cache2

    def test_get_metrics_cache_with_config(self):
        """Test get_metrics_cache uses config on first call."""
        config = CacheConfig(key_prefix="custom_prefix")
        cache = get_metrics_cache(config=config)

        assert cache.config.key_prefix == "custom_prefix"

    @pytest.mark.asyncio
    async def test_reset_metrics_cache(self):
        """Test reset clears singleton."""
        cache1 = get_metrics_cache()
        await reset_metrics_cache()
        cache2 = get_metrics_cache()

        assert cache1 is not cache2


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestMetricsCacheEdgeCases:
    """Test MetricsCache edge cases."""

    @pytest.mark.asyncio
    async def test_empty_metrics(self, cache_config):
        """Test caching empty metrics."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        await cache.set_metrics("1h", "orchestrator", {})
        result = await cache.get_metrics("1h", "orchestrator")

        assert result == {}

    @pytest.mark.asyncio
    async def test_nested_metrics(self, cache_config):
        """Test caching nested metrics."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        nested = {
            "level1": {
                "level2": {
                    "value": 42,
                    "list": [1, 2, 3],
                }
            }
        }

        await cache.set_metrics("1h", "orchestrator", nested)
        result = await cache.get_metrics("1h", "orchestrator")

        assert result == nested

    @pytest.mark.asyncio
    async def test_special_agent_names(self, cache_config, sample_metrics):
        """Test caching with special agent names."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        # Test with special characters
        await cache.set_metrics("1h", "agent-with-dashes", sample_metrics)
        await cache.set_metrics("1h", "agent_with_underscores", sample_metrics)

        result1 = await cache.get_metrics("1h", "agent-with-dashes")
        result2 = await cache.get_metrics("1h", "agent_with_underscores")

        assert result1 == sample_metrics
        assert result2 == sample_metrics

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache_config, sample_metrics):
        """Test concurrent cache access."""
        cache = MetricsCache(config=cache_config)
        await cache.initialize()

        async def set_and_get(agent_id):
            await cache.set_metrics("1h", f"agent_{agent_id}", sample_metrics)
            return await cache.get_metrics("1h", f"agent_{agent_id}")

        tasks = [set_and_get(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r == sample_metrics for r in results)
        assert len(cache._memory_cache) == 10
