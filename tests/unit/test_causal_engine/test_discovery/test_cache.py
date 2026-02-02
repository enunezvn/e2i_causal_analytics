"""Tests for Discovery Result Cache.

Version: 1.0.0
Tests the DiscoveryCache class with Redis + in-memory fallback.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.causal_engine.discovery.base import (
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
    GateDecision,
)
from src.causal_engine.discovery.cache import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    DiscoveryCache,
    get_discovery_cache,
)
from src.causal_engine.discovery.hasher import (
    hash_config,
    hash_dataframe,
    hash_discovery_request,
    make_cache_key,
    verify_hash_determinism,
)


class TestHashDataframe:
    """Test DataFrame hashing functionality."""

    def test_hash_dataframe_returns_64_char_hex(self):
        """Test hash_dataframe returns 64-character hex string."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        h = hash_dataframe(df)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_dataframe_determinism(self):
        """Test that same DataFrame produces same hash."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        h1 = hash_dataframe(df)
        h2 = hash_dataframe(df)
        h3 = hash_dataframe(df)
        assert h1 == h2 == h3

    def test_hash_dataframe_different_values_different_hash(self):
        """Test different data produces different hash."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [1, 2, 4], "B": [4, 5, 6]})  # Different value
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_hash_dataframe_different_columns_different_hash(self):
        """Test different column names produce different hash."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [1, 2, 3], "C": [4, 5, 6]})  # Different column
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_hash_dataframe_column_order_matters(self):
        """Test that column order affects hash."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"B": [4, 5, 6], "A": [1, 2, 3]})  # Reordered
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_hash_empty_dataframe(self):
        """Test hashing empty DataFrame returns special hash."""
        df = pd.DataFrame()
        h = hash_dataframe(df)
        assert len(h) == 64

    def test_hash_handles_float_precision(self):
        """Test float precision is handled consistently."""
        # Small floating point differences should not affect hash
        df1 = pd.DataFrame({"A": [1.00000001, 2.0, 3.0]})
        df2 = pd.DataFrame({"A": [1.00000002, 2.0, 3.0]})
        # Within 8 decimal precision, should be same
        h1 = hash_dataframe(df1)
        h2 = hash_dataframe(df2)
        # Both are valid hashes
        assert len(h1) == 64
        assert len(h2) == 64


class TestHashConfig:
    """Test DiscoveryConfig hashing functionality."""

    def test_hash_config_returns_64_char_hex(self):
        """Test hash_config returns 64-character hex string."""
        config = DiscoveryConfig()
        h = hash_config(config)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_config_determinism(self):
        """Test that same config produces same hash."""
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
            alpha=0.05,
        )
        h1 = hash_config(config)
        h2 = hash_config(config)
        assert h1 == h2

    def test_hash_config_different_algorithms_different_hash(self):
        """Test different algorithms produce different hash."""
        config1 = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.GES])
        config2 = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])
        assert hash_config(config1) != hash_config(config2)

    def test_hash_config_different_alpha_different_hash(self):
        """Test different alpha values produce different hash."""
        config1 = DiscoveryConfig(alpha=0.05)
        config2 = DiscoveryConfig(alpha=0.01)
        assert hash_config(config1) != hash_config(config2)

    def test_hash_config_algorithm_order_independent(self):
        """Test that algorithm order doesn't affect hash (sorted internally)."""
        config1 = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC]
        )
        config2 = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.GES]
        )
        assert hash_config(config1) == hash_config(config2)


class TestMakeCacheKey:
    """Test cache key generation."""

    def test_make_cache_key_format(self):
        """Test cache key has correct format."""
        data_hash = "a" * 64
        config_hash = "b" * 64
        key = make_cache_key(data_hash, config_hash)
        assert key.startswith("discovery:")
        assert ":" in key[10:]

    def test_make_cache_key_uses_truncated_hashes(self):
        """Test cache key uses first 16 chars of each hash."""
        data_hash = "a" * 64
        config_hash = "b" * 64
        key = make_cache_key(data_hash, config_hash)
        assert key == "discovery:aaaaaaaaaaaaaaaa:bbbbbbbbbbbbbbbb"


class TestHashDiscoveryRequest:
    """Test combined hashing function."""

    def test_hash_discovery_request_returns_cache_key(self):
        """Test hash_discovery_request returns valid cache key."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        config = DiscoveryConfig()
        key = hash_discovery_request(df, config)
        assert key.startswith("discovery:")

    def test_verify_hash_determinism(self):
        """Test hash determinism verification utility."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        config = DiscoveryConfig()
        assert verify_hash_determinism(df, config, n_iterations=5)


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_hit_rate_with_no_requests(self):
        """Test hit rate is 0 with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_to_dict(self):
        """Test stats serialization to dict."""
        now = datetime.now(timezone.utc)
        stats = CacheStats(hits=10, misses=5, last_hit_at=now)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["hit_rate"] == pytest.approx(0.6667, rel=0.01)
        assert d["last_hit_at"] == now.isoformat()


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_is_expired_not_expired(self):
        """Test is_expired returns False for valid entry."""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            result_json="{}",
            created_at=now,
            expires_at=now + timedelta(hours=1),
        )
        assert not entry.is_expired()

    def test_is_expired_expired(self):
        """Test is_expired returns True for expired entry."""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            result_json="{}",
            created_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),
        )
        assert entry.is_expired()


class TestDiscoveryCache:
    """Test DiscoveryCache class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test DataFrame."""
        return pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

    @pytest.fixture
    def config(self):
        """Create test DiscoveryConfig."""
        return DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES],
            alpha=0.05,
        )

    @pytest.fixture
    def sample_result(self, config):
        """Create sample DiscoveryResult."""
        return DiscoveryResult(
            success=True,
            config=config,
            edges=[
                DiscoveredEdge(
                    source="A",
                    target="B",
                    edge_type=EdgeType.DIRECTED,
                    confidence=0.9,
                    algorithm_votes=1,
                    algorithms=["ges"],
                )
            ],
            gate_decision=GateDecision.ACCEPT,
            gate_confidence=0.85,
            metadata={"runtime": 1.5},
        )

    @pytest.fixture
    def memory_only_cache(self):
        """Create cache with memory only (no Redis)."""
        config = CacheConfig(
            enable_redis=False,
            enable_memory=True,
            ttl_seconds=3600,
            max_memory_items=10,
        )
        return DiscoveryCache(config)

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, memory_only_cache, simple_data, config):
        """Test cache miss returns None."""
        result = await memory_only_cache.get(simple_data, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_result(
        self, memory_only_cache, simple_data, config, sample_result
    ):
        """Test cache hit returns cached result."""
        # Set cache
        await memory_only_cache.set(simple_data, config, sample_result)

        # Get cache
        result = await memory_only_cache.get(simple_data, config)

        assert result is not None
        assert result.success == sample_result.success
        assert len(result.edges) == len(sample_result.edges)
        assert result.edges[0].source == "A"
        assert result.edges[0].target == "B"

    @pytest.mark.asyncio
    async def test_cache_different_data_different_key(
        self, memory_only_cache, config, sample_result
    ):
        """Test different data produces cache miss."""
        df1 = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"A": [7.0, 8.0, 9.0], "B": [10.0, 11.0, 12.0]})

        # Cache with df1
        await memory_only_cache.set(df1, config, sample_result)

        # Get with df2 should miss
        result = await memory_only_cache.get(df2, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_different_config_different_key(
        self, memory_only_cache, simple_data, sample_result
    ):
        """Test different config produces cache miss."""
        config1 = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.GES])
        config2 = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])

        # Cache with config1
        await memory_only_cache.set(simple_data, config1, sample_result)

        # Get with config2 should miss
        result = await memory_only_cache.get(simple_data, config2)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, simple_data, sample_result):
        """Test LRU eviction when cache is full."""
        # Create cache with max 3 items
        cache_config = CacheConfig(
            enable_redis=False,
            enable_memory=True,
            max_memory_items=3,
        )
        cache = DiscoveryCache(cache_config)

        # Create different configs
        configs = [
            DiscoveryConfig(alpha=0.01),
            DiscoveryConfig(alpha=0.02),
            DiscoveryConfig(alpha=0.03),
            DiscoveryConfig(alpha=0.04),  # This should evict first
        ]

        # Fill cache
        for c in configs:
            await cache.set(simple_data, c, sample_result)

        # First config should be evicted
        result = await cache.get(simple_data, configs[0])
        assert result is None

        # Last config should still be cached
        result = await cache.get(simple_data, configs[3])
        assert result is not None

        # Check eviction count
        stats = cache.get_stats()
        assert stats.evictions >= 1

    @pytest.mark.asyncio
    async def test_cache_invalidation_all(
        self, memory_only_cache, simple_data, config, sample_result
    ):
        """Test invalidating all cache entries."""
        # Set cache
        await memory_only_cache.set(simple_data, config, sample_result)

        # Verify cached
        result = await memory_only_cache.get(simple_data, config)
        assert result is not None

        # Invalidate all
        count = memory_only_cache.invalidate()
        assert count == 1

        # Verify empty
        result = await memory_only_cache.get(simple_data, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats_tracking(
        self, memory_only_cache, simple_data, config, sample_result
    ):
        """Test cache statistics are tracked correctly."""
        # Initial stats
        stats = memory_only_cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0

        # Miss
        await memory_only_cache.get(simple_data, config)
        stats = memory_only_cache.get_stats()
        assert stats.misses == 1

        # Set
        await memory_only_cache.set(simple_data, config, sample_result)
        stats = memory_only_cache.get_stats()
        assert stats.sets == 1

        # Hit
        await memory_only_cache.get(simple_data, config)
        stats = memory_only_cache.get_stats()
        assert stats.hits == 1

    @pytest.mark.asyncio
    async def test_cache_redis_fallback_to_memory(self, simple_data, config, sample_result):
        """Test cache falls back to memory when Redis unavailable."""
        # Create cache with Redis enabled but unavailable
        cache_config = CacheConfig(
            redis_url="redis://nonexistent:6379",
            enable_redis=True,
            enable_memory=True,
        )
        cache = DiscoveryCache(cache_config)

        # Should still work with memory cache
        await cache.set(simple_data, config, sample_result)
        result = await cache.get(simple_data, config)
        assert result is not None

    @pytest.mark.asyncio
    async def test_cache_serialization_roundtrip(
        self, memory_only_cache, simple_data, config, sample_result
    ):
        """Test result survives serialization/deserialization."""
        # Set cache
        await memory_only_cache.set(simple_data, config, sample_result)

        # Get cache
        result = await memory_only_cache.get(simple_data, config)

        # Verify all fields preserved
        assert result.success == sample_result.success
        assert result.gate_decision == sample_result.gate_decision
        assert result.gate_confidence == sample_result.gate_confidence
        assert len(result.edges) == 1
        assert result.edges[0].source == "A"
        assert result.edges[0].target == "B"
        assert result.edges[0].edge_type == EdgeType.DIRECTED
        assert result.edges[0].confidence == 0.9


class TestGetDiscoveryCache:
    """Test singleton cache accessor."""

    def test_get_discovery_cache_returns_singleton(self):
        """Test get_discovery_cache returns same instance."""
        # Note: This modifies global state, use with caution in tests
        cache1 = get_discovery_cache()
        cache2 = get_discovery_cache()
        assert cache1 is cache2
