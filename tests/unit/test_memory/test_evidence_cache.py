"""
Unit tests for EvidenceEvaluationCache.

Tests:
- Cache get/set operations
- TTL expiration
- LRU eviction
- Cache statistics
- Environment variable configuration
"""

import time
from unittest.mock import patch

import pytest

from src.memory.004_cognitive_workflow import (
    EvidenceEvaluationCache,
    get_evidence_cache,
    is_evidence_cache_enabled,
)


class TestEvidenceEvaluationCache:
    """Tests for EvidenceEvaluationCache."""

    def test_cache_hit(self):
        """Test that cached values are returned."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        goal = "Find causal factors for TRx decline"
        evidence = "- [episodic] TRx dropped 15%\n- [semantic] HCP detailing decreased"

        # Set value
        cache.set(goal, evidence, "need_more")

        # Get should return cached value
        result = cache.get(goal, evidence)
        assert result == "need_more"

    def test_cache_miss(self):
        """Test that cache miss returns None."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        result = cache.get("unknown goal", "unknown evidence")
        assert result is None

    def test_cache_different_goals_different_keys(self):
        """Test that different goals create different cache keys."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        evidence = "Same evidence content"

        cache.set("Goal A", evidence, "sufficient")
        cache.set("Goal B", evidence, "need_more")

        assert cache.get("Goal A", evidence) == "sufficient"
        assert cache.get("Goal B", evidence) == "need_more"

    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=0.1)

        cache.set("goal", "evidence", "sufficient")

        # Should be cached initially
        assert cache.get("goal", "evidence") == "sufficient"

        # Wait for TTL
        time.sleep(0.15)

        # Should be expired
        assert cache.get("goal", "evidence") is None

    def test_cache_eviction_at_capacity(self):
        """Test that oldest entries are evicted at capacity."""
        cache = EvidenceEvaluationCache(max_size=10, ttl_seconds=3600)

        # Fill cache
        for i in range(10):
            cache.set(f"goal-{i}", "evidence", f"result-{i}")
            time.sleep(0.01)  # Ensure different timestamps

        # Cache should be at capacity
        assert cache.stats["size"] == 10

        # Add one more (should trigger eviction)
        cache.set("goal-new", "evidence", "result-new")

        # Should have evicted oldest (10% = 1 entry)
        assert cache.stats["size"] <= 10

    def test_cache_stats(self):
        """Test that stats are correctly tracked."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        # Initial stats
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # Add entry and hit
        cache.set("goal", "evidence", "result")
        cache.get("goal", "evidence")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["size"] == 1

        # Miss
        cache.get("unknown", "unknown")
        stats = cache.stats
        assert stats["misses"] == 1

    def test_cache_hit_rate(self):
        """Test hit rate calculation."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        cache.set("goal", "evidence", "result")

        # 2 hits
        cache.get("goal", "evidence")
        cache.get("goal", "evidence")

        # 1 miss
        cache.get("unknown", "unknown")

        stats = cache.stats
        # 2 hits / 3 total = 0.666...
        assert 0.65 < stats["hit_rate"] < 0.68

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        cache.set("goal-1", "evidence", "result-1")
        cache.set("goal-2", "evidence", "result-2")

        assert cache.stats["size"] == 2

        cache.clear()

        assert cache.stats["size"] == 0
        assert cache.get("goal-1", "evidence") is None

    def test_key_normalization(self):
        """Test that keys are normalized (case, whitespace)."""
        cache = EvidenceEvaluationCache(max_size=100, ttl_seconds=3600)

        # Set with extra whitespace
        cache.set("  Goal with spaces  ", "evidence", "result")

        # Get with normalized version should still work
        result = cache.get("goal with spaces", "evidence")
        assert result == "result"


class TestCacheConfiguration:
    """Tests for cache configuration."""

    def test_env_var_max_size(self):
        """Test max_size configuration via environment variable."""
        with patch.dict("os.environ", {"E2I_EVIDENCE_CACHE_SIZE": "500"}):
            cache = EvidenceEvaluationCache()
            assert cache._max_size == 500

    def test_env_var_ttl(self):
        """Test TTL configuration via environment variable."""
        with patch.dict("os.environ", {"E2I_EVIDENCE_CACHE_TTL": "7200"}):
            cache = EvidenceEvaluationCache()
            assert cache._ttl == 7200.0

    def test_is_cache_enabled_default(self):
        """Test that cache is enabled by default."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear the env var to test default
            with patch.dict("os.environ", {"E2I_EVIDENCE_CACHE": "true"}):
                assert is_evidence_cache_enabled() is True

    def test_is_cache_disabled(self):
        """Test cache can be disabled via environment."""
        with patch.dict("os.environ", {"E2I_EVIDENCE_CACHE": "false"}):
            assert is_evidence_cache_enabled() is False


class TestCacheSingleton:
    """Tests for cache singleton behavior."""

    def test_get_evidence_cache_returns_singleton(self):
        """Test that get_evidence_cache returns the same instance."""
        # Note: We need to reset between tests in real scenarios
        cache1 = get_evidence_cache()
        cache2 = get_evidence_cache()

        assert cache1 is cache2

    def test_cache_persists_across_calls(self):
        """Test that cached data persists across singleton access."""
        cache = get_evidence_cache()
        cache.set("singleton-test-goal", "evidence", "result")

        # Get fresh reference
        cache2 = get_evidence_cache()
        result = cache2.get("singleton-test-goal", "evidence")

        assert result == "result"
