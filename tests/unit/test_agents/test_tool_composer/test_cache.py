"""
Tests for Tool Composer Performance Cache.

Tests the caching layer for composition performance optimization (G6):
- LRU cache with TTL
- Decomposition result caching
- Plan similarity matching
- Deterministic tool output caching
- Unified cache manager
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import pytest

from src.agents.tool_composer.cache import (
    CacheEntry,
    DecompositionCache,
    LRUCache,
    PlanSimilarityCache,
    ToolComposerCacheManager,
    ToolOutputCache,
    get_cache_manager,
)

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def lru_cache():
    """Create a fresh LRU cache for testing."""
    return LRUCache(max_size=5, default_ttl_seconds=60.0)


@pytest.fixture
def decomposition_cache():
    """Create a fresh decomposition cache for testing."""
    return DecompositionCache(max_size=10, ttl_seconds=300.0)


@pytest.fixture
def plan_similarity_cache():
    """Create a fresh plan similarity cache for testing."""
    return PlanSimilarityCache(max_size=10, ttl_seconds=600.0, similarity_threshold=0.8)


@pytest.fixture
def tool_output_cache():
    """Create a fresh tool output cache for testing."""
    return ToolOutputCache(max_size=20, ttl_seconds=120.0)


@pytest.fixture
def cache_manager():
    """Create a fresh cache manager for testing (resets singleton)."""
    # Reset singleton for isolated testing
    ToolComposerCacheManager._instance = None
    manager = ToolComposerCacheManager(
        decomposition_max_size=10,
        decomposition_ttl=300.0,
        plan_max_size=5,
        plan_ttl=600.0,
        plan_similarity_threshold=0.7,
        output_max_size=20,
        output_ttl=120.0,
    )
    yield manager
    # Cleanup singleton after test
    ToolComposerCacheManager._instance = None


# ============================================================================
# CACHE ENTRY TESTS
# ============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(key="test_key", value={"data": 123})
        assert entry.key == "test_key"
        assert entry.value == {"data": 123}
        assert entry.access_count == 0
        assert entry.ttl_seconds == 300.0  # default

    def test_cache_entry_custom_ttl(self):
        """Test cache entry with custom TTL."""
        entry = CacheEntry(key="test", value="value", ttl_seconds=60.0)
        assert entry.ttl_seconds == 60.0

    def test_is_expired_false(self):
        """Test entry is not expired when within TTL."""
        entry = CacheEntry(key="test", value="value", ttl_seconds=60.0)
        assert entry.is_expired() is False

    def test_is_expired_true(self):
        """Test entry is expired when past TTL."""
        entry = CacheEntry(key="test", value="value", ttl_seconds=0.01)
        time.sleep(0.02)
        assert entry.is_expired() is True

    def test_access_increments_count(self):
        """Test access() increments access count."""
        entry = CacheEntry(key="test", value="value")
        assert entry.access_count == 0
        entry.access()
        assert entry.access_count == 1
        entry.access()
        assert entry.access_count == 2

    def test_access_updates_last_accessed(self):
        """Test access() updates last_accessed time."""
        entry = CacheEntry(key="test", value="value")
        initial_accessed = entry.last_accessed
        time.sleep(0.01)
        entry.access()
        assert entry.last_accessed > initial_accessed

    def test_access_returns_value(self):
        """Test access() returns the value."""
        entry = CacheEntry(key="test", value={"result": 42})
        result = entry.access()
        assert result == {"result": 42}


# ============================================================================
# LRU CACHE TESTS
# ============================================================================


class TestLRUCache:
    """Tests for LRU cache with TTL support."""

    def test_set_and_get(self, lru_cache):
        """Test basic set and get operations."""
        lru_cache.set("key1", "value1")
        assert lru_cache.get("key1") == "value1"

    def test_get_missing_key_returns_none(self, lru_cache):
        """Test get on missing key returns None."""
        assert lru_cache.get("nonexistent") is None

    def test_get_increments_hits(self, lru_cache):
        """Test successful get increments hits."""
        lru_cache.set("key1", "value1")
        lru_cache.get("key1")
        assert lru_cache._hits == 1
        assert lru_cache._misses == 0

    def test_get_missing_increments_misses(self, lru_cache):
        """Test missing get increments misses."""
        lru_cache.get("nonexistent")
        assert lru_cache._hits == 0
        assert lru_cache._misses == 1

    def test_expired_entry_returns_none(self, lru_cache):
        """Test expired entry returns None and is removed."""
        lru_cache.set("key1", "value1", ttl_seconds=0.01)
        time.sleep(0.02)
        assert lru_cache.get("key1") is None
        assert lru_cache.size == 0

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = LRUCache(max_size=3, default_ttl_seconds=60.0)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 and key3 to make key2 LRU
        cache.get("key1")
        cache.get("key3")

        # Add new entry, should evict key2
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing_key(self, lru_cache):
        """Test updating existing key doesn't trigger eviction."""
        lru_cache.set("key1", "value1")
        lru_cache.set("key1", "updated_value")
        assert lru_cache.get("key1") == "updated_value"
        assert lru_cache.size == 1

    def test_invalidate_existing_key(self, lru_cache):
        """Test invalidate removes existing key."""
        lru_cache.set("key1", "value1")
        result = lru_cache.invalidate("key1")
        assert result is True
        assert lru_cache.get("key1") is None

    def test_invalidate_missing_key(self, lru_cache):
        """Test invalidate returns False for missing key."""
        result = lru_cache.invalidate("nonexistent")
        assert result is False

    def test_clear(self, lru_cache):
        """Test clear removes all entries."""
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")
        lru_cache.clear()
        assert lru_cache.size == 0

    def test_cleanup_expired(self):
        """Test cleanup_expired removes expired entries."""
        cache = LRUCache(max_size=10, default_ttl_seconds=60.0)
        cache.set("short1", "value1", ttl_seconds=0.01)
        cache.set("short2", "value2", ttl_seconds=0.01)
        cache.set("long", "value3", ttl_seconds=60.0)

        time.sleep(0.02)
        removed = cache.cleanup_expired()

        assert removed == 2
        assert cache.size == 1
        assert cache.get("long") == "value3"

    def test_size_property(self, lru_cache):
        """Test size property returns correct count."""
        assert lru_cache.size == 0
        lru_cache.set("key1", "value1")
        assert lru_cache.size == 1
        lru_cache.set("key2", "value2")
        assert lru_cache.size == 2

    def test_hit_rate_calculation(self, lru_cache):
        """Test hit rate calculation."""
        lru_cache.set("key1", "value1")
        lru_cache.get("key1")  # hit
        lru_cache.get("key1")  # hit
        lru_cache.get("key2")  # miss

        assert lru_cache.hit_rate == pytest.approx(0.666, rel=0.01)

    def test_hit_rate_zero_when_no_accesses(self, lru_cache):
        """Test hit rate is 0 when no accesses."""
        assert lru_cache.hit_rate == 0.0

    def test_get_stats(self, lru_cache):
        """Test get_stats returns comprehensive statistics."""
        lru_cache.set("key1", "value1")
        lru_cache.get("key1")
        lru_cache.get("missing")

        stats = lru_cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ============================================================================
# DECOMPOSITION CACHE TESTS
# ============================================================================


class TestDecompositionCache:
    """Tests for query decomposition caching."""

    def test_cache_decomposition(self, decomposition_cache):
        """Test caching decomposition result."""
        query = "What is the causal effect of rep visits?"
        decomposition = {"sub_questions": ["q1", "q2"]}

        decomposition_cache.set(query, decomposition)
        result = decomposition_cache.get(query)

        assert result == decomposition

    def test_normalized_query_matching(self, decomposition_cache):
        """Test queries are normalized for matching."""
        decomposition = {"sub_questions": ["q1"]}

        # Set with one format
        decomposition_cache.set("What is the effect?", decomposition)

        # Get with different whitespace/case
        result = decomposition_cache.get("  WHAT   IS   THE   EFFECT?  ")
        assert result == decomposition

    def test_cache_miss_returns_none(self, decomposition_cache):
        """Test cache miss returns None."""
        result = decomposition_cache.get("never seen this query")
        assert result is None

    def test_get_stats(self, decomposition_cache):
        """Test get_stats includes type."""
        decomposition_cache.set("query", {"result": "data"})
        decomposition_cache.get("query")

        stats = decomposition_cache.get_stats()
        assert stats["type"] == "decomposition"
        assert "hits" in stats
        assert "misses" in stats


# ============================================================================
# PLAN SIMILARITY CACHE TESTS
# ============================================================================


@dataclass
class MockSubQuestion:
    """Mock sub-question for testing."""

    intent: str
    entities: List[str]
    depends_on: Optional[List[int]] = None


@dataclass
class MockDecomposition:
    """Mock decomposition for testing."""

    sub_questions: List[MockSubQuestion]


class TestPlanSimilarityCache:
    """Tests for plan similarity matching cache."""

    def test_exact_match(self, plan_similarity_cache):
        """Test exact decomposition match returns plan."""
        decomposition = MockDecomposition(
            sub_questions=[
                MockSubQuestion(intent="CAUSAL", entities=["rep_visits"]),
                MockSubQuestion(intent="COMPARATIVE", entities=["regions"]),
            ]
        )
        plan = {"steps": ["step1", "step2"]}

        plan_similarity_cache.set(decomposition, plan)
        result = plan_similarity_cache.get_similar(decomposition)

        assert result is not None
        matched_plan, similarity = result
        assert matched_plan == plan
        assert similarity >= 0.8

    def test_similar_decomposition_match(self, plan_similarity_cache):
        """Test similar decomposition matches with high similarity."""
        original = MockDecomposition(
            sub_questions=[
                MockSubQuestion(intent="CAUSAL", entities=["rep_visits", "Rx"]),
                MockSubQuestion(intent="COMPARATIVE", entities=["regions"]),
            ]
        )
        plan = {"steps": ["step1", "step2"]}
        plan_similarity_cache.set(original, plan)

        # Similar decomposition (same intents, overlapping entities)
        similar = MockDecomposition(
            sub_questions=[
                MockSubQuestion(intent="CAUSAL", entities=["rep_visits", "Rx", "volume"]),
                MockSubQuestion(intent="COMPARATIVE", entities=["regions", "midwest"]),
            ]
        )

        result = plan_similarity_cache.get_similar(similar)
        assert result is not None
        matched_plan, similarity = result
        assert matched_plan == plan
        assert similarity >= 0.7  # Lower threshold for partial match

    def test_dissimilar_decomposition_no_match(self, plan_similarity_cache):
        """Test very different decomposition doesn't match."""
        original = MockDecomposition(
            sub_questions=[
                MockSubQuestion(intent="CAUSAL", entities=["rep_visits"]),
            ]
        )
        plan_similarity_cache.set(original, {"steps": ["causal"]})

        # Very different decomposition
        different = MockDecomposition(
            sub_questions=[
                MockSubQuestion(intent="PREDICTIVE", entities=["sales", "forecast"]),
                MockSubQuestion(intent="EXPERIMENTAL", entities=["AB_test"]),
            ]
        )

        result = plan_similarity_cache.get_similar(different)
        assert result is None

    def test_empty_cache_returns_none(self, plan_similarity_cache):
        """Test empty cache returns None."""
        decomposition = MockDecomposition(sub_questions=[])
        result = plan_similarity_cache.get_similar(decomposition)
        assert result is None

    def test_get_stats(self, plan_similarity_cache):
        """Test get_stats includes similarity threshold."""
        stats = plan_similarity_cache.get_stats()
        assert stats["type"] == "plan_similarity"
        assert stats["similarity_threshold"] == 0.8


# ============================================================================
# TOOL OUTPUT CACHE TESTS
# ============================================================================


class TestToolOutputCache:
    """Tests for deterministic tool output caching."""

    def test_register_deterministic_tool(self, tool_output_cache):
        """Test registering a deterministic tool."""
        tool_output_cache.register_deterministic_tool("my_tool")
        assert tool_output_cache.is_deterministic("my_tool") is True

    def test_non_deterministic_tool(self, tool_output_cache):
        """Test non-registered tool is not deterministic."""
        assert tool_output_cache.is_deterministic("random_tool") is False

    def test_cache_deterministic_output(self, tool_output_cache):
        """Test caching output for deterministic tool."""
        tool_output_cache.register_deterministic_tool("psi_calculator")
        inputs = {"metric": "volume", "threshold": 0.1}
        output = {"psi_value": 0.05, "significant": False}

        tool_output_cache.set("psi_calculator", inputs, output)
        result = tool_output_cache.get("psi_calculator", inputs)

        assert result == output

    def test_no_cache_for_non_deterministic(self, tool_output_cache):
        """Test non-deterministic tool output is not cached."""
        inputs = {"data": [1, 2, 3]}
        output = {"result": 42}

        # Don't register as deterministic
        tool_output_cache.set("random_tool", inputs, output)
        result = tool_output_cache.get("random_tool", inputs)

        assert result is None

    def test_different_inputs_different_cache_keys(self, tool_output_cache):
        """Test different inputs create different cache entries."""
        tool_output_cache.register_deterministic_tool("calculator")

        tool_output_cache.set("calculator", {"a": 1}, {"result": 1})
        tool_output_cache.set("calculator", {"a": 2}, {"result": 2})

        assert tool_output_cache.get("calculator", {"a": 1})["result"] == 1
        assert tool_output_cache.get("calculator", {"a": 2})["result"] == 2

    def test_input_order_independence(self, tool_output_cache):
        """Test input dict order doesn't affect caching."""
        tool_output_cache.register_deterministic_tool("tool")
        output = {"result": 42}

        tool_output_cache.set("tool", {"a": 1, "b": 2}, output)
        result = tool_output_cache.get("tool", {"b": 2, "a": 1})

        assert result == output

    def test_get_stats(self, tool_output_cache):
        """Test get_stats includes deterministic tools list."""
        tool_output_cache.register_deterministic_tool("tool1")
        tool_output_cache.register_deterministic_tool("tool2")

        stats = tool_output_cache.get_stats()
        assert stats["type"] == "tool_output"
        assert "tool1" in stats["deterministic_tools"]
        assert "tool2" in stats["deterministic_tools"]


# ============================================================================
# CACHE MANAGER TESTS
# ============================================================================


class TestToolComposerCacheManager:
    """Tests for unified cache manager."""

    def test_singleton_pattern(self, cache_manager):
        """Test cache manager is singleton."""
        manager2 = ToolComposerCacheManager()
        assert cache_manager is manager2

    def test_decomposition_caching(self, cache_manager):
        """Test decomposition caching through manager."""
        query = "test query"
        decomposition = {"sub_questions": ["q1"]}

        cache_manager.cache_decomposition(query, decomposition)
        result = cache_manager.get_decomposition(query)

        assert result == decomposition

    def test_plan_caching(self, cache_manager):
        """Test plan caching through manager."""
        decomposition = MockDecomposition(
            sub_questions=[MockSubQuestion(intent="CAUSAL", entities=["test"])]
        )
        plan = {"steps": ["step1"]}

        cache_manager.cache_plan(decomposition, plan)
        result = cache_manager.get_similar_plan(decomposition)

        assert result is not None
        assert result[0] == plan

    def test_tool_output_caching(self, cache_manager):
        """Test tool output caching through manager."""
        # Use default deterministic tool
        inputs = {"metric": "volume"}
        output = {"psi": 0.1}

        cache_manager.cache_tool_output("psi_calculator", inputs, output)
        result = cache_manager.get_tool_output("psi_calculator", inputs)

        assert result == output

    def test_register_custom_deterministic_tool(self, cache_manager):
        """Test registering custom deterministic tool."""
        cache_manager.register_deterministic_tool("custom_tool")

        inputs = {"x": 1}
        output = {"y": 2}
        cache_manager.cache_tool_output("custom_tool", inputs, output)

        assert cache_manager.get_tool_output("custom_tool", inputs) == output

    def test_default_deterministic_tools(self, cache_manager):
        """Test default deterministic tools are registered."""
        default_tools = ["psi_calculator", "power_calculator", "segment_ranker"]
        for tool in default_tools:
            assert cache_manager.output_cache.is_deterministic(tool) is True

    def test_cleanup(self, cache_manager):
        """Test cleanup removes expired entries."""
        # Set short TTL entries
        cache_manager.decomposition_cache._cache.set("short", "value", ttl_seconds=0.01)
        time.sleep(0.02)

        result = cache_manager.cleanup()
        assert result["decomposition"] == 1

    def test_clear_all(self, cache_manager):
        """Test clear_all removes all entries."""
        cache_manager.cache_decomposition("query", {"data": 1})
        cache_manager.cache_tool_output("psi_calculator", {"x": 1}, {"y": 2})

        cache_manager.clear_all()

        assert cache_manager.decomposition_cache._cache.size == 0
        assert cache_manager.output_cache._cache.size == 0

    def test_get_all_stats(self, cache_manager):
        """Test get_all_stats returns comprehensive statistics."""
        cache_manager.cache_decomposition("query", {"data": 1})

        stats = cache_manager.get_all_stats()

        assert "decomposition" in stats
        assert "plan" in stats
        assert "output" in stats
        assert "timestamp" in stats


# ============================================================================
# GET_CACHE_MANAGER FUNCTION TESTS
# ============================================================================


class TestGetCacheManager:
    """Tests for the get_cache_manager function."""

    def test_returns_singleton(self):
        """Test get_cache_manager returns singleton instance."""
        # Reset singleton
        ToolComposerCacheManager._instance = None

        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        assert manager1 is manager2

        # Cleanup
        ToolComposerCacheManager._instance = None

    def test_creates_manager_with_defaults(self):
        """Test get_cache_manager creates manager with default config."""
        # Reset singleton
        ToolComposerCacheManager._instance = None

        manager = get_cache_manager()

        assert manager.decomposition_cache is not None
        assert manager.plan_cache is not None
        assert manager.output_cache is not None

        # Cleanup
        ToolComposerCacheManager._instance = None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestCacheIntegration:
    """Integration tests for cache components."""

    def test_full_caching_workflow(self, cache_manager):
        """Test complete caching workflow."""
        # 1. Cache decomposition
        query = "What is the effect of rep visits on Rx?"
        decomposition = MockDecomposition(
            sub_questions=[
                MockSubQuestion(intent="CAUSAL", entities=["rep_visits", "Rx"]),
            ]
        )
        cache_manager.cache_decomposition(query, decomposition)

        # 2. Cache plan
        plan = {"steps": [{"tool": "causal_effect_estimator"}]}
        cache_manager.cache_plan(decomposition, plan)

        # 3. Cache tool outputs
        cache_manager.cache_tool_output(
            "psi_calculator",
            {"metric": "rep_visits"},
            {"psi": 0.05},
        )

        # 4. Retrieve all cached data
        cached_decomp = cache_manager.get_decomposition(query)
        assert cached_decomp == decomposition

        cached_plan = cache_manager.get_similar_plan(decomposition)
        assert cached_plan is not None

        cached_output = cache_manager.get_tool_output("psi_calculator", {"metric": "rep_visits"})
        assert cached_output == {"psi": 0.05}

    def test_cache_eviction_under_load(self):
        """Test cache correctly evicts under load."""
        # Small cache for testing
        ToolComposerCacheManager._instance = None
        manager = ToolComposerCacheManager(
            decomposition_max_size=3,
            decomposition_ttl=300.0,
            plan_max_size=3,
            plan_ttl=300.0,
            output_max_size=3,
            output_ttl=300.0,
        )

        # Fill decomposition cache beyond capacity
        for i in range(5):
            manager.cache_decomposition(f"query_{i}", {"id": i})

        # Should have max 3 entries
        assert manager.decomposition_cache._cache.size == 3

        # Cleanup
        ToolComposerCacheManager._instance = None

    def test_concurrent_cache_access(self, cache_manager):
        """Test cache handles concurrent access."""
        import concurrent.futures

        def cache_operation(i):
            cache_manager.cache_decomposition(f"query_{i}", {"id": i})
            return cache_manager.get_decomposition(f"query_{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All operations should succeed
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == {"id": i}
