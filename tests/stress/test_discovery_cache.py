"""Discovery Cache Performance Tests.

Tests for caching behavior in causal discovery:
- Cache hit performance with repeated queries
- Cache hash stability (same data = same hash)
- Cache eviction behavior
- Cache overhead vs computation savings

Performance targets:
- Cache hit should return in <100ms
- Cache should reduce repeated query time by >90%
"""

import hashlib
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

# Mark all tests as stress tests
pytestmark = [pytest.mark.stress]


# =============================================================================
# DATA GENERATORS
# =============================================================================


def generate_causal_data(
    n_samples: int = 5000,
    n_variables: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate causal data for cache testing."""
    np.random.seed(seed)

    data = {}
    for i in range(n_variables):
        if i == 0:
            data[f"X{i}"] = np.random.normal(0, 1, n_samples)
        else:
            # Create dependencies
            parent = np.random.randint(0, i)
            data[f"X{i}"] = 0.5 * data[f"X{parent}"] + np.random.normal(0, 1, n_samples)

    return pd.DataFrame(data)


def compute_data_hash(data: pd.DataFrame) -> str:
    """Compute deterministic hash of DataFrame."""
    # Use numpy array bytes for hashing
    arr_bytes = data.values.tobytes()
    col_bytes = ",".join(data.columns).encode()
    return hashlib.sha256(arr_bytes + col_bytes).hexdigest()


# =============================================================================
# MOCK CACHE IMPLEMENTATION
# =============================================================================


class DiscoveryCache:
    """Simple in-memory cache for discovery results.

    Simulates the caching behavior of the discovery module.
    """

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Dict] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Dict:
        """Get cached result."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Dict) -> None:
        """Cache a result."""
        if len(self._cache) >= self._max_size:
            # Simple FIFO eviction
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


def run_discovery_with_cache(
    data: pd.DataFrame,
    cache: DiscoveryCache,
    algorithm: str = "ges",
) -> Tuple[Dict, float]:
    """Run discovery with caching.

    Returns (result, duration_ms).
    """
    data_hash = compute_data_hash(data)
    cache_key = f"{algorithm}:{data_hash}"

    start = time.time()

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        duration_ms = (time.time() - start) * 1000
        return cached, duration_ms

    # Run actual discovery (mocked for speed)
    # In production, this would call the actual algorithm
    time.sleep(0.5)  # Simulate computation time

    result = {
        "algorithm": algorithm,
        "n_edges": np.random.randint(5, 20),
        "adjacency": np.random.rand(10, 10),
    }

    cache.set(cache_key, result)

    duration_ms = (time.time() - start) * 1000
    return result, duration_ms


# =============================================================================
# CACHE TESTS
# =============================================================================


class TestCacheHitPerformance:
    """Tests for cache hit performance."""

    def test_cache_hit_under_100ms(self):
        """Cache hit should return in <100ms."""
        data = generate_causal_data()
        cache = DiscoveryCache()

        # First run (cache miss)
        result1, duration1 = run_discovery_with_cache(data, cache)

        # Second run (cache hit)
        result2, duration2 = run_discovery_with_cache(data, cache)

        assert duration2 < 100, f"Cache hit took {duration2:.1f}ms > 100ms"
        assert result1 == result2, "Cached result differs from original"

    def test_cache_reduces_time_by_90_percent(self):
        """Cache should reduce repeated query time by >90%."""
        data = generate_causal_data()
        cache = DiscoveryCache()

        # First run (cache miss)
        _, duration_miss = run_discovery_with_cache(data, cache)

        # Multiple cache hits
        hit_durations = []
        for _ in range(5):
            _, duration_hit = run_discovery_with_cache(data, cache)
            hit_durations.append(duration_hit)

        avg_hit = np.mean(hit_durations)
        reduction = (duration_miss - avg_hit) / duration_miss

        assert reduction > 0.90, f"Cache only reduced time by {reduction:.1%}"


class TestCacheHashStability:
    """Tests for cache hash stability."""

    def test_same_data_same_hash(self):
        """Same data should produce same hash."""
        data1 = generate_causal_data(seed=42)
        data2 = generate_causal_data(seed=42)

        hash1 = compute_data_hash(data1)
        hash2 = compute_data_hash(data2)

        assert hash1 == hash2, "Same data produced different hashes"

    def test_different_data_different_hash(self):
        """Different data should produce different hash."""
        data1 = generate_causal_data(seed=42)
        data2 = generate_causal_data(seed=43)

        hash1 = compute_data_hash(data1)
        hash2 = compute_data_hash(data2)

        assert hash1 != hash2, "Different data produced same hash"

    def test_hash_stable_across_calls(self):
        """Hash should be stable across multiple calls."""
        data = generate_causal_data()

        hashes = [compute_data_hash(data) for _ in range(10)]

        assert len(set(hashes)) == 1, "Hash not stable across calls"


class TestCacheEviction:
    """Tests for cache eviction behavior."""

    def test_fifo_eviction(self):
        """Cache should evict oldest entries when full."""
        cache = DiscoveryCache(max_size=3)

        # Fill cache
        for i in range(3):
            data = generate_causal_data(seed=i)
            run_discovery_with_cache(data, cache, algorithm=f"algo_{i}")

        # Verify cache is full
        assert len(cache._cache) == 3

        # Add one more (should evict oldest)
        data_new = generate_causal_data(seed=99)
        run_discovery_with_cache(data_new, cache, algorithm="algo_new")

        # Cache should still be at max size
        assert len(cache._cache) == 3

    def test_cache_hit_rate_tracking(self):
        """Cache should track hit rate accurately."""
        cache = DiscoveryCache()
        data = generate_causal_data()

        # One miss, four hits
        for _i in range(5):
            run_discovery_with_cache(data, cache)

        assert cache._misses == 1
        assert cache._hits == 4
        assert cache.hit_rate == 0.8


class TestCacheOverhead:
    """Tests for cache overhead vs savings."""

    def test_cache_overhead_minimal(self):
        """Cache operations should have minimal overhead."""
        cache = DiscoveryCache()

        # Measure cache set overhead
        set_times = []
        for i in range(100):
            key = f"key_{i}"
            value = {"data": np.random.rand(10, 10)}
            start = time.time()
            cache.set(key, value)
            set_times.append((time.time() - start) * 1000)

        avg_set = np.mean(set_times)
        assert avg_set < 1, f"Cache set takes {avg_set:.2f}ms > 1ms"

        # Measure cache get overhead
        get_times = []
        for i in range(100):
            key = f"key_{i}"
            start = time.time()
            cache.get(key)
            get_times.append((time.time() - start) * 1000)

        avg_get = np.mean(get_times)
        assert avg_get < 0.5, f"Cache get takes {avg_get:.2f}ms > 0.5ms"

    def test_hash_computation_time(self):
        """Data hash computation should be fast."""
        data = generate_causal_data(n_samples=10000)

        hash_times = []
        for _ in range(10):
            start = time.time()
            compute_data_hash(data)
            hash_times.append((time.time() - start) * 1000)

        avg_hash = np.mean(hash_times)
        assert avg_hash < 100, f"Hash computation takes {avg_hash:.1f}ms > 100ms"


class TestCacheWithLargeData:
    """Tests for cache behavior with large datasets."""

    @pytest.mark.memory_intensive
    def test_cache_with_50k_rows(self):
        """Cache should work efficiently with 50K row data."""
        data = generate_causal_data(n_samples=50000, n_variables=20)
        cache = DiscoveryCache()

        # First run
        _, duration1 = run_discovery_with_cache(data, cache)

        # Cache hit
        _, duration2 = run_discovery_with_cache(data, cache)

        assert duration2 < duration1 * 0.1, "Cache not effective for large data"

    def test_hash_stability_large_data(self):
        """Hash should be stable for large data."""
        data = generate_causal_data(n_samples=50000)

        hash1 = compute_data_hash(data)
        hash2 = compute_data_hash(data)

        assert hash1 == hash2
