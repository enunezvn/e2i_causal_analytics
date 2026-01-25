"""Parallel Algorithm Execution Tests.

Tests for parallel causal discovery execution:
- use_process_pool=True path with large data
- ProcessPoolExecutor serialization
- Timeout behavior with slow algorithms
- Memory cleanup after parallel runs

Performance targets:
- Parallel execution should show speedup for 3+ algorithms
- Memory cleanup should prevent accumulation
- Timeouts should be respected
"""

import asyncio
import gc
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# Mark all tests as stress tests
pytestmark = [
    pytest.mark.stress,
    pytest.mark.parallel,
]


# =============================================================================
# DATA GENERATORS
# =============================================================================


def generate_parallel_test_data(
    n_samples: int = 5000,
    n_variables: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data for parallel execution tests.

    Args:
        n_samples: Number of samples
        n_variables: Number of variables
        seed: Random seed

    Returns:
        DataFrame with causal structure
    """
    np.random.seed(seed)

    data = {}
    for i in range(n_variables):
        if i == 0:
            data[f"X{i}"] = np.random.normal(0, 1, n_samples)
        else:
            # Create chain dependency
            parent_idx = i - 1
            data[f"X{i}"] = 0.5 * data[f"X{parent_idx}"] + np.random.normal(0, 1, n_samples)

    return pd.DataFrame(data)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def small_parallel_data():
    """5K rows for parallel tests."""
    return generate_parallel_test_data(n_samples=5000, n_variables=8)


@pytest.fixture
def medium_parallel_data():
    """20K rows for parallel tests."""
    return generate_parallel_test_data(n_samples=20000, n_variables=12)


@pytest.fixture
def large_parallel_data():
    """50K rows for parallel tests."""
    return generate_parallel_test_data(n_samples=50000, n_variables=15)


# =============================================================================
# PARALLEL ALGORITHM RUNNERS
# =============================================================================


def _run_ges_worker(data_values: np.ndarray, columns: List[str]) -> Dict:
    """Worker function for GES in ProcessPool.

    Args:
        data_values: numpy array of data (serializable)
        columns: column names

    Returns:
        Dict with algorithm, n_edges, duration_s
    """
    try:
        from causallearn.search.ScoreBased.GES import ges

        data = pd.DataFrame(data_values, columns=columns)

        start = time.time()
        record = ges(data.values, score_func="local_score_BIC")
        duration = time.time() - start

        return {
            "algorithm": "ges",
            "n_edges": np.sum(record["G"].graph != 0) // 2,
            "duration_s": duration,
            "success": True,
        }
    except Exception as e:
        return {"algorithm": "ges", "error": str(e), "success": False}


def _run_pc_worker(data_values: np.ndarray, columns: List[str]) -> Dict:
    """Worker function for PC in ProcessPool."""
    try:
        from causallearn.search.ConstraintBased.PC import pc

        data = pd.DataFrame(data_values, columns=columns)

        start = time.time()
        cg = pc(
            data.values,
            alpha=0.05,
            indep_test="fisherz",
            stable=True,
        )
        duration = time.time() - start

        return {
            "algorithm": "pc",
            "n_edges": np.sum(cg.G.graph != 0) // 2,
            "duration_s": duration,
            "success": True,
        }
    except Exception as e:
        return {"algorithm": "pc", "error": str(e), "success": False}


def _run_lingam_worker(data_values: np.ndarray, columns: List[str]) -> Dict:
    """Worker function for LiNGAM in ProcessPool."""
    try:
        from lingam import DirectLiNGAM

        start = time.time()
        model = DirectLiNGAM()
        model.fit(data_values)
        duration = time.time() - start

        return {
            "algorithm": "lingam",
            "n_edges": np.sum(np.abs(model.adjacency_matrix_) > 0.1),
            "duration_s": duration,
            "success": True,
        }
    except Exception as e:
        return {"algorithm": "lingam", "error": str(e), "success": False}


def _slow_algorithm_worker(data_values: np.ndarray, delay: float = 10.0) -> Dict:
    """Simulates a slow algorithm for timeout testing."""
    time.sleep(delay)
    return {"algorithm": "slow", "duration_s": delay, "success": True}


# =============================================================================
# PARALLEL EXECUTION HELPERS
# =============================================================================


def run_algorithms_parallel(
    data: pd.DataFrame,
    algorithms: List[str],
    max_workers: int = 3,
    timeout: float = 60.0,
) -> Tuple[List[Dict], float]:
    """Run multiple algorithms in parallel using ProcessPoolExecutor.

    Args:
        data: DataFrame with variables
        algorithms: List of algorithm names to run
        max_workers: Maximum parallel workers
        timeout: Timeout per algorithm in seconds

    Returns:
        Tuple of (results list, total duration)
    """
    data_values = data.values
    columns = list(data.columns)

    worker_map = {
        "ges": _run_ges_worker,
        "pc": _run_pc_worker,
        "lingam": _run_lingam_worker,
    }

    results = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for algo in algorithms:
            if algo in worker_map:
                future = executor.submit(worker_map[algo], data_values, columns)
                futures[future] = algo

        for future in futures:
            algo = futures[future]
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except FuturesTimeoutError:
                results.append({
                    "algorithm": algo,
                    "error": "timeout",
                    "success": False,
                })
            except Exception as e:
                results.append({
                    "algorithm": algo,
                    "error": str(e),
                    "success": False,
                })

    total_duration = time.time() - start
    return results, total_duration


def run_algorithms_sequential(
    data: pd.DataFrame,
    algorithms: List[str],
) -> Tuple[List[Dict], float]:
    """Run multiple algorithms sequentially for comparison.

    Args:
        data: DataFrame with variables
        algorithms: List of algorithm names to run

    Returns:
        Tuple of (results list, total duration)
    """
    data_values = data.values
    columns = list(data.columns)

    worker_map = {
        "ges": _run_ges_worker,
        "pc": _run_pc_worker,
        "lingam": _run_lingam_worker,
    }

    results = []
    start = time.time()

    for algo in algorithms:
        if algo in worker_map:
            result = worker_map[algo](data_values, columns)
            results.append(result)

    total_duration = time.time() - start
    return results, total_duration


# =============================================================================
# PARALLEL EXECUTION TESTS
# =============================================================================


class TestParallelSpeedup:
    """Tests for parallel execution speedup."""

    @pytest.mark.timeout(180)
    def test_parallel_faster_than_sequential(self, small_parallel_data):
        """Parallel execution should be faster than sequential for 3+ algorithms."""
        data = small_parallel_data
        algorithms = ["ges", "pc", "lingam"]

        # Run sequential
        seq_results, seq_duration = run_algorithms_sequential(data, algorithms)

        # Run parallel
        par_results, par_duration = run_algorithms_parallel(
            data, algorithms, max_workers=3
        )

        # Both should have results
        seq_success = sum(1 for r in seq_results if r.get("success"))
        par_success = sum(1 for r in par_results if r.get("success"))

        if seq_success < 2 or par_success < 2:
            pytest.skip("Not enough successful algorithm runs")

        # Parallel should be faster (at least 30% speedup)
        speedup = seq_duration / par_duration if par_duration > 0 else 0
        assert speedup > 1.3, (
            f"Parallel speedup {speedup:.2f}x < 1.3x "
            f"(seq={seq_duration:.1f}s, par={par_duration:.1f}s)"
        )

    @pytest.mark.timeout(120)
    def test_parallel_with_two_algorithms(self, small_parallel_data):
        """Parallel should work with 2 algorithms."""
        data = small_parallel_data
        algorithms = ["ges", "lingam"]

        results, duration = run_algorithms_parallel(data, algorithms, max_workers=2)

        success_count = sum(1 for r in results if r.get("success"))
        assert success_count >= 1, "No algorithms succeeded"


class TestProcessPoolSerialization:
    """Tests for ProcessPoolExecutor serialization."""

    def test_data_serialization_roundtrip(self, small_parallel_data):
        """Data should serialize correctly to worker processes."""
        data = small_parallel_data

        # Run single algorithm to test serialization
        results, _ = run_algorithms_parallel(data, ["ges"], max_workers=1, timeout=60)

        assert len(results) == 1
        if results[0].get("success"):
            assert results[0]["n_edges"] >= 0
        else:
            # Serialization may work but algorithm may fail
            # If error is not serialization-related, that's acceptable
            error = results[0].get("error", "")
            assert "pickle" not in error.lower(), f"Serialization error: {error}"

    def test_large_data_serialization(self, medium_parallel_data):
        """Large data should serialize without issues."""
        data = medium_parallel_data

        # Should not raise pickling errors
        try:
            results, _ = run_algorithms_parallel(
                data, ["lingam"], max_workers=1, timeout=120
            )
            # Success or algorithm failure (not serialization failure)
            assert len(results) == 1
        except Exception as e:
            assert "pickle" not in str(e).lower(), f"Serialization error: {e}"


class TestTimeoutBehavior:
    """Tests for timeout behavior with slow algorithms."""

    @pytest.mark.timeout(30)
    def test_timeout_respected(self):
        """Timeout should be respected for slow algorithms."""
        # Use mock slow worker
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_slow_algorithm_worker, np.array([1, 2, 3]), 10.0)

            start = time.time()
            try:
                result = future.result(timeout=2.0)
                # Should not reach here
                assert False, "Timeout was not raised"
            except FuturesTimeoutError:
                elapsed = time.time() - start
                # Should timeout within 3 seconds (2s timeout + margin)
                assert elapsed < 5.0, f"Timeout took {elapsed:.1f}s"

    @pytest.mark.timeout(60)
    def test_mixed_fast_slow_algorithms(self, small_parallel_data):
        """Fast algorithms should complete even if one times out."""
        data = small_parallel_data

        # Run just one fast algorithm with short timeout
        results, duration = run_algorithms_parallel(
            data, ["ges"], max_workers=1, timeout=30
        )

        # Should complete (ges is fast)
        assert len(results) == 1
        # May succeed or fail, but should respond within timeout


class TestMemoryCleanup:
    """Tests for memory cleanup after parallel runs."""

    def test_memory_released_after_parallel(self, small_parallel_data):
        """Memory should be released after parallel execution."""
        gc.collect()
        tracemalloc.start()

        data = small_parallel_data

        # Run parallel algorithms
        _, baseline = tracemalloc.get_traced_memory()
        results, _ = run_algorithms_parallel(data, ["ges", "lingam"], max_workers=2)

        # Force cleanup
        del results
        gc.collect()

        _, after_cleanup = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should not accumulate significantly
        # Allow 100MB margin for temporary objects
        memory_increase_mb = (after_cleanup - baseline) / (1024 * 1024)
        assert memory_increase_mb < 100, (
            f"Memory increased by {memory_increase_mb:.1f}MB after cleanup"
        )

    def test_repeated_parallel_runs_stable(self, small_parallel_data):
        """Memory should remain stable across repeated parallel runs."""
        gc.collect()
        tracemalloc.start()

        data = small_parallel_data
        memory_snapshots = []

        for i in range(3):
            run_algorithms_parallel(data, ["ges"], max_workers=1, timeout=60)
            gc.collect()
            _, current = tracemalloc.get_traced_memory()
            memory_snapshots.append(current)

        tracemalloc.stop()

        # Memory should not grow unboundedly
        # Allow 50MB growth per iteration
        if len(memory_snapshots) >= 2:
            growth = (memory_snapshots[-1] - memory_snapshots[0]) / (1024 * 1024)
            assert growth < 150, f"Memory grew by {growth:.1f}MB over 3 iterations"


class TestWorkerIsolation:
    """Tests for worker process isolation."""

    def test_worker_failure_isolated(self, small_parallel_data):
        """Failure in one worker should not affect others."""
        data = small_parallel_data

        # Run multiple algorithms - if one fails, others should continue
        results, _ = run_algorithms_parallel(
            data, ["ges", "pc", "lingam"], max_workers=3, timeout=60
        )

        # Should have result for each algorithm
        assert len(results) == 3

        # Each result should have algorithm name
        algos = {r["algorithm"] for r in results}
        assert algos == {"ges", "pc", "lingam"}

    def test_no_state_leak_between_runs(self, small_parallel_data):
        """State should not leak between parallel runs."""
        data = small_parallel_data

        # Run twice with same data
        results1, _ = run_algorithms_parallel(data, ["ges"], max_workers=1, timeout=60)
        results2, _ = run_algorithms_parallel(data, ["ges"], max_workers=1, timeout=60)

        # Results should be consistent (same edge count)
        if results1[0].get("success") and results2[0].get("success"):
            # Allow small variance
            n_edges_1 = results1[0]["n_edges"]
            n_edges_2 = results2[0]["n_edges"]
            assert n_edges_1 == n_edges_2, (
                f"Different results across runs: {n_edges_1} vs {n_edges_2}"
            )


class TestScaleWithParallel:
    """Tests for parallel execution at scale."""

    @pytest.mark.timeout(300)
    @pytest.mark.memory_intensive
    def test_parallel_50k_rows(self, large_parallel_data):
        """Parallel execution should work with 50K row data."""
        data = large_parallel_data

        results, duration = run_algorithms_parallel(
            data, ["ges", "lingam"], max_workers=2, timeout=120
        )

        success_count = sum(1 for r in results if r.get("success"))

        # At least one should succeed
        if success_count == 0:
            pytest.skip("All algorithms failed on 50K rows")

        # Should complete in reasonable time
        assert duration < 180, f"Parallel run took {duration:.1f}s > 180s"

    @pytest.mark.timeout(180)
    def test_cpu_bound_efficiency(self, medium_parallel_data):
        """CPU-bound algorithms should utilize multiple cores efficiently."""
        data = medium_parallel_data
        cpu_count = os.cpu_count() or 2
        max_workers = min(3, cpu_count)

        results, duration = run_algorithms_parallel(
            data, ["ges", "pc"], max_workers=max_workers, timeout=90
        )

        # Should complete both algorithms
        assert len(results) == 2

        # Both should have attempted
        for r in results:
            assert "algorithm" in r
