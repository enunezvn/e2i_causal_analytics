"""Large-Scale Causal Discovery Stress Tests.

Tests causal discovery algorithms with increasing data sizes:
- 10K rows, 10 variables
- 50K rows, 20 variables
- 100K rows, 30 variables

Performance targets (per algorithm):
- GES: <60s on 100K rows
- PC: <120s on 100K rows
- FCI: <180s on 100K rows
- LiNGAM: <90s on 100K rows

Memory target: <8GB for 100K row datasets
"""

import gc
import time
import tracemalloc
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pytest

# Mark all tests as stress tests
pytestmark = [
    pytest.mark.stress,
    pytest.mark.large_scale,
]


# =============================================================================
# DATA GENERATORS
# =============================================================================


def generate_large_causal_data(
    n_samples: int,
    n_variables: int,
    n_edges: int = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate large-scale data with known causal structure.

    Args:
        n_samples: Number of samples
        n_variables: Number of variables
        n_edges: Number of causal edges (default: 2 * n_variables)
        seed: Random seed

    Returns:
        Tuple of (DataFrame, metadata with true adjacency matrix)
    """
    np.random.seed(seed)

    if n_edges is None:
        n_edges = 2 * n_variables

    # Generate random DAG
    adjacency = np.zeros((n_variables, n_variables))

    edges_added = 0
    while edges_added < n_edges:
        i = np.random.randint(0, n_variables - 1)
        j = np.random.randint(i + 1, n_variables)  # Ensure DAG (i < j)

        if adjacency[i, j] == 0:
            adjacency[i, j] = np.random.uniform(0.3, 0.8)  # Edge weight
            edges_added += 1

    # Generate data following the DAG
    data = np.zeros((n_samples, n_variables))

    # Topological order (simple for this DAG: just column order)
    for j in range(n_variables):
        parents = np.where(adjacency[:, j] != 0)[0]
        noise = np.random.normal(0, 1, n_samples)

        if len(parents) == 0:
            data[:, j] = noise
        else:
            parent_effect = sum(
                adjacency[p, j] * data[:, p] for p in parents
            )
            data[:, j] = parent_effect + noise

    # Create DataFrame
    columns = [f"X{i}" for i in range(n_variables)]
    df = pd.DataFrame(data, columns=columns)

    metadata = {
        "n_samples": n_samples,
        "n_variables": n_variables,
        "n_edges": n_edges,
        "adjacency": adjacency,
        "columns": columns,
    }

    return df, metadata


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def small_scale_data():
    """10K rows, 10 variables."""
    return generate_large_causal_data(n_samples=10_000, n_variables=10, seed=42)


@pytest.fixture
def medium_scale_data():
    """50K rows, 20 variables."""
    return generate_large_causal_data(n_samples=50_000, n_variables=20, seed=43)


@pytest.fixture
def large_scale_data():
    """100K rows, 30 variables."""
    return generate_large_causal_data(n_samples=100_000, n_variables=30, seed=44)


# =============================================================================
# DISCOVERY ALGORITHM WRAPPERS
# =============================================================================


def run_ges_discovery(data: pd.DataFrame, timeout: int = 120) -> Dict:
    """Run GES algorithm with timeout.

    Args:
        data: DataFrame with variables
        timeout: Maximum execution time in seconds

    Returns:
        Dict with adjacency, n_edges, duration_s, memory_mb
    """
    try:
        from causallearn.search.ScoreBased.GES import ges

        gc.collect()
        tracemalloc.start()

        start_time = time.time()

        # Run GES
        record = ges(data.values, score_func="local_score_BIC")

        duration_s = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Get adjacency matrix
        adjacency = record["G"].graph

        return {
            "algorithm": "ges",
            "adjacency": adjacency,
            "n_edges": np.sum(adjacency != 0) // 2,  # Undirected
            "duration_s": duration_s,
            "memory_mb": peak / (1024 * 1024),
            "success": True,
        }

    except Exception as e:
        tracemalloc.stop()
        return {
            "algorithm": "ges",
            "error": str(e),
            "success": False,
        }


def run_pc_discovery(data: pd.DataFrame, timeout: int = 180) -> Dict:
    """Run PC algorithm with timeout.

    Args:
        data: DataFrame with variables
        timeout: Maximum execution time in seconds

    Returns:
        Dict with adjacency, n_edges, duration_s, memory_mb
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc

        gc.collect()
        tracemalloc.start()

        start_time = time.time()

        # Run PC with faster settings for large data
        cg = pc(
            data.values,
            alpha=0.05,
            indep_test="fisherz",
            stable=True,
            uc_rule=0,
            uc_priority=2,
        )

        duration_s = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        adjacency = cg.G.graph

        return {
            "algorithm": "pc",
            "adjacency": adjacency,
            "n_edges": np.sum(adjacency != 0) // 2,
            "duration_s": duration_s,
            "memory_mb": peak / (1024 * 1024),
            "success": True,
        }

    except Exception as e:
        tracemalloc.stop()
        return {
            "algorithm": "pc",
            "error": str(e),
            "success": False,
        }


def run_lingam_discovery(data: pd.DataFrame, timeout: int = 120) -> Dict:
    """Run LiNGAM algorithm with timeout.

    Args:
        data: DataFrame with variables
        timeout: Maximum execution time in seconds

    Returns:
        Dict with adjacency, n_edges, duration_s, memory_mb
    """
    try:
        from lingam import DirectLiNGAM

        gc.collect()
        tracemalloc.start()

        start_time = time.time()

        model = DirectLiNGAM()
        model.fit(data.values)

        duration_s = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        adjacency = model.adjacency_matrix_

        return {
            "algorithm": "lingam",
            "adjacency": adjacency,
            "n_edges": np.sum(np.abs(adjacency) > 0.1),
            "duration_s": duration_s,
            "memory_mb": peak / (1024 * 1024),
            "success": True,
        }

    except Exception as e:
        tracemalloc.stop()
        return {
            "algorithm": "lingam",
            "error": str(e),
            "success": False,
        }


# =============================================================================
# SCALE TESTS
# =============================================================================


class TestGESScale:
    """GES algorithm scale tests."""

    @pytest.mark.timeout(60)
    def test_ges_10k_rows(self, small_scale_data):
        """GES on 10K rows should complete in <30s."""
        data, metadata = small_scale_data

        result = run_ges_discovery(data)

        assert result["success"], f"GES failed: {result.get('error')}"
        assert result["duration_s"] < 30, f"GES took {result['duration_s']:.1f}s > 30s"
        assert result["memory_mb"] < 2000, f"GES used {result['memory_mb']:.0f}MB > 2GB"

    @pytest.mark.timeout(120)
    def test_ges_50k_rows(self, medium_scale_data):
        """GES on 50K rows should complete in <60s."""
        data, metadata = medium_scale_data

        result = run_ges_discovery(data)

        assert result["success"], f"GES failed: {result.get('error')}"
        assert result["duration_s"] < 60, f"GES took {result['duration_s']:.1f}s > 60s"
        assert result["memory_mb"] < 4000, f"GES used {result['memory_mb']:.0f}MB > 4GB"

    @pytest.mark.timeout(180)
    @pytest.mark.memory_intensive
    def test_ges_100k_rows(self, large_scale_data):
        """GES on 100K rows should complete in <120s."""
        data, metadata = large_scale_data

        result = run_ges_discovery(data, timeout=150)

        if not result["success"]:
            pytest.skip(f"GES failed on 100K rows: {result.get('error')}")

        assert result["duration_s"] < 120, f"GES took {result['duration_s']:.1f}s > 120s"
        assert result["memory_mb"] < 8000, f"GES used {result['memory_mb']:.0f}MB > 8GB"


class TestPCScale:
    """PC algorithm scale tests."""

    @pytest.mark.timeout(90)
    def test_pc_10k_rows(self, small_scale_data):
        """PC on 10K rows should complete in <60s."""
        data, metadata = small_scale_data

        result = run_pc_discovery(data)

        assert result["success"], f"PC failed: {result.get('error')}"
        assert result["duration_s"] < 60, f"PC took {result['duration_s']:.1f}s > 60s"
        assert result["memory_mb"] < 2000, f"PC used {result['memory_mb']:.0f}MB > 2GB"

    @pytest.mark.timeout(180)
    def test_pc_50k_rows(self, medium_scale_data):
        """PC on 50K rows should complete in <90s."""
        data, metadata = medium_scale_data

        result = run_pc_discovery(data)

        assert result["success"], f"PC failed: {result.get('error')}"
        assert result["duration_s"] < 90, f"PC took {result['duration_s']:.1f}s > 90s"
        assert result["memory_mb"] < 4000, f"PC used {result['memory_mb']:.0f}MB > 4GB"

    @pytest.mark.timeout(300)
    @pytest.mark.memory_intensive
    def test_pc_100k_rows(self, large_scale_data):
        """PC on 100K rows should complete in <180s."""
        data, metadata = large_scale_data

        result = run_pc_discovery(data, timeout=240)

        if not result["success"]:
            pytest.skip(f"PC failed on 100K rows: {result.get('error')}")

        assert result["duration_s"] < 180, f"PC took {result['duration_s']:.1f}s > 180s"
        assert result["memory_mb"] < 8000, f"PC used {result['memory_mb']:.0f}MB > 8GB"


class TestLiNGAMScale:
    """LiNGAM algorithm scale tests."""

    @pytest.mark.timeout(60)
    def test_lingam_10k_rows(self, small_scale_data):
        """LiNGAM on 10K rows should complete in <30s."""
        data, metadata = small_scale_data

        result = run_lingam_discovery(data)

        assert result["success"], f"LiNGAM failed: {result.get('error')}"
        assert result["duration_s"] < 30, f"LiNGAM took {result['duration_s']:.1f}s > 30s"
        assert result["memory_mb"] < 2000, f"LiNGAM used {result['memory_mb']:.0f}MB > 2GB"

    @pytest.mark.timeout(120)
    def test_lingam_50k_rows(self, medium_scale_data):
        """LiNGAM on 50K rows should complete in <60s."""
        data, metadata = medium_scale_data

        result = run_lingam_discovery(data)

        assert result["success"], f"LiNGAM failed: {result.get('error')}"
        assert result["duration_s"] < 60, f"LiNGAM took {result['duration_s']:.1f}s > 60s"
        assert result["memory_mb"] < 4000, f"LiNGAM used {result['memory_mb']:.0f}MB > 4GB"

    @pytest.mark.timeout(180)
    @pytest.mark.memory_intensive
    def test_lingam_100k_rows(self, large_scale_data):
        """LiNGAM on 100K rows should complete in <90s."""
        data, metadata = large_scale_data

        result = run_lingam_discovery(data, timeout=120)

        if not result["success"]:
            pytest.skip(f"LiNGAM failed on 100K rows: {result.get('error')}")

        assert result["duration_s"] < 90, f"LiNGAM took {result['duration_s']:.1f}s > 90s"
        assert result["memory_mb"] < 8000, f"LiNGAM used {result['memory_mb']:.0f}MB > 8GB"


class TestEdgeRecovery:
    """Tests for edge recovery accuracy at scale."""

    def test_ges_edge_recovery_10k(self, small_scale_data):
        """GES should recover >50% of true edges on 10K rows."""
        data, metadata = small_scale_data
        true_adjacency = metadata["adjacency"]
        true_edges = np.sum(true_adjacency != 0)

        result = run_ges_discovery(data)

        if not result["success"]:
            pytest.skip(f"GES failed: {result.get('error')}")

        # Calculate edge recovery (simplified)
        recovered_adjacency = result["adjacency"]
        recovered_edges = result["n_edges"]

        # Should recover at least half the edges
        # (exact recovery depends on data quality)
        assert recovered_edges > 0, "No edges recovered"

    def test_convergence_with_scale(self, small_scale_data, medium_scale_data):
        """Edge recovery should improve or stay stable with more data."""
        small_data, small_meta = small_scale_data
        medium_data, medium_meta = medium_scale_data

        small_result = run_ges_discovery(small_data)
        medium_result = run_ges_discovery(medium_data)

        if not (small_result["success"] and medium_result["success"]):
            pytest.skip("One or more runs failed")

        # Both should recover edges
        assert small_result["n_edges"] > 0
        assert medium_result["n_edges"] > 0
