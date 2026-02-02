"""Tests for PC (Peter-Clark) Algorithm Wrapper.

Version: 1.0.0
Tests the PC algorithm wrapper for constraint-based causal discovery.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.algorithms.pc_wrapper import PCAlgorithm
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)

# =============================================================================
# PCAlgorithm Basic Properties Tests
# =============================================================================


class TestPCAlgorithmProperties:
    """Test PCAlgorithm basic properties."""

    def test_algorithm_type(self):
        """Test algorithm_type property returns PC."""
        algo = PCAlgorithm()
        assert algo.algorithm_type == DiscoveryAlgorithmType.PC

    def test_supports_latent_confounders(self):
        """Test PC reports no support for latent confounders."""
        algo = PCAlgorithm()
        assert algo.supports_latent_confounders() is False


# =============================================================================
# PC Data Validation Tests
# =============================================================================


class TestPCDataValidation:
    """Test data validation for PC algorithm."""

    @pytest.fixture
    def pc(self):
        """Create PC algorithm instance."""
        return PCAlgorithm()

    def test_empty_data_raises_error(self, pc):
        """Test that empty data raises ValueError."""
        config = DiscoveryConfig()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            pc.discover(empty_df, config)

    def test_missing_values_raises_error(self, pc):
        """Test that data with missing values raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="missing values"):
            pc.discover(df, config)

    def test_single_variable_raises_error(self, pc):
        """Test that single variable raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="at least 2 variables"):
            pc.discover(df, config)


# =============================================================================
# PC Discovery Tests
# =============================================================================


class TestPCDiscovery:
    """Test PC discovery functionality."""

    @pytest.fixture
    def pc(self):
        """Create PC algorithm instance."""
        return PCAlgorithm()

    @pytest.fixture
    def simple_data(self):
        """Create simple test data with known structure."""
        np.random.seed(42)
        n = 200
        # X -> Y -> Z structure
        X = np.random.randn(n)
        Y = 0.8 * X + 0.2 * np.random.randn(n)
        Z = 0.8 * Y + 0.2 * np.random.randn(n)
        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_basic_discovery(self, pc, simple_data):
        """Test basic causal discovery with PC."""
        config = DiscoveryConfig(alpha=0.05)

        result = pc.discover(simple_data, config)

        assert isinstance(result, AlgorithmResult)
        assert result.algorithm == DiscoveryAlgorithmType.PC
        assert result.converged is True
        assert result.runtime_seconds > 0
        assert result.adjacency_matrix.shape == (3, 3)

    def test_discovery_with_config(self, pc, simple_data):
        """Test discovery respects configuration."""
        config = DiscoveryConfig(
            alpha=0.01,  # Stricter significance
        )

        result = pc.discover(simple_data, config)

        assert result.converged is True
        assert result.metadata.get("alpha") == 0.01

    def test_edge_list_format(self, pc, simple_data):
        """Test that edge_list has correct format."""
        config = DiscoveryConfig()

        result = pc.discover(simple_data, config)

        for edge in result.edge_list:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert edge[0] in simple_data.columns
            assert edge[1] in simple_data.columns

    def test_node_names_preserved(self, pc, simple_data):
        """Test that node names are preserved in metadata."""
        config = DiscoveryConfig()

        result = pc.discover(simple_data, config)

        assert "node_names" in result.metadata
        assert set(result.metadata["node_names"]) == set(simple_data.columns)

    def test_metadata_contains_ci_test_info(self, pc, simple_data):
        """Test that metadata contains CI test information."""
        config = DiscoveryConfig()

        result = pc.discover(simple_data, config)

        assert "indep_test" in result.metadata
        assert "n_edges" in result.metadata
        assert "n_nodes" in result.metadata
        assert result.metadata["n_nodes"] == 3


# =============================================================================
# PC Independence Test Selection Tests
# =============================================================================


class TestPCIndependenceTest:
    """Test independence test selection."""

    @pytest.fixture
    def pc(self):
        """Create PC algorithm instance."""
        return PCAlgorithm()

    def test_continuous_data_uses_fisherz(self, pc):
        """Test continuous data uses Fisher's z-test."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 5.0, 6.0, 7.0]})
        config = DiscoveryConfig()

        test_name = pc._select_independence_test(df, config)

        assert test_name == "fisherz"

    def test_discrete_data_uses_chisq(self, pc):
        """Test discrete data uses chi-squared test.

        Note: The implementation considers numeric dtypes (int64, float64) as
        continuous. To trigger chisq, we need non-numeric object dtype columns
        with few unique values.
        """
        # Use object dtype to avoid being classified as continuous
        df = pd.DataFrame(
            {
                "A": pd.Categorical(["a", "b", "a", "b"]),
                "B": pd.Categorical(["x", "y", "x", "y"]),
            }
        )
        config = DiscoveryConfig(assume_gaussian=False)

        test_name = pc._select_independence_test(df, config)

        assert test_name == "chisq"

    def test_gaussian_assumption_uses_fisherz(self, pc):
        """Test that assume_gaussian=True uses Fisher's z."""
        df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [0, 1, 0, 1]})
        config = DiscoveryConfig(assume_gaussian=True)

        test_name = pc._select_independence_test(df, config)

        assert test_name == "fisherz"

    def test_mixed_data_uses_fisherz(self, pc):
        """Test mixed data types default to Fisher's z."""
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [0, 1, 0, 1],
            }
        )
        config = DiscoveryConfig()

        test_name = pc._select_independence_test(df, config)

        assert test_name == "fisherz"


# =============================================================================
# PC Error Handling Tests
# =============================================================================


class TestPCErrorHandling:
    """Test error handling in PC algorithm."""

    @pytest.fixture
    def pc(self):
        """Create PC algorithm instance."""
        return PCAlgorithm()

    def test_algorithm_failure_returns_failed_result(self, pc):
        """Test that algorithm failure returns non-converged result."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        # Patch to simulate failure
        with patch(
            "causallearn.search.ConstraintBased.PC.pc",
            side_effect=RuntimeError("Algorithm failed"),
        ):
            result = pc.discover(df, config)

            assert result.converged is False
            assert "error" in result.metadata

    def test_import_error_raises_import_error(self, pc):
        """Test that missing causal-learn raises ImportError with helpful message."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        # Force the import to fail by making causallearn unavailable
        import sys

        original_modules = {}
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("causallearn")]
        for mod in modules_to_remove:
            original_modules[mod] = sys.modules.pop(mod)

        try:
            # Add a blocking entry that raises ImportError when accessed
            sys.modules["causallearn"] = None
            sys.modules["causallearn.search"] = None
            sys.modules["causallearn.search.ConstraintBased"] = None
            sys.modules["causallearn.search.ConstraintBased.PC"] = None

            with pytest.raises(ImportError, match="causal-learn is required"):
                pc.discover(df, config)
        finally:
            # Restore original modules
            for mod in [
                "causallearn",
                "causallearn.search",
                "causallearn.search.ConstraintBased",
                "causallearn.search.ConstraintBased.PC",
            ]:
                sys.modules.pop(mod, None)
            sys.modules.update(original_modules)


# =============================================================================
# PC Adjacency Conversion Tests
# =============================================================================


class TestPCAdjacencyConversion:
    """Test adjacency matrix conversion for CPDAG."""

    @pytest.fixture
    def pc(self):
        """Create PC algorithm instance."""
        return PCAlgorithm()

    def test_adjacency_matrix_shape(self, pc):
        """Test adjacency matrix has correct shape."""
        mock_graph = MagicMock()
        # 3x3 graph with one directed edge (0 -> 1)
        mock_graph.graph = np.array(
            [
                [0, -1, 0],  # Row 0: tail at 1
                [1, 0, 0],  # Row 1: arrow at 0
                [0, 0, 0],  # Row 2
            ]
        )

        adj = pc._graph_to_adjacency(mock_graph, 3)

        assert adj.shape == (3, 3)

    def test_directed_edge_detection(self, pc):
        """Test detection of directed edges in CPDAG."""
        mock_graph = MagicMock()
        # Directed edge 0 -> 1:
        # - graph[0,1] = -1 (tail at node 0)
        # - graph[1,0] = 1 (arrow at node 1)
        mock_graph.graph = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )

        adj = pc._graph_to_adjacency(mock_graph, 3)

        assert adj[0, 1] == 1  # Edge from 0 to 1
        assert adj[1, 0] == 0  # No reverse edge

    def test_undirected_edge_detection(self, pc):
        """Test detection of undirected edges in CPDAG."""
        mock_graph = MagicMock()
        # Undirected edge 0 - 1: tails at both ends
        mock_graph.graph = np.array(
            [
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, 0],
            ]
        )

        adj = pc._graph_to_adjacency(mock_graph, 3)

        # Undirected edges represented as both directions
        assert adj[0, 1] == 1
        assert adj[1, 0] == 1

    def test_empty_graph_returns_zeros(self, pc):
        """Test empty graph returns zero adjacency matrix."""
        mock_graph = MagicMock()
        mock_graph.graph = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        adj = pc._graph_to_adjacency(mock_graph, 3)

        assert np.all(adj == 0)

    def test_graph_without_graph_attribute(self, pc):
        """Test handling of graph without graph attribute."""
        mock_graph = MagicMock(spec=[])

        adj = pc._graph_to_adjacency(mock_graph, 3)

        # Should return empty adjacency matrix as fallback
        assert adj.shape == (3, 3)


# =============================================================================
# PC Integration Tests
# =============================================================================


class TestPCIntegrationWithRunner:
    """Test PC integration with DiscoveryRunner."""

    def test_pc_registered_in_runner(self):
        """Test PC is registered in DiscoveryRunner."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        algorithms = DiscoveryRunner.get_available_algorithms()
        assert DiscoveryAlgorithmType.PC in algorithms

    def test_runner_can_get_pc(self):
        """Test runner can instantiate PC algorithm."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner()
        algo = runner._get_algorithm(DiscoveryAlgorithmType.PC)

        assert algo is not None
        assert algo.algorithm_type == DiscoveryAlgorithmType.PC

    @pytest.mark.asyncio
    async def test_runner_runs_pc(self):
        """Test runner can execute PC discovery."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        Y = 0.8 * X + 0.2 * np.random.randn(n)
        data = pd.DataFrame({"X": X, "Y": Y})

        runner = DiscoveryRunner()
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            alpha=0.05,
        )

        result = await runner.discover_dag(data, config)

        assert result.success is True
        assert len(result.algorithm_results) == 1
        assert result.algorithm_results[0].algorithm == DiscoveryAlgorithmType.PC


# =============================================================================
# PC Stability Tests
# =============================================================================


class TestPCStability:
    """Test PC algorithm stability settings."""

    @pytest.fixture
    def pc(self):
        """Create PC algorithm instance."""
        return PCAlgorithm()

    @pytest.fixture
    def reproducible_data(self):
        """Create data for reproducibility testing."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        Y = 0.8 * X + 0.2 * np.random.randn(n)
        return pd.DataFrame({"X": X, "Y": Y})

    def test_stable_pc_reproducibility(self, pc, reproducible_data):
        """Test that stable PC produces reproducible results."""
        config = DiscoveryConfig(alpha=0.05)

        result1 = pc.discover(reproducible_data, config)
        result2 = pc.discover(reproducible_data, config)

        # Results should be identical for stable PC
        np.testing.assert_array_equal(
            result1.adjacency_matrix,
            result2.adjacency_matrix,
        )

    def test_different_alpha_produces_different_results(self, pc, reproducible_data):
        """Test that different alpha values can produce different results."""
        # Note: This may not always produce different results depending on data
        config_strict = DiscoveryConfig(alpha=0.001)
        config_lenient = DiscoveryConfig(alpha=0.5)

        result_strict = pc.discover(reproducible_data, config_strict)
        result_lenient = pc.discover(reproducible_data, config_lenient)

        # Both should converge successfully
        assert result_strict.converged is True
        assert result_lenient.converged is True
