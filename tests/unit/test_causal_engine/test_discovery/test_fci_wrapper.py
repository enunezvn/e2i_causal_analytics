"""Tests for FCI (Fast Causal Inference) Algorithm Wrapper.

Version: 1.0.0
Tests the FCI algorithm wrapper for causal discovery with latent confounders.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.algorithms.fci_wrapper import FCIAlgorithm
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    EdgeType,
)


class TestFCIAlgorithmProperties:
    """Test FCIAlgorithm basic properties."""

    def test_algorithm_type(self):
        """Test algorithm_type property returns FCI."""
        algo = FCIAlgorithm()
        assert algo.algorithm_type == DiscoveryAlgorithmType.FCI

    def test_supports_latent_confounders(self):
        """Test FCI reports support for latent confounders."""
        algo = FCIAlgorithm()
        assert algo.supports_latent_confounders() is True


class TestFCIDataValidation:
    """Test data validation for FCI algorithm."""

    @pytest.fixture
    def fci(self):
        """Create FCI algorithm instance."""
        return FCIAlgorithm()

    def test_empty_data_raises_error(self, fci):
        """Test that empty data raises ValueError."""
        config = DiscoveryConfig()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            fci.discover(empty_df, config)

    def test_missing_values_raises_error(self, fci):
        """Test that data with missing values raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="missing values"):
            fci.discover(df, config)

    def test_single_variable_raises_error(self, fci):
        """Test that single variable raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="at least 2 variables"):
            fci.discover(df, config)


class TestFCIDiscovery:
    """Test FCI discovery functionality."""

    @pytest.fixture
    def fci(self):
        """Create FCI algorithm instance."""
        return FCIAlgorithm()

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

    @pytest.fixture
    def latent_confounder_data(self):
        """Create data with latent confounder structure.

        Structure: L -> X, L -> Y (L is latent)
        This should produce X <-> Y (bidirected edge).
        """
        np.random.seed(42)
        n = 300
        # Latent confounder L
        L = np.random.randn(n)
        # X and Y both caused by L
        X = 0.8 * L + 0.3 * np.random.randn(n)
        Y = 0.8 * L + 0.3 * np.random.randn(n)
        # Z caused by Y only (no latent confounder)
        Z = 0.8 * Y + 0.3 * np.random.randn(n)
        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_basic_discovery(self, fci, simple_data):
        """Test basic causal discovery with FCI."""
        config = DiscoveryConfig(alpha=0.05)

        result = fci.discover(simple_data, config)

        assert isinstance(result, AlgorithmResult)
        assert result.algorithm == DiscoveryAlgorithmType.FCI
        assert result.converged is True
        assert result.runtime_seconds > 0
        assert result.adjacency_matrix.shape == (3, 3)

    def test_discovery_with_config(self, fci, simple_data):
        """Test discovery respects configuration."""
        config = DiscoveryConfig(
            alpha=0.01,  # Stricter significance
            max_cond_vars=2,
        )

        result = fci.discover(simple_data, config)

        assert result.converged is True
        assert result.metadata.get("alpha") == 0.01

    def test_edge_list_format(self, fci, simple_data):
        """Test that edge_list has correct format."""
        config = DiscoveryConfig()

        result = fci.discover(simple_data, config)

        for edge in result.edge_list:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert edge[0] in simple_data.columns
            assert edge[1] in simple_data.columns

    def test_metadata_contains_edge_types(self, fci, simple_data):
        """Test that metadata contains edge type information."""
        config = DiscoveryConfig()

        result = fci.discover(simple_data, config)

        assert "n_directed_edges" in result.metadata
        assert "n_bidirected_edges" in result.metadata
        assert "n_undirected_edges" in result.metadata
        assert "supports_latent_confounders" in result.metadata
        assert result.metadata["supports_latent_confounders"] is True

    def test_node_names_preserved(self, fci, simple_data):
        """Test that node names are preserved in metadata."""
        config = DiscoveryConfig()

        result = fci.discover(simple_data, config)

        assert "node_names" in result.metadata
        assert set(result.metadata["node_names"]) == set(simple_data.columns)


class TestFCIIndependenceTest:
    """Test independence test selection."""

    @pytest.fixture
    def fci(self):
        """Create FCI algorithm instance."""
        return FCIAlgorithm()

    def test_continuous_data_uses_fisherz(self, fci):
        """Test continuous data uses Fisher's z-test."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        config = DiscoveryConfig()

        test_name = fci._select_independence_test(df, config)

        assert test_name == "fisherz"

    def test_discrete_data_uses_chisq(self, fci):
        """Test discrete data uses chi-squared test."""
        df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [0, 1, 0, 1]})
        config = DiscoveryConfig(assume_gaussian=False)

        test_name = fci._select_independence_test(df, config)

        assert test_name == "chisq"

    def test_gaussian_assumption_uses_fisherz(self, fci):
        """Test that assume_gaussian=True uses Fisher's z."""
        df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [0, 1, 0, 1]})
        config = DiscoveryConfig(assume_gaussian=True)

        test_name = fci._select_independence_test(df, config)

        assert test_name == "fisherz"


class TestFCIErrorHandling:
    """Test error handling in FCI algorithm."""

    @pytest.fixture
    def fci(self):
        """Create FCI algorithm instance."""
        return FCIAlgorithm()

    def test_algorithm_failure_returns_failed_result(self, fci):
        """Test that algorithm failure returns non-converged result."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        # Patch to simulate failure
        with patch(
            "causallearn.search.ConstraintBased.FCI.fci",
            side_effect=RuntimeError("Algorithm failed"),
        ):
            result = fci.discover(df, config)

            assert result.converged is False
            assert "error" in result.metadata


class TestFCIAdjacencyConversion:
    """Test adjacency matrix conversion for PAG."""

    @pytest.fixture
    def fci(self):
        """Create FCI algorithm instance."""
        return FCIAlgorithm()

    def test_adjacency_matrix_shape(self, fci):
        """Test adjacency matrix has correct shape."""
        # Create mock graph
        mock_graph = MagicMock()
        # 3x3 graph with one directed edge (0 -> 1)
        mock_graph.graph = np.array(
            [
                [0, -1, 0],  # Row 0: tail at 1
                [1, 0, 0],  # Row 1: arrow at 0
                [0, 0, 0],  # Row 2
            ]
        )

        adj, edge_types = fci._graph_to_adjacency_with_types(mock_graph, 3)

        assert adj.shape == (3, 3)

    def test_directed_edge_detection(self, fci):
        """Test detection of directed edges."""
        mock_graph = MagicMock()
        # Directed edge 0 -> 1 in PAG encoding:
        # - graph[0,1] = 1 (arrow at node 1 end)
        # - graph[1,0] = -1 (tail at node 0 end)
        mock_graph.graph = np.array(
            [
                [0, 1, 0],  # graph[0,1] = 1 (arrow at j=1)
                [-1, 0, 0],  # graph[1,0] = -1 (tail at i=0)
                [0, 0, 0],
            ]
        )

        adj, edge_types = fci._graph_to_adjacency_with_types(mock_graph, 3)

        assert adj[0, 1] == 1  # Edge from 0 to 1
        assert adj[1, 0] == 0  # No reverse edge

    def test_bidirected_edge_detection(self, fci):
        """Test detection of bidirected edges (latent confounders)."""
        mock_graph = MagicMock()
        # Bidirected edge 0 <-> 1: graph[0,1] = 1 (arrow), graph[1,0] = 1 (arrow)
        mock_graph.graph = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )

        adj, edge_types = fci._graph_to_adjacency_with_types(mock_graph, 3)

        assert adj[0, 1] == 1
        assert adj[1, 0] == 1
        assert edge_types.get((0, 1)) == EdgeType.BIDIRECTED

    def test_undirected_edge_detection(self, fci):
        """Test detection of undirected edges."""
        mock_graph = MagicMock()
        # Undirected edge 0 - 1: graph[0,1] = -1 (tail), graph[1,0] = -1 (tail)
        mock_graph.graph = np.array(
            [
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, 0],
            ]
        )

        adj, edge_types = fci._graph_to_adjacency_with_types(mock_graph, 3)

        assert adj[0, 1] == 1
        assert adj[1, 0] == 1
        assert edge_types.get((0, 1)) == EdgeType.UNDIRECTED


class TestFCIIntegrationWithRunner:
    """Test FCI integration with DiscoveryRunner."""

    def test_fci_registered_in_runner(self):
        """Test FCI is registered in DiscoveryRunner."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        algorithms = DiscoveryRunner.get_available_algorithms()
        assert DiscoveryAlgorithmType.FCI in algorithms

    def test_runner_can_get_fci(self):
        """Test runner can instantiate FCI algorithm."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner()
        algo = runner._get_algorithm(DiscoveryAlgorithmType.FCI)

        assert algo is not None
        assert algo.algorithm_type == DiscoveryAlgorithmType.FCI

    @pytest.mark.asyncio
    async def test_runner_runs_fci(self):
        """Test runner can execute FCI discovery."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        Y = 0.8 * X + 0.2 * np.random.randn(n)
        data = pd.DataFrame({"X": X, "Y": Y})

        runner = DiscoveryRunner()
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.FCI],
            alpha=0.05,
        )

        result = await runner.discover_dag(data, config)

        assert result.success is True
        assert len(result.algorithm_results) == 1
        assert result.algorithm_results[0].algorithm == DiscoveryAlgorithmType.FCI


class TestFCIBidirectedEdgeHelper:
    """Test helper method for extracting bidirected edges."""

    @pytest.fixture
    def fci(self):
        """Create FCI algorithm instance."""
        return FCIAlgorithm()

    def test_get_bidirected_edges_empty(self, fci):
        """Test extracting bidirected edges when none exist."""
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.FCI,
            adjacency_matrix=np.array([[0, 1], [0, 0]]),
            edge_list=[("X", "Y")],
            runtime_seconds=0.1,
            metadata={"edge_types": {"X->Y": "directed"}},
        )

        bidirected = fci.get_bidirected_edges(result)
        assert bidirected == []

    def test_get_bidirected_edges_present(self, fci):
        """Test extracting bidirected edges when they exist."""
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.FCI,
            adjacency_matrix=np.array([[0, 1], [1, 0]]),
            edge_list=[("X", "Y")],
            runtime_seconds=0.1,
            metadata={"edge_types": {"X->Y": "bidirected"}},
        )

        bidirected = fci.get_bidirected_edges(result)
        assert ("X", "Y") in bidirected
