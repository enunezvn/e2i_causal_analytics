"""Tests for GES (Greedy Equivalence Search) Algorithm Wrapper.

Version: 1.0.0
Tests the GES algorithm wrapper for score-based causal discovery.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.algorithms.ges_wrapper import GESAlgorithm
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)

# =============================================================================
# GESAlgorithm Basic Properties Tests
# =============================================================================


class TestGESAlgorithmProperties:
    """Test GESAlgorithm basic properties."""

    def test_algorithm_type(self):
        """Test algorithm_type property returns GES."""
        algo = GESAlgorithm()
        assert algo.algorithm_type == DiscoveryAlgorithmType.GES

    def test_supports_latent_confounders(self):
        """Test GES reports no support for latent confounders."""
        algo = GESAlgorithm()
        assert algo.supports_latent_confounders() is False


# =============================================================================
# GES Data Validation Tests
# =============================================================================


class TestGESDataValidation:
    """Test data validation for GES algorithm."""

    @pytest.fixture
    def ges(self):
        """Create GES algorithm instance."""
        return GESAlgorithm()

    def test_empty_data_raises_error(self, ges):
        """Test that empty data raises ValueError."""
        config = DiscoveryConfig()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            ges.discover(empty_df, config)

    def test_missing_values_raises_error(self, ges):
        """Test that data with missing values raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="missing values"):
            ges.discover(df, config)

    def test_single_variable_raises_error(self, ges):
        """Test that single variable raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="at least 2 variables"):
            ges.discover(df, config)


# =============================================================================
# GES Discovery Tests
# =============================================================================


class TestGESDiscovery:
    """Test GES discovery functionality."""

    @pytest.fixture
    def ges(self):
        """Create GES algorithm instance."""
        return GESAlgorithm()

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

    def test_basic_discovery(self, ges, simple_data):
        """Test basic causal discovery with GES."""
        config = DiscoveryConfig()

        result = ges.discover(simple_data, config)

        assert isinstance(result, AlgorithmResult)
        assert result.algorithm == DiscoveryAlgorithmType.GES
        assert result.converged is True
        assert result.runtime_seconds > 0
        assert result.adjacency_matrix.shape == (3, 3)

    def test_discovery_with_config(self, ges, simple_data):
        """Test discovery respects configuration."""
        config = DiscoveryConfig(
            max_cond_vars=2,
            score_func="BIC",
        )

        result = ges.discover(simple_data, config)

        assert result.converged is True
        assert result.metadata.get("score_func") == "BIC"

    def test_edge_list_format(self, ges, simple_data):
        """Test that edge_list has correct format."""
        config = DiscoveryConfig()

        result = ges.discover(simple_data, config)

        for edge in result.edge_list:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert edge[0] in simple_data.columns
            assert edge[1] in simple_data.columns

    def test_node_names_preserved(self, ges, simple_data):
        """Test that node names are preserved in metadata."""
        config = DiscoveryConfig()

        result = ges.discover(simple_data, config)

        assert "node_names" in result.metadata
        assert set(result.metadata["node_names"]) == set(simple_data.columns)

    def test_n_edges_in_metadata(self, ges, simple_data):
        """Test that n_edges count is in metadata."""
        config = DiscoveryConfig()

        result = ges.discover(simple_data, config)

        assert "n_edges" in result.metadata
        assert "n_nodes" in result.metadata
        assert result.metadata["n_nodes"] == 3


# =============================================================================
# GES Score Function Tests
# =============================================================================


class TestGESScoreFunction:
    """Test score function selection."""

    @pytest.fixture
    def ges(self):
        """Create GES algorithm instance."""
        return GESAlgorithm()

    def test_bic_score_mapping(self, ges):
        """Test BIC score function mapping."""
        assert ges._get_score_func("BIC") == "local_score_BIC"
        assert ges._get_score_func("local_score_BIC") == "local_score_BIC"

    def test_bdeu_score_mapping(self, ges):
        """Test BDeu score function mapping."""
        assert ges._get_score_func("BDeu") == "local_score_BDeu"
        assert ges._get_score_func("local_score_BDeu") == "local_score_BDeu"

    def test_unknown_score_defaults_to_bic(self, ges):
        """Test unknown score function defaults to BIC."""
        assert ges._get_score_func("unknown") == "local_score_BIC"


# =============================================================================
# GES Error Handling Tests
# =============================================================================


class TestGESErrorHandling:
    """Test error handling in GES algorithm."""

    @pytest.fixture
    def ges(self):
        """Create GES algorithm instance."""
        return GESAlgorithm()

    def test_algorithm_failure_returns_failed_result(self, ges):
        """Test that algorithm failure returns non-converged result."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        # Patch to simulate failure
        with patch(
            "causallearn.search.ScoreBased.GES.ges",
            side_effect=RuntimeError("Algorithm failed"),
        ):
            result = ges.discover(df, config)

            assert result.converged is False
            assert "error" in result.metadata

    def test_import_error_raises_import_error(self, ges):
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
            # Add a blocking entry that raises ImportError
            sys.modules["causallearn"] = None
            sys.modules["causallearn.search"] = None
            sys.modules["causallearn.search.ScoreBased"] = None
            sys.modules["causallearn.search.ScoreBased.GES"] = None

            with pytest.raises(ImportError, match="causal-learn is required"):
                ges.discover(df, config)
        finally:
            # Restore original modules
            for mod in [
                "causallearn",
                "causallearn.search",
                "causallearn.search.ScoreBased",
                "causallearn.search.ScoreBased.GES",
            ]:
                sys.modules.pop(mod, None)
            sys.modules.update(original_modules)


# =============================================================================
# GES Adjacency Conversion Tests
# =============================================================================


class TestGESAdjacencyConversion:
    """Test adjacency matrix conversion for CPDAG."""

    @pytest.fixture
    def ges(self):
        """Create GES algorithm instance."""
        return GESAlgorithm()

    def test_adjacency_matrix_shape(self, ges):
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

        adj = ges._graph_to_adjacency(mock_graph, 3)

        assert adj.shape == (3, 3)

    def test_directed_edge_detection(self, ges):
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

        adj = ges._graph_to_adjacency(mock_graph, 3)

        assert adj[0, 1] == 1  # Edge from 0 to 1
        assert adj[1, 0] == 0  # No reverse edge

    def test_undirected_edge_detection(self, ges):
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

        adj = ges._graph_to_adjacency(mock_graph, 3)

        # Undirected edges represented as both directions
        assert adj[0, 1] == 1
        assert adj[1, 0] == 1

    def test_empty_graph_returns_zeros(self, ges):
        """Test empty graph returns zero adjacency matrix."""
        mock_graph = MagicMock()
        mock_graph.graph = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        adj = ges._graph_to_adjacency(mock_graph, 3)

        assert np.all(adj == 0)

    def test_graph_without_graph_attribute(self, ges):
        """Test handling of graph without graph attribute."""
        mock_graph = MagicMock(spec=[])

        adj = ges._graph_to_adjacency(mock_graph, 3)

        # Should return empty adjacency matrix as fallback
        assert adj.shape == (3, 3)


# =============================================================================
# GES Integration Tests
# =============================================================================


class TestGESIntegrationWithRunner:
    """Test GES integration with DiscoveryRunner."""

    def test_ges_registered_in_runner(self):
        """Test GES is registered in DiscoveryRunner."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        algorithms = DiscoveryRunner.get_available_algorithms()
        assert DiscoveryAlgorithmType.GES in algorithms

    def test_runner_can_get_ges(self):
        """Test runner can instantiate GES algorithm."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner()
        algo = runner._get_algorithm(DiscoveryAlgorithmType.GES)

        assert algo is not None
        assert algo.algorithm_type == DiscoveryAlgorithmType.GES

    @pytest.mark.asyncio
    async def test_runner_runs_ges(self):
        """Test runner can execute GES discovery."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        Y = 0.8 * X + 0.2 * np.random.randn(n)
        data = pd.DataFrame({"X": X, "Y": Y})

        runner = DiscoveryRunner()
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES],
        )

        result = await runner.discover_dag(data, config)

        assert result.success is True
        assert len(result.algorithm_results) == 1
        assert result.algorithm_results[0].algorithm == DiscoveryAlgorithmType.GES
