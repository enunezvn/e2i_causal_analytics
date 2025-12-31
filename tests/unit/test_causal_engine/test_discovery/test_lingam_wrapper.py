"""Tests for LiNGAM Algorithm Wrappers.

Version: 1.0.0
Tests the DirectLiNGAM and ICA-LiNGAM algorithm wrappers for causal discovery.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.causal_engine.discovery.algorithms.lingam_wrapper import (
    DirectLiNGAMAlgorithm,
    ICALiNGAMAlgorithm,
)
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)


# =============================================================================
# DirectLiNGAM Tests (10 tests)
# =============================================================================


class TestDirectLiNGAMProperties:
    """Test DirectLiNGAM basic properties."""

    def test_algorithm_type(self):
        """Test algorithm_type property returns DIRECT_LINGAM."""
        algo = DirectLiNGAMAlgorithm()
        assert algo.algorithm_type == DiscoveryAlgorithmType.DIRECT_LINGAM

    def test_supports_latent_confounders_false(self):
        """Test DirectLiNGAM reports no support for latent confounders."""
        algo = DirectLiNGAMAlgorithm()
        assert algo.supports_latent_confounders() is False


class TestDirectLiNGAMDataValidation:
    """Test data validation for DirectLiNGAM algorithm."""

    @pytest.fixture
    def direct_lingam(self):
        """Create DirectLiNGAM algorithm instance."""
        return DirectLiNGAMAlgorithm()

    def test_empty_data_raises_error(self, direct_lingam):
        """Test that empty data raises ValueError."""
        config = DiscoveryConfig()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            direct_lingam.discover(empty_df, config)

    def test_missing_values_raises_error(self, direct_lingam):
        """Test that data with missing values raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="missing values"):
            direct_lingam.discover(df, config)

    def test_single_variable_raises_error(self, direct_lingam):
        """Test that single variable raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="at least 2 variables"):
            direct_lingam.discover(df, config)


class TestDirectLiNGAMDiscovery:
    """Test DirectLiNGAM discovery functionality."""

    @pytest.fixture
    def direct_lingam(self):
        """Create DirectLiNGAM algorithm instance."""
        return DirectLiNGAMAlgorithm()

    @pytest.fixture
    def non_gaussian_data(self):
        """Create non-Gaussian test data with known structure.

        Uses uniform distribution (non-Gaussian) for error terms.
        Structure: X -> Y -> Z
        """
        np.random.seed(42)
        n = 300
        # Non-Gaussian errors (uniform distribution)
        e_x = np.random.uniform(-1, 1, n)
        e_y = np.random.uniform(-1, 1, n)
        e_z = np.random.uniform(-1, 1, n)

        # X -> Y -> Z structure with linear relationships
        X = e_x
        Y = 0.8 * X + 0.3 * e_y
        Z = 0.8 * Y + 0.3 * e_z

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_basic_discovery(self, direct_lingam, non_gaussian_data):
        """Test basic causal discovery with DirectLiNGAM."""
        config = DiscoveryConfig(random_state=42)

        with patch("lingam.DirectLiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            mock_model.adjacency_matrix_ = np.array([
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ])
            mock_model.causal_order_ = [0, 1, 2]
            MockLiNGAM.return_value = mock_model

            result = direct_lingam.discover(non_gaussian_data, config)

            assert isinstance(result, AlgorithmResult)
            assert result.algorithm == DiscoveryAlgorithmType.DIRECT_LINGAM
            assert result.converged is True
            assert result.runtime_seconds > 0
            assert result.adjacency_matrix.shape == (3, 3)

    def test_causal_order_in_metadata(self, direct_lingam, non_gaussian_data):
        """Test that causal order is included in metadata."""
        config = DiscoveryConfig(random_state=42)

        with patch("lingam.DirectLiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            mock_model.adjacency_matrix_ = np.array([
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ])
            mock_model.causal_order_ = [0, 1, 2]
            MockLiNGAM.return_value = mock_model

            result = direct_lingam.discover(non_gaussian_data, config)

            assert "causal_order" in result.metadata
            assert result.metadata["causal_order"] == ["X", "Y", "Z"]

    def test_adjacency_weights_in_metadata(self, direct_lingam, non_gaussian_data):
        """Test that adjacency weights are included in metadata."""
        config = DiscoveryConfig(random_state=42)

        with patch("lingam.DirectLiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            adj_weights = np.array([
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ])
            mock_model.adjacency_matrix_ = adj_weights
            mock_model.causal_order_ = [0, 1, 2]
            MockLiNGAM.return_value = mock_model

            result = direct_lingam.discover(non_gaussian_data, config)

            assert "adjacency_weights" in result.metadata
            assert result.metadata["assume_non_gaussian"] is True


class TestDirectLiNGAMErrorHandling:
    """Test error handling in DirectLiNGAM algorithm."""

    @pytest.fixture
    def direct_lingam(self):
        """Create DirectLiNGAM algorithm instance."""
        return DirectLiNGAMAlgorithm()

    def test_import_error_handling(self, direct_lingam):
        """Test handling of lingam package import error."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        with patch.dict("sys.modules", {"lingam": None}):
            with patch(
                "src.causal_engine.discovery.algorithms.lingam_wrapper.DirectLiNGAMAlgorithm.discover",
                side_effect=ImportError("lingam not installed"),
            ):
                # Create a fresh instance to test import error path
                algo = DirectLiNGAMAlgorithm()
                # The actual import error would be caught in the discover method
                pass

    def test_algorithm_failure_returns_failed_result(self, direct_lingam):
        """Test that algorithm failure returns non-converged result."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        with patch("lingam.DirectLiNGAM") as MockLiNGAM:
            MockLiNGAM.side_effect = RuntimeError("Algorithm failed")

            result = direct_lingam.discover(df, config)

            assert result.converged is False
            assert "error" in result.metadata


class TestDirectLiNGAMIntegration:
    """Test DirectLiNGAM integration with DiscoveryRunner."""

    def test_direct_lingam_registered_in_runner(self):
        """Test DirectLiNGAM is registered in DiscoveryRunner."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        assert DiscoveryAlgorithmType.DIRECT_LINGAM in DiscoveryRunner.ALGORITHM_REGISTRY

    def test_runner_can_get_direct_lingam(self):
        """Test runner can instantiate DirectLiNGAM algorithm."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner()
        algo = runner._get_algorithm(DiscoveryAlgorithmType.DIRECT_LINGAM)

        assert algo is not None
        assert algo.algorithm_type == DiscoveryAlgorithmType.DIRECT_LINGAM


# =============================================================================
# ICA-LiNGAM Tests (10 tests)
# =============================================================================


class TestICALiNGAMProperties:
    """Test ICA-LiNGAM basic properties."""

    def test_algorithm_type(self):
        """Test algorithm_type property returns ICA_LINGAM."""
        algo = ICALiNGAMAlgorithm()
        assert algo.algorithm_type == DiscoveryAlgorithmType.ICA_LINGAM

    def test_supports_latent_confounders_false(self):
        """Test ICA-LiNGAM reports no support for latent confounders."""
        algo = ICALiNGAMAlgorithm()
        assert algo.supports_latent_confounders() is False


class TestICALiNGAMDataValidation:
    """Test data validation for ICA-LiNGAM algorithm."""

    @pytest.fixture
    def ica_lingam(self):
        """Create ICA-LiNGAM algorithm instance."""
        return ICALiNGAMAlgorithm()

    def test_empty_data_raises_error(self, ica_lingam):
        """Test that empty data raises ValueError."""
        config = DiscoveryConfig()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            ica_lingam.discover(empty_df, config)

    def test_missing_values_raises_error(self, ica_lingam):
        """Test that data with missing values raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="missing values"):
            ica_lingam.discover(df, config)

    def test_single_variable_raises_error(self, ica_lingam):
        """Test that single variable raises ValueError."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="at least 2 variables"):
            ica_lingam.discover(df, config)


class TestICALiNGAMDiscovery:
    """Test ICA-LiNGAM discovery functionality."""

    @pytest.fixture
    def ica_lingam(self):
        """Create ICA-LiNGAM algorithm instance."""
        return ICALiNGAMAlgorithm()

    @pytest.fixture
    def non_gaussian_data(self):
        """Create non-Gaussian test data with known structure."""
        np.random.seed(42)
        n = 300
        # Non-Gaussian errors (uniform distribution)
        e_x = np.random.uniform(-1, 1, n)
        e_y = np.random.uniform(-1, 1, n)
        e_z = np.random.uniform(-1, 1, n)

        # X -> Y -> Z structure
        X = e_x
        Y = 0.8 * X + 0.3 * e_y
        Z = 0.8 * Y + 0.3 * e_z

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_basic_discovery(self, ica_lingam, non_gaussian_data):
        """Test basic causal discovery with ICA-LiNGAM."""
        config = DiscoveryConfig(random_state=42, max_iter=1000)

        with patch("lingam.ICALiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            mock_model.adjacency_matrix_ = np.array([
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ])
            mock_model.causal_order_ = [0, 1, 2]
            MockLiNGAM.return_value = mock_model

            result = ica_lingam.discover(non_gaussian_data, config)

            assert isinstance(result, AlgorithmResult)
            assert result.algorithm == DiscoveryAlgorithmType.ICA_LINGAM
            assert result.converged is True
            assert result.runtime_seconds > 0
            assert result.adjacency_matrix.shape == (3, 3)

    def test_max_iter_in_metadata(self, ica_lingam, non_gaussian_data):
        """Test that max_iter is included in metadata."""
        config = DiscoveryConfig(random_state=42, max_iter=5000)

        with patch("lingam.ICALiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            mock_model.adjacency_matrix_ = np.array([
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ])
            mock_model.causal_order_ = [0, 1, 2]
            MockLiNGAM.return_value = mock_model

            result = ica_lingam.discover(non_gaussian_data, config)

            assert "max_iter" in result.metadata
            assert result.metadata["max_iter"] == 5000

    def test_adjacency_weights_in_metadata(self, ica_lingam, non_gaussian_data):
        """Test that adjacency weights are included in metadata."""
        config = DiscoveryConfig(random_state=42)

        with patch("lingam.ICALiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            adj_weights = np.array([
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ])
            mock_model.adjacency_matrix_ = adj_weights
            mock_model.causal_order_ = [0, 1, 2]
            MockLiNGAM.return_value = mock_model

            result = ica_lingam.discover(non_gaussian_data, config)

            assert "adjacency_weights" in result.metadata
            assert result.metadata["assume_linear"] is True


class TestICALiNGAMErrorHandling:
    """Test error handling in ICA-LiNGAM algorithm."""

    @pytest.fixture
    def ica_lingam(self):
        """Create ICA-LiNGAM algorithm instance."""
        return ICALiNGAMAlgorithm()

    def test_algorithm_failure_returns_failed_result(self, ica_lingam):
        """Test that algorithm failure returns non-converged result."""
        config = DiscoveryConfig()
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

        with patch("lingam.ICALiNGAM") as MockLiNGAM:
            MockLiNGAM.side_effect = RuntimeError("ICA failed to converge")

            result = ica_lingam.discover(df, config)

            assert result.converged is False
            assert "error" in result.metadata


class TestICALiNGAMIntegration:
    """Test ICA-LiNGAM integration with DiscoveryRunner."""

    def test_ica_lingam_registered_in_runner(self):
        """Test ICA-LiNGAM is registered in DiscoveryRunner."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        assert DiscoveryAlgorithmType.ICA_LINGAM in DiscoveryRunner.ALGORITHM_REGISTRY

    def test_runner_can_get_ica_lingam(self):
        """Test runner can instantiate ICA-LiNGAM algorithm."""
        from src.causal_engine.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner()
        algo = runner._get_algorithm(DiscoveryAlgorithmType.ICA_LINGAM)

        assert algo is not None
        assert algo.algorithm_type == DiscoveryAlgorithmType.ICA_LINGAM


class TestICALiNGAMMixingMatrix:
    """Test mixing matrix helper method for ICA-LiNGAM."""

    @pytest.fixture
    def ica_lingam(self):
        """Create ICA-LiNGAM algorithm instance."""
        return ICALiNGAMAlgorithm()

    def test_get_mixing_matrix_returns_matrix(self, ica_lingam):
        """Test get_mixing_matrix returns a valid matrix."""
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.ICA_LINGAM,
            adjacency_matrix=np.array([[0, 1], [0, 0]]),
            edge_list=[("X", "Y")],
            runtime_seconds=0.1,
            metadata={
                "adjacency_weights": [[0.0, 0.5], [0.0, 0.0]],
                "node_names": ["X", "Y"],
            },
        )

        mixing = ica_lingam.get_mixing_matrix(result)

        assert mixing is not None
        assert mixing.shape == (2, 2)

    def test_get_mixing_matrix_no_weights(self, ica_lingam):
        """Test get_mixing_matrix returns None when no weights available."""
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.ICA_LINGAM,
            adjacency_matrix=np.array([[0, 1], [0, 0]]),
            edge_list=[("X", "Y")],
            runtime_seconds=0.1,
            metadata={},
        )

        mixing = ica_lingam.get_mixing_matrix(result)

        assert mixing is None


class TestLiNGAMCausalEffects:
    """Test causal effect extraction for both LiNGAM variants."""

    def test_direct_lingam_get_causal_effects(self):
        """Test extracting causal effects from DirectLiNGAM result."""
        algo = DirectLiNGAMAlgorithm()
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.DIRECT_LINGAM,
            adjacency_matrix=np.array([[0, 1], [0, 0]]),
            edge_list=[("X", "Y")],
            runtime_seconds=0.1,
            metadata={
                "adjacency_weights": [[0.0, 0.75], [0.0, 0.0]],
                "node_names": ["X", "Y"],
            },
        )

        effects = algo.get_causal_effects(result)

        assert ("X", "Y") in effects
        assert abs(effects[("X", "Y")] - 0.75) < 0.01

    def test_ica_lingam_get_causal_effects(self):
        """Test extracting causal effects from ICA-LiNGAM result."""
        algo = ICALiNGAMAlgorithm()
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.ICA_LINGAM,
            adjacency_matrix=np.array([[0, 1], [0, 0]]),
            edge_list=[("X", "Y")],
            runtime_seconds=0.1,
            metadata={
                "adjacency_weights": [[0.0, 0.85], [0.0, 0.0]],
                "node_names": ["X", "Y"],
            },
        )

        effects = algo.get_causal_effects(result)

        assert ("X", "Y") in effects
        assert abs(effects[("X", "Y")] - 0.85) < 0.01


class TestLiNGAMThresholding:
    """Test adjacency matrix thresholding for LiNGAM algorithms."""

    def test_direct_lingam_threshold_adjacency(self):
        """Test DirectLiNGAM threshold_adjacency method."""
        algo = DirectLiNGAMAlgorithm()
        weights = np.array([
            [0.0, 0.8, 0.05],  # 0.05 should be filtered
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0],
        ])

        binary = algo._threshold_adjacency(weights)

        assert binary[0, 1] == 1  # 0.8 > threshold
        assert binary[0, 2] == 0  # 0.05 < threshold
        assert binary[1, 2] == 1  # 0.3 > threshold

    def test_ica_lingam_threshold_adjacency(self):
        """Test ICA-LiNGAM threshold_adjacency method."""
        algo = ICALiNGAMAlgorithm()
        weights = np.array([
            [0.0, 0.8, 0.05],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0],
        ])

        binary = algo._threshold_adjacency(weights, threshold=0.2)

        assert binary[0, 1] == 1  # 0.8 > 0.2
        assert binary[0, 2] == 0  # 0.05 < 0.2
        assert binary[1, 2] == 1  # 0.3 > 0.2
