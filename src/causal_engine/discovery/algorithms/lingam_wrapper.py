"""
E2I Causal Analytics - LiNGAM Algorithm Wrappers
=================================================

Wrappers for LiNGAM (Linear Non-Gaussian Acyclic Model) algorithms.

LiNGAM algorithms exploit non-Gaussian distributions to identify
causal directions in linear systems. Unlike constraint-based methods
(PC, FCI), LiNGAM can uniquely identify the causal structure when:
- Relationships are linear
- Error terms are non-Gaussian (enables identifiability)
- Structure is acyclic

Two variants:
1. DirectLiNGAM: Iteratively identifies root variables via regression
2. ICA-LiNGAM: Uses Independent Component Analysis for unmixing

Author: E2I Causal Analytics Team
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)


class DirectLiNGAMAlgorithm(BaseDiscoveryAlgorithm):
    """DirectLiNGAM algorithm wrapper.

    DirectLiNGAM identifies the causal order by iteratively:
    1. Finding the variable with most independent residuals
    2. Removing its effect from other variables
    3. Repeating until all variables are ordered

    Advantages:
    - Fast and stable
    - No ICA optimization issues
    - Works well with moderate non-Gaussianity

    Assumptions:
    - Linear causal relationships
    - Non-Gaussian error terms
    - Acyclic structure
    """

    # Threshold for converting continuous weights to binary adjacency
    WEIGHT_THRESHOLD = 0.1

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return DirectLiNGAM algorithm type."""
        return DiscoveryAlgorithmType.DIRECT_LINGAM

    def supports_latent_confounders(self) -> bool:
        """DirectLiNGAM does not detect latent confounders."""
        return False

    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run DirectLiNGAM causal discovery.

        Args:
            data: Input DataFrame with variables as columns
            config: Discovery configuration

        Returns:
            AlgorithmResult with discovered DAG structure
        """
        self._validate_data(data)
        start_time = time.time()

        try:
            # Import DirectLiNGAM from lingam package
            from lingam import DirectLiNGAM

            # Prepare data as numpy array
            X = data.values
            node_names = list(data.columns)

            # Create and fit DirectLiNGAM model
            model = DirectLiNGAM(random_state=config.random_state)
            model.fit(X)

            # Get adjacency matrix (continuous weights)
            adj_weights = model.adjacency_matrix_

            # Convert to binary adjacency using threshold
            adj_matrix = self._threshold_adjacency(adj_weights)

            # Get causal order
            causal_order = model.causal_order_

            # Convert to edge list
            edge_list = self._adjacency_to_edge_list(adj_matrix, node_names)

            runtime = time.time() - start_time

            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=adj_matrix,
                edge_list=edge_list,
                runtime_seconds=runtime,
                converged=True,
                metadata={
                    "n_edges": len(edge_list),
                    "n_nodes": len(node_names),
                    "node_names": node_names,
                    "causal_order": [node_names[i] for i in causal_order],
                    "weight_threshold": self.WEIGHT_THRESHOLD,
                    "adjacency_weights": adj_weights.tolist(),
                    "supports_latent_confounders": False,
                    "assume_linear": True,
                    "assume_non_gaussian": True,
                },
            )

        except ImportError as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=np.zeros((len(data.columns), len(data.columns)), dtype=int),
                edge_list=[],
                runtime_seconds=runtime,
                converged=False,
                metadata={
                    "error": f"lingam package not installed: {e}",
                    "install_hint": "pip install lingam",
                },
            )

        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=np.zeros((len(data.columns), len(data.columns)), dtype=int),
                edge_list=[],
                runtime_seconds=runtime,
                converged=False,
                metadata={"error": str(e)},
            )

    def _threshold_adjacency(
        self,
        adj_weights: NDArray[np.float64],
        threshold: Optional[float] = None,
    ) -> NDArray[np.int_]:
        """Convert continuous adjacency weights to binary.

        Args:
            adj_weights: Continuous adjacency matrix from LiNGAM
            threshold: Threshold for edge inclusion (default: WEIGHT_THRESHOLD)

        Returns:
            Binary adjacency matrix
        """
        if threshold is None:
            threshold = self.WEIGHT_THRESHOLD

        adj_binary = (np.abs(adj_weights) > threshold).astype(int)
        return adj_binary

    def get_causal_effects(
        self,
        result: AlgorithmResult,
    ) -> Dict[Tuple[str, str], float]:
        """Extract causal effect strengths from result.

        Args:
            result: AlgorithmResult from DirectLiNGAM discovery

        Returns:
            Dict mapping (source, target) to effect strength
        """
        effects: Dict[Tuple[str, str], float] = {}
        adj_weights = result.metadata.get("adjacency_weights", [])
        node_names = result.metadata.get("node_names", [])

        if not adj_weights or not node_names:
            return effects

        adj_weights = np.array(adj_weights)
        n = len(node_names)

        for i in range(n):
            for j in range(n):
                if abs(adj_weights[i, j]) > self.WEIGHT_THRESHOLD:
                    effects[(node_names[i], node_names[j])] = adj_weights[i, j]

        return effects


class ICALiNGAMAlgorithm(BaseDiscoveryAlgorithm):
    """ICA-LiNGAM algorithm wrapper.

    ICA-LiNGAM uses Independent Component Analysis to:
    1. Estimate the unmixing matrix W such that S = WX
    2. Permute and scale W to find the causal structure
    3. Extract the adjacency matrix B where X = BX + E

    Advantages:
    - Theoretically grounded in ICA
    - Can handle stronger non-Gaussianity
    - Provides connection weights

    Limitations:
    - Can be sensitive to ICA optimization
    - Requires sufficient non-Gaussianity
    - May struggle with nearly Gaussian data
    """

    # Threshold for converting continuous weights to binary adjacency
    WEIGHT_THRESHOLD = 0.1

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return ICA-LiNGAM algorithm type."""
        return DiscoveryAlgorithmType.ICA_LINGAM

    def supports_latent_confounders(self) -> bool:
        """ICA-LiNGAM does not detect latent confounders."""
        return False

    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run ICA-LiNGAM causal discovery.

        Args:
            data: Input DataFrame with variables as columns
            config: Discovery configuration

        Returns:
            AlgorithmResult with discovered DAG structure
        """
        self._validate_data(data)
        start_time = time.time()

        try:
            # Import ICALiNGAM from lingam package
            from lingam import ICALiNGAM

            # Prepare data as numpy array
            X = data.values
            node_names = list(data.columns)

            # Create and fit ICA-LiNGAM model
            model = ICALiNGAM(random_state=config.random_state, max_iter=config.max_iter)
            model.fit(X)

            # Get adjacency matrix (continuous weights)
            adj_weights = model.adjacency_matrix_

            # Convert to binary adjacency using threshold
            adj_matrix = self._threshold_adjacency(adj_weights)

            # Get causal order
            causal_order = model.causal_order_

            # Convert to edge list
            edge_list = self._adjacency_to_edge_list(adj_matrix, node_names)

            runtime = time.time() - start_time

            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=adj_matrix,
                edge_list=edge_list,
                runtime_seconds=runtime,
                converged=True,
                metadata={
                    "n_edges": len(edge_list),
                    "n_nodes": len(node_names),
                    "node_names": node_names,
                    "causal_order": [node_names[i] for i in causal_order],
                    "weight_threshold": self.WEIGHT_THRESHOLD,
                    "adjacency_weights": adj_weights.tolist(),
                    "supports_latent_confounders": False,
                    "assume_linear": True,
                    "assume_non_gaussian": True,
                    "max_iter": config.max_iter,
                },
            )

        except ImportError as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=np.zeros((len(data.columns), len(data.columns)), dtype=int),
                edge_list=[],
                runtime_seconds=runtime,
                converged=False,
                metadata={
                    "error": f"lingam package not installed: {e}",
                    "install_hint": "pip install lingam",
                },
            )

        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=np.zeros((len(data.columns), len(data.columns)), dtype=int),
                edge_list=[],
                runtime_seconds=runtime,
                converged=False,
                metadata={"error": str(e)},
            )

    def _threshold_adjacency(
        self,
        adj_weights: NDArray[np.float64],
        threshold: Optional[float] = None,
    ) -> NDArray[np.int_]:
        """Convert continuous adjacency weights to binary.

        Args:
            adj_weights: Continuous adjacency matrix from ICA-LiNGAM
            threshold: Threshold for edge inclusion (default: WEIGHT_THRESHOLD)

        Returns:
            Binary adjacency matrix
        """
        if threshold is None:
            threshold = self.WEIGHT_THRESHOLD

        adj_binary = (np.abs(adj_weights) > threshold).astype(int)
        return adj_binary

    def get_mixing_matrix(
        self,
        result: AlgorithmResult,
    ) -> Optional[NDArray[np.float64]]:
        """Extract the estimated mixing matrix from result.

        The mixing matrix represents how independent components
        (causal mechanisms) mix to produce observed variables.

        Args:
            result: AlgorithmResult from ICA-LiNGAM discovery

        Returns:
            Mixing matrix if available, None otherwise
        """
        adj_weights = result.metadata.get("adjacency_weights")
        if adj_weights is None:
            return None

        # In ICA-LiNGAM, the adjacency matrix B is related to mixing by:
        # X = BX + E => X = (I - B)^{-1} E
        # So mixing matrix A = (I - B)^{-1}
        B = np.array(adj_weights)
        try:
            I = np.eye(B.shape[0])
            A = np.linalg.inv(I - B)
            return A
        except np.linalg.LinAlgError:
            return None

    def get_causal_effects(
        self,
        result: AlgorithmResult,
    ) -> Dict[Tuple[str, str], float]:
        """Extract causal effect strengths from result.

        Args:
            result: AlgorithmResult from ICA-LiNGAM discovery

        Returns:
            Dict mapping (source, target) to effect strength
        """
        effects: Dict[Tuple[str, str], float] = {}
        adj_weights = result.metadata.get("adjacency_weights", [])
        node_names = result.metadata.get("node_names", [])

        if not adj_weights or not node_names:
            return effects

        adj_weights = np.array(adj_weights)
        n = len(node_names)

        for i in range(n):
            for j in range(n):
                if abs(adj_weights[i, j]) > self.WEIGHT_THRESHOLD:
                    effects[(node_names[i], node_names[j])] = adj_weights[i, j]

        return effects
