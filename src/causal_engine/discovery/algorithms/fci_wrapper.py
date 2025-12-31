"""
E2I Causal Analytics - FCI Algorithm Wrapper
=============================================

Wrapper for Fast Causal Inference (FCI) causal discovery algorithm.

FCI extends the PC algorithm to handle latent confounders and selection
bias. It outputs a Partial Ancestral Graph (PAG) that can represent:
- Directed edges: X -> Y (definite causal direction)
- Bidirected edges: X <-> Y (latent confounder present)
- Circle marks: X o-> Y (uncertain endpoint)

Key features:
- Handles latent confounders (unlike PC/GES)
- Outputs PAG with more edge types
- Sound under weaker assumptions than PC

Author: E2I Causal Analytics Team
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from numpy.typing import NDArray

from ..base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    EdgeType,
)


class FCIAlgorithm(BaseDiscoveryAlgorithm):
    """Fast Causal Inference (FCI) algorithm wrapper.

    FCI extends PC to handle latent confounders by:
    1. Running modified PC with possible ancestors (PAG)
    2. Orienting edges using FCI orientation rules (R1-R10)
    3. Outputting PAG with directed, bidirected, and circle marks

    Output edge types:
    - Directed (->): Definite causal direction
    - Bidirected (<->): Latent confounder between X and Y
    - Circle (o->): Uncertain endpoint (could be tail or arrowhead)

    Suitable for:
    - Data with potential hidden confounders
    - When causal sufficiency cannot be assumed
    - When you need to detect latent variables
    """

    # PAG edge mark constants from causal-learn
    # -1 = tail, 1 = arrowhead, 2 = circle
    TAIL = -1
    ARROW = 1
    CIRCLE = 2

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return FCI algorithm type."""
        return DiscoveryAlgorithmType.FCI

    def supports_latent_confounders(self) -> bool:
        """FCI can detect latent confounders via bidirected edges."""
        return True

    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run FCI causal discovery.

        Args:
            data: Input DataFrame with variables as columns
            config: Discovery configuration

        Returns:
            AlgorithmResult with discovered PAG structure
        """
        self._validate_data(data)
        start_time = time.time()

        try:
            # Prepare data as numpy array
            X = data.values
            node_names = list(data.columns)

            # Select independence test
            indep_test = self._select_independence_test(data, config)

            # Run FCI algorithm
            # FCI returns a tuple: (G, edges) where G is the PAG
            result = fci(
                X,
                independence_test_method=indep_test,
                alpha=config.alpha,
                depth=config.max_cond_vars if config.max_cond_vars else -1,
                verbose=False,
            )

            # Extract graph from result
            # fci returns (G, edges) tuple
            if isinstance(result, tuple):
                graph = result[0]
            else:
                graph = result

            # Extract adjacency matrix and edge types
            adj_matrix, edge_types = self._graph_to_adjacency_with_types(
                graph, len(node_names)
            )

            # Convert to edge list
            edge_list = self._adjacency_to_edge_list(adj_matrix, node_names)

            # Count edge types for metadata
            n_directed = sum(1 for et in edge_types.values() if et == EdgeType.DIRECTED)
            n_bidirected = sum(
                1 for et in edge_types.values() if et == EdgeType.BIDIRECTED
            )
            n_undirected = sum(
                1 for et in edge_types.values() if et == EdgeType.UNDIRECTED
            )

            runtime = time.time() - start_time

            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=adj_matrix,
                edge_list=edge_list,
                runtime_seconds=runtime,
                converged=True,
                metadata={
                    "alpha": config.alpha,
                    "indep_test": indep_test,
                    "n_edges": len(edge_list),
                    "n_nodes": len(node_names),
                    "node_names": node_names,
                    "n_directed_edges": n_directed,
                    "n_bidirected_edges": n_bidirected,
                    "n_undirected_edges": n_undirected,
                    "edge_types": {
                        f"{s}->{t}": et.value for (s, t), et in edge_types.items()
                    },
                    "supports_latent_confounders": True,
                },
            )

        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=np.zeros(
                    (len(data.columns), len(data.columns)), dtype=int
                ),
                edge_list=[],
                runtime_seconds=runtime,
                converged=False,
                metadata={"error": str(e)},
            )

    def _select_independence_test(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> str:
        """Select appropriate independence test based on data characteristics.

        Args:
            data: Input data
            config: Discovery configuration

        Returns:
            Independence test name for causal-learn FCI
        """
        # If user explicitly wants Gaussian assumption, use Fisher's z
        if config.assume_gaussian:
            return "fisherz"

        # Check if data is float (treat as continuous regardless of cardinality)
        is_float = all(
            data[col].dtype in [np.float64, np.float32]
            for col in data.columns
        )

        # Float data is treated as continuous
        if is_float:
            return "fisherz"

        # Check for integer columns with low cardinality (treat as discrete)
        is_integer = all(
            data[col].dtype in [np.int64, np.int32]
            for col in data.columns
        )
        n_unique_per_col = [data[col].nunique() for col in data.columns]
        is_low_cardinality = all(n <= 10 for n in n_unique_per_col)

        # Integer data with low cardinality is treated as discrete
        if is_integer and is_low_cardinality:
            return "chisq"

        # Default to Fisher's z for other cases
        return "fisherz"

    def _graph_to_adjacency_with_types(
        self,
        graph: Any,
        n_nodes: int,
    ) -> Tuple[NDArray[np.int_], Dict[Tuple[int, int], EdgeType]]:
        """Convert causal-learn PAG to adjacency matrix with edge types.

        PAG edge encoding in causal-learn:
        - graph[i,j] = -1, graph[j,i] = 1: i -> j (directed)
        - graph[i,j] = 1, graph[j,i] = 1: i <-> j (bidirected, latent confounder)
        - graph[i,j] = -1, graph[j,i] = -1: i - j (undirected)
        - graph[i,j] = 2, graph[j,i] = 1: i o-> j (circle to arrow)
        - graph[i,j] = 2, graph[j,i] = -1: i o- j (circle to tail)
        - graph[i,j] = 2, graph[j,i] = 2: i o-o j (both circles)

        Args:
            graph: causal-learn GeneralGraph (PAG)
            n_nodes: Number of nodes

        Returns:
            Tuple of (adjacency matrix, edge type dict)
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=int)
        edge_types: Dict[Tuple[int, int], EdgeType] = {}

        try:
            if hasattr(graph, "graph"):
                g = graph.graph
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        mark_ij = g[i, j]  # Mark at j end of edge from i
                        mark_ji = g[j, i]  # Mark at i end of edge from j

                        if mark_ij == 0 and mark_ji == 0:
                            # No edge
                            continue

                        # Directed edge i -> j
                        if mark_ji == self.TAIL and mark_ij == self.ARROW:
                            adj[i, j] = 1
                            edge_types[(i, j)] = EdgeType.DIRECTED

                        # Directed edge j -> i
                        elif mark_ij == self.TAIL and mark_ji == self.ARROW:
                            adj[j, i] = 1
                            edge_types[(j, i)] = EdgeType.DIRECTED

                        # Bidirected edge i <-> j (latent confounder)
                        elif mark_ij == self.ARROW and mark_ji == self.ARROW:
                            adj[i, j] = 1
                            adj[j, i] = 1
                            edge_types[(i, j)] = EdgeType.BIDIRECTED

                        # Undirected edge i - j
                        elif mark_ij == self.TAIL and mark_ji == self.TAIL:
                            adj[i, j] = 1
                            adj[j, i] = 1
                            edge_types[(i, j)] = EdgeType.UNDIRECTED

                        # Circle marks - treat conservatively
                        elif self.CIRCLE in (mark_ij, mark_ji):
                            # Circle to arrow (o->) - treat as possible directed
                            if mark_ij == self.ARROW and mark_ji == self.CIRCLE:
                                adj[i, j] = 1
                                edge_types[(i, j)] = EdgeType.UNDIRECTED  # Uncertain
                            elif mark_ji == self.ARROW and mark_ij == self.CIRCLE:
                                adj[j, i] = 1
                                edge_types[(j, i)] = EdgeType.UNDIRECTED  # Uncertain
                            else:
                                # Other circle combinations - treat as undirected
                                adj[i, j] = 1
                                adj[j, i] = 1
                                edge_types[(i, j)] = EdgeType.UNDIRECTED

        except Exception:
            pass

        return adj, edge_types

    def _graph_to_adjacency(
        self,
        graph: Any,
        n_nodes: int,
    ) -> NDArray[np.int_]:
        """Convert causal-learn PAG to adjacency matrix (ignoring edge types).

        Args:
            graph: causal-learn GeneralGraph (PAG)
            n_nodes: Number of nodes

        Returns:
            Adjacency matrix (n_nodes x n_nodes)
        """
        adj, _ = self._graph_to_adjacency_with_types(graph, n_nodes)
        return adj

    def get_bidirected_edges(
        self,
        result: AlgorithmResult,
    ) -> List[Tuple[str, str]]:
        """Extract bidirected edges (latent confounders) from result.

        Args:
            result: AlgorithmResult from FCI discovery

        Returns:
            List of (X, Y) pairs with latent confounders
        """
        bidirected = []
        edge_types = result.metadata.get("edge_types", {})
        for edge_str, edge_type in edge_types.items():
            if edge_type == EdgeType.BIDIRECTED.value:
                # Parse "X->Y" format
                parts = edge_str.split("->")
                if len(parts) == 2:
                    bidirected.append((parts[0], parts[1]))
        return bidirected
