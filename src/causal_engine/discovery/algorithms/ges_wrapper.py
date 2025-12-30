"""
E2I Causal Analytics - GES Algorithm Wrapper
=============================================

Wrapper for Greedy Equivalence Search (GES) causal discovery.

GES is a score-based algorithm that searches over the space of
equivalence classes of DAGs to find the structure that maximizes
a scoring criterion (BIC, BDeu).

Key features:
- Score-based: Uses BIC/BDeu to evaluate structures
- Equivalence class: Returns CPDAG (completed partially directed acyclic graph)
- Efficient: Greedy search with forward/backward phases

Author: E2I Causal Analytics Team
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)


class GESAlgorithm(BaseDiscoveryAlgorithm):
    """Greedy Equivalence Search algorithm wrapper.

    GES searches the space of equivalence classes (represented as CPDAGs)
    using a two-phase greedy approach:
    1. Forward phase: Adds edges that improve score
    2. Backward phase: Removes edges that improve score

    Suitable for:
    - Medium to large datasets
    - When causal sufficiency is assumed (no hidden confounders)
    - When you want a score-based approach
    """

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return GES algorithm type."""
        return DiscoveryAlgorithmType.GES

    def supports_latent_confounders(self) -> bool:
        """GES assumes causal sufficiency (no hidden confounders)."""
        return False

    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run GES causal discovery.

        Args:
            data: Input DataFrame with variables as columns
            config: Discovery configuration

        Returns:
            AlgorithmResult with discovered CPDAG structure
        """
        self._validate_data(data)
        start_time = time.time()

        try:
            # Import causal-learn GES
            from causallearn.search.ScoreBased.GES import ges

            # Prepare data as numpy array
            X = data.values
            node_names = list(data.columns)

            # Map score function name
            score_func = self._get_score_func(config.score_func)

            # Run GES
            record = ges(
                X,
                score_func=score_func,
                maxP=config.max_cond_vars,
            )

            # Extract adjacency matrix from the result
            # GES returns a GeneralGraph object with get_graph_edges() method
            adj_matrix = self._graph_to_adjacency(record["G"], len(node_names))

            # Convert to edge list
            edge_list = self._adjacency_to_edge_list(adj_matrix, node_names)

            runtime = time.time() - start_time

            return AlgorithmResult(
                algorithm=self.algorithm_type,
                adjacency_matrix=adj_matrix,
                edge_list=edge_list,
                runtime_seconds=runtime,
                converged=True,
                score=record.get("score"),
                metadata={
                    "score_func": config.score_func,
                    "n_edges": len(edge_list),
                    "n_nodes": len(node_names),
                    "node_names": node_names,
                },
            )

        except ImportError as e:
            raise ImportError(
                "causal-learn is required for GES. Install with: pip install causal-learn"
            ) from e
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

    def _get_score_func(self, score_name: str) -> str:
        """Map score function name to causal-learn format.

        Args:
            score_name: Score function name from config

        Returns:
            causal-learn score function string
        """
        score_map = {
            "local_score_BIC": "local_score_BIC",
            "local_score_BDeu": "local_score_BDeu",
            "BIC": "local_score_BIC",
            "BDeu": "local_score_BDeu",
        }
        return score_map.get(score_name, "local_score_BIC")

    def _graph_to_adjacency(
        self,
        graph: Any,
        n_nodes: int,
    ) -> NDArray[np.int_]:
        """Convert causal-learn GeneralGraph to adjacency matrix.

        Args:
            graph: causal-learn GeneralGraph object
            n_nodes: Number of nodes

        Returns:
            Adjacency matrix (n_nodes x n_nodes)
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=int)

        try:
            # Get the graph matrix from causal-learn
            # The graph object has a graph attribute that is the adjacency matrix
            if hasattr(graph, "graph"):
                # graph.graph is a numpy array where:
                # -1 means tail (no arrowhead), 1 means arrowhead
                # So i -> j means graph[j,i] = 1 and graph[i,j] = -1
                g = graph.graph
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if g[j, i] == 1 and g[i, j] == -1:
                            # Directed edge i -> j
                            adj[i, j] = 1
                        elif g[i, j] == -1 and g[j, i] == -1:
                            # Undirected edge i - j (in CPDAG)
                            # Represent as both directions for now
                            adj[i, j] = 1
                            adj[j, i] = 1
            elif hasattr(graph, "get_graph_edges"):
                # Alternative: Use edge list
                edges = graph.get_graph_edges()
                for edge in edges:
                    i = edge.get_node1().get_name()
                    j = edge.get_node2().get_name()
                    # Get indices if names are X1, X2, etc.
                    i_idx = int(i.replace("X", "")) - 1 if i.startswith("X") else int(i)
                    j_idx = int(j.replace("X", "")) - 1 if j.startswith("X") else int(j)
                    adj[i_idx, j_idx] = 1

        except Exception:
            # Fallback: return empty adjacency matrix
            pass

        return adj
