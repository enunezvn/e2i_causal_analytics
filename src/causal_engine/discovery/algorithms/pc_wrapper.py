"""
E2I Causal Analytics - PC Algorithm Wrapper
============================================

Wrapper for Peter-Clark (PC) causal discovery algorithm.

PC is a constraint-based algorithm that uses conditional independence
tests to discover the causal structure from observational data.

Key features:
- Constraint-based: Uses CI tests to eliminate edges
- Sound and complete: Under faithfulness assumption
- Efficient pruning: Removes edges based on d-separation

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


class PCAlgorithm(BaseDiscoveryAlgorithm):
    """Peter-Clark (PC) algorithm wrapper.

    The PC algorithm works in two phases:
    1. Skeleton phase: Starts with complete graph, removes edges
       based on conditional independence tests
    2. Orientation phase: Orients edges using orientation rules
       (v-structures, Meek rules)

    Suitable for:
    - Small to medium datasets (CI tests are expensive)
    - When conditional independence is reliable
    - When causal sufficiency is assumed
    """

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return PC algorithm type."""
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        """PC assumes causal sufficiency (no hidden confounders)."""
        return False

    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run PC causal discovery.

        Args:
            data: Input DataFrame with variables as columns
            config: Discovery configuration

        Returns:
            AlgorithmResult with discovered CPDAG structure
        """
        self._validate_data(data)
        start_time = time.time()

        try:
            # Import causal-learn PC
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import (
                chisq,
                fisherz,
                gsq,
                kci,
                mv_fisherz,
            )

            # Prepare data as numpy array
            X = data.values
            node_names = list(data.columns)

            # Select independence test based on data type
            indep_test = self._select_independence_test(data, config)

            # Run PC algorithm
            cg = pc(
                X,
                alpha=config.alpha,
                indep_test=indep_test,
                stable=True,  # Stable PC for reproducibility
                uc_rule=0,  # Orientation rule (0 = standard PC rules)
                uc_priority=-1,  # No priority for unshielded colliders
            )

            # Extract adjacency matrix from the result
            adj_matrix = self._graph_to_adjacency(cg.G, len(node_names))

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
                    "alpha": config.alpha,
                    "indep_test": indep_test,
                    "n_edges": len(edge_list),
                    "n_nodes": len(node_names),
                    "node_names": node_names,
                    "n_ci_tests": getattr(cg, "no_of_ci_tests", None),
                },
            )

        except ImportError as e:
            raise ImportError(
                "causal-learn is required for PC. Install with: pip install causal-learn"
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
            Independence test name for causal-learn
        """
        # Check if data is continuous or discrete
        is_continuous = all(
            data[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            for col in data.columns
        )

        # Check for categorical columns
        n_unique_per_col = [data[col].nunique() for col in data.columns]
        is_discrete = all(n <= 10 for n in n_unique_per_col)

        if config.assume_gaussian or is_continuous:
            return "fisherz"  # Fisher's z-test for continuous data
        elif is_discrete:
            return "chisq"  # Chi-squared test for discrete data
        else:
            return "fisherz"  # Default to Fisher's z

    def _graph_to_adjacency(
        self,
        graph: Any,
        n_nodes: int,
    ) -> NDArray[np.int_]:
        """Convert causal-learn CausalGraph to adjacency matrix.

        Args:
            graph: causal-learn GeneralGraph object
            n_nodes: Number of nodes

        Returns:
            Adjacency matrix (n_nodes x n_nodes)
        """
        adj = np.zeros((n_nodes, n_nodes), dtype=int)

        try:
            # Get the graph matrix from causal-learn
            if hasattr(graph, "graph"):
                g = graph.graph
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        # Directed edge i -> j
                        if g[j, i] == 1 and g[i, j] == -1:
                            adj[i, j] = 1
                        # Undirected edge (in CPDAG)
                        elif g[i, j] == -1 and g[j, i] == -1:
                            adj[i, j] = 1
                            adj[j, i] = 1

        except Exception:
            pass

        return adj
