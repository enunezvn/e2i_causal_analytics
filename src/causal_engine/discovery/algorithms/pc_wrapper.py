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
from typing import Any

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
                chisq,  # noqa: F401
                fisherz,  # noqa: F401
                gsq,  # noqa: F401
                kci,  # noqa: F401
                mv_fisherz,  # noqa: F401
            )

            # Prepare data as numpy array
            X = data.values
            node_names = list(data.columns)

            # Select independence test based on data type
            indep_test = self._select_independence_test(data, config)

            # Run PC algorithm. When the caller supplies domain priors (GUIDED
            # discovery), translate them into a causal-learn BackgroundKnowledge
            # so edge orientation honors the known tiers / required / forbidden
            # edges instead of an arbitrary member of the equivalence class.
            pc_kwargs: dict[str, Any] = {
                "alpha": config.alpha,
                "indep_test": indep_test,
                "stable": True,  # Stable PC for reproducibility
                "uc_rule": 0,  # Orientation rule (0 = standard PC rules)
                "uc_priority": -1,  # No priority for unshielded colliders
                "show_progress": False,
            }
            prior = config.prior_knowledge
            guided = prior is not None and not prior.is_empty()
            if guided and prior is not None:
                from ..background_knowledge import build_background_knowledge

                pc_kwargs["node_names"] = node_names
                pc_kwargs["background_knowledge"] = build_background_knowledge(prior, node_names)

            cg = pc(X, **pc_kwargs)

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
                    "guided": guided,
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
        """Select the causal-learn conditional-independence test for ``data``.

        Every live frame is MIXED — 0/1 treatment and outcome flags next to
        continuous covariates — and every such frame selects ``fisherz``
        (all columns numeric -> ``is_continuous``). That is a MEASURED choice,
        not an oversight; the selection is pinned by
        ``tests/unit/test_causal_engine/test_discovery/test_structural_recovery.py``
        (``TestBinaryFramesGetAGaussianTest``, module docstring items 5 and 7):

        - 2026-09-11 (#2009, ``docs/demos/results/2026-09-11_pc_indep_test/``):
          on the structural-recovery DGP (binary T/Y, continuous covariates),
          n in {500, 2000} x seeds 1-10, guided production shape at B=20,
          driven through the real ``GraphBuilderNode`` — fisherz mean F1 0.933 /
          recall 0.929 / wall 1.15 s per point, versus ``chisq`` on 10-level
          quantile-binned covariates 0.832 / 0.764 / 3.7 s and ``gsq`` 0.872 /
          0.843 / 3.9 s. Paired per (n, seed) on SHD of the shipped DAG,
          fisherz is better on 10 points, tied on 9, worse on 1 against either
          alternative; the loss sits at n=500, where binning turns each
          conditional test into a sparse contingency table and PC drops true
          conf->T / conf->Y edges. ``kci`` converged on a single unbootstrapped
          frame in 83 s, i.e. ~29 min per production-shape point (1 + B=20
          fits) — ~1500x fisherz, out of bounds on time alone. No alternative
          beat fisherz on any recovery number or on wall-clock.
        - 2026-09-02 (item 5): on an ALL-binary variant of the same DGP, chisq
          F1 0.943 vs fisherz 0.953 — no gain either.

        The ``chisq`` branch below is unreachable for any frame PC can run on
        (it needs a non-numeric dtype, which the ``data.values`` path cannot
        consume). It is documented as a dead branch in item 5 and is left as
        is on purpose: removing or "fixing" it is a separate decision from
        pinning the measured selection.

        Args:
            data: Input data
            config: Discovery configuration

        Returns:
            Independence test name for causal-learn
        """
        # Check if data is continuous or discrete
        is_continuous = all(
            data[col].dtype in [np.float64, np.float32, np.int64, np.int32] for col in data.columns
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

        # Get the graph matrix from causal-learn.
        # NOTE (M-fo4): do NOT wrap this in `try/except: pass`. A genuine parse
        # failure must propagate to discover()'s outer except so the result is
        # marked converged=False instead of a silently-empty converged=True DAG.
        # A graph object without a `.graph` attribute is a legitimate empty
        # result and returns a zeros matrix via the guard below.
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

        return adj
