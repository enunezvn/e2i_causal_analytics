"""
E2I Causal Analytics - Causal Discovery Runner
===============================================

Orchestrates causal structure learning with multi-algorithm ensemble.

The DiscoveryRunner:
1. Runs multiple discovery algorithms in parallel
2. Combines results using ensemble voting
3. Computes confidence scores per edge
4. Returns a unified DAG with edge metadata

Author: E2I Causal Analytics Team
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
from uuid import UUID

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .algorithms import (
    DirectLiNGAMAlgorithm,
    FCIAlgorithm,
    GESAlgorithm,
    ICALiNGAMAlgorithm,
    PCAlgorithm,
)
from .base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
)

if TYPE_CHECKING:
    from .observability import DiscoveryTracer


def _run_algorithm_in_process(
    algo_class: Type["BaseDiscoveryAlgorithm"],
    data_dict: Dict[str, Any],
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single algorithm in a separate process.

    This function is defined at module level to enable pickling for ProcessPoolExecutor.

    Args:
        algo_class: The algorithm class to instantiate
        data_dict: DataFrame as dict for serialization
        config_dict: Config as dict for serialization

    Returns:
        AlgorithmResult as dict for serialization
    """
    import pandas as pd

    from .base import DiscoveryAlgorithmType, DiscoveryConfig

    # Reconstruct objects from dicts
    data = pd.DataFrame(data_dict)
    config = DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType(a) for a in config_dict["algorithms"]],
        alpha=config_dict["alpha"],
        max_cond_vars=config_dict.get("max_cond_vars"),
        ensemble_threshold=config_dict["ensemble_threshold"],
        max_iter=config_dict["max_iter"],
        random_state=config_dict["random_state"],
        score_func=config_dict["score_func"],
        assume_linear=config_dict["assume_linear"],
        assume_gaussian=config_dict["assume_gaussian"],
    )

    # Run algorithm
    algorithm = algo_class()
    result = algorithm.discover(data, config)

    # Convert result to dict for serialization
    return {
        "algorithm": result.algorithm.value,
        "adjacency_matrix": result.adjacency_matrix.tolist(),
        "edge_list": [
            (e.source, e.target, e.edge_type.value, e.confidence) for e in result.edge_list
        ],
        "runtime_seconds": result.runtime_seconds,
        "converged": result.converged,
        "metadata": result.metadata,
    }


class DiscoveryRunner:
    """Orchestrates causal structure learning with multi-algorithm ensemble.

    The runner manages multiple discovery algorithms and combines their
    results into a single ensemble DAG with confidence scores.

    Example:
        >>> runner = DiscoveryRunner()
        >>> config = DiscoveryConfig(
        ...     algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
        ...     ensemble_threshold=0.5,
        ... )
        >>> result = await runner.discover_dag(data, config)
        >>> print(f"Found {result.n_edges} edges with {result.algorithm_agreement:.2%} agreement")
    """

    # Registry of available algorithms
    ALGORITHM_REGISTRY: Dict[DiscoveryAlgorithmType, Type[BaseDiscoveryAlgorithm]] = {
        DiscoveryAlgorithmType.GES: GESAlgorithm,
        DiscoveryAlgorithmType.PC: PCAlgorithm,
        DiscoveryAlgorithmType.FCI: FCIAlgorithm,
        DiscoveryAlgorithmType.DIRECT_LINGAM: DirectLiNGAMAlgorithm,
        DiscoveryAlgorithmType.ICA_LINGAM: ICALiNGAMAlgorithm,
    }

    def __init__(
        self,
        max_workers: int = 4,
        timeout_seconds: float = 300.0,
        tracer: Optional["DiscoveryTracer"] = None,
        enable_tracing: bool = True,
    ):
        """Initialize DiscoveryRunner.

        Args:
            max_workers: Maximum parallel workers for algorithm execution
            timeout_seconds: Timeout for each algorithm
            tracer: Optional DiscoveryTracer for Opik observability
            enable_tracing: Whether to enable tracing (default True)
        """
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self._algorithms: Dict[DiscoveryAlgorithmType, BaseDiscoveryAlgorithm] = {}
        self._tracer = tracer
        self._enable_tracing = enable_tracing

        # Initialize tracer if enabled and not provided
        if enable_tracing and tracer is None:
            self._init_tracer()

    def _init_tracer(self) -> None:
        """Initialize DiscoveryTracer for observability."""
        try:
            from .observability import get_discovery_tracer

            self._tracer = get_discovery_tracer()
            logger.debug("DiscoveryTracer initialized for runner")
        except ImportError:
            logger.warning("DiscoveryTracer not available, tracing disabled")
            self._enable_tracing = False
        except Exception as e:
            logger.warning(f"Failed to initialize DiscoveryTracer: {e}")
            self._enable_tracing = False

    def _get_algorithm(self, algo_type: DiscoveryAlgorithmType) -> BaseDiscoveryAlgorithm:
        """Get or create algorithm instance.

        Args:
            algo_type: Algorithm type to get

        Returns:
            Algorithm instance

        Raises:
            ValueError: If algorithm type is not supported
        """
        if algo_type not in self._algorithms:
            if algo_type not in self.ALGORITHM_REGISTRY:
                raise ValueError(
                    f"Algorithm {algo_type.value} not supported. "
                    f"Available: {list(self.ALGORITHM_REGISTRY.keys())}"
                )
            self._algorithms[algo_type] = self.ALGORITHM_REGISTRY[algo_type]()

        return self._algorithms[algo_type]

    async def discover_dag(
        self,
        data: pd.DataFrame,
        config: Optional[DiscoveryConfig] = None,
        session_id: Optional[UUID] = None,
    ) -> DiscoveryResult:
        """Run causal discovery with ensemble of algorithms.

        Args:
            data: Input DataFrame with variables as columns
            config: Discovery configuration. If None, uses defaults.
            session_id: Session ID for tracking

        Returns:
            DiscoveryResult with ensemble DAG and confidence scores
        """
        if config is None:
            config = DiscoveryConfig()

        logger.info(
            f"Starting causal discovery with {len(config.algorithms)} algorithms: "
            f"{[a.value for a in config.algorithms]}"
        )

        start_time = time.time()
        node_names = list(data.columns)

        # Execute with optional tracing
        if self._enable_tracing and self._tracer:
            return await self._discover_dag_with_tracing(
                data, config, session_id, node_names, start_time
            )
        else:
            return await self._discover_dag_internal(
                data, config, session_id, node_names, start_time
            )

    async def _discover_dag_internal(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
        session_id: Optional[UUID],
        node_names: List[str],
        start_time: float,
    ) -> DiscoveryResult:
        """Internal discovery implementation without tracing."""
        # Run algorithms (potentially in parallel)
        algorithm_results = await self._run_algorithms(data, config)

        # Combine results into ensemble
        edges, ensemble_dag = self._build_ensemble(
            algorithm_results,
            node_names,
            config.ensemble_threshold,
        )

        total_runtime = time.time() - start_time
        logger.info(f"Causal discovery complete: {len(edges)} edges found in {total_runtime:.2f}s")

        return DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=ensemble_dag,
            edges=edges,
            algorithm_results=algorithm_results,
            session_id=session_id,
            metadata={
                "total_runtime_seconds": total_runtime,
                "node_names": node_names,
                "n_samples": len(data),
            },
        )

    async def _discover_dag_with_tracing(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
        session_id: Optional[UUID],
        node_names: List[str],
        start_time: float,
    ) -> DiscoveryResult:
        """Discovery implementation with Opik tracing."""
        async with self._tracer.trace_discovery(
            session_id=session_id,
            algorithms=[a.value for a in config.algorithms],
            n_variables=len(node_names),
            n_samples=len(data),
            config=config,
            tags=["causal_discovery", "ensemble"],
        ) as span:
            # Run algorithms with individual tracing
            algorithm_results = await self._run_algorithms_with_tracing(data, config, span)

            # Combine results into ensemble
            edges, ensemble_dag = self._build_ensemble(
                algorithm_results,
                node_names,
                config.ensemble_threshold,
            )

            total_runtime = time.time() - start_time

            # Calculate algorithm agreement
            n_converged = len([r for r in algorithm_results if r.converged])
            agreement = 0.0
            if edges and n_converged > 0:
                agreement = sum(e.confidence for e in edges) / len(edges)

            # Log ensemble result to tracer
            await self._tracer.log_ensemble_result(
                parent_span=span,
                n_edges=len(edges),
                agreement=agreement,
                runtime_seconds=total_runtime,
            )

            # Update span with final results
            span.n_edges_discovered = len(edges)
            span.algorithm_agreement = agreement

            logger.info(
                f"Causal discovery complete: {len(edges)} edges found in {total_runtime:.2f}s"
            )

            return DiscoveryResult(
                success=True,
                config=config,
                ensemble_dag=ensemble_dag,
                edges=edges,
                algorithm_results=algorithm_results,
                session_id=session_id,
                metadata={
                    "total_runtime_seconds": total_runtime,
                    "node_names": node_names,
                    "n_samples": len(data),
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                },
            )

    async def _run_algorithms_with_tracing(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
        parent_span: Any,
    ) -> List[AlgorithmResult]:
        """Run algorithms with individual result tracing."""
        results = []

        for algo_type in config.algorithms:
            try:
                algorithm = self._get_algorithm(algo_type)
                logger.debug(f"Running {algo_type.value} algorithm...")

                # Run in executor to not block event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: algorithm.discover(data, config),  # noqa: B023
                )

                results.append(result)

                # Log algorithm result to tracer
                if self._tracer:
                    await self._tracer.log_algorithm_result(parent_span, result)

                logger.debug(
                    f"{algo_type.value} found {len(result.edge_list)} edges "
                    f"in {result.runtime_seconds:.2f}s"
                )

            except Exception as e:
                logger.error(f"Algorithm {algo_type.value} failed: {e}")
                # Create failed result
                failed_result = AlgorithmResult(
                    algorithm=algo_type,
                    adjacency_matrix=np.zeros((len(data.columns), len(data.columns)), dtype=int),
                    edge_list=[],
                    runtime_seconds=0.0,
                    converged=False,
                    metadata={"error": str(e)},
                )
                results.append(failed_result)

                # Log failed result
                if self._tracer:
                    await self._tracer.log_algorithm_result(parent_span, failed_result)

        return results

    async def _run_algorithms(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> List[AlgorithmResult]:
        """Run all configured algorithms.

        Args:
            data: Input data
            config: Discovery configuration

        Returns:
            List of results from each algorithm
        """
        results = []

        # Use ProcessPoolExecutor for true parallelism when configured
        # (causal-learn is not thread-safe, so processes are preferred)
        if config.use_process_pool and len(config.algorithms) > 1:
            logger.info(
                f"Using ProcessPoolExecutor with {config.max_workers or 'auto'} workers "
                f"for {len(config.algorithms)} algorithms"
            )
            results = await self._run_algorithms_parallel(data, config)
        else:
            # Sequential execution (default, safer for single algorithm or debugging)
            for algo_type in config.algorithms:
                try:
                    algorithm = self._get_algorithm(algo_type)
                    logger.debug(f"Running {algo_type.value} algorithm...")

                    # Run in executor to not block event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda a=algorithm: a.discover(data, config),
                    )

                    results.append(result)
                    logger.debug(
                        f"{algo_type.value} found {len(result.edge_list)} edges "
                        f"in {result.runtime_seconds:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Algorithm {algo_type.value} failed: {e}")
                    # Create failed result
                    results.append(
                        AlgorithmResult(
                            algorithm=algo_type,
                            adjacency_matrix=np.zeros(
                                (len(data.columns), len(data.columns)), dtype=int
                            ),
                            edge_list=[],
                            runtime_seconds=0.0,
                            converged=False,
                            metadata={"error": str(e)},
                        )
                    )

        return results

    async def _run_algorithms_parallel(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> List[AlgorithmResult]:
        """Run algorithms in parallel using ProcessPoolExecutor.

        This provides true parallelism since causal-learn is not thread-safe.

        Args:
            data: Input data
            config: Discovery configuration

        Returns:
            List of results from each algorithm
        """
        results = []
        loop = asyncio.get_event_loop()

        # Prepare serializable data
        data_dict = data.to_dict()
        config_dict = config.to_dict()

        # Create process pool
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all algorithms
            futures = []
            for algo_type in config.algorithms:
                algo_class = self.ALGORITHM_REGISTRY.get(algo_type)
                if not algo_class:
                    logger.warning(f"Unknown algorithm type: {algo_type}")
                    continue

                future = loop.run_in_executor(
                    executor,
                    _run_algorithm_in_process,
                    algo_class,
                    data_dict,
                    config_dict,
                )
                futures.append((algo_type, future))

            # Gather results
            for algo_type, future in futures:
                try:
                    result_dict = await future

                    # Reconstruct AlgorithmResult from dict
                    result = AlgorithmResult(
                        algorithm=DiscoveryAlgorithmType(result_dict["algorithm"]),
                        adjacency_matrix=np.array(result_dict["adjacency_matrix"]),
                        edge_list=[
                            DiscoveredEdge(
                                source=e[0],
                                target=e[1],
                                edge_type=EdgeType(e[2]),
                                confidence=e[3],
                            )
                            for e in result_dict["edge_list"]
                        ],
                        runtime_seconds=result_dict["runtime_seconds"],
                        converged=result_dict["converged"],
                        metadata=result_dict["metadata"],
                    )
                    results.append(result)
                    logger.debug(
                        f"{algo_type.value} found {len(result.edge_list)} edges "
                        f"in {result.runtime_seconds:.2f}s (parallel)"
                    )

                except Exception as e:
                    logger.error(f"Algorithm {algo_type.value} failed in process: {e}")
                    results.append(
                        AlgorithmResult(
                            algorithm=algo_type,
                            adjacency_matrix=np.zeros(
                                (len(data.columns), len(data.columns)), dtype=int
                            ),
                            edge_list=[],
                            runtime_seconds=0.0,
                            converged=False,
                            metadata={"error": str(e)},
                        )
                    )

        return results

    def _build_ensemble(
        self,
        results: List[AlgorithmResult],
        node_names: List[str],
        threshold: float,
    ) -> Tuple[List[DiscoveredEdge], nx.DiGraph]:
        """Build ensemble DAG from algorithm results.

        Uses voting across algorithms to determine which edges to include.
        Edges found by >= threshold fraction of algorithms are included.

        Args:
            results: Results from individual algorithms
            node_names: Names of nodes
            threshold: Minimum fraction of algorithms that must agree

        Returns:
            Tuple of (edge list with confidence, networkx DiGraph)
        """
        n_algorithms = len(results)
        if n_algorithms == 0:
            return [], nx.DiGraph()

        # Count votes for each edge
        edge_votes: Dict[Tuple[str, str], List[str]] = {}

        for result in results:
            if not result.converged:
                continue

            for source, target in result.edge_list:
                edge_key = (source, target)
                if edge_key not in edge_votes:
                    edge_votes[edge_key] = []
                edge_votes[edge_key].append(result.algorithm.value)

        # Filter edges by threshold and create DiscoveredEdge objects
        min_votes = max(1, int(n_algorithms * threshold))
        edges = []

        for (source, target), algorithms in edge_votes.items():
            n_votes = len(algorithms)
            if n_votes >= min_votes:
                confidence = n_votes / n_algorithms
                edges.append(
                    DiscoveredEdge(
                        source=source,
                        target=target,
                        edge_type=EdgeType.DIRECTED,
                        confidence=confidence,
                        algorithm_votes=n_votes,
                        algorithms=algorithms,
                    )
                )

        # Build networkx DiGraph
        dag = nx.DiGraph()
        dag.add_nodes_from(node_names)

        for edge in edges:
            dag.add_edge(
                edge.source,
                edge.target,
                confidence=edge.confidence,
                votes=edge.algorithm_votes,
                algorithms=edge.algorithms,
            )

        # Check for cycles and remove lowest-confidence edge if found
        dag = self._remove_cycles(dag)

        return edges, dag

    def _remove_cycles(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles from graph by removing lowest-confidence edges.

        Args:
            dag: Graph that may contain cycles

        Returns:
            Acyclic graph
        """
        while True:
            try:
                cycle = nx.find_cycle(dag, orientation="original")
                # Find edge with lowest confidence in cycle
                min_conf = float("inf")
                min_edge = None

                for u, v, _ in cycle:
                    conf = dag.edges[u, v].get("confidence", 1.0)
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (u, v)

                if min_edge:
                    logger.warning(
                        f"Removing cycle edge {min_edge[0]} -> {min_edge[1]} "
                        f"(confidence: {min_conf:.2f})"
                    )
                    dag.remove_edge(*min_edge)

            except nx.NetworkXNoCycle:
                break

        return dag

    def discover_dag_sync(
        self,
        data: pd.DataFrame,
        config: Optional[DiscoveryConfig] = None,
        session_id: Optional[UUID] = None,
    ) -> DiscoveryResult:
        """Synchronous version of discover_dag.

        Args:
            data: Input DataFrame
            config: Discovery configuration
            session_id: Session ID

        Returns:
            DiscoveryResult
        """
        return asyncio.run(self.discover_dag(data, config, session_id))

    @classmethod
    def get_available_algorithms(cls) -> List[DiscoveryAlgorithmType]:
        """Get list of available algorithms.

        Returns:
            List of supported algorithm types
        """
        return list(cls.ALGORITHM_REGISTRY.keys())

    @classmethod
    def register_algorithm(
        cls,
        algo_type: DiscoveryAlgorithmType,
        algo_class: Type[BaseDiscoveryAlgorithm],
    ) -> None:
        """Register a new algorithm.

        Args:
            algo_type: Algorithm type identifier
            algo_class: Algorithm implementation class
        """
        cls.ALGORITHM_REGISTRY[algo_type] = algo_class
        logger.info(f"Registered algorithm: {algo_type.value}")
