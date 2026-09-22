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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, cast
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

    from .base import DiscoveryConfig

    # Reconstruct objects from dicts. from_dict is the single reconstruction
    # path: the previous hand-enumerated copy dropped prior_knowledge (guided
    # PC in a process worker would have run unguided), bootstrap_resamples,
    # and latent_diagnostic.
    data = pd.DataFrame(data_dict)
    config = DiscoveryConfig.from_dict(config_dict)

    # Run algorithm
    algorithm = algo_class()
    result = algorithm.discover(data, config)

    # Convert result to dict for serialization
    return {
        "algorithm": result.algorithm.value,
        "adjacency_matrix": result.adjacency_matrix.tolist(),
        "edge_list": [
            (e.source, e.target, e.edge_type.value, e.confidence)  # type: ignore[attr-defined]
            for e in result.edge_list
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

    @staticmethod
    def _run_outcome(algorithm_results: List[AlgorithmResult]) -> Tuple[bool, Dict[str, Any]]:
        """``(success, metadata)`` for a finished run, derived from what the
        algorithms actually did.

        A run is successful only if at least one algorithm CONVERGED. Before
        this, ``success`` was hard-coded ``True`` on both result paths, so a run
        in which every algorithm raised (measured 2026-09-22 on the real Optum
        persistence cohort: fisherz refuses the singular correlation matrix of
        the claims comorbidity families) came back as a successful discovery
        with zero edges. The gate then reported "Too few edges discovered" and
        the API answered ``dag_source='domain_knowledge'`` with no reason --
        the error string never left ``algorithm_results[].metadata``.

        ``algorithm_errors`` (``{algorithm: message}``) is recorded whenever
        any algorithm failed; ``error`` (one line naming each failed algorithm)
        is set only when NONE converged, because that is the case in which the
        gate must say "could not run" rather than "found nothing".
        """
        errors = {
            r.algorithm.value: str(r.metadata.get("error") or "did not converge")
            for r in algorithm_results
            if not r.converged
        }
        success = any(r.converged for r in algorithm_results)
        metadata: Dict[str, Any] = {}
        if errors:
            metadata["algorithm_errors"] = errors
        if not success:
            metadata["error"] = (
                "; ".join(f"{algo}: {msg}" for algo, msg in errors.items())
                if errors
                else "no discovery algorithm ran"
            )
        return success, metadata

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

        bootstrap_metadata = await self._maybe_bootstrap(
            data, config, algorithm_results, edges, ensemble_dag
        )

        latent_metadata = await self._maybe_latent_diagnostic(
            data, config, elapsed_s=time.time() - start_time
        )

        total_runtime = time.time() - start_time
        success, outcome_metadata = self._run_outcome(algorithm_results)
        if success:
            logger.info(
                f"Causal discovery complete: {len(edges)} edges found in {total_runtime:.2f}s"
            )
        else:
            logger.warning(
                f"Causal discovery could not run ({outcome_metadata['error']}) "
                f"after {total_runtime:.2f}s"
            )

        return DiscoveryResult(
            success=success,
            config=config,
            ensemble_dag=ensemble_dag,
            edges=edges,
            algorithm_results=algorithm_results,
            session_id=session_id,
            metadata={
                "total_runtime_seconds": total_runtime,
                "node_names": node_names,
                "n_samples": len(data),
                **bootstrap_metadata,
                **latent_metadata,
                **outcome_metadata,
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
        assert self._tracer is not None, "Tracer must be initialized for traced discovery"
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

            bootstrap_metadata = await self._maybe_bootstrap(
                data, config, algorithm_results, edges, ensemble_dag
            )

            latent_metadata = await self._maybe_latent_diagnostic(
                data, config, elapsed_s=time.time() - start_time
            )

            total_runtime = time.time() - start_time

            # Calculate algorithm agreement. On a bootstrapped single-algorithm
            # run, _maybe_bootstrap has already overwritten e.confidence with
            # bootstrap_stability above, so this reports mean bootstrap
            # stability rather than the (vacuous) 1.0 multi-vote agreement.
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

            success, outcome_metadata = self._run_outcome(algorithm_results)
            if success:
                logger.info(
                    f"Causal discovery complete: {len(edges)} edges found in {total_runtime:.2f}s"
                )
            else:
                logger.warning(
                    f"Causal discovery could not run ({outcome_metadata['error']}) "
                    f"after {total_runtime:.2f}s"
                )

            return DiscoveryResult(
                success=success,
                config=config,
                ensemble_dag=ensemble_dag,
                edges=edges,
                algorithm_results=algorithm_results,
                session_id=session_id,
                metadata={
                    **outcome_metadata,
                    "total_runtime_seconds": total_runtime,
                    "node_names": node_names,
                    "n_samples": len(data),
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    **bootstrap_metadata,
                    **latent_metadata,
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

                    # Run in executor to not block event loop. Bound each
                    # algorithm by ``timeout_seconds`` (#1978): the attribute
                    # was documented as a per-algorithm timeout but never
                    # enforced, so a runaway causal-learn call was bounded only
                    # by the caller's wall-clock cap. A timeout is a FAILED run
                    # (converged=False) so the gate scores no evidence; never a
                    # silently empty converged DAG. The worker thread itself
                    # cannot be cancelled; its result is abandoned.
                    loop = asyncio.get_event_loop()
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda a=algorithm: a.discover(data, config),  # type: ignore[misc]
                            ),
                            timeout=self.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Algorithm {algo_type.value} timed out after "
                            f"{self.timeout_seconds:.0f}s"
                        )
                        results.append(
                            AlgorithmResult(
                                algorithm=algo_type,
                                adjacency_matrix=np.zeros(
                                    (len(data.columns), len(data.columns)), dtype=int
                                ),
                                edge_list=[],
                                runtime_seconds=float(self.timeout_seconds),
                                converged=False,
                                metadata={
                                    "error": f"timeout after {self.timeout_seconds:.0f}s",
                                    "timeout_seconds": self.timeout_seconds,
                                },
                            )
                        )
                        continue

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
        # #1978: the runner's own max_workers is the fallback when the config
        # does not pin one (it used to be stored and never read).
        with ProcessPoolExecutor(max_workers=config.max_workers or self.max_workers) as executor:
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
                    # Note: edge_list type annotation in AlgorithmResult is List[Tuple[str, str]]
                    # but actual implementation uses DiscoveredEdge objects
                    edge_list_data = [
                        DiscoveredEdge(
                            source=e[0],
                            target=e[1],
                            edge_type=EdgeType(e[2]),
                            confidence=e[3],
                        )
                        for e in result_dict["edge_list"]
                    ]
                    result = AlgorithmResult(
                        algorithm=DiscoveryAlgorithmType(result_dict["algorithm"]),
                        adjacency_matrix=np.array(result_dict["adjacency_matrix"]),
                        edge_list=cast(List[Tuple[str, str]], edge_list_data),
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

        # MED: the agreement threshold + edge confidence must be relative to the
        # algorithms that actually CONVERGED, not the total (which includes
        # failed / non-converged ones). Dividing by the total deflated every
        # edge's confidence whenever an algorithm failed (e.g. 2 of 2 converged
        # algorithms agreeing reported as 0.5 on a 4-algorithm run where 2
        # crashed).
        n_converged = sum(1 for r in results if r.converged)
        if n_converged == 0:
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
        min_votes = max(1, int(n_converged * threshold))
        edges = []

        for (source, target), algorithms in edge_votes.items():
            n_votes = len(algorithms)
            if n_votes >= min_votes:
                confidence = n_votes / n_converged
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

    def _bootstrap_edge_stability(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
        algorithm: BaseDiscoveryAlgorithm,
        edges: List[DiscoveredEdge],
        ensemble_dag: nx.DiGraph,
        elapsed_before_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Measure per-edge stability by re-running the single converged
        algorithm on up to ``config.bootstrap_resamples`` bootstrap resamples.

        Mutates ``edges`` in place when the run is corroborated:
        ``bootstrap_stability`` becomes the directed-match frequency over the
        SUCCEEDED resamples, and ``confidence`` — vacuously 1.0 for a
        single-algorithm run — is overwritten with it. The frequencies are
        left unwritten (None) when fewer than ``min_resamples`` resamples
        succeeded: they would be noise, so the gate must treat the run as
        uncorroborated (failing toward caution, not toward ACCEPT).
        ``min_resamples`` is ``config.min_resamples`` when set, else the legacy
        ``max(2, B // 2)``.

        Time budget (Lane D item 2): ``config.time_budget_s`` bounds the whole
        discovery run, so the primary fits' wall (``elapsed_before_s``) is
        charged first and the loop stops before a resample that would overrun
        it, estimating the next resample's cost as the mean resample so far
        (the primary fits' wall before any resample ran); a resample that is
        still running when the budget ends is abandoned (``n_abandoned``). Measured on the real Optum
        persistence frame at 43 covariates, one PC fit is 230 s: under
        production's 20 resamples that is ~81 min against a 900 s agent
        timeout, which is why the loop must be bounded and the ACHIEVED count
        reported rather than the requested one.

        Always returns the summary — achieved counts are reported whether or
        not the run was corroborated — with ``corroborated`` saying which.
        """
        n_resamples = config.bootstrap_resamples
        if config.min_resamples is not None:
            min_required = max(1, int(config.min_resamples))
        else:
            min_required = max(2, n_resamples // 2)
        budget = config.time_budget_s
        rng = np.random.default_rng(config.random_state)
        counts: Dict[Tuple[str, str], int] = {(e.source, e.target): 0 for e in edges}
        succeeded = 0
        attempted = 0
        abandoned = 0
        budget_exhausted = False
        loop_start = time.monotonic()
        resample_wall: List[float] = []
        # Under a budget each resample fit is WAITED FOR only as long as the
        # budget has left: the estimate below cannot see a fit that is slower
        # than its predecessors (measured on the real Optum persistence frame,
        # one gsq resample fit ran > 53 min after an 8 s primary fit —
        # docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/
        # d7_gsq_arm_stopped.txt). An overrun is abandoned: the worker thread
        # cannot be cancelled and finishes on its own (the same contract as the
        # per-algorithm timeout in ``_run_algorithms``); it is counted as
        # attempted and abandoned, never as succeeded.
        pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="discovery-bootstrap")
            if budget is not None
            else None
        )
        try:
            for _ in range(n_resamples):
                if budget is not None:
                    loop_elapsed = time.monotonic() - loop_start
                    spent = elapsed_before_s + loop_elapsed
                    # Next resample's cost: the mean resample so far, or — before
                    # any ran — the primary fits' wall (same algorithm, same frame
                    # size, so the first guess is the fit already measured).
                    if resample_wall:
                        estimate = sum(resample_wall) / len(resample_wall)
                    else:
                        estimate = elapsed_before_s
                    if spent + estimate > budget:
                        budget_exhausted = True
                        break
                attempted += 1
                indices = rng.integers(0, len(data), len(data))
                resample = data.iloc[indices].reset_index(drop=True)
                fit_start = time.monotonic()
                try:
                    if pool is not None and budget is not None:
                        remaining = budget - (elapsed_before_s + (fit_start - loop_start))
                        future = pool.submit(algorithm.discover, resample, config)
                        try:
                            result = future.result(timeout=max(0.0, remaining))
                        except FuturesTimeoutError:
                            abandoned += 1
                            budget_exhausted = True
                            resample_wall.append(time.monotonic() - fit_start)
                            logger.warning(
                                f"Bootstrap resample abandoned: still running after the "
                                f"{remaining:.1f}s the budget had left ({budget:.0f}s); "
                                "its worker thread finishes on its own"
                            )
                            # The pool is single-threaded and its worker is busy
                            # with the abandoned fit: release it without waiting.
                            pool.shutdown(wait=False)
                            pool = None
                            break
                    else:
                        result = algorithm.discover(resample, config)
                except Exception as exc:
                    logger.debug(f"Bootstrap resample failed: {exc}")
                    resample_wall.append(time.monotonic() - fit_start)
                    continue
                resample_wall.append(time.monotonic() - fit_start)
                if not result.converged:
                    continue
                succeeded += 1
                found = {(source, target) for source, target in result.edge_list}
                for key in counts:
                    if key in found:
                        counts[key] += 1
        finally:
            if pool is not None:
                pool.shutdown(wait=False)
        loop_elapsed = time.monotonic() - loop_start
        corroborated = succeeded >= min_required
        summary: Dict[str, Any] = {
            "n_resamples": n_resamples,
            "n_attempted": attempted,
            "n_succeeded": succeeded,
            "n_abandoned": abandoned,
            "min_resamples": min_required,
            "corroborated": corroborated,
            "time_budget_s": budget,
            "elapsed_s": elapsed_before_s + loop_elapsed,
            "budget_exhausted": budget_exhausted,
        }
        if budget_exhausted:
            logger.warning(
                f"Bootstrap stopped by the time budget ({budget:.0f}s): "
                f"{attempted}/{n_resamples} resamples attempted, {succeeded} succeeded"
            )
        if not corroborated:
            logger.warning(
                f"Bootstrap stability unknown: {succeeded}/{n_resamples} resamples "
                f"succeeded (fewer than {min_required})"
            )
            return summary
        for edge in edges:
            stability = counts[(edge.source, edge.target)] / succeeded
            edge.bootstrap_stability = stability
            edge.confidence = stability
            if ensemble_dag.has_edge(edge.source, edge.target):
                ensemble_dag.edges[edge.source, edge.target]["confidence"] = stability
        return summary

    async def _maybe_bootstrap(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
        algorithm_results: List[AlgorithmResult],
        edges: List[DiscoveredEdge],
        ensemble_dag: nx.DiGraph,
    ) -> Dict[str, Any]:
        """Run stability measurement when configured and exactly one
        algorithm converged (multi-algorithm runs already have agreement).
        Returns extra metadata entries ({} when bootstrap did not apply).
        The primary fits' wall is charged against ``config.time_budget_s``."""
        converged = [r for r in algorithm_results if r.converged]
        if config.bootstrap_resamples <= 0 or len(converged) != 1 or not edges:
            return {}
        algorithm = self._get_algorithm(converged[0].algorithm)
        elapsed_before = float(sum(r.runtime_seconds for r in algorithm_results))
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None,
            lambda: self._bootstrap_edge_stability(
                data, config, algorithm, edges, ensemble_dag, elapsed_before_s=elapsed_before
            ),
        )
        return {"bootstrap": summary}

    def _run_latent_diagnostic(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> Dict[str, Any]:
        """Run FCI once, unguided, as a latent-confounding diagnostic.

        PC (and the guided production path) assume causal sufficiency; FCI's
        PAG is the only signal in the toolbox that can even represent a latent
        confounder (a bidirected edge). The diagnostic strips the guided
        priors — the point is the data's OWN testimony about latent structure,
        not the priors echoed back — and never multiplies by the bootstrap
        resample count. Failure modes are distinct and neither fails
        discovery: an exception (data invalid, algorithm missing) degrades to
        ``{"ran": False, "error": ...}``, while an FCI run that executed but
        did not converge reports ``{"ran": True, "converged": False}`` — it
        DID run, and the distinction tells the consumer whether to blame the
        infrastructure or the data.

        Returns the ``latent_diagnostic`` payload; graph_builder later
        annotates it with the estimand and the flag (the runner does not know
        treatment/outcome).
        """
        from dataclasses import replace

        diagnostic_config = replace(
            config,
            algorithms=[DiscoveryAlgorithmType.FCI],
            prior_knowledge=None,
            bootstrap_resamples=0,
            latent_diagnostic=False,
        )
        try:
            algorithm = self._get_algorithm(DiscoveryAlgorithmType.FCI)
            result = algorithm.discover(data, diagnostic_config)
            if not result.converged:
                return {
                    "ran": True,
                    "converged": False,
                    "runtime_seconds": result.runtime_seconds,
                    "bidirected_edges": [],
                    "error": result.metadata.get("error"),
                }
            # get_bidirected_edges maps index-keyed edge_types to column names.
            pairs = algorithm.get_bidirected_edges(result)  # type: ignore[attr-defined]
            return {
                "ran": True,
                "converged": True,
                "runtime_seconds": result.runtime_seconds,
                "bidirected_edges": [[source, target] for source, target in pairs],
            }
        except Exception as exc:
            logger.warning(f"Latent-confounding diagnostic (FCI) failed: {exc}")
            return {"ran": False, "error": str(exc)}

    async def _maybe_latent_diagnostic(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
        elapsed_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Run the FCI latent diagnostic when configured (off by default).
        Returns extra metadata entries ({} when the diagnostic is off).

        Lane D item 2: the diagnostic falls under ``config.time_budget_s``
        like the bootstrap. Measured on the capped real Optum persistence
        frame (22 columns, n = 15,209) one unguided FCI fit is 382.5 s —
        more than twice the 180 s production budget — so with the budget
        already spent it is not started (``ran=False``, the reason says
        so), and otherwise it is bounded by the remaining budget: a timeout
        is reported ``ran=True, converged=False`` with the reason. The
        worker thread cannot be cancelled; its result is abandoned (the same
        contract as the per-algorithm timeout in ``_run_algorithms``)."""
        if not config.latent_diagnostic:
            return {}
        budget = config.time_budget_s
        remaining: Optional[float] = None
        if budget is not None:
            remaining = float(budget) - float(elapsed_s)
            if remaining <= 0.0:
                logger.warning(
                    f"Latent-confounding diagnostic (FCI) not started: discovery time "
                    f"budget {budget:.0f}s exhausted after {elapsed_s:.1f}s"
                )
                return {
                    "latent_diagnostic": {
                        "ran": False,
                        "error": (
                            f"skipped: discovery time budget {budget:.0f}s exhausted "
                            f"after {elapsed_s:.1f}s"
                        ),
                        "time_budget_s": budget,
                        "elapsed_before_s": elapsed_s,
                    }
                }
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, lambda: self._run_latent_diagnostic(data, config))
        try:
            payload = await asyncio.wait_for(future, timeout=remaining)
        except asyncio.TimeoutError:
            logger.warning(
                f"Latent-confounding diagnostic (FCI) timed out after {remaining:.1f}s "
                f"(discovery time budget {budget:.0f}s)"
            )
            return {
                "latent_diagnostic": {
                    "ran": True,
                    "converged": False,
                    "bidirected_edges": [],
                    "error": (
                        f"timeout after {remaining:.1f}s (discovery time budget "
                        f"{budget:.0f}s, {elapsed_s:.1f}s already spent)"
                    ),
                    "time_budget_s": budget,
                    "elapsed_before_s": elapsed_s,
                }
            }
        if budget is not None:
            payload["time_budget_s"] = budget
            payload["elapsed_before_s"] = elapsed_s
        return {"latent_diagnostic": payload}

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
