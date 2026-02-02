"""Parallel Pipeline implementation.

Implements simultaneous execution of all four causal libraries
with confidence-weighted result aggregation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from .orchestrator import (
    CausalLibrary,
    LibraryExecutor,
    PipelineOrchestrator,
)
from .router import LibraryRouter
from .state import (
    LibraryExecutionResult,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineState,
)

logger = logging.getLogger(__name__)


class ParallelPipeline(PipelineOrchestrator):
    """Parallel pipeline executor.

    Executes all enabled libraries simultaneously using asyncio.gather.
    Results are aggregated using confidence-weighted consensus.

    Reference: docs/Data Architecture & Integration.html
    """

    def __init__(
        self,
        router: Optional[LibraryRouter] = None,
        executors: Optional[Dict[CausalLibrary, LibraryExecutor]] = None,
        max_parallel: int = 4,
        fail_fast: bool = False,
    ):
        """Initialize parallel pipeline.

        Args:
            router: Library router for question classification
            executors: Map of library to executor
            max_parallel: Maximum libraries to run in parallel
            fail_fast: If True, cancel remaining on first failure
        """
        super().__init__(router, executors)
        self.max_parallel = max_parallel
        self.fail_fast = fail_fast

    async def execute(self, input_data: PipelineInput) -> PipelineOutput:
        """Execute the parallel pipeline.

        Args:
            input_data: Pipeline input with query and configuration

        Returns:
            PipelineOutput with aggregated results from all libraries
        """
        start_time = time.time()

        # Route query to determine libraries
        routing_decision = await self.route(
            input_data["query"],
            force_libraries=input_data.get("libraries_enabled"),
        )

        # Create initial state
        state = self._create_initial_state(input_data, routing_decision)
        state["status"] = "running"
        state["current_stage"] = PipelineStage.ROUTING

        # Override mode to parallel
        state["config"]["mode"] = "parallel"

        # Get libraries to execute
        libraries = self._get_libraries_to_execute(state)

        logger.info(
            f"Parallel pipeline starting with libraries: {[lib.value for lib in libraries]}"
        )

        # Execute all libraries in parallel
        state = await self._execute_parallel(state, libraries)

        # Aggregate results
        state["current_stage"] = PipelineStage.AGGREGATING
        state = self._aggregate_parallel_results(state)

        state["total_latency_ms"] = int((time.time() - start_time) * 1000)
        state["current_stage"] = PipelineStage.COMPLETED

        return self._create_output(state)

    def _get_libraries_to_execute(self, state: PipelineState) -> List[CausalLibrary]:
        """Get list of libraries to execute.

        Args:
            state: Current pipeline state

        Returns:
            List of CausalLibrary to execute
        """
        enabled = set(state["config"]["libraries_enabled"])
        libraries = [lib for lib in CausalLibrary if lib.value in enabled]

        # Limit to max_parallel
        return libraries[: self.max_parallel]

    async def _execute_parallel(
        self,
        state: PipelineState,
        libraries: List[CausalLibrary],
    ) -> PipelineState:
        """Execute libraries in parallel.

        Args:
            state: Current pipeline state
            libraries: Libraries to execute

        Returns:
            Updated state with all results
        """
        # Create tasks for each library
        tasks: List[asyncio.Task[Tuple[CausalLibrary, LibraryExecutionResult]]] = []

        for library in libraries:
            executor = self.executors.get(library)
            if not executor:
                state["libraries_skipped"].append(library.value)
                continue

            # Validate input
            is_valid, error_msg = executor.validate_input(state)
            if not is_valid:
                state["warnings"].append(f"{library.value}: {error_msg}")
                state["libraries_skipped"].append(library.value)
                continue

            # Create task
            task = asyncio.create_task(
                self._execute_library_with_timeout(
                    library,
                    executor,
                    state,
                    state["config"]["stage_timeout_ms"],
                )
            )
            tasks.append(task)

        if not tasks:
            logger.warning("No libraries to execute in parallel")
            state["status"] = "failed"
            state["errors"].append(
                {"library": "pipeline", "error": "No valid libraries to execute"}
            )
            return state

        # Execute all tasks
        if self.fail_fast:
            # Cancel remaining tasks on first failure
            results = await self._gather_fail_fast(tasks)
        else:
            # Wait for all tasks, continue on failures
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                state["errors"].append({"library": "unknown", "error": str(result)})
            elif isinstance(result, tuple):
                library, exec_result = result
                state = self._update_state_with_result(state, library, exec_result)

        return state

    async def _execute_library_with_timeout(
        self,
        library: CausalLibrary,
        executor: LibraryExecutor,
        state: PipelineState,
        timeout_ms: int,
    ) -> Tuple[CausalLibrary, LibraryExecutionResult]:
        """Execute a single library with timeout.

        Args:
            library: Library to execute
            executor: Executor for the library
            state: Current pipeline state
            timeout_ms: Timeout in milliseconds

        Returns:
            Tuple of (library, result)
        """
        try:
            result = await asyncio.wait_for(
                executor.execute(state, state["config"]),
                timeout=timeout_ms / 1000,
            )
            return (library, result)
        except asyncio.TimeoutError:
            logger.error(f"{library.value} timed out")
            return (
                library,
                LibraryExecutionResult(
                    library=library.value,
                    success=False,
                    latency_ms=timeout_ms,
                    result=None,
                    error="Execution timed out",
                    confidence=0.0,
                    warnings=[],
                ),
            )
        except Exception as e:
            logger.error(f"{library.value} failed: {e}")
            return (
                library,
                LibraryExecutionResult(
                    library=library.value,
                    success=False,
                    latency_ms=0,
                    result=None,
                    error=str(e),
                    confidence=0.0,
                    warnings=[],
                ),
            )

    async def _gather_fail_fast(
        self,
        tasks: List[asyncio.Task[Tuple[CausalLibrary, LibraryExecutionResult]]],
    ) -> List[Tuple[CausalLibrary, LibraryExecutionResult]]:
        """Gather results with fail-fast behavior.

        Args:
            tasks: Tasks to gather

        Returns:
            List of results (may be partial if fail-fast triggered)
        """
        results: List[Tuple[CausalLibrary, LibraryExecutionResult]] = []

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    result = task.result()
                    results.append(result)

                    # Check for failure
                    if isinstance(result, tuple):
                        _, exec_result = result
                        if not exec_result["success"]:
                            # Cancel remaining tasks
                            for p in pending:
                                p.cancel()
                            return results
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    # Cancel remaining tasks
                    for p in pending:
                        p.cancel()
                    return results

        return results

    def _aggregate_parallel_results(self, state: PipelineState) -> PipelineState:
        """Aggregate results from parallel execution.

        Uses confidence-weighted consensus for effect estimates.

        Args:
            state: Pipeline state with all library results

        Returns:
            Updated state with aggregated results
        """
        # Collect effect estimates with confidence
        effect_estimates: List[Tuple[str, float, float]] = []  # (library, effect, conf)

        if state["dowhy_result"] and state["dowhy_result"]["success"]:
            if state["causal_effect"] is not None:
                effect_estimates.append(
                    ("dowhy", state["causal_effect"], state["dowhy_result"]["confidence"])
                )

        if state["econml_result"] and state["econml_result"]["success"]:
            if state["overall_ate"] is not None:
                effect_estimates.append(
                    ("econml", state["overall_ate"], state["econml_result"]["confidence"])
                )

        # Calculate consensus
        if effect_estimates:
            state = self._calculate_consensus(state, effect_estimates)

        # Calculate library agreement
        state = self._calculate_agreement(state, effect_estimates)

        # Generate summary
        state["executive_summary"] = self._generate_parallel_summary(state)
        state["key_insights"] = self._generate_parallel_insights(state)
        state["recommended_actions"] = self._generate_parallel_recommendations(state)

        return state

    def _calculate_consensus(
        self,
        state: PipelineState,
        estimates: List[Tuple[str, float, float]],
    ) -> PipelineState:
        """Calculate confidence-weighted consensus effect.

        Args:
            state: Current pipeline state
            estimates: List of (library, effect, confidence)

        Returns:
            Updated state with consensus values
        """
        if not estimates:
            return state

        # Confidence-weighted average
        total_weight = sum(conf for _, _, conf in estimates)
        if total_weight > 0:
            weighted_sum = sum(effect * conf for _, effect, conf in estimates)
            state["consensus_effect"] = weighted_sum / total_weight

            # Average confidence as consensus confidence
            state["consensus_confidence"] = total_weight / len(estimates)

        return state

    def _calculate_agreement(
        self,
        state: PipelineState,
        estimates: List[Tuple[str, float, float]],
    ) -> PipelineState:
        """Calculate pairwise library agreement.

        Args:
            state: Current pipeline state
            estimates: List of (library, effect, confidence)

        Returns:
            Updated state with agreement metrics
        """
        if len(estimates) < 2:
            return state

        agreement: Dict[str, float] = {}

        for i, (lib1, effect1, _) in enumerate(estimates):
            for lib2, effect2, _ in estimates[i + 1 :]:
                pair_name = f"{lib1}_{lib2}"
                # Agreement = 1 - normalized difference
                max_abs = max(abs(effect1), abs(effect2), 0.001)
                diff = abs(effect1 - effect2) / max_abs
                agreement[pair_name] = max(0.0, 1.0 - diff)

        state["library_agreement"] = agreement
        return state

    def _generate_parallel_summary(self, state: PipelineState) -> str:
        """Generate executive summary for parallel execution."""
        parts = ["Parallel Analysis Results"]

        if state["libraries_executed"]:
            parts.append(f"Libraries: {', '.join(state['libraries_executed'])}")

        if state["consensus_effect"] is not None:
            parts.append(
                f"Consensus effect: {state['consensus_effect']:.4f} "
                f"(confidence: {state['consensus_confidence']:.2%})"
            )

        if state["library_agreement"]:
            avg_agreement = sum(state["library_agreement"].values()) / len(
                state["library_agreement"]
            )
            parts.append(f"Average library agreement: {avg_agreement:.2%}")

        return ". ".join(parts)

    def _generate_parallel_insights(self, state: PipelineState) -> List[str]:
        """Generate insights from parallel execution."""
        insights = []

        # Consensus insight
        if state["consensus_effect"] is not None:
            direction = (
                "positive"
                if state["consensus_effect"] > 0
                else "negative"
                if state["consensus_effect"] < 0
                else "neutral"
            )
            insights.append(f"Consensus analysis indicates {direction} treatment effect")

        # Agreement insight
        if state["library_agreement"]:
            min_agreement = min(state["library_agreement"].values())
            if min_agreement < 0.7:
                insights.append(
                    "Library disagreement detected - results may need further investigation"
                )
            elif min_agreement >= 0.9:
                insights.append("High agreement across libraries - results are robust")

        # Individual library insights
        if state["causal_graph"]:
            insights.append("Causal structure identified via NetworkX")

        if state["refutation_results"]:
            insights.append("DoWhy refutation tests completed")

        if state["heterogeneity_score"] is not None and state["heterogeneity_score"] > 0.3:
            insights.append(
                f"EconML detected significant effect heterogeneity "
                f"(score: {state['heterogeneity_score']:.2f})"
            )

        if state["auuc"] is not None:
            insights.append(f"CausalML uplift model AUUC: {state['auuc']:.4f}")

        return insights

    def _generate_parallel_recommendations(self, state: PipelineState) -> List[str]:
        """Generate recommendations from parallel execution."""
        recommendations = []

        # Based on agreement
        if state["library_agreement"]:
            min_agreement = min(state["library_agreement"].values())
            if min_agreement < 0.7:
                recommendations.append(
                    "Consider running sequential pipeline to validate discrepancies"
                )

        # Based on confidence
        if state["consensus_confidence"] is not None:
            if state["consensus_confidence"] < 0.7:
                recommendations.append(
                    "Low overall confidence - consider additional data collection"
                )
            elif state["consensus_confidence"] >= 0.85:
                recommendations.append(
                    "High confidence consensus - results suitable for decision-making"
                )

        # Based on heterogeneity
        if state["heterogeneity_score"] is not None and state["heterogeneity_score"] > 0.5:
            recommendations.append(
                "Significant heterogeneity detected - consider segment-specific strategies"
            )

        if not recommendations:
            recommendations.append("Review individual library results for detailed insights")

        return recommendations


def create_parallel_pipeline(
    router: Optional[LibraryRouter] = None,
    max_parallel: int = 4,
    fail_fast: bool = False,
) -> ParallelPipeline:
    """Factory function for creating parallel pipelines.

    Args:
        router: Custom router (uses default if not provided)
        max_parallel: Maximum libraries to run in parallel
        fail_fast: Cancel remaining tasks on first failure

    Returns:
        Configured ParallelPipeline instance
    """
    return ParallelPipeline(
        router=router,
        max_parallel=max_parallel,
        fail_fast=fail_fast,
    )


class ParallelPipelineBuilder:
    """Builder for configuring parallel pipelines."""

    def __init__(self):
        self._router: Optional[LibraryRouter] = None
        self._executors: Dict[CausalLibrary, LibraryExecutor] = {}
        self._max_parallel: int = 4
        self._fail_fast: bool = False

    def with_router(self, router: LibraryRouter) -> "ParallelPipelineBuilder":
        """Set custom router."""
        self._router = router
        return self

    def with_executor(
        self, library: CausalLibrary, executor: LibraryExecutor
    ) -> "ParallelPipelineBuilder":
        """Add custom executor for a library."""
        self._executors[library] = executor
        return self

    def with_max_parallel(self, max_parallel: int) -> "ParallelPipelineBuilder":
        """Set maximum parallel libraries."""
        self._max_parallel = max_parallel
        return self

    def with_fail_fast(self, fail_fast: bool = True) -> "ParallelPipelineBuilder":
        """Enable fail-fast mode."""
        self._fail_fast = fail_fast
        return self

    def build(self) -> ParallelPipeline:
        """Build the pipeline."""
        return ParallelPipeline(
            router=self._router,
            executors=self._executors if self._executors else None,
            max_parallel=self._max_parallel,
            fail_fast=self._fail_fast,
        )
