"""Sequential Pipeline implementation.

Implements the NetworkX → DoWhy → EconML → CausalML sequential flow
as defined in the Data Architecture & Integration documentation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from .orchestrator import (
    LibraryExecutor,
    PipelineOrchestrator,
)
from .router import CausalLibrary, LibraryRouter
from .state import (
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineState,
)

logger = logging.getLogger(__name__)


# Standard sequential order for end-to-end pipeline
SEQUENTIAL_ORDER = [
    CausalLibrary.NETWORKX,  # Step 1: Graph analysis
    CausalLibrary.DOWHY,  # Step 2: Causal validation
    CausalLibrary.ECONML,  # Step 3: Effect estimation
    CausalLibrary.CAUSALML,  # Step 4: Uplift modeling
]

# Stage mapping for each library
LIBRARY_STAGES = {
    CausalLibrary.NETWORKX: PipelineStage.GRAPH_ANALYSIS,
    CausalLibrary.DOWHY: PipelineStage.CAUSAL_VALIDATION,
    CausalLibrary.ECONML: PipelineStage.EFFECT_ESTIMATION,
    CausalLibrary.CAUSALML: PipelineStage.UPLIFT_MODELING,
}


class SequentialPipeline(PipelineOrchestrator):
    """Sequential pipeline executor.

    Executes libraries in order: NetworkX → DoWhy → EconML → CausalML
    Each stage's output becomes input for the next stage.

    Reference: docs/Data Architecture & Integration.html
    """

    def __init__(
        self,
        router: Optional[LibraryRouter] = None,
        executors: Optional[Dict[CausalLibrary, LibraryExecutor]] = None,
        fail_fast: bool = False,
    ):
        """Initialize sequential pipeline.

        Args:
            router: Library router for question classification
            executors: Map of library to executor
            fail_fast: If True, stop on first failure
        """
        super().__init__(router, executors)
        self.fail_fast = fail_fast

    async def execute(self, input_data: PipelineInput) -> PipelineOutput:
        """Execute the sequential pipeline.

        Args:
            input_data: Pipeline input with query and configuration

        Returns:
            PipelineOutput with results from all executed libraries
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

        # Determine execution order
        libraries_to_run = self._get_execution_order(state)

        logger.info(
            f"Sequential pipeline starting with libraries: {[lib.value for lib in libraries_to_run]}"
        )

        # Execute each library in sequence
        for library in libraries_to_run:
            state["current_stage"] = LIBRARY_STAGES.get(library, PipelineStage.PENDING)

            executor = self.executors.get(library)
            if not executor:
                logger.warning(f"No executor for library {library.value}, skipping")
                state["libraries_skipped"].append(library.value)
                continue

            # Validate input for this library
            is_valid, error_msg = executor.validate_input(state)
            if not is_valid:
                logger.warning(f"Validation failed for {library.value}: {error_msg}")
                state["warnings"].append(f"{library.value}: {error_msg}")
                state["libraries_skipped"].append(library.value)

                if self.fail_fast:
                    state["errors"].append({"library": library.value, "error": error_msg})
                    break
                continue

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    executor.execute(state, state["config"]),
                    timeout=state["config"]["stage_timeout_ms"] / 1000,
                )

                # Update state with result
                state = self._update_state_with_result(state, library, result)

                if not result["success"]:
                    logger.error(f"{library.value} failed: {result['error']}")
                    if self.fail_fast:
                        break
                else:
                    logger.info(f"{library.value} completed in {result['latency_ms']}ms")

            except asyncio.TimeoutError:
                logger.error(f"{library.value} timed out")
                state["errors"].append({"library": library.value, "error": "Execution timed out"})
                state["libraries_skipped"].append(library.value)

                if self.fail_fast:
                    break

        # Finalize state
        state["current_stage"] = PipelineStage.AGGREGATING
        state = self._aggregate_results(state)

        state["total_latency_ms"] = int((time.time() - start_time) * 1000)
        state["current_stage"] = PipelineStage.COMPLETED

        return self._create_output(state)

    def _get_execution_order(self, state: PipelineState) -> List[CausalLibrary]:
        """Determine which libraries to run and in what order.

        Args:
            state: Current pipeline state

        Returns:
            List of libraries in execution order
        """
        enabled = set(state["config"]["libraries_enabled"])

        # Filter to only enabled libraries, maintaining sequential order
        return [lib for lib in SEQUENTIAL_ORDER if lib.value in enabled]

    def _aggregate_results(self, state: PipelineState) -> PipelineState:
        """Aggregate results from all libraries.

        Args:
            state: Pipeline state with library results

        Returns:
            Updated state with aggregated results
        """
        # Calculate consensus effect if multiple libraries produced estimates
        effects: List[tuple[float, float]] = []  # (effect, confidence)

        if state["causal_effect"] is not None:
            confidence = state["dowhy_result"]["confidence"] if state["dowhy_result"] else 0.8
            effects.append((state["causal_effect"], confidence))

        if state["overall_ate"] is not None:
            confidence = state["econml_result"]["confidence"] if state["econml_result"] else 0.8
            effects.append((state["overall_ate"], confidence))

        if effects:
            # Confidence-weighted average
            total_weight = sum(conf for _, conf in effects)
            if total_weight > 0:
                state["consensus_effect"] = (
                    sum(effect * conf for effect, conf in effects) / total_weight
                )
                state["consensus_confidence"] = total_weight / len(effects)

                # Calculate pairwise agreement
                if len(effects) >= 2:
                    state["library_agreement"] = {
                        "dowhy_econml": 1.0
                        - abs(effects[0][0] - effects[1][0])
                        / max(abs(effects[0][0]), abs(effects[1][0]), 0.001)
                    }

        # Generate executive summary
        state["executive_summary"] = self._generate_summary(state)

        # Generate key insights
        state["key_insights"] = self._generate_insights(state)

        # Generate recommendations
        state["recommended_actions"] = self._generate_recommendations(state)

        return state

    def _generate_summary(self, state: PipelineState) -> str:
        """Generate executive summary from results."""
        parts = []

        question_type = state.get("question_type", "unknown")
        parts.append(f"Analysis type: {question_type.replace('_', ' ').title()}")

        if state["libraries_executed"]:
            parts.append(f"Libraries used: {', '.join(state['libraries_executed'])}")

        if state["consensus_effect"] is not None:
            parts.append(
                f"Consensus effect estimate: {state['consensus_effect']:.4f} "
                f"(confidence: {state['consensus_confidence']:.2%})"
            )
        elif state["causal_effect"] is not None:
            parts.append(f"Causal effect estimate: {state['causal_effect']:.4f}")

        if state["errors"]:
            parts.append(f"Warnings: {len(state['errors'])} library errors occurred")

        return ". ".join(parts)

    def _generate_insights(self, state: PipelineState) -> List[str]:
        """Generate key insights from results."""
        insights = []

        # Graph insights
        if state["causal_graph"]:
            node_count = len(state["causal_graph"].get("nodes", []))
            edge_count = len(state["causal_graph"].get("edges", []))
            if node_count > 0:
                insights.append(
                    f"Causal graph identified {node_count} variables and {edge_count} relationships"
                )

        # Causal effect insights
        if state["causal_effect"] is not None:
            effect = state["causal_effect"]
            direction = "positive" if effect > 0 else "negative" if effect < 0 else "neutral"
            insights.append(f"Treatment shows {direction} causal effect ({effect:.4f})")

        # Heterogeneity insights
        if state["heterogeneity_score"] is not None and state["heterogeneity_score"] > 0.3:
            insights.append(
                f"High treatment effect heterogeneity detected "
                f"(score: {state['heterogeneity_score']:.2f})"
            )

        # Uplift insights
        if state["auuc"] is not None:
            insights.append(f"Uplift model AUUC: {state['auuc']:.4f}")

        return insights

    def _generate_recommendations(self, state: PipelineState) -> List[str]:
        """Generate recommended actions from results."""
        recommendations = []

        # Based on question type
        question_type = state.get("question_type", "")

        if question_type == "targeting_optimization" and state["targeting_recommendations"]:
            recommendations.extend([str(rec) for rec in state["targeting_recommendations"][:3]])

        if question_type == "effect_heterogeneity" and state["cate_by_segment"]:
            recommendations.append("Segment-specific targeting recommended based on CATE analysis")

        if state["heterogeneity_score"] is not None and state["heterogeneity_score"] > 0.5:
            recommendations.append(
                "Consider personalized treatment strategies due to high effect heterogeneity"
            )

        if state["consensus_confidence"] is not None and state["consensus_confidence"] < 0.7:
            recommendations.append(
                "Low confidence in estimates - consider additional data collection"
            )

        if not recommendations:
            recommendations.append("Review detailed results for actionable insights")

        return recommendations


class SequentialPipelineBuilder:
    """Builder for configuring sequential pipelines."""

    def __init__(self):
        self._router: Optional[LibraryRouter] = None
        self._executors: Dict[CausalLibrary, LibraryExecutor] = {}
        self._fail_fast: bool = False

    def with_router(self, router: LibraryRouter) -> "SequentialPipelineBuilder":
        """Set custom router."""
        self._router = router
        return self

    def with_executor(
        self, library: CausalLibrary, executor: LibraryExecutor
    ) -> "SequentialPipelineBuilder":
        """Add custom executor for a library."""
        self._executors[library] = executor
        return self

    def with_fail_fast(self, fail_fast: bool = True) -> "SequentialPipelineBuilder":
        """Enable fail-fast mode."""
        self._fail_fast = fail_fast
        return self

    def build(self) -> SequentialPipeline:
        """Build the pipeline."""
        return SequentialPipeline(
            router=self._router,
            executors=self._executors if self._executors else None,
            fail_fast=self._fail_fast,
        )


def create_sequential_pipeline(
    router: Optional[LibraryRouter] = None,
    fail_fast: bool = False,
) -> SequentialPipeline:
    """Factory function for creating sequential pipelines.

    Args:
        router: Custom router (uses default if not provided)
        fail_fast: Stop on first library failure

    Returns:
        Configured SequentialPipeline instance
    """
    return SequentialPipeline(router=router, fail_fast=fail_fast)
