"""Pipeline Orchestrator base class.

This module provides the base orchestrator for coordinating multi-library
causal analysis pipelines per the Data Architecture & Integration documentation.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast

from .router import CausalLibrary, LibraryRouter, RoutingDecision
from .state import (
    LibraryExecutionResult,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineState,
)

logger = logging.getLogger(__name__)


class LibraryExecutor(ABC):
    """Abstract base class for library-specific executors."""

    @property
    @abstractmethod
    def library(self) -> CausalLibrary:
        """Return the library this executor handles."""
        pass

    @abstractmethod
    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute the library's analysis and return results.

        Args:
            state: Current pipeline state with input data
            config: Pipeline configuration

        Returns:
            LibraryExecutionResult with success/failure and result data
        """
        pass

    @abstractmethod
    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate that input state has required fields for this library.

        Args:
            state: Current pipeline state

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class NetworkXExecutor(LibraryExecutor):
    """Executor for NetworkX graph analysis."""

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.NETWORKX

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute NetworkX graph construction and analysis."""
        start_time = time.time()
        try:
            # Placeholder implementation - actual graph analysis would go here
            # In production, this would:
            # 1. Build causal DAG from confounders/effect_modifiers
            # 2. Calculate centrality metrics
            # 3. Identify causal paths
            nodes: List[Any] = []
            edges: List[Dict[str, Any]] = []
            result: Dict[str, Any] = {
                "nodes": nodes,
                "edges": edges,
                "centrality": {},
                "paths": [],
            }

            confounders = state.get("confounders")
            if confounders:
                nodes = list(confounders)
                result["nodes"] = nodes
            if state.get("treatment_var") and state.get("outcome_var"):
                edges.append({"from": state["treatment_var"], "to": state["outcome_var"]})
                nodes.extend([state["treatment_var"], state["outcome_var"]])
                result["nodes"] = list(set(nodes))

            latency_ms = int((time.time() - start_time) * 1000)
            return LibraryExecutionResult(
                library="networkx",
                success=True,
                latency_ms=latency_ms,
                result=result,
                error=None,
                confidence=0.8,
                warnings=[],
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"NetworkX execution failed: {e}")
            return LibraryExecutionResult(
                library="networkx",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(e),
                confidence=0.0,
                warnings=[],
            )

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for NetworkX analysis."""
        if not state.get("treatment_var") and not state.get("confounders"):
            return False, "NetworkX requires treatment_var or confounders"
        return True, ""


class DoWhyExecutor(LibraryExecutor):
    """Executor for DoWhy causal inference."""

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.DOWHY

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute DoWhy causal identification and estimation."""
        start_time = time.time()
        try:
            # Placeholder implementation - actual DoWhy analysis would go here
            # In production, this would:
            # 1. Build causal model from graph structure
            # 2. Identify causal effect
            # 3. Estimate effect
            # 4. Run refutation tests
            result = {
                "identified_estimand": "backdoor",
                "causal_effect": 0.0,
                "confidence_interval": [0.0, 0.0],
                "refutation_results": {},
            }

            # Use graph from NetworkX if available
            if state.get("causal_graph"):
                result["graph_source"] = "networkx"

            latency_ms = int((time.time() - start_time) * 1000)
            return LibraryExecutionResult(
                library="dowhy",
                success=True,
                latency_ms=latency_ms,
                result=result,
                error=None,
                confidence=0.85,
                warnings=[],
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"DoWhy execution failed: {e}")
            return LibraryExecutionResult(
                library="dowhy",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(e),
                confidence=0.0,
                warnings=[],
            )

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for DoWhy analysis."""
        if not state.get("treatment_var"):
            return False, "DoWhy requires treatment_var"
        if not state.get("outcome_var"):
            return False, "DoWhy requires outcome_var"
        return True, ""


class EconMLExecutor(LibraryExecutor):
    """Executor for EconML heterogeneous treatment effects."""

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.ECONML

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute EconML CATE estimation."""
        start_time = time.time()
        try:
            # Placeholder implementation - actual EconML analysis would go here
            # In production, this would:
            # 1. Select appropriate CATE estimator (DML, CausalForest, etc.)
            # 2. Fit model with treatment/outcome/confounders
            # 3. Estimate heterogeneous effects by segment
            result = {
                "estimator": "CausalForestDML",
                "ate": 0.0,
                "cate_by_segment": {},
                "heterogeneity_score": 0.0,
            }

            # Use validated effect from DoWhy if available
            if state.get("causal_effect") is not None:
                result["ate"] = state["causal_effect"]

            latency_ms = int((time.time() - start_time) * 1000)
            return LibraryExecutionResult(
                library="econml",
                success=True,
                latency_ms=latency_ms,
                result=result,
                error=None,
                confidence=0.82,
                warnings=[],
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"EconML execution failed: {e}")
            return LibraryExecutionResult(
                library="econml",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(e),
                confidence=0.0,
                warnings=[],
            )

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for EconML analysis."""
        if not state.get("treatment_var"):
            return False, "EconML requires treatment_var"
        if not state.get("outcome_var"):
            return False, "EconML requires outcome_var"
        return True, ""


class CausalMLExecutor(LibraryExecutor):
    """Executor for CausalML uplift modeling."""

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.CAUSALML

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute CausalML uplift modeling."""
        start_time = time.time()
        try:
            # Placeholder implementation - actual CausalML analysis would go here
            # In production, this would:
            # 1. Select uplift model (Random Forest, XGBoost, etc.)
            # 2. Train on treatment/outcome data
            # 3. Calculate uplift scores per segment
            # 4. Generate targeting recommendations
            result = {
                "model": "UpliftRandomForest",
                "auuc": 0.0,
                "qini": 0.0,
                "uplift_by_segment": {},
                "targeting_recommendations": [],
            }

            # Use CATE from EconML if available for comparison
            if state.get("cate_by_segment"):
                result["econml_comparison"] = "available"

            latency_ms = int((time.time() - start_time) * 1000)
            return LibraryExecutionResult(
                library="causalml",
                success=True,
                latency_ms=latency_ms,
                result=result,
                error=None,
                confidence=0.78,
                warnings=[],
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"CausalML execution failed: {e}")
            return LibraryExecutionResult(
                library="causalml",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(e),
                confidence=0.0,
                warnings=[],
            )

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for CausalML analysis."""
        if not state.get("treatment_var"):
            return False, "CausalML requires treatment_var"
        if not state.get("outcome_var"):
            return False, "CausalML requires outcome_var"
        return True, ""


class PipelineOrchestrator(ABC):
    """Base class for pipeline orchestration.

    The orchestrator coordinates execution across multiple causal libraries,
    handling routing, execution, and result aggregation.

    Reference: docs/Data Architecture & Integration.html
    """

    def __init__(
        self,
        router: Optional[LibraryRouter] = None,
        executors: Optional[Dict[CausalLibrary, LibraryExecutor]] = None,
    ):
        """Initialize the orchestrator.

        Args:
            router: Library router for question classification
            executors: Map of library to executor (uses defaults if not provided)
        """
        self.router = router or LibraryRouter()
        self.executors = executors or self._default_executors()

    def _default_executors(self) -> Dict[CausalLibrary, LibraryExecutor]:
        """Create default executors for all libraries."""
        return {
            CausalLibrary.NETWORKX: NetworkXExecutor(),
            CausalLibrary.DOWHY: DoWhyExecutor(),
            CausalLibrary.ECONML: EconMLExecutor(),
            CausalLibrary.CAUSALML: CausalMLExecutor(),
        }

    def _create_initial_state(
        self,
        input_data: PipelineInput,
        routing_decision: RoutingDecision,
    ) -> PipelineState:
        """Create initial pipeline state from input and routing decision."""
        libraries = [routing_decision.primary_library.value]
        libraries.extend([lib.value for lib in routing_decision.secondary_libraries])

        config: PipelineConfig = {
            "mode": routing_decision.recommended_mode,  # type: ignore[typeddict-item]
            "libraries_enabled": libraries,
            "primary_library": routing_decision.primary_library.value,
            "stage_timeout_ms": cast(int, input_data.get("stage_timeout_ms", 30000) or 30000),
            "total_timeout_ms": cast(int, input_data.get("total_timeout_ms", 120000) or 120000),
            "cross_validate": bool(input_data.get("cross_validate", True)),
            "min_agreement_threshold": 0.85,
            "max_parallel_libraries": 4,
            "fail_fast": False,
            "segment_by_uplift": False,
            "nested_ci_level": 0.95,
        }

        return PipelineState(
            # Input
            query=input_data["query"],
            question_type=routing_decision.question_type.value,
            treatment_var=input_data.get("treatment_var"),
            outcome_var=input_data.get("outcome_var"),
            confounders=input_data.get("confounders"),
            effect_modifiers=input_data.get("effect_modifiers"),
            data_source=input_data["data_source"],
            filters=input_data.get("filters"),
            # Configuration
            config=config,
            # Routing
            routed_libraries=libraries,
            routing_confidence=routing_decision.confidence,
            routing_rationale=routing_decision.rationale,
            # Library results (initially empty)
            networkx_result=None,
            causal_graph=None,
            graph_metrics=None,
            dowhy_result=None,
            causal_effect=None,
            refutation_results=None,
            identification_method=None,
            econml_result=None,
            cate_by_segment=None,
            overall_ate=None,
            heterogeneity_score=None,
            causalml_result=None,
            uplift_scores=None,
            auuc=None,
            qini=None,
            targeting_recommendations=None,
            # Aggregated outputs
            consensus_effect=None,
            consensus_confidence=None,
            library_agreement=None,
            nested_cate=None,
            segment_confidence_intervals=None,
            executive_summary=None,
            key_insights=None,
            recommended_actions=None,
            # Execution metadata
            current_stage=PipelineStage.PENDING,
            stage_latencies={},
            total_latency_ms=0,
            libraries_executed=[],
            libraries_skipped=[],
            # Error handling
            errors=[],
            warnings=[],
            status="pending",
        )

    def _update_state_with_result(
        self,
        state: PipelineState,
        library: CausalLibrary,
        result: LibraryExecutionResult,
    ) -> PipelineState:
        """Update state with library execution result."""
        # Update library-specific result
        if library == CausalLibrary.NETWORKX:
            state["networkx_result"] = result
            if result["success"] and result["result"]:
                state["causal_graph"] = result["result"]
                state["graph_metrics"] = result["result"].get("centrality", {})
        elif library == CausalLibrary.DOWHY:
            state["dowhy_result"] = result
            if result["success"] and result["result"]:
                state["causal_effect"] = result["result"].get("causal_effect")
                state["refutation_results"] = result["result"].get("refutation_results")
                state["identification_method"] = result["result"].get("identified_estimand")
        elif library == CausalLibrary.ECONML:
            state["econml_result"] = result
            if result["success"] and result["result"]:
                state["cate_by_segment"] = result["result"].get("cate_by_segment")
                state["overall_ate"] = result["result"].get("ate")
                state["heterogeneity_score"] = result["result"].get("heterogeneity_score")
        elif library == CausalLibrary.CAUSALML:
            state["causalml_result"] = result
            if result["success"] and result["result"]:
                state["uplift_scores"] = result["result"].get("uplift_by_segment")
                state["auuc"] = result["result"].get("auuc")
                state["qini"] = result["result"].get("qini")
                state["targeting_recommendations"] = result["result"].get(
                    "targeting_recommendations"
                )

        # Update metadata
        state["libraries_executed"].append(library.value)
        state["stage_latencies"][library.value] = result["latency_ms"]

        if not result["success"]:
            state["errors"].append({"library": library.value, "error": result["error"]})

        if result["warnings"]:
            state["warnings"].extend(result["warnings"])

        return state

    def _create_output(self, state: PipelineState) -> PipelineOutput:
        """Create output from final state."""
        # Determine primary result based on primary library
        primary_lib = state["config"]["primary_library"]
        primary_result: Dict[str, Any] = {}

        if primary_lib == "networkx" and state["networkx_result"]:
            primary_result = state["networkx_result"].get("result") or {}
        elif primary_lib == "dowhy" and state["dowhy_result"]:
            primary_result = state["dowhy_result"].get("result") or {}
        elif primary_lib == "econml" and state["econml_result"]:
            primary_result = state["econml_result"].get("result") or {}
        elif primary_lib == "causalml" and state["causalml_result"]:
            primary_result = state["causalml_result"].get("result") or {}

        # Determine status
        if state["errors"]:
            if state["libraries_executed"]:
                status = "partial"
            else:
                status = "failed"
        else:
            status = "completed"

        return PipelineOutput(
            question_type=state["question_type"] or "unknown",
            primary_result=primary_result,
            libraries_used=state["libraries_executed"],
            consensus_effect=state["consensus_effect"],
            consensus_confidence=state["consensus_confidence"],
            executive_summary=state["executive_summary"] or "",
            key_insights=state["key_insights"] or [],
            recommended_actions=state["recommended_actions"] or [],
            total_latency_ms=state["total_latency_ms"],
            status=status,  # type: ignore
            warnings=state["warnings"],
            errors=state["errors"],
        )

    @abstractmethod
    async def execute(self, input_data: PipelineInput) -> PipelineOutput:
        """Execute the pipeline.

        Args:
            input_data: Pipeline input with query and configuration

        Returns:
            PipelineOutput with results from all executed libraries
        """
        pass

    async def route(self, query: str, **kwargs: Any) -> RoutingDecision:
        """Route a query to appropriate libraries.

        Args:
            query: Natural language query
            **kwargs: Additional routing parameters

        Returns:
            RoutingDecision with primary/secondary libraries
        """
        return self.router.route(query, **kwargs)
