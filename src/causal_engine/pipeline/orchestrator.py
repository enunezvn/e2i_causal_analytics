"""Pipeline Orchestrator base class.

This module provides the base orchestrator for coordinating multi-library
causal analysis pipelines per the Data Architecture & Integration documentation.

The per-library executor classes (`LibraryExecutor` ABC, `NetworkXExecutor`,
`DoWhyExecutor`, `EconMLExecutor`, `CausalMLExecutor`) live in the sibling
``executors/`` package as of phase C-1 of GH #354. They are re-exported here
for backward compatibility — existing call sites and tests that imported them
from ``src.causal_engine.pipeline.orchestrator`` continue to work without
modification. New code should prefer the canonical paths under
``src.causal_engine.pipeline.executors``.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, cast

from .executors import (
    CausalMLExecutor,
    DoWhyExecutor,
    EconMLExecutor,
    LibraryExecutor,
    NetworkXExecutor,
)
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

# Re-export executor classes for backward compatibility with callers that
# import them from this module (e.g. tests/unit/test_causal_engine/test_pipeline/
# test_orchestrator.py and any external callers).
__all__ = [
    "LibraryExecutor",
    "NetworkXExecutor",
    "DoWhyExecutor",
    "EconMLExecutor",
    "CausalMLExecutor",
    "PipelineOrchestrator",
]


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
            # R6-F1 (#740): thread the opt-in refutation flag into config so the
            # DoWhy executor can run the real refutation suite while the live
            # model/estimand/estimate objects are in scope. Explicit False default
            # keeps the fast path when the request omits the flag.
            "run_refutation": bool(input_data.get("run_refutation") or False),
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
            # First-class DataFrame slot (#458). Propagated from
            # `PipelineInput.estimation_data` so executors can resolve it via
            # `resolve_estimation_dataframe()` without per-executor key drift.
            estimation_data=input_data.get("estimation_data"),
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
            # C-6 extracted channels (NetworkX graph quality, CausalML uplift)
            graph_quality=None,
            uplift_summary=None,
            library_metric_types=None,
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
        """Update state with library execution result.

        Phase C-6 extension (GH #354): in addition to the per-library
        backward-compatible state fields, populates two new top-level
        channels (``state["graph_quality"]`` and ``state["uplift_summary"]``)
        from the rich post-Wave-1 result payloads, and records each
        library's metric type in ``state["library_metric_types"]`` so the
        downstream consensus aggregator can distinguish ATE-track
        contributions from uplift-track contributions.
        """
        # Update library-specific result
        if library == CausalLibrary.NETWORKX:
            state["networkx_result"] = result
            if result["success"] and result["result"]:
                state["causal_graph"] = result["result"]
                state["graph_metrics"] = result["result"].get("centrality", {})
                # C-6: extract structural-quality channel for the
                # consensus-confidence modulator. Read explicit fields
                # rather than relying on truthiness — codex iter-N
                # anti-pattern is to treat None / 0 / False as "skipped"
                # when the executor explicitly emitted them as data.
                self._extract_graph_quality(state, result["result"])
        elif library == CausalLibrary.DOWHY:
            state["dowhy_result"] = result
            if result["success"] and result["result"]:
                state["causal_effect"] = result["result"].get("causal_effect")
                state["refutation_results"] = result["result"].get("refutation_results")
                state["identification_method"] = result["result"].get("identified_estimand")
                # C-6: register DoWhy's metric type for the aggregator.
                self._record_metric_type(state, "dowhy", "ate")
        elif library == CausalLibrary.ECONML:
            state["econml_result"] = result
            if result["success"] and result["result"]:
                state["cate_by_segment"] = result["result"].get("cate_by_segment")
                state["overall_ate"] = result["result"].get("ate")
                state["heterogeneity_score"] = result["result"].get("heterogeneity_score")
                self._record_metric_type(state, "econml", "ate")
        elif library == CausalLibrary.CAUSALML:
            state["causalml_result"] = result
            if result["success"] and result["result"]:
                # C-6: read the post-C-4 key `uplift_scores_summary`
                # (the pre-C-4 `uplift_by_segment` is gone from the
                # post-Wave-1 result payload; reading the legacy key
                # silently dropped real data on the floor).
                state["uplift_scores"] = result["result"].get("uplift_scores_summary")
                state["auuc"] = result["result"].get("auuc")
                state["qini"] = result["result"].get("qini")
                state["targeting_recommendations"] = result["result"].get(
                    "targeting_recommendations"
                )
                # C-6: extract uplift-channel summary distinct from ATE
                # consensus, and register CausalML's ate-track metric.
                self._extract_uplift_summary(state, result["result"])
                if result["result"].get("ate") is not None:
                    self._record_metric_type(state, "causalml", "ate")

        # Update metadata
        state["libraries_executed"].append(library.value)
        state["stage_latencies"][library.value] = result["latency_ms"]

        if not result["success"]:
            state["errors"].append({"library": library.value, "error": result["error"]})

        if result["warnings"]:
            state["warnings"].extend(result["warnings"])

        return state

    @staticmethod
    def _extract_graph_quality(state: PipelineState, payload: Dict[str, Any]) -> None:
        """Populate ``state["graph_quality"]`` from a NetworkX result payload.

        The post-C-5 NetworkX executor emits ``n_nodes``, ``n_edges``,
        ``is_dag``, ``has_treatment_outcome_path``, and ``cycles`` in its
        result. C-6 funnels these into a compact ``graph_quality`` dict
        that includes a derived ``structural_quality`` score:

        - 1.0 when DAG + treatment->outcome path + n_nodes >= 3
        - 0.5 when DAG but path missing OR n_nodes < 3
        - 0.0 when not a DAG (cycles present)

        Mirrors the NetworkX executor's confidence-derivation logic (see
        ``executors/networkx.py::_compute_confidence``) so this aggregator
        sees the same signal.
        """
        is_dag = bool(payload.get("is_dag", False))
        has_path = bool(payload.get("has_treatment_outcome_path", False))
        n_nodes = int(payload.get("n_nodes", 0) or 0)
        n_edges = int(payload.get("n_edges", 0) or 0)
        cycles = payload.get("cycles") or []
        n_cycles = len(cycles) if isinstance(cycles, list) else 0

        if not is_dag:
            structural_quality = 0.0
        elif has_path and n_nodes >= 3:
            structural_quality = 1.0
        else:
            structural_quality = 0.5

        state["graph_quality"] = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "is_dag": is_dag,
            "has_treatment_outcome_path": has_path,
            "structural_quality": structural_quality,
            "n_cycles": n_cycles,
        }

    @staticmethod
    def _extract_uplift_summary(state: PipelineState, payload: Dict[str, Any]) -> None:
        """Populate ``state["uplift_summary"]`` from a CausalML result payload.

        Pulls the load-bearing uplift-quality fields the aggregator and
        downstream consumers need without forcing them to dig through the
        full executor result. ``auuc``/``qini`` may legitimately be
        ``None`` when the metrics-helper raised post-fit (per the
        CausalML executor's documented warning path) — propagate ``None``
        rather than substituting a zero.
        """
        state["uplift_summary"] = {
            "auuc": payload.get("auuc"),
            "qini": payload.get("qini"),
            "ate": payload.get("ate"),
            "ate_ci_lower": payload.get("ate_ci_lower"),
            "ate_ci_upper": payload.get("ate_ci_upper"),
            "n_samples": payload.get("n_samples"),
            "treatment_groups": payload.get("treatment_groups"),
            "control_name": payload.get("control_name"),
            "model": payload.get("model"),
        }

    @staticmethod
    def _record_metric_type(state: PipelineState, library_name: str, metric_type: str) -> None:
        """Record a library's metric type for the aggregator.

        Each library contributes either to the ATE-track (DoWhy, EconML,
        CausalML's population ATE) or to a non-ATE-track (future
        executors may report different metric types). The aggregator
        reads this dict to know which libraries' effects to combine.
        """
        existing = state.get("library_metric_types")
        if not isinstance(existing, dict):
            existing = {}
        existing[library_name] = metric_type
        state["library_metric_types"] = existing

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
