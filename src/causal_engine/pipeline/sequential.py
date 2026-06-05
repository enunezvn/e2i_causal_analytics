"""Sequential Pipeline implementation.

Implements the NetworkX → DoWhy → EconML → CausalML sequential flow
as defined in the Data Architecture & Integration documentation.
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

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


# ----------------------------------------------------------------------------- #
# C-6 aggregation helpers (shared by sequential.py and parallel.py).
#
# These live at module level rather than as methods so the parallel pipeline
# can import + reuse the same logic without forcing both pipeline classes
# under a common mixin. The helpers operate on PipelineState directly and
# mutate it in place; they return nothing.
# ----------------------------------------------------------------------------- #


def _is_valid_confidence(value: object) -> bool:
    """Confidence is valid iff it's a finite numeric in [0, 1].

    Returns False for: None, NaN, inf, non-numeric, out-of-range. This is
    the rule that REPLACES the pre-C-6 ``else 0.8`` silent-default fallback
    (Wave-3 anti-mocking pattern #2). A library with an invalid confidence
    is EXCLUDED from consensus, not coerced.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        # bool is a subclass of int but logically not a confidence value.
        return False
    if not isinstance(value, (int, float)):
        return False
    f = float(value)
    if not math.isfinite(f):
        return False
    return 0.0 <= f <= 1.0


def _collect_ate_estimates(state: PipelineState) -> List[Tuple[str, float, float]]:
    """Collect ATE-track effect estimates from successful libraries.

    Returns a list of ``(library_name, effect, confidence)`` triples — one
    per library that produced a finite numeric effect AND has a valid
    confidence. The output is suitable for both consensus averaging and
    pairwise agreement.

    Per C-6: includes DoWhy (``causal_effect``), EconML (``overall_ate``),
    AND CausalML (``uplift_summary.ate``). CausalML's auuc/qini are NOT
    included here — they live in the separate uplift channel.

    A library is INCLUDED iff:
    1. Its result dict is present in state and ``success`` is truthy
       (or has no ``success`` key but a populated effect — preserves
       back-compat with tests that wire result dicts directly)
    2. The effect value is finite numeric
    3. The result's ``confidence`` passes ``_is_valid_confidence``
    """
    effects: List[Tuple[str, float, float]] = []

    # --- DoWhy: causal_effect + dowhy_result.confidence ---
    dowhy_effect = state.get("causal_effect")
    dowhy_result = state.get("dowhy_result")
    if (
        dowhy_effect is not None
        and isinstance(dowhy_effect, (int, float))
        and math.isfinite(float(dowhy_effect))
        and isinstance(dowhy_result, dict)
        and dowhy_result.get("success") is not False  # treat missing 'success' as truthy
        and _is_valid_confidence(dowhy_result.get("confidence"))
    ):
        effects.append(("dowhy", float(dowhy_effect), float(dowhy_result["confidence"])))

    # --- EconML: overall_ate + econml_result.confidence ---
    econml_effect = state.get("overall_ate")
    econml_result = state.get("econml_result")
    if (
        econml_effect is not None
        and isinstance(econml_effect, (int, float))
        and math.isfinite(float(econml_effect))
        and isinstance(econml_result, dict)
        and econml_result.get("success") is not False
        and _is_valid_confidence(econml_result.get("confidence"))
    ):
        effects.append(("econml", float(econml_effect), float(econml_result["confidence"])))

    # --- CausalML: ate (population ATE from uplift fit) + causalml_result.confidence ---
    # CausalML's ATE is read from `state["uplift_summary"]["ate"]` (the
    # extraction channel populated by `_update_state_with_result` from
    # the post-C-4 result_payload) — and, as a fallback for callers that
    # exercise the aggregator directly without going through the orchestrator
    # update path (e.g., unit tests that pre-populate raw library result
    # dicts), from `causalml_result["result"]["ate"]`. We explicitly do
    # NOT use auuc/qini here — those are model-quality metrics, not
    # effect-magnitude estimates, and live in the SEPARATE uplift channel.
    causalml_result = state.get("causalml_result")
    causalml_ate: object = None
    uplift_summary = state.get("uplift_summary")
    if isinstance(uplift_summary, dict) and uplift_summary.get("ate") is not None:
        causalml_ate = uplift_summary.get("ate")
    elif isinstance(causalml_result, dict):
        # Fallback path — aggregator invoked without orchestrator extraction.
        raw_result = causalml_result.get("result")
        if isinstance(raw_result, dict):
            causalml_ate = raw_result.get("ate")
    if (
        causalml_ate is not None
        and isinstance(causalml_ate, (int, float))
        and math.isfinite(float(causalml_ate))
        and isinstance(causalml_result, dict)
        and causalml_result.get("success") is not False
        and _is_valid_confidence(causalml_result.get("confidence"))
    ):
        effects.append(("causalml", float(causalml_ate), float(causalml_result["confidence"])))

    return effects


_Z_95 = 1.959963984540054  # z for a 95% normal CI (half-width / z = SE)


def _se_for_library(state: PipelineState, lib: str) -> Optional[float]:
    """Resolve a per-library standard error of the ATE for inverse-variance
    weighting (H9). DoWhy emits ``standard_error`` directly; EconML / CausalML
    expose a 95% CI, so SE = (ci_upper − ci_lower) / (2·z). Returns None when no
    valid positive SE is available."""

    def _ci_to_se(lo: object, hi: object) -> Optional[float]:
        try:
            lo_f, hi_f = float(lo), float(hi)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if math.isfinite(lo_f) and math.isfinite(hi_f) and hi_f > lo_f:
            return (hi_f - lo_f) / (2.0 * _Z_95)
        return None

    def _payload(key: str) -> Optional[Dict[str, object]]:
        r = state.get(key)
        if isinstance(r, dict):
            p = r.get("result")
            return p if isinstance(p, dict) else None
        return None

    if lib == "dowhy":
        p = _payload("dowhy_result")
        if p is not None:
            try:
                se = float(p.get("standard_error"))  # type: ignore[arg-type]
                if math.isfinite(se) and se > 0:
                    return se
            except (TypeError, ValueError):
                pass
        return None
    if lib == "econml":
        p = _payload("econml_result")
        return _ci_to_se(p.get("ate_ci_lower"), p.get("ate_ci_upper")) if p else None
    if lib == "causalml":
        us = state.get("uplift_summary")
        if isinstance(us, dict):
            # Distinct name: the dowhy branch binds ``se: float``; reusing it here
            # for the Optional return of _ci_to_se would conflict (mypy).
            se_uplift = _ci_to_se(us.get("ate_ci_lower"), us.get("ate_ci_upper"))
            if se_uplift is not None:
                return se_uplift
        p = _payload("causalml_result")
        return _ci_to_se(p.get("ate_ci_lower"), p.get("ate_ci_upper")) if p else None
    return None


def _apply_consensus(state: PipelineState, effects: List[Tuple[str, float, float]]) -> None:
    """Compute consensus_effect + consensus_confidence and write into state.

    - ``consensus_effect`` = INVERSE-VARIANCE-weighted average of ATE estimates
      (weight ∝ 1/SE²) when every contributing library exposes a valid SE, so a
      library's influence reflects its PRECISION (H9) — NOT the incommensurable
      per-library confidence scale (DoWhy hardcodes confidence=1.0 and would
      otherwise structurally dominate). Falls back to confidence-weighting when
      any SE is unavailable.
    - ``consensus_confidence`` = arithmetic mean of contributing confidences,
      OPTIONALLY MODULATED by NetworkX structural quality:

      ``modulated = mean_confidence * (0.5 + 0.5 * structural_quality)``

      Effects:
      - structural_quality = 1.0 (perfect DAG + path)  =>  no penalty
      - structural_quality = 0.5 (DAG but path missing) => 25% penalty
      - structural_quality = 0.0 (non-DAG, cycles)     => 50% penalty
      - graph_quality not set (NetworkX skipped)        => no penalty
        (absence of evidence != evidence of absence)
    """
    total_weight = sum(conf for _, _, conf in effects)
    if total_weight <= 0:
        return

    # H9: inverse-variance weighting only when EVERY library has a valid positive
    # SE. This is intentionally all-or-nothing, NOT per-library: mixing 1/SE²
    # weights (for libraries with an SE) and confidence weights (for those
    # without) on the same average would re-introduce exactly the
    # incommensurable-scale problem this fix removes. A zero SE (e.g. a
    # constant-prediction estimator) also fails the gate, avoiding an infinite
    # 1/SE² weight.
    ses = {lib: _se_for_library(state, lib) for lib, _, _ in effects}
    # Walrus binds a local mypy can narrow — a bare ``ses[lib] > 0`` after
    # ``ses[lib] is not None`` is not narrowed (subscript, not a name).
    use_inverse_variance = all((se := ses[lib]) is not None and se > 0 for lib, _, _ in effects)
    if use_inverse_variance:
        weights = [1.0 / (ses[lib] ** 2) for lib, _, _ in effects]  # type: ignore[operator]
        weight_sum = sum(weights)
        consensus_effect = (
            sum(eff * w for (_, eff, _), w in zip(effects, weights, strict=False)) / weight_sum
        )
        state["consensus_weighting"] = "inverse_variance"
    else:
        consensus_effect = sum(effect * conf for _, effect, conf in effects) / total_weight
        state["consensus_weighting"] = "confidence"

    base_confidence = total_weight / len(effects)

    # Apply NetworkX structural-quality modulation (C-6).
    graph_quality = state.get("graph_quality")
    if isinstance(graph_quality, dict):
        sq = graph_quality.get("structural_quality")
        if isinstance(sq, (int, float)) and math.isfinite(float(sq)):
            sq_f = max(0.0, min(1.0, float(sq)))
            # Linear mix: at sq=1.0, multiplier=1.0 (no penalty); at sq=0.0,
            # multiplier=0.5 (50% penalty). Intermediate values interpolate.
            multiplier = 0.5 + 0.5 * sq_f
            base_confidence = base_confidence * multiplier

    state["consensus_effect"] = float(consensus_effect)
    state["consensus_confidence"] = float(max(0.0, min(1.0, base_confidence)))


def _apply_pairwise_agreement(
    state: PipelineState, effects: List[Tuple[str, float, float]]
) -> None:
    """Populate state['library_agreement'] with all pairs of contributing libraries.

    Agreement between two effects ``a`` and ``b`` is
    ``1.0 - |a - b| / max(|a|, |b|, 0.001)`` clipped to [0, 1].
    """
    if len(effects) < 2:
        return

    agreement: Dict[str, float] = {}
    for i, (lib_a, eff_a, _) in enumerate(effects):
        for lib_b, eff_b, _ in effects[i + 1 :]:
            pair = f"{lib_a}_{lib_b}"
            denom = max(abs(eff_a), abs(eff_b), 0.001)
            diff = abs(eff_a - eff_b) / denom
            agreement[pair] = max(0.0, min(1.0, 1.0 - diff))

    state["library_agreement"] = agreement


def _apply_agreement_score(state: PipelineState) -> None:
    """Surface a REAL library-agreement metric for ``library_agreement_score`` (H8).

    This is the mean of the pairwise concordances in ``state['library_agreement']``
    — a genuine measure of whether the libraries AGREE on the effect. It is a
    SEPARATE quantity from ``consensus_confidence`` (the mean of per-library
    confidences), which the API previously mislabeled as the agreement score:
    two libraries reporting +0.5 and −0.5 cancel to ~0 effect yet keep a high
    consensus_confidence, so surfacing that as "library agreement" was misleading.
    """
    agreement = state.get("library_agreement")
    if isinstance(agreement, dict) and agreement:
        vals = [float(v) for v in agreement.values() if isinstance(v, (int, float))]
        if vals:
            state["library_agreement_score"] = float(sum(vals) / len(vals))
            return
    state["library_agreement_score"] = None


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
        """Aggregate results from all libraries (4-library ATE consensus +
        separate uplift channel + structural-quality modulation).

        Per phase C-6 of GH #354:

        - Effect estimates are collected from DoWhy (``causal_effect``),
          EconML (``overall_ate``), and CausalML (``uplift_summary.ate``
          — population ATE from the uplift fit). CausalML's auuc/qini
          live in ``state["uplift_summary"]`` (set by
          ``_update_state_with_result``) — they are NEVER averaged into
          ``consensus_effect`` because uplift metrics answer a different
          question (population-targeting quality vs effect magnitude).
        - A library is INCLUDED iff: (1) it produced a finite numeric
          effect, AND (2) its ``LibraryExecutionResult.confidence`` is
          present and finite (numeric in [0, 1]). A missing/None/NaN
          confidence EXCLUDES the library; we never silent-default to
          0.8 (Wave-3 anti-mocking pattern #2).
        - Pairwise agreement is computed for ALL pairs of contributing
          libraries (not the hardcoded ``dowhy_econml`` pair only).
        - ``consensus_confidence`` is modulated by NetworkX's
          ``structural_quality`` when ``state["graph_quality"]`` is
          populated. A non-DAG (structural_quality=0.0) penalizes
          consensus confidence by 50% (multiplicative). A partial DAG
          (0.5) penalizes by 25%. A perfect DAG (1.0) applies no
          penalty. When NetworkX did not run (graph_quality is None),
          no penalty is applied — absence of evidence is not evidence
          of absence.

        Args:
            state: Pipeline state with library results

        Returns:
            Updated state with aggregated results
        """
        # Collect effect estimates with VALID confidences (no 0.8 fallback).
        effects = _collect_ate_estimates(state)

        if effects:
            _apply_consensus(state, effects)
            _apply_pairwise_agreement(state, effects)
            _apply_agreement_score(state)

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

        question_type = state.get("question_type") or "unknown"
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
