"""Segment Analyzer Node for Heterogeneous Optimizer Agent.

This node analyzes segments to identify high/low responders based on CATE estimates.
Pure computation - no LLM needed.

V4.4: Added DAG validation for segment-specific effects.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..state import CATEResult, HeterogeneousOptimizerState, SegmentProfile

# V4.4: DAG validation constants
LATENT_CONFOUNDER_WARNING_THRESHOLD = 0.3  # Warn if >30% segments have latent confounders

logger = logging.getLogger(__name__)


class SegmentAnalyzerNode:
    """Analyze segments to identify genuine differential responders.

    Classification is significance-gated against the overall ATE (the SSOT shared
    with the policy learner's expected-lift gate), so the responder tiles can never
    contradict the lift the page reports:

    - High (above-average) responder: CATE CI entirely above the ATE *and* above 0.
    - Harmful responder: CATE CI entirely below 0.
    - Average responder: CI overlaps the ATE (statistically indistinguishable).

    A homogeneous effect honestly yields zero high/harmful responders. See
    ``_classify_by_significance``.
    """

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute segment analysis."""
        start_time = time.time()
        logger.info(
            "Starting segment analysis",
            extra={
                "node": "segment_analyzer",
                "overall_ate": state.get("overall_ate"),
                "segment_count": len(state.get("cate_by_segment") or {}),
            },
        )

        if state.get("status") == "failed":
            logger.warning("Skipping segment analysis - previous node failed")
            return state

        # #437 fail-close: if upstream cate_estimator did NOT set status=failed
        # but produced no overall_ate, downstream MUST fail-close rather than
        # synthesize a successful-looking response. Heterogeneity analysis is
        # undefined without a baseline ATE; an empty/populated cate_by_segment
        # does not redeem a missing ATE. Honest zero (overall_ate == 0.0) is
        # distinct from absence (None) and remains supported.
        if state.get("overall_ate") is None:
            logger.error(
                "Fail-closed: upstream cate_estimator produced no overall_ate",
                extra={"node": "segment_analyzer"},
            )
            return {
                **state,
                "errors": [
                    {
                        "node": "segment_analyzer",
                        "error": (
                            "upstream cate_estimator produced overall_ate=None "
                            "without setting status=failed; refusing to "
                            "synthesize heterogeneity analysis without baseline ATE"
                        ),
                    }
                ],
                "status": "failed",
            }

        try:
            raw_ate = state.get("overall_ate")
            # Past the fail-close gate above, ``overall_ate`` is guaranteed to
            # be present (a real float, including legitimate 0.0). ``or 0.0``
            # would silently turn a legitimate 0.0 into 0.0 — same outcome —
            # but using an explicit ``is None`` guard keeps the semantic of
            # "honest zero is allowed" load-bearing.
            ate: float = float(raw_ate) if raw_ate is not None else 0.0
            cate_by_segment_raw = state.get("cate_by_segment")
            cate_by_segment: Dict[str, List[CATEResult]] = (
                cate_by_segment_raw if cate_by_segment_raw is not None else {}
            )
            top_count = state.get("top_segments_count", 10)

            # #437 row 4: real ATE but empty cate_by_segment is an honest
            # partial-data case. Emit an explicit warning rather than silently
            # treating absence as "no heterogeneity."
            partial_data_warning: Optional[str] = None
            if not cate_by_segment:
                partial_data_warning = (
                    "cate_by_segment is empty; heterogeneity analysis will "
                    "report zero responder segments and heterogeneity_score=0"
                )
                logger.warning(
                    partial_data_warning,
                    extra={"node": "segment_analyzer"},
                )

            # Flatten all segment results
            all_segments = []
            total_size = 0

            for segment_var, results in (cate_by_segment or {}).items():
                for result in results:
                    total_size += result["sample_size"]
                    all_segments.append({"segment_var": segment_var, "result": result})

            # V4.4: DAG validation if available
            dag_validation_warnings: List[str] = []
            dag_validated_segments: Optional[List[str]] = None
            dag_invalid_segments: Optional[List[str]] = None
            latent_confounder_segments: Optional[List[str]] = None

            if self._has_dag_evidence(state):
                logger.info(
                    "DAG evidence available, validating segment effects",
                    extra={"node": "segment_analyzer"},
                )
                (
                    dag_validated_segments,
                    dag_invalid_segments,
                    latent_confounder_segments,
                    dag_validation_warnings,
                ) = self._validate_segment_effects(all_segments, state)

            # Classify by the SAME above-ATE significance gate the policy/lift uses
            # (SSOT for "is this a genuine differential responder"), so the responder
            # TILES can never contradict the expected lift the page reports. A
            # homogeneous effect honestly yields ZERO high/harmful responders (matching
            # ~0 lift) instead of a relative top/bottom-half fallback that always
            # fabricated responders. The full effect-size ranking is still available on
            # the CATE plot for exploration.
            high_responders, low_responders, mid_responders = self._classify_by_significance(
                all_segments, ate, total_size
            )
            high_responders = high_responders[:top_count]
            low_responders = low_responders[:top_count]
            mid_responders = mid_responders[:top_count]

            # Create segment comparison
            comparison = self._create_comparison(
                high_responders, low_responders, ate, mid_responders
            )

            analysis_time = int((time.time() - start_time) * 1000)

            logger.info(
                "Segment analysis complete",
                extra={
                    "node": "segment_analyzer",
                    "high_responder_count": len(high_responders),
                    "low_responder_count": len(low_responders),
                    "effect_ratio": comparison.get("effect_ratio"),
                    "dag_validated_count": len(dag_validated_segments)
                    if dag_validated_segments
                    else 0,
                    "dag_invalid_count": len(dag_invalid_segments) if dag_invalid_segments else 0,
                    "latent_confounder_count": len(latent_confounder_segments)
                    if latent_confounder_segments
                    else 0,
                    "latency_ms": analysis_time,
                },
            )

            output_state: Dict[str, Any] = {
                **state,
                "high_responders": high_responders,
                "mid_responders": mid_responders,
                "low_responders": low_responders,
                "segment_comparison": comparison,
                "analysis_latency_ms": analysis_time,
                "status": "optimizing",
            }

            # #437 row 4: surface the partial-data warning in the standard
            # ``warnings`` channel. LangGraph reduces ``warnings`` with
            # ``operator.add``; in unit tests called directly on the node,
            # the returned dict carries the warning as a single-element list.
            if partial_data_warning is not None:
                existing_warnings = state.get("warnings") or []
                output_state["warnings"] = [*existing_warnings, partial_data_warning]

            # V4.4: Add DAG validation results if available
            if dag_validated_segments is not None:
                output_state["dag_validated_segments"] = dag_validated_segments
            if dag_invalid_segments is not None:
                output_state["dag_invalid_segments"] = dag_invalid_segments
            if latent_confounder_segments is not None:
                output_state["latent_confounder_segments"] = latent_confounder_segments
            if dag_validation_warnings:
                output_state["dag_validation_warnings"] = dag_validation_warnings

            return output_state  # type: ignore[return-value]

        except Exception as e:
            logger.error(
                "Segment analysis failed",
                extra={"node": "segment_analyzer", "error": str(e)},
                exc_info=True,
            )
            return {
                **state,
                "errors": [{"node": "segment_analyzer", "error": str(e)}],
                "status": "failed",
            }

    def _classify_by_significance(
        self,
        all_segments: List[Dict],
        ate: float,
        total_size: int,
    ) -> Tuple[List[SegmentProfile], List[SegmentProfile], List[SegmentProfile]]:
        """Classify every segment by the SAME above-ATE significance gate the policy
        learner uses for the expected lift (the SSOT for "is this a genuine
        differential responder"), so the responder tiles can NEVER contradict the lift:

        - HIGH (above-average): CATE CI lies ENTIRELY above the ATE *and* above zero
          (``cate_ci_lower > ate`` and ``> 0``) -> exactly the policy's INCREASE set.
        - HARMFUL (low): CATE CI lies ENTIRELY below zero (``cate_ci_upper < 0``) ->
          exactly the policy's DECREASE set.
        - AVERAGE (mid): CI overlaps the ATE -> statistically indistinguishable from
          the average effect; maintained.

        There is NO relative top/bottom-half fallback: a homogeneous effect honestly
        yields ZERO high/harmful responders (matching ~0 expected lift) instead of a
        fabricated top-half. The full effect-size ranking remains on the CATE plot.

        Returns (high, harmful, average) profile lists, each sorted for display.
        """
        high: List[SegmentProfile] = []
        harmful: List[SegmentProfile] = []
        average: List[SegmentProfile] = []
        for seg in all_segments:
            result = seg["result"]
            ci_lower = result.get("cate_ci_lower")
            ci_upper = result.get("cate_ci_upper")
            if ci_lower is not None and ci_lower > ate and ci_lower > 0:
                high.append(self._build_profile(seg, ate, total_size, "high"))
            elif ci_upper is not None and ci_upper < 0:
                harmful.append(self._build_profile(seg, ate, total_size, "low"))
            else:
                average.append(self._build_profile(seg, ate, total_size, "average"))
        high.sort(key=lambda x: x["cate_estimate"], reverse=True)
        harmful.sort(key=lambda x: x["cate_estimate"])
        average.sort(key=lambda x: abs(x["cate_estimate"] - ate))
        return high, harmful, average

    def _build_profile(
        self, seg: Dict, ate: float, total_size: int, responder_type: str
    ) -> SegmentProfile:
        """Build a SegmentProfile for a classified segment (one shared shape for the
        high / harmful / average buckets)."""
        result = seg["result"]
        cate = result["cate_estimate"]
        return SegmentProfile(
            segment_id=f"{seg['segment_var']}_{result['segment_value']}",
            responder_type=responder_type,  # type: ignore[typeddict-item]
            cate_estimate=cate,
            defining_features=[
                {
                    "variable": seg["segment_var"],
                    "value": result["segment_value"],
                    "effect_size": cate / ate if ate != 0 else 0,
                }
            ],
            size=result["sample_size"],
            size_percentage=result["sample_size"] / total_size * 100 if total_size > 0 else 0,
            recommendation=self._generate_recommendation(
                seg["segment_var"], result, responder_type
            ),
        )

    def _generate_recommendation(
        self, segment_var: str, result: CATEResult, responder_type: str
    ) -> str:
        """Generate action recommendation for segment.

        Args:
            segment_var: Segment variable name
            result: CATE result for this segment
            responder_type: 'high' or 'low'

        Returns:
            Action recommendation string
        """

        segment_value = result["segment_value"]
        cate = result["cate_estimate"]

        if responder_type == "high":
            return f"Prioritize treatment for {segment_var}={segment_value} (CATE: {cate:.3f}). High response expected."
        elif responder_type == "average":
            return f"Maintain current targeting for {segment_var}={segment_value} (CATE: {cate:.3f}). Near-average response — monitor, no reallocation indicated."
        else:
            return f"De-prioritize treatment for {segment_var}={segment_value} (CATE: {cate:.3f}). Consider alternative interventions."

    def _create_comparison(
        self,
        high_responders: List[SegmentProfile],
        low_responders: List[SegmentProfile],
        ate: float,
        mid_responders: Optional[List[SegmentProfile]] = None,
    ) -> Dict[str, Any]:
        """Create comparison summary between high, mid and low responders.

        Args:
            high_responders: High responder segments
            low_responders: Low responder segments
            ate: Overall ATE
            mid_responders: Mid ("average") responder segments (optional; counted
                so downstream narratives reflect three buckets, not two)

        Returns:
            Comparison dictionary
        """

        mid_responders = mid_responders or []

        high_avg_cate = (
            sum(h["cate_estimate"] for h in high_responders) / len(high_responders)
            if high_responders
            else 0
        )
        low_avg_cate = (
            sum(l["cate_estimate"] for l in low_responders) / len(low_responders)
            if low_responders
            else 0
        )
        mid_avg_cate = (
            sum(m["cate_estimate"] for m in mid_responders) / len(mid_responders)
            if mid_responders
            else 0
        )

        return {
            "overall_ate": ate,
            "high_responder_avg_cate": high_avg_cate,
            "mid_responder_avg_cate": mid_avg_cate,
            "low_responder_avg_cate": low_avg_cate,
            "effect_ratio": high_avg_cate / low_avg_cate if low_avg_cate != 0 else float("inf"),
            "high_responder_count": len(high_responders),
            "mid_responder_count": len(mid_responders),
            "low_responder_count": len(low_responders),
        }

    # =========================================================================
    # V4.4: DAG Validation Methods
    # =========================================================================

    def _has_dag_evidence(self, state: HeterogeneousOptimizerState) -> bool:
        """Check if DAG evidence is available for validation.

        Args:
            state: Current optimizer state

        Returns:
            True if DAG evidence is available and valid
        """
        dag_adjacency = state.get("discovered_dag_adjacency")
        dag_nodes = state.get("discovered_dag_nodes")
        discovery_gate_decision = state.get("discovery_gate_decision")

        # DAG evidence is available if:
        # 1. We have DAG adjacency matrix and nodes
        # 2. Discovery gate decision is accept or review (not reject)
        return (
            dag_adjacency is not None
            and dag_nodes is not None
            and len(dag_adjacency) > 0
            and len(dag_nodes) > 0
            and discovery_gate_decision in ("accept", "review")
        )

    def _validate_segment_effects(
        self,
        all_segments: List[Dict],
        state: HeterogeneousOptimizerState,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Validate segment-specific effects against discovered DAG.

        V4.4: Check if treatment → segment path exists in DAG and detect
        latent confounders from FCI bidirected edges.

        Args:
            all_segments: All segment results
            state: Current optimizer state with DAG information

        Returns:
            Tuple of (validated_segments, invalid_segments, latent_confounder_segments, warnings)
        """
        validated_segments: List[str] = []
        invalid_segments: List[str] = []
        latent_confounder_segments: List[str] = []
        warnings: List[str] = []

        dag_adjacency: List[List[int]] = state.get("discovered_dag_adjacency") or []
        dag_nodes: List[str] = state.get("discovered_dag_nodes") or []
        edge_types: Dict[str, str] = state.get("discovered_dag_edge_types") or {}
        treatment_var = state.get("treatment_var", "")
        outcome_var = state.get("outcome_var", "")

        # Build node index lookup
        node_to_idx = {node: idx for idx, node in enumerate(dag_nodes)}

        for seg in all_segments:
            segment_var = seg["segment_var"]
            segment_value = seg["result"]["segment_value"]
            segment_id = f"{segment_var}_{segment_value}"

            # Check if segment variable is in DAG
            if segment_var not in node_to_idx:
                # Segment variable not discovered - add warning but don't invalidate
                warnings.append(
                    f"Segment '{segment_var}' not in discovered DAG. "
                    f"CATE for {segment_id} may lack causal support."
                )
                validated_segments.append(segment_id)  # Include but warn
                continue

            node_to_idx[segment_var]

            # Check for treatment → segment path
            has_treatment_path = self._has_causal_path(
                treatment_var, segment_var, dag_adjacency, node_to_idx
            )

            # Check for segment → outcome path
            has_outcome_path = self._has_causal_path(
                segment_var, outcome_var, dag_adjacency, node_to_idx
            )

            # Check for bidirected edges (latent confounders from FCI)
            has_latent_confounder = self._has_latent_confounder(
                segment_var, treatment_var, outcome_var, edge_types
            )

            if has_latent_confounder:
                latent_confounder_segments.append(segment_id)
                warnings.append(
                    f"Segment '{segment_id}' has bidirected edge indicating latent confounder. "
                    f"CATE estimate may be biased."
                )

            # Validate based on path existence
            if has_treatment_path or has_outcome_path:
                validated_segments.append(segment_id)
            else:
                invalid_segments.append(segment_id)
                warnings.append(
                    f"Segment '{segment_id}' has no causal path from treatment or to outcome. "
                    f"Heterogeneous effect may be spurious."
                )

        # Add summary warning if many segments have latent confounders
        if len(latent_confounder_segments) > 0:
            confounder_ratio = len(latent_confounder_segments) / len(all_segments)
            if confounder_ratio > LATENT_CONFOUNDER_WARNING_THRESHOLD:
                warnings.append(
                    f"{len(latent_confounder_segments)}/{len(all_segments)} segments "
                    f"({confounder_ratio:.0%}) have latent confounders. "
                    f"Consider sensitivity analysis or instrumental variables."
                )

        return validated_segments, invalid_segments, latent_confounder_segments, warnings

    def _has_causal_path(
        self,
        source: str,
        target: str,
        dag_adjacency: List[List[int]],
        node_to_idx: Dict[str, int],
    ) -> bool:
        """Check if there's a directed path from source to target in the DAG.

        Uses BFS to find if any path exists.

        Args:
            source: Source node name
            target: Target node name
            dag_adjacency: DAG adjacency matrix
            node_to_idx: Node name to index mapping

        Returns:
            True if a directed path exists
        """
        if source not in node_to_idx or target not in node_to_idx:
            return False

        if source == target:
            return True

        source_idx = node_to_idx[source]
        target_idx = node_to_idx[target]
        n_nodes = len(dag_adjacency)

        # BFS to find path
        visited = set()
        queue = [source_idx]

        while queue:
            current = queue.pop(0)
            if current == target_idx:
                return True

            if current in visited:
                continue
            visited.add(current)

            # Check all outgoing edges
            for neighbor in range(n_nodes):
                if dag_adjacency[current][neighbor] == 1 and neighbor not in visited:
                    queue.append(neighbor)

        return False

    def _has_latent_confounder(
        self,
        segment_var: str,
        treatment_var: str,
        outcome_var: str,
        edge_types: Dict[str, str],
    ) -> bool:
        """Check if segment has bidirected edges indicating latent confounders.

        FCI algorithm produces bidirected edges (↔) when there's an
        unobserved common cause.

        Args:
            segment_var: Segment variable to check
            treatment_var: Treatment variable
            outcome_var: Outcome variable
            edge_types: Dict mapping edge keys to types (DIRECTED, BIDIRECTED, UNDIRECTED)

        Returns:
            True if latent confounder detected
        """
        # Check for bidirected edges involving the segment
        vars_to_check = [treatment_var, outcome_var, segment_var]

        for v1 in vars_to_check:
            for v2 in vars_to_check:
                if v1 == v2:
                    continue
                # Check both orderings of the edge key
                edge_key1 = f"{v1}->{v2}"
                edge_key2 = f"{v2}->{v1}"
                edge_key_bi1 = f"{v1}<->{v2}"
                edge_key_bi2 = f"{v2}<->{v1}"

                for key in [edge_key1, edge_key2, edge_key_bi1, edge_key_bi2]:
                    if key in edge_types and edge_types[key] == "BIDIRECTED":
                        return True

        return False
