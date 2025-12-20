"""Causal Impact Agent - Tier 2 Hybrid Agent.

Estimates causal effects using DoWhy/EconML with natural language interpretation.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.agents.causal_impact.state import (
    CausalImpactState,
    CausalImpactInput,
    CausalImpactOutput,
)
from src.agents.causal_impact.graph import create_causal_impact_graph


class CausalImpactAgent:
    """Causal Impact Agent - Causal effect estimation and interpretation.

    Tier: 2 (Causal Analytics)
    Type: Hybrid (Computation + Deep Reasoning)
    SLA: 120s total (60s computation + 30s interpretation)

    Pipeline:
    1. graph_builder: Construct causal DAG (10s)
    2. estimation: Estimate ATE/CATE (30s)
    3. refutation: Robustness tests (15s)
    4. sensitivity: E-value analysis (5s)
    5. interpretation: Natural language output (30s)
    """

    tier = 2
    tier_name = "causal_analytics"
    agent_type = "hybrid"  # Computation + Deep Reasoning
    sla_seconds = 120

    def __init__(
        self,
        enable_checkpointing: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Causal Impact Agent.

        Args:
            enable_checkpointing: Whether to enable state checkpointing
            config: Optional configuration overrides
        """
        self.enable_checkpointing = enable_checkpointing
        self.config = config or {}

        # Create workflow graph
        self.graph = create_causal_impact_graph(enable_checkpointing)

    async def run(self, input_data: Dict[str, Any]) -> CausalImpactOutput:
        """Execute causal impact analysis.

        Args:
            input_data: Input conforming to CausalImpactInput contract

        Returns:
            Output conforming to CausalImpactOutput contract

        Raises:
            ValueError: If required input fields are missing
        """
        start_time = time.time()

        # Validate input
        if "query" not in input_data:
            raise ValueError("Missing required field: query")

        # Initialize state
        initial_state = self._initialize_state(input_data)

        try:
            # Run workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Build output
            output = self._build_output(final_state, start_time)

            return output

        except Exception as e:
            # Error handling
            latency_ms = (time.time() - start_time) * 1000
            error_output = self._build_error_output(str(e), latency_ms, input_data)
            return error_output

    def _initialize_state(self, input_data: Dict[str, Any]) -> CausalImpactState:
        """Initialize workflow state from input.

        Args:
            input_data: Input data

        Returns:
            Initial state
        """
        state: CausalImpactState = {
            "query": input_data["query"],
            "query_id": input_data.get("query_id", self._generate_query_id()),
            # Contract-aligned field names
            "treatment_var": input_data.get("treatment_var"),
            "outcome_var": input_data.get("outcome_var"),
            "confounders": input_data.get("confounders", []),
            "mediators": input_data.get("mediators", []),
            "effect_modifiers": input_data.get("effect_modifiers", []),
            "instruments": input_data.get("instruments", []),
            "segment_filters": input_data.get("segment_filters", {}),
            "interpretation_depth": input_data.get("interpretation_depth", "standard"),
            "user_context": input_data.get("user_context", {}),
            "parameters": input_data.get("parameters", {}),
            "data_source": input_data.get("data_source"),
            "time_period": input_data.get("time_period"),
            "brand": input_data.get("brand"),
            "current_phase": "graph_building",
            "status": "pending",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return state

    def _build_output(
        self, state: CausalImpactState, start_time: float
    ) -> CausalImpactOutput:
        """Build output from final state.

        Args:
            state: Final workflow state
            start_time: Workflow start time

        Returns:
            Output conforming to contract
        """
        # Extract results
        interpretation = state.get("interpretation", {})
        estimation_result = state.get("estimation_result", {})
        refutation_results = state.get("refutation_results", {})
        sensitivity_analysis = state.get("sensitivity_analysis", {})
        causal_graph = state.get("causal_graph", {})

        # Calculate latencies
        computation_latency_ms = (
            state.get("graph_builder_latency_ms", 0)
            + state.get("estimation_latency_ms", 0)
            + state.get("refutation_latency_ms", 0)
            + state.get("sensitivity_latency_ms", 0)
        )

        interpretation_latency_ms = state.get("interpretation_latency_ms", 0)
        total_latency_ms = (time.time() - start_time) * 1000

        # Determine overall confidence
        refutation_confidence = refutation_results.get("confidence_adjustment", 1.0)
        sensitivity_robust = sensitivity_analysis.get("robust_to_confounding", False)
        statistical_significance = estimation_result.get(
            "statistical_significance", False
        )

        if statistical_significance and sensitivity_robust:
            base_confidence = 0.9
        elif statistical_significance:
            base_confidence = 0.75
        else:
            base_confidence = 0.5

        overall_confidence = base_confidence * refutation_confidence

        # Build output with contract-aligned field names
        output: CausalImpactOutput = {
            "query_id": state.get("query_id", "unknown"),
            "status": "completed",
            # Contract-aligned output field names
            "causal_narrative": interpretation.get(
                "narrative", "Analysis completed successfully."
            ),
            "ate_estimate": estimation_result.get("ate"),
            "confidence_interval": (
                estimation_result.get("ate_ci_lower", 0.0),
                estimation_result.get("ate_ci_upper", 0.0),
            )
            if "ate_ci_lower" in estimation_result
            else None,
            "standard_error": estimation_result.get("standard_error"),
            "statistical_significance": estimation_result.get(
                "statistical_significance", False
            ),
            "p_value": estimation_result.get("p_value"),
            "effect_type": estimation_result.get("effect_type", "ate"),
            "estimation_method": estimation_result.get("method"),
            "mechanism_explanation": interpretation.get("mechanism_explanation"),
            "causal_graph_summary": self._summarize_causal_graph(causal_graph),
            "key_assumptions": interpretation.get("assumptions_made", []),
            "limitations": interpretation.get("limitations", []),
            "recommendations": interpretation.get("recommendations", []),
            "computation_latency_ms": computation_latency_ms,
            "interpretation_latency_ms": interpretation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "refutation_tests_passed": refutation_results.get("tests_passed"),
            "refutation_tests_total": refutation_results.get("total_tests"),
            "sensitivity_e_value": sensitivity_analysis.get("e_value"),
            "overall_confidence": overall_confidence,
            "visualizations": [],  # Would add visualizations in production
            "follow_up_suggestions": interpretation.get("recommendations", []),
            "citations": ["E2I Causal Analytics System v4.1"],
        }

        return output

    def _build_error_output(
        self, error_message: str, latency_ms: float, input_data: Dict[str, Any]
    ) -> CausalImpactOutput:
        """Build error output.

        Args:
            error_message: Error description
            latency_ms: Time elapsed before error
            input_data: Original input

        Returns:
            Error output
        """
        output: CausalImpactOutput = {
            "query_id": input_data.get("query_id", "unknown"),
            "status": "failed",
            # Contract-aligned field name
            "causal_narrative": f"Analysis failed: {error_message}",
            "statistical_significance": False,
            "key_assumptions": [],
            "limitations": ["Analysis failed before completion"],
            "recommendations": ["Review error and retry with valid input"],
            "computation_latency_ms": latency_ms,
            "interpretation_latency_ms": 0.0,
            "total_latency_ms": latency_ms,
            "overall_confidence": 0.0,
            "follow_up_suggestions": [],
            "citations": [],
            "error_message": error_message,
            "partial_results": False,
        }

        return output

    def _summarize_causal_graph(self, causal_graph: Dict) -> Optional[str]:
        """Summarize causal graph structure.

        Args:
            causal_graph: CausalGraph dict

        Returns:
            Human-readable summary
        """
        if not causal_graph:
            return None

        nodes = causal_graph.get("nodes", [])
        edges = causal_graph.get("edges", [])
        treatment_nodes = causal_graph.get("treatment_nodes", [])
        outcome_nodes = causal_graph.get("outcome_nodes", [])
        adjustment_sets = causal_graph.get("adjustment_sets", [])

        summary = (
            f"Causal graph with {len(nodes)} variables and {len(edges)} relationships. "
            f"Treatment: {', '.join(treatment_nodes)}. "
            f"Outcome: {', '.join(outcome_nodes)}. "
            f"{len(adjustment_sets)} valid adjustment sets identified."
        )

        return summary

    def _generate_query_id(self) -> str:
        """Generate unique query ID.

        Returns:
            Query ID (format: q-{12 hex chars})
        """
        import secrets

        return f"q-{secrets.token_hex(6)}"

    # Helper methods for orchestrator integration

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """Helper: Classify query intent (not used by causal_impact).

        This is primarily used by orchestrator. Causal impact agent
        receives pre-classified queries.
        """
        return {
            "primary_intent": "causal_effect",
            "confidence": 0.95,
            "secondary_intents": [],
            "requires_multi_agent": False,
        }

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper: Simplified interface for orchestrator.

        Args:
            input_data: Input data

        Returns:
            Simplified output for orchestrator synthesis
        """
        output = await self.run(input_data)

        # Return simplified format for orchestrator (contract-aligned)
        return {
            "narrative": output["causal_narrative"],
            "recommendations": output.get("recommendations", []),
            "confidence": output.get("overall_confidence", 0.0),
            "key_findings": [
                f"Causal effect: {output.get('ate_estimate', 'N/A')}",
                f"Significance: {output.get('statistical_significance', False)}",
            ],
        }
