"""Causal Impact Agent - Tier 2 Hybrid Agent.

Estimates causal effects using DoWhy/EconML with natural language interpretation.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agents.causal_impact.graph import create_causal_impact_graph
from src.agents.causal_impact.state import (
    CausalImpactOutput,
    CausalImpactState,
)


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
    agent_name = "causal_impact"  # Contract REQUIRED: BaseAgentState.agent_name
    tools = ["dowhy", "econml", "networkx"]  # Contract: AgentConfig.tools
    primary_model = "claude-sonnet-4-20250514"  # Contract: AgentConfig.primary_model
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

        # Validate required input fields per contract
        required_fields = ["query", "treatment_var", "outcome_var", "confounders", "data_source"]
        missing_fields = [f for f in required_fields if f not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required field(s): {', '.join(missing_fields)}")

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
            input_data: Input data (must have required fields validated)

        Returns:
            Initial state conforming to contract
        """
        state: CausalImpactState = {
            # Required input fields (contract)
            "query": input_data["query"],
            "query_id": input_data.get("query_id", self._generate_query_id()),
            "treatment_var": input_data["treatment_var"],  # REQUIRED
            "outcome_var": input_data["outcome_var"],  # REQUIRED
            "confounders": input_data["confounders"],  # REQUIRED
            "data_source": input_data["data_source"],  # REQUIRED
            # Optional input fields
            "mediators": input_data.get("mediators", []),
            "effect_modifiers": input_data.get("effect_modifiers", []),
            "instruments": input_data.get("instruments", []),
            "segment_filters": input_data.get("segment_filters", {}),
            "interpretation_depth": input_data.get("interpretation_depth", "standard"),
            "user_context": input_data.get("user_context", {}),
            "parameters": input_data.get("parameters", {}),
            "time_period": input_data.get("time_period"),
            "brand": input_data.get("brand"),
            # Workflow state
            "current_phase": "graph_building",
            "status": "pending",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Agent identity (Contract REQUIRED: BaseAgentState.agent_name)
            "agent_name": self.agent_name,
            # Error accumulators (contract: operator.add)
            "errors": [],
            "warnings": [],
            "fallback_used": False,
            "retry_count": 0,
        }

        return state

    def _build_output(self, state: CausalImpactState, start_time: float) -> CausalImpactOutput:
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
        statistical_significance = estimation_result.get("statistical_significance", False)

        if statistical_significance and sensitivity_robust:
            base_confidence = 0.9
        elif statistical_significance:
            base_confidence = 0.75
        else:
            base_confidence = 0.5

        overall_confidence = base_confidence * refutation_confidence

        # Determine refutation status
        refutation_passed = refutation_results.get("overall_robust", False)

        # Build output with contract field names
        output: CausalImpactOutput = {
            "query_id": state.get("query_id", "unknown"),
            "status": "completed",
            # Core results
            "causal_narrative": interpretation.get("narrative", "Analysis completed successfully."),
            "ate_estimate": estimation_result.get("ate"),
            "confidence_interval": (
                (
                    estimation_result.get("ate_ci_lower", 0.0),
                    estimation_result.get("ate_ci_upper", 0.0),
                )
                if "ate_ci_lower" in estimation_result
                else None
            ),
            "standard_error": estimation_result.get("standard_error"),
            "statistical_significance": estimation_result.get("statistical_significance", False),
            "p_value": estimation_result.get("p_value"),
            "effect_type": estimation_result.get("effect_type", "ate"),
            "estimation_method": estimation_result.get("method"),
            # Contract REQUIRED fields
            "confidence": overall_confidence,  # Contract field (was overall_confidence)
            "model_used": estimation_result.get("method", "unknown"),  # Contract REQUIRED
            "key_insights": interpretation.get("key_findings", []),  # Contract REQUIRED
            "assumption_warnings": self._extract_assumption_warnings(
                interpretation, estimation_result, refutation_results
            ),  # Contract REQUIRED
            "actionable_recommendations": interpretation.get(
                "recommendations", []
            ),  # Contract field (was recommendations)
            "requires_further_analysis": overall_confidence < 0.7,  # Contract REQUIRED
            "refutation_passed": refutation_passed,  # Contract REQUIRED
            "executive_summary": self._generate_executive_summary(
                interpretation, estimation_result, overall_confidence
            ),  # Contract REQUIRED
            # Rich metadata
            "mechanism_explanation": interpretation.get("mechanism_explanation"),
            "causal_graph_summary": self._summarize_causal_graph(causal_graph),
            "key_assumptions": interpretation.get("assumptions_made", []),
            "limitations": interpretation.get("limitations", []),
            # Performance metrics
            "computation_latency_ms": computation_latency_ms,
            "interpretation_latency_ms": interpretation_latency_ms,
            "total_latency_ms": total_latency_ms,
            # Robustness indicators
            "refutation_tests_passed": refutation_results.get("tests_passed"),
            "refutation_tests_total": refutation_results.get("total_tests"),
            "sensitivity_e_value": sensitivity_analysis.get("e_value"),
            # Follow-up
            "visualizations": [],
            "follow_up_suggestions": interpretation.get("recommendations", []),
            "citations": ["E2I Causal Analytics System v4.2"],
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
            "causal_narrative": f"Analysis failed: {error_message}",
            "statistical_significance": False,
            # Contract REQUIRED fields
            "confidence": 0.0,  # Contract field (was overall_confidence)
            "model_used": "none",  # Contract REQUIRED
            "key_insights": [],  # Contract REQUIRED
            "assumption_warnings": ["Analysis failed - unable to verify assumptions"],
            "actionable_recommendations": [
                "Review error and retry with valid input"
            ],  # Contract field (was recommendations)
            "requires_further_analysis": True,  # Contract REQUIRED
            "refutation_passed": False,  # Contract REQUIRED
            "executive_summary": f"Analysis failed: {error_message}",  # Contract REQUIRED
            # Metadata
            "key_assumptions": [],
            "limitations": ["Analysis failed before completion"],
            "computation_latency_ms": latency_ms,
            "interpretation_latency_ms": 0.0,
            "total_latency_ms": latency_ms,
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

    def _extract_assumption_warnings(
        self, interpretation: Dict, estimation_result: Dict, refutation_results: Dict
    ) -> List[str]:
        """Extract assumption warnings from analysis results.

        Contract REQUIRED field: assumption_warnings

        Args:
            interpretation: Interpretation node output
            estimation_result: Estimation node output
            refutation_results: Refutation node output

        Returns:
            List of assumption warning strings
        """
        warnings = []

        # Check for low sample size
        sample_size = estimation_result.get("sample_size", 0)
        if sample_size < 100:
            warnings.append(f"Low sample size ({sample_size}) may limit reliability")

        # Check for failed refutation tests
        tests_failed = refutation_results.get("tests_failed", 0)
        if tests_failed > 0:
            warnings.append(f"{tests_failed} refutation test(s) failed - causal claim may be weak")

        # Check for weak effect size
        effect_size = estimation_result.get("effect_size", "")
        if effect_size == "small":
            warnings.append("Small effect size detected - practical significance may be limited")

        # Check assumptions from interpretation
        assumptions = interpretation.get("assumptions_made", [])
        for assumption in assumptions:
            if "unverified" in assumption.lower() or "assumed" in assumption.lower():
                warnings.append(f"Unverified assumption: {assumption}")

        # If no warnings, indicate clean status
        if not warnings:
            warnings.append("No critical assumption violations detected")

        return warnings

    def _generate_executive_summary(
        self, interpretation: Dict, estimation_result: Dict, confidence: float
    ) -> str:
        """Generate executive summary for causal impact analysis.

        Contract REQUIRED field: executive_summary (2-3 sentences)

        Args:
            interpretation: Interpretation node output
            estimation_result: Estimation node output
            confidence: Overall confidence score (0-1)

        Returns:
            Executive summary string
        """
        ate = estimation_result.get("ate")
        significance = estimation_result.get("statistical_significance", False)
        method = estimation_result.get("method", "causal inference")

        # Build summary based on results
        if ate is not None and significance:
            effect_direction = "positive" if ate > 0 else "negative"
            summary = (
                f"Analysis identified a statistically significant {effect_direction} "
                f"causal effect (ATE: {ate:.4f}) using {method}. "
                f"Overall confidence in this finding is {confidence:.0%}."
            )
        elif ate is not None:
            summary = (
                f"Analysis estimated a causal effect (ATE: {ate:.4f}) using {method}, "
                f"but the result is not statistically significant. "
                f"Confidence in this finding is {confidence:.0%}."
            )
        else:
            narrative = interpretation.get("narrative", "Analysis completed")
            summary = f"{narrative[:200]}... " if len(narrative) > 200 else narrative

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

        # Return simplified format for orchestrator (contract field names)
        return {
            "narrative": output["causal_narrative"],
            "recommendations": output.get("actionable_recommendations", []),  # Contract field
            "confidence": output.get("confidence", 0.0),  # Contract field
            "key_findings": output.get("key_insights", []),  # Contract field
        }
