"""Heterogeneous Optimizer Agent.

This agent estimates treatment effect heterogeneity using EconML's CausalForestDML
and recommends optimal treatment allocation across segments.

Tier: 2 (Causal Analytics)
Type: Standard (Computational)
Latency: Up to 150s
"""

from typing import Dict, Any, Optional, Literal
import time

from .graph import create_heterogeneous_optimizer_graph
from .state import HeterogeneousOptimizerState


class HeterogeneousOptimizerAgent:
    """Heterogeneous Optimizer Agent for segment-level CATE analysis.

    This agent:
    1. Estimates CATE using EconML CausalForestDML
    2. Identifies high/low responder segments
    3. Generates optimal treatment allocation policy
    4. Creates visualization data and insights
    """

    def __init__(self, data_connector=None):
        """Initialize the Heterogeneous Optimizer agent.

        Args:
            data_connector: Data connector for fetching data (optional, uses mock if None)
        """
        self.graph = create_heterogeneous_optimizer_graph(data_connector)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run heterogeneous optimization analysis.

        Args:
            input_data: Input dictionary matching HeterogeneousOptimizerInput contract

        Returns:
            Output dictionary matching HeterogeneousOptimizerOutput contract

        Raises:
            ValueError: If input validation fails
        """
        # Validate input
        self._validate_input(input_data)

        # Build initial state
        initial_state = self._build_initial_state(input_data)

        # Execute workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Build output
        output = self._build_output(final_state)

        return output

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data against contract.

        Args:
            input_data: Input dictionary to validate

        Raises:
            ValueError: If validation fails
        """
        # Required fields
        required_fields = [
            "query",
            "treatment_var",
            "outcome_var",
            "segment_vars",
            "effect_modifiers",
            "data_source",
        ]

        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(input_data["segment_vars"], list) or not input_data["segment_vars"]:
            raise ValueError("segment_vars must be a non-empty list")

        if (
            not isinstance(input_data["effect_modifiers"], list)
            or not input_data["effect_modifiers"]
        ):
            raise ValueError("effect_modifiers must be a non-empty list")

        # Validate optional numeric fields
        if "n_estimators" in input_data:
            n_est = input_data["n_estimators"]
            if not isinstance(n_est, int) or not (50 <= n_est <= 500):
                raise ValueError("n_estimators must be an integer between 50 and 500")

        if "min_samples_leaf" in input_data:
            min_leaf = input_data["min_samples_leaf"]
            if not isinstance(min_leaf, int) or not (5 <= min_leaf <= 100):
                raise ValueError("min_samples_leaf must be an integer between 5 and 100")

        if "significance_level" in input_data:
            alpha = input_data["significance_level"]
            if not isinstance(alpha, (int, float)) or not (0.01 <= alpha <= 0.10):
                raise ValueError("significance_level must be a number between 0.01 and 0.10")

        if "top_segments_count" in input_data:
            top_count = input_data["top_segments_count"]
            if not isinstance(top_count, int) or not (5 <= top_count <= 50):
                raise ValueError("top_segments_count must be an integer between 5 and 50")

    def _build_initial_state(
        self, input_data: Dict[str, Any]
    ) -> HeterogeneousOptimizerState:
        """Build initial state from input data.

        Args:
            input_data: Input dictionary

        Returns:
            Initial state dictionary
        """
        return {
            # Input
            "query": input_data["query"],
            "treatment_var": input_data["treatment_var"],
            "outcome_var": input_data["outcome_var"],
            "segment_vars": input_data["segment_vars"],
            "effect_modifiers": input_data["effect_modifiers"],
            "data_source": input_data["data_source"],
            "filters": input_data.get("filters"),
            # Configuration
            "n_estimators": input_data.get("n_estimators", 100),
            "min_samples_leaf": input_data.get("min_samples_leaf", 10),
            "significance_level": input_data.get("significance_level", 0.05),
            "top_segments_count": input_data.get("top_segments_count", 10),
            # Outputs (initialized as None)
            "cate_by_segment": None,
            "overall_ate": None,
            "heterogeneity_score": None,
            "feature_importance": None,
            "high_responders": None,
            "low_responders": None,
            "segment_comparison": None,
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            # Metadata
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            # Error handling
            "errors": [],
            "warnings": [],
            "status": "pending",
            # Contract-required fields
            "confidence": None,
            "requires_further_analysis": None,
            "suggested_next_agent": None,
        }

    def _build_output(
        self, final_state: HeterogeneousOptimizerState
    ) -> Dict[str, Any]:
        """Build output from final state.

        Args:
            final_state: Final state from workflow execution

        Returns:
            Output dictionary matching HeterogeneousOptimizerOutput contract
        """
        # Calculate confidence based on results
        confidence = self._calculate_confidence(final_state)

        # Determine if further analysis is needed
        requires_further_analysis = self._check_further_analysis(final_state)

        # Suggest next agent
        suggested_next_agent = self._suggest_next_agent(final_state)

        return {
            # Core results
            "overall_ate": final_state.get("overall_ate", 0.0),
            "heterogeneity_score": final_state.get("heterogeneity_score", 0.0),
            # Segment analysis
            "high_responders": final_state.get("high_responders", []),
            "low_responders": final_state.get("low_responders", []),
            "cate_by_segment": final_state.get("cate_by_segment", {}),
            # Policy recommendations
            "policy_recommendations": final_state.get("policy_recommendations", []),
            "expected_total_lift": final_state.get("expected_total_lift", 0.0),
            "optimal_allocation_summary": final_state.get("optimal_allocation_summary", ""),
            # Feature importance
            "feature_importance": final_state.get("feature_importance", {}),
            # Summaries
            "executive_summary": final_state.get(
                "executive_summary", "Heterogeneous effect analysis completed."
            ),
            "key_insights": final_state.get("key_insights", []),
            # Metadata
            "estimation_latency_ms": final_state.get("estimation_latency_ms", 0),
            "analysis_latency_ms": final_state.get("analysis_latency_ms", 0),
            "total_latency_ms": final_state.get("total_latency_ms", 0),
            # Common fields
            "confidence": confidence,
            "warnings": final_state.get("warnings", []),
            "requires_further_analysis": requires_further_analysis,
            "suggested_next_agent": suggested_next_agent,
            # Status fields (contract-required)
            "status": "failed" if final_state.get("errors") else "completed",
            "errors": final_state.get("errors", []),
        }

    def _calculate_confidence(
        self, state: HeterogeneousOptimizerState
    ) -> float:
        """Calculate confidence score.

        Args:
            state: Final state

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.7  # Base confidence

        # Factor 1: Errors reduce confidence
        errors = state.get("errors", [])
        if errors:
            confidence -= 0.3

        # Factor 2: Sample size in segments
        high_responders = state.get("high_responders", [])
        low_responders = state.get("low_responders", [])

        if high_responders and low_responders:
            # Check if segments have adequate sample size
            min_size = min(
                [h["size"] for h in high_responders] + [l["size"] for l in low_responders]
            )
            if min_size >= 100:
                confidence += 0.1
            elif min_size < 30:
                confidence -= 0.1

        # Factor 3: Statistical significance
        cate_by_segment = state.get("cate_by_segment", {})
        if cate_by_segment:
            total_results = sum(len(results) for results in cate_by_segment.values())
            significant_results = sum(
                1
                for results in cate_by_segment.values()
                for result in results
                if result.get("statistical_significance")
            )
            if total_results > 0:
                sig_ratio = significant_results / total_results
                confidence += sig_ratio * 0.1

        # Factor 4: Heterogeneity score
        heterogeneity = state.get("heterogeneity_score") or 0
        if heterogeneity > 0.5:
            confidence += 0.05  # High heterogeneity means clear signal

        return max(0.0, min(1.0, confidence))

    def _check_further_analysis(
        self, state: HeterogeneousOptimizerState
    ) -> bool:
        """Check if further analysis is needed.

        Args:
            state: Final state

        Returns:
            True if further analysis recommended
        """
        # Further analysis needed if:
        # 1. High heterogeneity detected
        # 2. Significant policy recommendations
        # 3. High expected lift

        heterogeneity = state.get("heterogeneity_score") or 0
        if heterogeneity > 0.6:
            return True

        policy_recs = state.get("policy_recommendations") or []
        if len(policy_recs) > 5:
            return True

        expected_lift = state.get("expected_total_lift") or 0
        if abs(expected_lift) > 100:
            return True

        return False

    def _suggest_next_agent(
        self, state: HeterogeneousOptimizerState
    ) -> Optional[str]:
        """Suggest next agent based on results.

        Args:
            state: Final state

        Returns:
            Suggested next agent name or None
        """
        if not self._check_further_analysis(state):
            return None

        policy_recs = state.get("policy_recommendations") or []
        heterogeneity = state.get("heterogeneity_score") or 0

        # If high heterogeneity and many recommendations → experiment_designer
        if heterogeneity > 0.6 and len(policy_recs) > 5:
            return "experiment_designer"

        # If significant expected lift → resource_optimizer
        expected_lift = state.get("expected_total_lift") or 0
        if abs(expected_lift) > 100:
            return "resource_optimizer"

        return None

    def classify_intent(self, query: str) -> Literal["in_scope", "out_of_scope"]:
        """Classify if query is in scope for heterogeneous optimization.

        Args:
            query: Natural language query

        Returns:
            'in_scope' or 'out_of_scope'
        """
        # Keywords for CATE/heterogeneity analysis
        in_scope_keywords = [
            "heterogeneous",
            "segment",
            "responder",
            "cate",
            "effect modifier",
            "which segments",
            "who responds",
            "differential effect",
            "treatment allocation",
            "optimal policy",
            "targeting",
        ]

        query_lower = query.lower()
        return (
            "in_scope"
            if any(keyword in query_lower for keyword in in_scope_keywords)
            else "out_of_scope"
        )

    async def analyze(self, query: str, **kwargs) -> Dict[str, Any]:
        """Convenience method for quick analysis.

        Args:
            query: Natural language query
            **kwargs: Additional input parameters

        Returns:
            Analysis output

        Raises:
            ValueError: If required parameters missing
        """
        input_data = {"query": query, **kwargs}
        return await self.run(input_data)
