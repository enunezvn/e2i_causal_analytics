"""Heterogeneous Optimizer Agent.

This agent estimates treatment effect heterogeneity using EconML's CausalForestDML
and recommends optimal treatment allocation across segments.

Tier: 2 (Causal Analytics)
Type: Standard (Computational)
Latency: Up to 150s
"""

import logging
import uuid
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

from .graph import create_heterogeneous_optimizer_graph
from .memory_hooks import (
    contribute_to_memory,
    get_heterogeneous_optimizer_memory_hooks,
)
from .state import HeterogeneousOptimizerState

if TYPE_CHECKING:
    from .mlflow_tracker import HeterogeneousOptimizerMLflowTracker
    from .opik_tracer import HeterogeneousOptimizerOpikTracer

logger = logging.getLogger(__name__)


class HeterogeneousOptimizerAgent:
    """Heterogeneous Optimizer Agent for segment-level CATE analysis.

    This agent:
    1. Estimates CATE using EconML CausalForestDML
    2. Identifies high/low responder segments
    3. Generates optimal treatment allocation policy
    4. Creates visualization data and insights
    """

    def __init__(
        self,
        data_connector=None,
        enable_memory: bool = True,
        enable_mlflow: bool = True,
        enable_opik: bool = True,
    ):
        """Initialize the Heterogeneous Optimizer agent.

        Args:
            data_connector: Data connector for fetching data (optional, uses mock if None)
            enable_memory: Whether to enable tri-memory integration (default: True)
            enable_mlflow: Whether to enable MLflow tracking (default: True)
            enable_opik: Whether to enable Opik distributed tracing (default: True)
        """
        self.graph = create_heterogeneous_optimizer_graph(data_connector)
        self.enable_memory = enable_memory
        self.enable_mlflow = enable_mlflow
        self.enable_opik = enable_opik
        self._memory_hooks = None
        self._mlflow_tracker: Optional["HeterogeneousOptimizerMLflowTracker"] = None
        self._opik_tracer: Optional["HeterogeneousOptimizerOpikTracer"] = None

    def _get_mlflow_tracker(self) -> Optional["HeterogeneousOptimizerMLflowTracker"]:
        """Get or create MLflow tracker instance (lazy initialization)."""
        if not self.enable_mlflow:
            return None

        if self._mlflow_tracker is None:
            try:
                from .mlflow_tracker import HeterogeneousOptimizerMLflowTracker

                self._mlflow_tracker = HeterogeneousOptimizerMLflowTracker()
            except ImportError:
                logger.warning("MLflow tracker not available")
                return None

        return self._mlflow_tracker

    def _get_opik_tracer(self) -> Optional["HeterogeneousOptimizerOpikTracer"]:
        """Get or create Opik tracer instance (lazy initialization)."""
        if not self.enable_opik:
            return None

        if self._opik_tracer is None:
            try:
                from .opik_tracer import HeterogeneousOptimizerOpikTracer

                self._opik_tracer = HeterogeneousOptimizerOpikTracer()
            except ImportError:
                logger.warning("Opik tracer not available")
                return None

        return self._opik_tracer

    @property
    def memory_hooks(self):
        """Lazy-load memory hooks."""
        if self._memory_hooks is None and self.enable_memory:
            self._memory_hooks = get_heterogeneous_optimizer_memory_hooks()
        return self._memory_hooks

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

        # Generate session ID if not provided
        session_id = input_data.get("session_id") or str(uuid.uuid4())

        # Retrieve memory context (if enabled)
        memory_context = None
        if self.enable_memory and self.memory_hooks:
            try:
                memory_context = await self.memory_hooks.get_context(
                    session_id=session_id,
                    query=input_data["query"],
                    treatment_var=input_data.get("treatment_var"),
                    outcome_var=input_data.get("outcome_var"),
                )
                logger.debug(
                    f"Retrieved memory context: "
                    f"working={len(memory_context.working_memory)}, "
                    f"episodic={len(memory_context.episodic_context)}"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve memory context: {e}")

        # Build initial state with memory context
        initial_state = self._build_initial_state(input_data, session_id, memory_context)

        # Get trackers
        tracker = self._get_mlflow_tracker()
        opik_tracer = self._get_opik_tracer()

        # Helper function for memory contribution
        async def _contribute_to_memory(output: Dict[str, Any], final_state) -> None:
            if self.enable_memory and output.get("status") != "failed":
                try:
                    await contribute_to_memory(
                        result=output,
                        state=final_state,
                        memory_hooks=self.memory_hooks,
                        session_id=session_id,
                        brand=input_data.get("brand"),
                        region=input_data.get("region"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to contribute to memory: {e}")

        # Execute with Opik tracing if available
        if opik_tracer:
            async with opik_tracer.trace_analysis(
                query=input_data["query"],
                treatment_var=input_data["treatment_var"],
                outcome_var=input_data.get("outcome_var"),
                segment_vars=input_data.get("segment_vars"),
                brand=input_data.get("brand"),
                session_id=session_id,
            ) as trace_ctx:
                # Execute with MLflow tracking if available
                if tracker:
                    async with tracker.start_analysis_run(
                        experiment_name=input_data.get("experiment_name", "default"),
                        brand=input_data.get("brand"),
                        region=input_data.get("region"),
                        treatment_var=input_data.get("treatment_var"),
                        outcome_var=input_data.get("outcome_var"),
                        query_id=session_id,
                    ):
                        # Execute workflow
                        final_state = await self.graph.ainvoke(initial_state)
                        output = self._build_output(final_state)
                        await tracker.log_analysis_result(output, final_state)
                else:
                    # Execute workflow without MLflow
                    final_state = await self.graph.ainvoke(initial_state)
                    output = self._build_output(final_state)

                # Log analysis results to Opik
                trace_ctx.log_analysis_complete(
                    status=output.get("status", "completed"),
                    success=output.get("status") != "failed",
                    total_duration_ms=output.get("total_latency_ms", 0),
                    overall_ate=output.get("overall_ate", 0.0),
                    heterogeneity_score=output.get("heterogeneity_score", 0.0),
                    high_responders_count=len(output.get("high_responders") or []),
                    low_responders_count=len(output.get("low_responders") or []),
                    recommendations_count=len(output.get("policy_recommendations") or []),
                    expected_total_lift=output.get("expected_total_lift", 0.0),
                    confidence=output.get("confidence", 0.0),
                    errors=output.get("errors"),
                    suggested_next_agent=output.get("suggested_next_agent"),
                )

                # Contribute to memory
                await _contribute_to_memory(output, final_state)
                return output
        else:
            # Run without Opik tracing
            if tracker:
                async with tracker.start_analysis_run(
                    experiment_name=input_data.get("experiment_name", "default"),
                    brand=input_data.get("brand"),
                    region=input_data.get("region"),
                    treatment_var=input_data.get("treatment_var"),
                    outcome_var=input_data.get("outcome_var"),
                    query_id=session_id,
                ):
                    final_state = await self.graph.ainvoke(initial_state)
                    output = self._build_output(final_state)
                    await tracker.log_analysis_result(output, final_state)
                    await _contribute_to_memory(output, final_state)
                    return output
            else:
                # Execute workflow without any tracking
                final_state = await self.graph.ainvoke(initial_state)
                output = self._build_output(final_state)
                await _contribute_to_memory(output, final_state)
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
        self,
        input_data: Dict[str, Any],
        session_id: str,
        memory_context: Optional[Any] = None,
    ) -> HeterogeneousOptimizerState:
        """Build initial state from input data.

        Args:
            input_data: Input dictionary
            session_id: Session identifier for memory operations
            memory_context: Optional CATEAnalysisContext from memory hooks

        Returns:
            Initial state dictionary
        """
        # Extract memory context if available
        working_memory_context = None
        episodic_context = None
        if memory_context:
            working_memory_context = {
                "messages": memory_context.working_memory,
                "timestamp": memory_context.retrieval_timestamp.isoformat(),
            }
            episodic_context = memory_context.episodic_context

        return {
            # Input
            "query": input_data["query"],
            "treatment_var": input_data["treatment_var"],
            "outcome_var": input_data["outcome_var"],
            "segment_vars": input_data["segment_vars"],
            "effect_modifiers": input_data["effect_modifiers"],
            "data_source": input_data["data_source"],
            "filters": input_data.get("filters"),
            "tier0_data": input_data.get("tier0_data"),  # Passthrough for tier0 testing
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
            # Memory context
            "session_id": session_id,
            "working_memory_context": working_memory_context,
            "episodic_context": episodic_context,
        }

    def _build_output(self, final_state: HeterogeneousOptimizerState) -> Dict[str, Any]:
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

    def _calculate_confidence(self, state: HeterogeneousOptimizerState) -> float:
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

    def _check_further_analysis(self, state: HeterogeneousOptimizerState) -> bool:
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

    def _suggest_next_agent(self, state: HeterogeneousOptimizerState) -> Optional[str]:
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
