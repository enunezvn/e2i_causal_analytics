"""Feature Analyzer Agent - HYBRID.

Analyzes trained models using SHAP for interpretability.

Hybrid execution:
- Computation nodes (1-2): NO LLM - Deterministic SHAP computation
- Interpretation node (3): LLM - Natural language explanations

Outputs:
- SHAPAnalysis: Global importance, interactions, feature directions
- InterpretabilityReport: Executive summary, insights, recommendations
- Semantic Memory: Feature relationships for downstream agents

Integration:
- Upstream: model_trainer (consumes TrainedModel)
- Downstream: model_deployer, explainer, causal_impact
- Memory: Semantic memory (feature relationships)
- Database: ml_shap_analyses table
"""

from typing import Any, Dict

from .graph import create_feature_analyzer_graph
from .state import FeatureAnalyzerState


class FeatureAnalyzerAgent:
    """Feature Analyzer: SHAP-based model interpretability.

    This is a HYBRID agent with 3 nodes:
    1. SHAP Computation (NO LLM) - Compute SHAP values
    2. Interaction Detection (NO LLM) - Detect feature interactions
    3. NL Interpretation (LLM) - Generate human-readable explanations
    """

    tier = 0
    tier_name = "ml_foundation"
    agent_type = "hybrid"  # Computation + LLM
    sla_seconds = 120

    def __init__(self):
        """Initialize feature_analyzer agent."""
        self.graph = create_feature_analyzer_graph()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature analysis workflow.

        Args:
            input_data: Input data conforming to FeatureAnalyzerInput contract
                Required fields:
                - model_uri: str (MLflow model URI)
                - experiment_id: str
                Optional fields:
                - max_samples: int (default 1000)
                - compute_interactions: bool (default True)
                - store_in_semantic_memory: bool (default True)
                - training_run_id: str
                - X_sample: Data for SHAP computation
                - y_sample: Labels for SHAP computation

        Returns:
            Output data conforming to FeatureAnalyzerOutput contract
        """
        # Validate required inputs
        required_fields = ["model_uri", "experiment_id"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Prepare initial state
        initial_state: FeatureAnalyzerState = {
            # Input fields
            "model_uri": input_data["model_uri"],
            "experiment_id": input_data["experiment_id"],
            "training_run_id": input_data.get("training_run_id", "unknown"),
            # Configuration
            "max_samples": input_data.get("max_samples", 1000),
            "compute_interactions": input_data.get("compute_interactions", True),
            "store_in_semantic_memory": input_data.get("store_in_semantic_memory", True),
            # Optional data
            "X_sample": input_data.get("X_sample"),
            "y_sample": input_data.get("y_sample"),
            # Status
            "status": "in_progress",
        }

        # Execute LangGraph workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            error_msg = final_state["error"]
            error_type = final_state.get("error_type", "unknown")
            raise RuntimeError(f"{error_type}: {error_msg}")

        # Update semantic memory if requested
        semantic_memory_updated = False
        semantic_memory_entries = 0

        if final_state.get("store_in_semantic_memory", True):
            semantic_memory_updated, semantic_memory_entries = await self._update_semantic_memory(
                final_state
            )

        # Construct structured outputs
        shap_analysis = self._build_shap_analysis(final_state)
        feature_importance_list = self._build_feature_importance_list(final_state)
        interaction_list = self._build_interaction_list(final_state)

        # Calculate total computation time
        total_time = (
            final_state.get("shap_computation_time_seconds", 0.0)
            + final_state.get("interaction_computation_time_seconds", 0.0)
            + final_state.get("interpretation_time_seconds", 0.0)
        )

        # Build output
        output = {
            # SHAP Analysis
            "shap_analysis": shap_analysis,
            "feature_importance": feature_importance_list,
            "interactions": interaction_list,
            # Interpretation
            "interpretation": final_state.get("interpretation", ""),
            "executive_summary": final_state.get("executive_summary", ""),
            "key_insights": final_state.get("key_insights", []),
            "recommendations": final_state.get("recommendations", []),
            "cautions": final_state.get("cautions", []),
            # Top features/interactions
            "top_features": final_state.get("top_features", []),
            "top_interactions": interaction_list[:3],  # Top 3
            # Semantic memory
            "semantic_memory_updated": semantic_memory_updated,
            "semantic_memory_entries": semantic_memory_entries,
            # Metadata
            "shap_analysis_id": final_state.get("shap_analysis_id"),
            "experiment_id": final_state["experiment_id"],
            "model_version": final_state.get("model_version", "unknown"),
            "samples_analyzed": final_state.get("samples_analyzed", 0),
            "explainer_type": final_state.get("explainer_type", "unknown"),
            "computation_time_seconds": total_time,
            # Status
            "status": "completed",
        }

        # Store to database (ml_shap_analyses table)
        await self._store_to_database(output)

        return output

    def _build_shap_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build SHAPAnalysis output structure.

        Args:
            state: Final agent state

        Returns:
            SHAPAnalysis dict conforming to contract
        """
        return {
            "experiment_id": state["experiment_id"],
            "model_version": state.get("model_version", "unknown"),
            "shap_analysis_id": state.get("shap_analysis_id"),
            "feature_importance": self._build_feature_importance_list(state),
            "interactions": self._build_interaction_list(state),
            "samples_analyzed": state.get("samples_analyzed", 0),
            "computation_time_seconds": (
                state.get("shap_computation_time_seconds", 0.0)
                + state.get("interaction_computation_time_seconds", 0.0)
            ),
        }

    def _build_feature_importance_list(self, state: Dict[str, Any]) -> list:
        """Build FeatureImportance list.

        Args:
            state: Final agent state

        Returns:
            List of FeatureImportance dicts
        """
        global_importance_ranked = state.get("global_importance_ranked", [])

        feature_importance_list = []
        for rank, (feature, importance) in enumerate(global_importance_ranked, 1):
            feature_importance_list.append(
                {
                    "feature": feature,
                    "importance": importance,
                    "rank": rank,
                }
            )

        return feature_importance_list

    def _build_interaction_list(self, state: Dict[str, Any]) -> list:
        """Build FeatureInteraction list.

        Args:
            state: Final agent state

        Returns:
            List of FeatureInteraction dicts
        """
        interaction_interpretations = state.get("interaction_interpretations", [])

        # If we have LLM interpretations, use those
        if interaction_interpretations:
            return interaction_interpretations

        # Otherwise, build from raw interactions
        top_interactions_raw = state.get("top_interactions_raw", [])

        interaction_list = []
        for feat1, feat2, strength in top_interactions_raw[:5]:
            interaction_type = "amplifying" if strength > 0 else "opposing"
            interaction_list.append(
                {
                    "features": [feat1, feat2],
                    "interaction_strength": float(strength),
                    "interpretation": f"{feat1} and {feat2} {interaction_type} (strength: {abs(strength):.3f})",
                }
            )

        return interaction_list

    async def _update_semantic_memory(self, state: Dict[str, Any]) -> tuple:
        """Update semantic memory with feature relationships.

        Args:
            state: Final agent state

        Returns:
            Tuple of (updated: bool, entries: int)
        """
        # TODO: Implement semantic memory integration with FalkorDB/Graphity
        # This would store:
        # - Feature importance relationships
        # - Feature interaction graphs
        # - Directional effects
        #
        # For now, return placeholder
        return False, 0

    async def _store_to_database(self, output: Dict[str, Any]) -> None:
        """Store SHAP analysis to ml_shap_analyses table.

        Args:
            output: Agent output to store
        """
        # TODO: Implement database storage
        # This would write to ml_shap_analyses table:
        # - analysis_id
        # - training_run_id
        # - experiment_id
        # - global_importance (JSONB)
        # - feature_directions (JSONB)
        # - interaction_matrix (JSONB)
        # - samples_analyzed
        # - computation_time_seconds
        #
        # For now, pass (will be implemented in integration phase)
        pass
