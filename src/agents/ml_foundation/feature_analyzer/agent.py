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
- Observability: Opik tracing
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .graph import create_feature_analyzer_graph, create_shap_analysis_graph
from .state import FeatureAnalyzerState

logger = logging.getLogger(__name__)


def _get_shap_repository():
    """Get ShapAnalysisRepository (lazy import to avoid circular deps)."""
    try:
        from src.repositories.shap_analysis import get_shap_analysis_repository
        return get_shap_analysis_repository()
    except Exception as e:
        logger.warning(f"Could not get SHAP repository: {e}")
        return None


def _get_opik_connector():
    """Get OpikConnector (lazy import to avoid circular deps)."""
    try:
        from src.mlops.opik_connector import get_opik_connector
        return get_opik_connector()
    except Exception as e:
        logger.warning(f"Could not get Opik connector: {e}")
        return None


def _get_semantic_memory():
    """Get semantic memory client (lazy import with graceful degradation)."""
    try:
        from src.memory.semantic_memory import get_semantic_memory_client
        return get_semantic_memory_client()
    except Exception as e:
        logger.debug(f"Semantic memory not available: {e}")
        return None


class FeatureAnalyzerAgent:
    """Feature Analyzer: SHAP-based model interpretability.

    This is a HYBRID agent with 3 nodes:
    1. SHAP Computation (NO LLM) - Compute SHAP values
    2. Interaction Detection (NO LLM) - Detect feature interactions
    3. NL Interpretation (LLM) - Generate human-readable explanations
    """

    # Class attributes per contract
    tier = 0
    tier_name = "ml_foundation"
    agent_name = "feature_analyzer"
    agent_type = "hybrid"  # Computation + LLM
    sla_seconds = 120
    tools = ["shap", "pandas", "numpy", "scipy"]
    primary_model = "claude-sonnet-4-20250514"  # For NL interpretation node

    def __init__(self):
        """Initialize feature_analyzer agent."""
        self._full_graph = None  # Lazy load
        self._shap_graph = None  # Lazy load

    def _get_full_graph(self):
        """Get full feature analyzer graph (lazy loaded)."""
        if self._full_graph is None:
            self._full_graph = create_feature_analyzer_graph()
        return self._full_graph

    def _get_shap_graph(self):
        """Get SHAP-only analysis graph (lazy loaded)."""
        if self._shap_graph is None:
            self._shap_graph = create_shap_analysis_graph()
        return self._shap_graph

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
        start_time = datetime.now()
        logger.info("Starting feature analysis pipeline")

        # Validate required inputs - model_uri is optional (SHAP skipped if not provided)
        required_fields = ["experiment_id"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        experiment_id = input_data["experiment_id"]
        model_uri = input_data.get("model_uri")

        # Warn if model_uri is missing (SHAP analysis will be skipped)
        if not model_uri:
            logger.warning(
                "model_uri not provided - SHAP analysis will be skipped. "
                "Only basic feature statistics will be computed."
            )

        # Prepare initial state
        initial_state: FeatureAnalyzerState = {
            # Input fields
            "model_uri": model_uri,
            "experiment_id": experiment_id,
            "training_run_id": input_data.get("training_run_id", "unknown"),
            # Configuration
            "max_samples": input_data.get("max_samples", 1000),
            "compute_interactions": input_data.get("compute_interactions", True),
            "store_in_semantic_memory": input_data.get("store_in_semantic_memory", True),
            # Optional data - support both X_train (for full pipeline) and X_sample (for SHAP-only)
            "X_sample": input_data.get("X_sample"),
            "y_sample": input_data.get("y_sample"),
            "X_train": input_data.get("X_train"),
            "y_train": input_data.get("y_train"),
            # Feature names from data_preparer (for SHAP output)
            "feature_columns": input_data.get("feature_columns"),
            # Status
            "status": "in_progress",
        }

        # Select appropriate workflow based on inputs
        # If X_train is provided, use full pipeline (feature generation -> selection -> SHAP)
        # If only model_uri is provided, use SHAP-only pipeline
        if input_data.get("X_train") is not None:
            graph = self._get_full_graph()
        else:
            graph = self._get_shap_graph()

        # Execute the graph with optional Opik tracing
        opik = _get_opik_connector()
        try:
            # Wrap execution in Opik trace if available
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="analyze_features",
                    metadata={
                        "experiment_id": experiment_id,
                        "model_uri": model_uri,
                        "tier": self.tier,
                        "max_samples": initial_state["max_samples"],
                        "compute_interactions": initial_state["compute_interactions"],
                    },
                    tags=[self.agent_name, "tier_0", "shap", "interpretability"],
                    input_data={"model_uri": model_uri},
                ) as span:
                    final_state = await graph.ainvoke(initial_state)
                    # Set output on span
                    span.set_output({
                        "samples_analyzed": final_state.get("samples_analyzed"),
                        "explainer_type": final_state.get("explainer_type"),
                        "top_features_count": len(final_state.get("top_features", [])),
                    })
            else:
                final_state = await graph.ainvoke(initial_state)

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

            # Check if SHAP was skipped
            shap_skipped = final_state.get("shap_skipped", False)

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
                # SHAP skip status
                "shap_skipped": shap_skipped,
                "shap_skip_reason": final_state.get("skip_reason") if shap_skipped else None,
                # Status
                "status": "completed" if not shap_skipped else "completed_without_shap",
            }

            # Store to database (ml_shap_analyses table)
            await self._store_to_database(output)

            # Log execution time
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Feature analysis completed in {duration:.2f}s (SLA: {self.sla_seconds}s)"
            )

            # Check SLA
            if duration > self.sla_seconds:
                logger.warning(f"SLA violation: {duration:.2f}s > {self.sla_seconds}s")

            return output

        except Exception as e:
            logger.error(f"Feature analysis failed: {e}", exc_info=True)
            raise RuntimeError(f"Feature analysis failed: {str(e)}") from e

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

    async def _update_semantic_memory(self, state: Dict[str, Any]) -> Tuple[bool, int]:
        """Update semantic memory with feature relationships.

        Graceful degradation: If semantic memory is unavailable,
        logs a debug message and continues without error.

        Args:
            state: Final agent state

        Returns:
            Tuple of (updated: bool, entries: int)
        """
        try:
            memory = _get_semantic_memory()
            if memory is None:
                logger.debug("Semantic memory not available, skipping update")
                return False, 0

            entries_added = 0
            experiment_id = state.get("experiment_id")

            # Store feature importance relationships
            global_importance_ranked = state.get("global_importance_ranked", [])
            for rank, (feature, importance) in enumerate(global_importance_ranked[:10], 1):
                try:
                    await memory.add_relationship(
                        source="experiment",
                        source_id=experiment_id,
                        target="feature",
                        target_id=feature,
                        relationship_type="has_important_feature",
                        properties={
                            "importance": float(importance),
                            "rank": rank,
                            "model_version": state.get("model_version", "unknown"),
                        }
                    )
                    entries_added += 1
                except Exception as e:
                    logger.debug(f"Failed to add feature relationship: {e}")

            # Store feature interactions
            top_interactions_raw = state.get("top_interactions_raw", [])
            for feat1, feat2, strength in top_interactions_raw[:5]:
                try:
                    await memory.add_relationship(
                        source="feature",
                        source_id=feat1,
                        target="feature",
                        target_id=feat2,
                        relationship_type="interacts_with",
                        properties={
                            "interaction_strength": float(strength),
                            "experiment_id": experiment_id,
                        }
                    )
                    entries_added += 1
                except Exception as e:
                    logger.debug(f"Failed to add interaction relationship: {e}")

            logger.info(f"Updated semantic memory with {entries_added} entries")
            return True, entries_added

        except Exception as e:
            logger.warning(f"Failed to update semantic memory: {e}")
            return False, 0

    async def _store_to_database(self, output: Dict[str, Any]) -> None:
        """Store SHAP analysis to ml_shap_analyses table.

        Graceful degradation: If repository is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output to store
        """
        try:
            repo = _get_shap_repository()
            if repo is None:
                logger.debug("Skipping SHAP analysis persistence (no repository)")
                return

            # Build analysis dict for repository
            analysis_dict = {
                "experiment_id": output.get("experiment_id"),
                "feature_importance": output.get("feature_importance", []),
                "interactions": output.get("interactions", []),
                "interpretation": output.get("interpretation"),
                "top_features": output.get("top_features", []),
                "samples_analyzed": output.get("samples_analyzed"),
                "computation_time_seconds": output.get("computation_time_seconds"),
                "explainer_type": output.get("explainer_type"),
                "model_version": output.get("model_version"),
            }

            # Get model_registry_id from shap_analysis if available
            model_registry_id = output.get("shap_analysis", {}).get("model_registry_id")

            result = await repo.store_analysis(
                analysis_dict=analysis_dict,
                model_registry_id=model_registry_id,
            )

            if result:
                logger.info(f"Persisted SHAP analysis for {output.get('experiment_id')}")
            else:
                logger.debug("SHAP analysis not persisted (no result returned)")

        except Exception as e:
            logger.warning(f"Failed to persist SHAP analysis: {e}")
