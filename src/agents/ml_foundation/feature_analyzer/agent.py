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
from typing import Any, Dict, List, Tuple, cast
from uuid import uuid4

from .graph import create_feature_analyzer_graph, create_shap_analysis_graph
from .memory_hooks import FeatureAnalyzerMemoryHooks
from .state import FeatureAnalyzerState

logger = logging.getLogger(__name__)


async def _get_shap_repository():
    """Get ShapAnalysisRepository (lazy import to avoid circular deps)."""
    try:
        from src.repositories.shap_analysis import get_shap_analysis_repository

        return await get_shap_analysis_repository()
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


async def _get_feast_client():
    """Get FeastClient instance (lazy import with graceful degradation).

    Block 5 (#14): used to auto-register surviving tier0 features as
    Feast FeatureViews so downstream serving can retrieve them via the
    same store the trainer used.
    """
    try:
        from src.feature_store.feast_client import get_feast_client

        return await get_feast_client()
    except Exception as e:
        logger.debug(f"Feast client not available: {e}")
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
    primary_model = "claude-sonnet-4-6"  # For NL interpretation node

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
            # D1.2: thread caller-provided audit_workflow_id (see scope_definer
            # for the rationale). Backlog #1 (closed 2026-05-09) tightened the
            # State to required-no-default to fix the LangGraph channel-reducer
            # bug (default_factory firing on every Schema reconstruction).
            # Caller-provided UUID is preferred; absent that, generate one at
            # the agent boundary. Either way the UUID is set ONCE before
            # graph.ainvoke, so LangGraph's reducer pins it across nodes.
            **(
                {"audit_workflow_id": input_data["audit_workflow_id"]}
                if input_data.get("audit_workflow_id") is not None
                else {"audit_workflow_id": uuid4()}
            ),
            # Input fields
            "model_uri": model_uri,  # type: ignore[typeddict-item]
            "experiment_id": experiment_id,
            "training_run_id": input_data.get("training_run_id", "unknown"),
            # In-memory model passthrough (avoids round-trip through MLflow loaders).
            # Accept either "trained_model" (canonical caller key, e.g.
            # scripts/run_tier0_test.py) or "loaded_model" (state-key form).
            "loaded_model": input_data.get("trained_model") or input_data.get("loaded_model"),
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
            "feature_columns": input_data.get("feature_columns"),  # type: ignore[typeddict-item]
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
                    span.set_output(
                        {
                            "samples_analyzed": final_state.get("samples_analyzed"),
                            "explainer_type": final_state.get("explainer_type"),
                            "top_features_count": len(final_state.get("top_features", [])),
                        }
                    )
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
                (
                    semantic_memory_updated,
                    semantic_memory_entries,
                ) = await self._update_semantic_memory(final_state)

            # Record the feature analysis to episodic memory (#749 — store_feature_analysis
            # was defined but never called from run() and used a non-existent insert API).
            await self._update_episodic_memory(final_state)

            # Block 5 (#14): auto-register surviving features in Feast as a
            # FeatureView so downstream serving can read them via the same
            # store the trainer used. Best-effort — failures don't block
            # the tier0 run; they're surfaced in feast_registration.error.
            feast_registration = await self._auto_register_in_feast(final_state, input_data)

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
                # Block 5 (#14): Feast FeatureView round-trip metadata.
                "feast_registration": feast_registration,
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
                # F8: surface SHAP data provenance end-to-end so API / chatbot /
                # tool_composer consumers can tell real importances from a
                # synthetic-background (opt-in) or unavailable (skipped) run.
                "data_provenance": final_state.get("data_provenance", "unknown"),
                # Status
                "status": "completed" if not shap_skipped else "completed_without_shap",
            }

            # Store to database (ml_shap_analyses table)
            # Pass model_registry_id from input if available (set by upstream agents)
            output["model_registry_id"] = input_data.get("model_registry_id")
            await self._store_to_database(output)

            # Log execution time
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Feature analysis completed in {duration:.2f}s (SLA: {self.sla_seconds}s)")

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
            # F8: carry SHAP data provenance ('real' | 'synthetic' | 'unavailable')
            # into the contract dict so it is never silently dropped.
            "data_provenance": state.get("data_provenance", "unknown"),
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

    def _build_interaction_list(self, state: Dict[str, Any]) -> List[Any]:
        """Build FeatureInteraction list.

        Args:
            state: Final agent state

        Returns:
            List of FeatureInteraction dicts
        """
        interaction_interpretations = state.get("interaction_interpretations", [])

        # If we have LLM interpretations, use those
        if interaction_interpretations:
            return cast(List[Any], interaction_interpretations)

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
        """Populate the semantic knowledge graph (FalkorDB ``e2i_causal``) with the
        feature-importance findings (#749).

        Drives the ``store_feature_importance_patterns`` hook (typed ``Feature``
        entities + ``HAS_IMPORTANCE`` / ``INTERACTS_WITH`` edges). The previous
        implementation called ``semantic_memory.add_relationship(source=…, target=…)``
        — a signature that never existed on ``FalkorDBSemanticMemory`` — so it raised,
        was swallowed, and wrote NO typed nodes; ``e2i_causal`` stayed unchanged.

        Graceful degradation: returns ``(False, 0)`` if semantic memory is unavailable
        or the write fails. Preserves the ``(updated, entries)`` contract that ``run()``
        unpacks at the call site.

        Args:
            state: Final agent state carrying experiment_id,
                ``global_importance_ranked`` (List[(feature, importance)]) and
                ``top_interactions_raw`` (List[(feat1, feat2, strength)]).

        Returns:
            Tuple of (updated: bool, entries: int).
        """
        try:
            experiment_id = state.get("experiment_id")
            if not experiment_id:
                logger.debug("No experiment_id; skipping semantic-graph update")
                return False, 0

            global_importance = {
                str(feature): float(importance)
                for feature, importance in (state.get("global_importance_ranked", []) or [])
            }
            interactions = [
                {
                    "feature_1": feat1,
                    "feature_2": feat2,
                    "interaction_strength": float(strength),
                }
                for feat1, feat2, strength in (state.get("top_interactions_raw", []) or [])
            ]

            if not global_importance and not interactions:
                logger.debug("No feature importance/interactions; skipping semantic-graph update")
                return False, 0

            hooks = FeatureAnalyzerMemoryHooks()
            ok = await hooks.store_feature_importance_patterns(
                experiment_id=str(experiment_id),
                global_importance=global_importance,
                interactions=interactions,
            )
            if not ok:
                return False, 0

            entries = min(len(global_importance), 10) + min(len(interactions), 10)
            logger.info(f"Updated semantic graph (e2i_causal) with {entries} feature entries")
            return True, entries

        except Exception as e:
            logger.warning(f"Failed to update semantic memory: {e}")
            return False, 0

    async def _update_episodic_memory(self, final_state: Dict[str, Any]) -> None:
        """Record the feature analysis to EPISODIC memory (#749).

        ``store_feature_analysis`` was defined but never called from ``run()`` AND
        called a non-existent ``insert_episodic_memory`` signature — both fixed
        (compat shim + migration 039). Graceful degradation. ``session_id`` is the
        ``audit_workflow_id`` (uuid column) or a fresh UUID.
        """
        try:
            experiment_id = final_state.get("experiment_id")
            if not experiment_id:
                return
            session_id = str(final_state.get("audit_workflow_id") or uuid4())
            hooks = FeatureAnalyzerMemoryHooks()
            await hooks.store_feature_analysis(
                session_id=session_id, result=final_state, state=final_state
            )
        except Exception as e:
            logger.debug(f"Failed to update episodic memory: {e}")

    async def _auto_register_in_feast(
        self, state: Dict[str, Any], input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Auto-register surviving features as a Feast FeatureView (Block 5 #14).

        Reads ``selected_features`` (or ``selected_features_all``) from
        the final agent state and asks ``FeastClient.register_feature_view``
        to apply a new FeatureView for them. Best-effort: any failure is
        captured in the returned dict and logged, never raised.

        The required Feast inputs (entity name, batch source name) are
        sourced from ``input_data["feast_registration_config"]`` so the
        caller (run_tier0_test.py / orchestrator) can declare which
        Feast Entity and existing batch source to bind tier0 features
        to. When the config is absent, registration is skipped — this
        keeps unit tests that don't talk to Feast unaffected.

        Args:
            state: Final agent state from the LangGraph run.
            input_data: Original input_data passed to ``run`` (carries
                ``feast_registration_config`` when the caller wants
                round-trip registration).

        Returns:
            Dict with ``registered`` (bool), ``feature_view_name`` (str
            or None), ``features_count`` (int), ``error`` (str | None),
            ``skipped_reason`` (str | None). Never raises.
        """
        result: Dict[str, Any] = {
            "registered": False,
            "feature_view_name": None,
            "features_count": 0,
            "error": None,
            "skipped_reason": None,
        }

        config = input_data.get("feast_registration_config")
        if not config:
            result["skipped_reason"] = "feast_registration_config not provided"
            return result

        selected_features = (
            state.get("selected_features") or state.get("selected_features_all") or []
        )
        if not selected_features:
            result["skipped_reason"] = "no surviving features in state"
            return result

        try:
            feast_client = await _get_feast_client()
            if feast_client is None:
                result["skipped_reason"] = "feast client unavailable"
                return result

            experiment_id = state.get("experiment_id", "unknown")
            fv_name = config.get("feature_view_name") or f"tier0_{experiment_id}_features"

            registration = await feast_client.register_feature_view(
                name=fv_name,
                entity_name=config["entity_name"],
                feature_names=list(selected_features),
                batch_source_name=config["batch_source_name"],
                ttl=config.get("ttl"),
                feature_dtypes=config.get("feature_dtypes"),
                tags={
                    "owner": "feature_analyzer",
                    "experiment_id": str(experiment_id),
                    "auto_registered": "true",
                },
                description=(f"Tier-0 features surviving selection for experiment {experiment_id}"),
            )

            result["registered"] = registration.get("registered", False)
            result["feature_view_name"] = registration.get("feature_view_name")
            result["features_count"] = registration.get("features_count", 0)
            result["error"] = registration.get("error")
            return result

        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"Feast auto-registration failed: {e}")
            return result

    async def _store_to_database(self, output: Dict[str, Any]) -> None:
        """Store SHAP analysis to ml_shap_analyses table.

        Graceful degradation: If repository is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output to store
        """
        try:
            # F8: do not persist a skipped run — it has no importances, and an empty
            # global_importance row in ml_shap_analyses is indistinguishable from a
            # genuine real-zero result. Skip persistence (fail closed on lineage too).
            if output.get("shap_skipped"):
                logger.info(
                    "SHAP skipped (%s); not persisting an empty ml_shap_analyses row.",
                    output.get("shap_skip_reason") or "no reason",
                )
                return

            repo = await _get_shap_repository()
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
                # F8: provenance of the persisted importances. The repo currently
                # whitelists columns (drops this until a lineage column is added via
                # migration — tracked follow-up), but we pass it through the contract.
                "data_provenance": output.get("data_provenance"),
            }

            # Get model_registry_id from input (passed by upstream agent or test)
            model_registry_id = output.get("model_registry_id")

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
