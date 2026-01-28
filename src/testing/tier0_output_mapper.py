"""Tier0 Output Mapper for Tier 1-5 Agent Testing.

Maps tier0 synthetic data outputs to each agent's required inputs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd


class Tier0OutputMapper:
    """Maps tier0 state dictionary to agent-specific inputs.

    The tier0 test script produces a state dictionary containing:
    - trained_model: sklearn/xgboost/lightgbm model
    - model_uri: str (MLflow URI)
    - validation_metrics: dict (auc_roc, precision, recall, f1_score)
    - feature_importance: list[{feature, importance}]
    - eligible_df: DataFrame (patient cohort)
    - qc_report: dict (data quality)
    - experiment_id: str
    - cohort_result: CohortExecutionResult
    - scope_spec: dict (brand, indication, etc.)
    - class_imbalance_info: dict (ratio, strategy)

    This class provides mapping methods for each Tier 1-5 agent.
    """

    # Required keys from tier0 state
    REQUIRED_KEYS = [
        "experiment_id",
        "eligible_df",
    ]

    # Optional but useful keys
    OPTIONAL_KEYS = [
        "trained_model",
        "model_uri",
        "validation_metrics",
        "feature_importance",
        "qc_report",
        "cohort_result",
        "scope_spec",
        "class_imbalance_info",
    ]

    def __init__(self, tier0_state: dict[str, Any]):
        """Initialize with tier0 state dictionary.

        Args:
            tier0_state: State dictionary from tier0 test run
        """
        self.state = tier0_state
        self._validate_required_keys()

    def _validate_required_keys(self) -> None:
        """Validate that required keys exist in state."""
        missing = [k for k in self.REQUIRED_KEYS if k not in self.state]
        if missing:
            raise ValueError(f"Missing required tier0 state keys: {missing}")

    def _get_feature_names(self) -> list[str]:
        """Extract feature names from feature_importance or eligible_df."""
        if self.state.get("feature_importance"):
            return [
                f["feature"]
                for f in self.state["feature_importance"]
                if isinstance(f, dict) and "feature" in f
            ]
        # Fallback to DataFrame columns (exclude non-feature columns)
        df = self.state.get("eligible_df")
        if df is not None:
            exclude = {"patient_journey_id", "patient_id", "brand", "discontinuation_flag"}
            return [c for c in df.columns if c not in exclude]
        return []

    def _get_top_features(self, n: int = 5) -> list[str]:
        """Get top N features by importance."""
        features = self._get_feature_names()
        return features[:n] if features else []

    # =========================================================================
    # TIER 1: Orchestrator Agents
    # =========================================================================

    def map_to_orchestrator(self) -> dict[str, Any]:
        """Map to OrchestratorState input.

        Orchestrator expects:
        - messages: list[dict] with user query
        - current_agent: Optional[str]
        - agent_outputs: dict
        """
        brand = self.state.get("scope_spec", {}).get("brand", "Kisqali")
        return {
            "query": f"What factors drive therapy discontinuation for {brand}?",
            "messages": [
                {
                    "role": "user",
                    "content": f"What factors drive therapy discontinuation for {brand}?",
                }
            ],
            "current_agent": None,
            "agent_outputs": {},
            "experiment_id": self.state["experiment_id"],
        }

    def map_to_tool_composer(self) -> dict[str, Any]:
        """Map to ToolComposer input.

        ToolComposer handles MULTI_FACETED queries requiring multiple tools.
        """
        brand = self.state.get("scope_spec", {}).get("brand", "Kisqali")
        return {
            "query": (
                f"Compare the causal impact of HCP visits vs prior treatments "
                f"on discontinuation for {brand}, and identify high-risk segments"
            ),
            "experiment_id": self.state["experiment_id"],
            "available_tools": [
                "causal_effect_estimator",
                "cate_analyzer",
                "segment_ranker",
                "gap_calculator",
            ],
        }

    # =========================================================================
    # TIER 2: Causal Agents
    # =========================================================================

    def map_to_causal_impact(self) -> dict[str, Any]:
        """Map to CausalImpactState input.

        CausalImpact expects:
        - query, query_id
        - treatment_var, outcome_var
        - confounders: list[str]
        - data_source: str
        """
        df = self.state["eligible_df"]
        features = self._get_top_features(5)

        # Use actual columns from the DataFrame
        treatment_var = "hcp_visits" if "hcp_visits" in df.columns else features[0] if features else "treatment"
        outcome_var = "discontinuation_flag" if "discontinuation_flag" in df.columns else "outcome"

        confounders = [f for f in features if f not in {treatment_var, outcome_var}]

        return {
            "query": f"What is the causal effect of {treatment_var} on {outcome_var}?",
            "query_id": str(uuid.uuid4()),
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "confounders": confounders[:5],
            "data_source": "patient_journeys",
            "experiment_id": self.state["experiment_id"],
            "data": df,  # Pass actual DataFrame for analysis
        }

    def map_to_gap_analyzer(self) -> dict[str, Any]:
        """Map to GapAnalyzerState input.

        GapAnalyzer expects:
        - query: str
        - metrics: List[str] - KPI names to analyze
        - segments: List[str] - Segmentation dimensions
        - brand: str
        - time_period: str (optional, default "current_quarter")
        - filters: Optional[Dict] (optional)
        - gap_type: Literal (optional, default "vs_potential")
        """
        df = self.state["eligible_df"]

        # metrics should be a list of KPI names (strings)
        metrics = ["trx", "market_share", "conversion_rate"]

        # segments should be a list of dimension names (strings)
        segments = []
        if "geographic_region" in df.columns:
            segments.append("geographic_region")
        if "age_group" in df.columns:
            segments.append("age_group")
        if "prior_treatments" in df.columns:
            segments.append("prior_treatments")
        if not segments:
            segments = ["all"]  # Default segment

        return {
            "query": "Identify performance gaps and ROI opportunities",
            "metrics": metrics,
            "segments": segments,
            "brand": self.state.get("scope_spec", {}).get("brand", "Kisqali"),
            "time_period": "current_quarter",
            "gap_type": "vs_potential",
        }

    def map_to_heterogeneous_optimizer(self) -> dict[str, Any]:
        """Map to HeterogeneousOptimizerState input.

        HeterogeneousOptimizer expects:
        - query: str
        - treatment_var: str
        - outcome_var: str
        - segment_vars: List[str] - Variables to segment by
        - effect_modifiers: List[str] - Variables that modify treatment effect
        - data_source: str
        - filters: Optional[Dict]

        Note: When Supabase is not configured, the agent falls back to MockDataConnector
        which has specific column names. We use those as defaults for testing.

        MockDataConnector columns:
        - Treatment: hcp_engagement_frequency (binary 0/1)
        - Outcome: trx_total (continuous)
        - Segments: hcp_specialty, patient_volume_decile, region
        - Effect modifiers: hcp_tenure, competitive_pressure, formulary_status
        """
        df = self.state["eligible_df"]
        features = self._get_top_features(5)

        # ALWAYS use MockDataConnector-compatible column names for testing
        # The agent falls back to MockDataConnector when Supabase isn't configured,
        # so we must use column names that MockDataConnector provides.
        # Do NOT override with tier0 columns as they won't exist in mock data.
        treatment_var = "hcp_engagement_frequency"  # MockDataConnector treatment (binary 0/1)
        outcome_var = "trx_total"  # MockDataConnector outcome (continuous)

        # Segment variables for CATE analysis
        # MockDataConnector provides: hcp_specialty, patient_volume_decile, region
        segment_vars = ["hcp_specialty", "region"]

        # Effect modifiers - variables that can modify treatment effect
        # MockDataConnector provides: hcp_tenure, competitive_pressure, formulary_status
        effect_modifiers = ["hcp_tenure", "competitive_pressure", "formulary_status"]

        return {
            "query": f"Analyze heterogeneous treatment effects of {treatment_var}",
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "segment_vars": segment_vars[:3],
            "effect_modifiers": effect_modifiers,
            "data_source": "patient_journeys",  # MockDataConnector ignores this
            "filters": None,
        }

    # =========================================================================
    # TIER 3: Monitoring Agents
    # =========================================================================

    def map_to_drift_monitor(self) -> dict[str, Any]:
        """Map to DriftMonitorInput (Pydantic model).

        DriftMonitorInput schema:
        - query: str (required)
        - features_to_monitor: list[str] (required)
        - model_id: Optional[str]
        - time_window: str (default "7d")
        - brand: Optional[str]
        - significance_level: float (default 0.05)
        """
        feature_cols = self._get_feature_names()

        return {
            "query": "Detect data and model drift in patient features",
            "features_to_monitor": feature_cols[:10],  # Limit features
            "model_id": None,
            "time_window": "30d",
            "brand": self.state.get("scope_spec", {}).get("brand"),
            "significance_level": 0.05,
        }

    def map_to_experiment_designer(self) -> dict[str, Any]:
        """Map to ExperimentDesignerInput (Pydantic model).

        ExperimentDesignerInput schema:
        - business_question: str (required, min 10 chars)
        - constraints: dict (optional) - budget, timeline, ethical, operational
        - available_data: dict (optional)
        - preregistration_formality: "light" | "medium" | "heavy" (default "medium")
        - max_redesign_iterations: int (default 2)
        - enable_validity_audit: bool (default True)
        - brand: Optional[str]
        """
        df = self.state["eligible_df"]
        validation_metrics = self.state.get("validation_metrics", {})
        brand = self.state.get("scope_spec", {}).get("brand")

        return {
            "business_question": "Does personalized HCP outreach improve patient retention rates compared to standard outreach?",
            "constraints": {
                "budget": 50000,
                "timeline": {"max_duration_days": 90},
                "operational": {
                    "min_sample_size": 100,
                    "max_sample_size": int(len(df) * 0.5),
                },
                "expected_effect_size": 0.10,
            },
            "available_data": {
                "total_patients": len(df),
                "historical_retention_rate": 1 - validation_metrics.get("recall", 0.3) * 0.4,
                "features": self._get_feature_names()[:10],
            },
            "preregistration_formality": "medium",
            "max_redesign_iterations": 2,
            "enable_validity_audit": True,
            "brand": brand,
        }

    def map_to_health_score(self) -> dict[str, Any]:
        """Map to HealthScoreAgent.check_health() kwargs.

        check_health signature:
        - scope: Literal["full", "quick", "models", "pipelines", "agents"]
        - query: str
        - experiment_name: str
        """
        return {
            "scope": "full",
            "query": "Check system health status",
            "experiment_name": self.state["experiment_id"],
        }

    # =========================================================================
    # TIER 4: ML Prediction Agents
    # =========================================================================

    def map_to_prediction_synthesizer(self) -> dict[str, Any]:
        """Map to PredictionSynthesizerAgent.synthesize() kwargs.

        synthesize signature:
        - entity_id: str
        - prediction_target: str
        - features: Optional[Dict[str, Any]]
        - entity_type: str (hcp, territory, patient)
        - time_horizon: str
        - models_to_use: Optional[List[str]]
        - ensemble_method: str
        - include_context: bool
        - query: str
        - session_id: Optional[str]
        """
        df = self.state["eligible_df"]

        # Get a sample entity
        sample_entity_id = str(df.iloc[0].get("patient_journey_id", "test_patient_001"))

        # Get feature data for the sample entity
        feature_cols = self._get_feature_names()
        sample_features = df.iloc[0][feature_cols].to_dict() if feature_cols else {}

        return {
            "entity_id": sample_entity_id,
            "prediction_target": "discontinuation_risk",
            "features": sample_features,
            "entity_type": "patient",
            "time_horizon": "30d",
            "models_to_use": None,  # Use all available
            "ensemble_method": "weighted",
            "include_context": True,
            "query": f"Predict discontinuation risk for patient {sample_entity_id}",
            "session_id": self.state["experiment_id"],
        }

    def map_to_resource_optimizer(self) -> dict[str, Any]:
        """Map to ResourceOptimizerAgent.optimize() kwargs.

        optimize signature:
        - allocation_targets: List[AllocationTarget]
        - constraints: List[Constraint]
        - resource_type: str
        - objective: str
        - solver_type: str
        - run_scenarios: bool
        - scenario_count: int
        - query: str
        - session_id: Optional[str]
        """
        df = self.state["eligible_df"]

        # Create allocation targets based on regions if available
        allocation_targets = []
        if "geographic_region" in df.columns:
            for region in df["geographic_region"].unique()[:5]:
                region_df = df[df["geographic_region"] == region]
                allocation_targets.append({
                    "entity_id": f"territory_{region}",
                    "entity_type": "territory",
                    "current_allocation": 50000.0,
                    "expected_response": len(region_df) / len(df),
                    "min_allocation": 25000.0,
                    "max_allocation": 100000.0,
                })
        else:
            # Default allocation targets
            allocation_targets = [
                {
                    "entity_id": "territory_northeast",
                    "entity_type": "territory",
                    "current_allocation": 50000.0,
                    "expected_response": 0.3,
                    "min_allocation": 25000.0,
                    "max_allocation": 100000.0,
                },
                {
                    "entity_id": "territory_midwest",
                    "entity_type": "territory",
                    "current_allocation": 40000.0,
                    "expected_response": 0.25,
                    "min_allocation": 20000.0,
                    "max_allocation": 80000.0,
                },
            ]

        return {
            "allocation_targets": allocation_targets,
            "constraints": [
                {"constraint_type": "budget", "value": 200000.0, "scope": "global"},
            ],
            "resource_type": "budget",
            "objective": "maximize_roi",
            "solver_type": "linear",
            "run_scenarios": False,
            "scenario_count": 3,
            "query": "Optimize budget allocation across territories",
            "session_id": self.state["experiment_id"],
        }

    # =========================================================================
    # TIER 5: Self-Improvement Agents
    # =========================================================================

    def map_to_explainer(self) -> dict[str, Any]:
        """Map to ExplainerAgent.explain() kwargs.

        explain signature:
        - analysis_results: List[Dict[str, Any]]
        - query: str
        - user_expertise: str (executive, analyst, data_scientist)
        - output_format: str (narrative, structured, presentation, brief)
        - focus_areas: Optional[List[str]]
        - session_id: Optional[str]
        - memory_config: Optional[Dict[str, Any]]
        """
        validation_metrics = self.state.get("validation_metrics", {})
        feature_importance = self.state.get("feature_importance", [])

        # Build analysis_results as a list of dicts (format expected by explain())
        analysis_results = [
            {
                "agent": "causal_impact",
                "analysis_type": "causal_analysis",
                "treatment_var": "hcp_visits",
                "outcome_var": "discontinuation_flag",
                "ate": 0.127,
                "ate_ci": [0.089, 0.165],
                "p_value": 0.0023,
                "confounders_identified": self._get_top_features(3),
            },
            {
                "agent": "model_trainer",
                "analysis_type": "model_performance",
                **validation_metrics,
            },
            {
                "agent": "feature_analyzer",
                "analysis_type": "feature_importance",
                "top_features": feature_importance[:5] if feature_importance else [],
            },
        ]

        return {
            "analysis_results": analysis_results,
            "query": "Explain the discontinuation risk analysis results",
            "user_expertise": "analyst",
            "output_format": "structured",
            "focus_areas": ["causal_effects", "feature_importance"],
            "session_id": self.state["experiment_id"],
            "memory_config": {
                "brand": self.state.get("scope_spec", {}).get("brand", "Kisqali"),
            },
        }

    def map_to_feedback_learner(self) -> dict[str, Any]:
        """Map to FeedbackLearnerAgent.learn() kwargs.

        learn signature:
        - time_range_start: str (ISO format)
        - time_range_end: str (ISO format)
        - batch_id: Optional[str]
        - focus_agents: Optional[List[str]]
        """
        now = datetime.now(UTC)

        return {
            "time_range_start": (now - timedelta(days=1)).isoformat(),
            "time_range_end": now.isoformat(),
            "batch_id": f"batch_{self.state['experiment_id'][:8]}",
            "focus_agents": ["causal_impact", "gap_analyzer", "explainer"],
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_all_mappings(self) -> dict[str, dict[str, Any]]:
        """Get all agent mappings as a dictionary.

        Returns:
            Dict mapping agent_name -> mapped_input
        """
        return {
            # Tier 1
            "orchestrator": self.map_to_orchestrator(),
            "tool_composer": self.map_to_tool_composer(),
            # Tier 2
            "causal_impact": self.map_to_causal_impact(),
            "gap_analyzer": self.map_to_gap_analyzer(),
            "heterogeneous_optimizer": self.map_to_heterogeneous_optimizer(),
            # Tier 3
            "drift_monitor": self.map_to_drift_monitor(),
            "experiment_designer": self.map_to_experiment_designer(),
            "health_score": self.map_to_health_score(),
            # Tier 4
            "prediction_synthesizer": self.map_to_prediction_synthesizer(),
            "resource_optimizer": self.map_to_resource_optimizer(),
            # Tier 5
            "explainer": self.map_to_explainer(),
            "feedback_learner": self.map_to_feedback_learner(),
        }

    def get_agent_mapping(self, agent_name: str) -> dict[str, Any]:
        """Get mapping for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'causal_impact')

        Returns:
            Mapped input dictionary for the agent

        Raises:
            ValueError: If agent_name is not supported
        """
        method_name = f"map_to_{agent_name}"
        if not hasattr(self, method_name):
            raise ValueError(f"Unknown agent: {agent_name}. Supported: {list(self.get_all_mappings().keys())}")
        return getattr(self, method_name)()
