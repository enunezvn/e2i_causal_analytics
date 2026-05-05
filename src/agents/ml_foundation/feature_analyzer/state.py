"""State definition for feature_analyzer agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard C of the migration. Inherits from ``BaseAgentSchema``.

The ``arbitrary_types_allowed=True`` setting (inherited) is load-bearing
here for ``shap_values: Optional[np.ndarray]`` and the various
``X_*: Optional[Any]`` fields holding pandas DataFrames / numpy arrays.

This agent is a HYBRID agent with 5 nodes:
1. Feature Generation (NO LLM) - Generate engineered features
2. Feature Selection (NO LLM) - Select optimal features
3. SHAP Computation (NO LLM) - Compute SHAP values
4. Interaction Detection (NO LLM) - Detect feature interactions
5. NL Interpretation (LLM) - Generate human-readable explanations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from pydantic import Field

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)


class FeatureAnalyzerState(BaseAgentSchema):
    """State for feature_analyzer agent.

    Follows hybrid execution pattern:
    - Feature engineering nodes (1-2): Generate and select features
    - SHAP analysis nodes (3-4): Compute importance and interactions
    - LLM node (5): Add interpretation fields
    """

    # === INPUT FIELDS (from data_preparer) ===

    X_train: Optional[Any] = None  # Training features (DataFrame or ndarray)
    X_val: Optional[Any] = None  # Validation features (optional)
    X_test: Optional[Any] = None  # Test features (optional)
    y_train: Optional[Any] = None  # Training target
    y_val: Optional[Any] = None  # Validation target (optional)
    y_test: Optional[Any] = None  # Test target (optional)
    problem_type: Optional[str] = None  # "classification" | "regression"
    feature_columns: Optional[List[str]] = None  # Feature names from data_preparer

    # === INPUT FIELDS (for SHAP - from model_trainer) ===

    model_uri: Optional[str] = None  # MLflow model URI (e.g., "runs:/abc123/model")
    experiment_id: Optional[str] = None
    training_run_id: Optional[str] = None

    # === FEATURE GENERATION CONFIGURATION ===

    feature_config: Optional[Dict[str, Any]] = None
    temporal_columns: Optional[List[str]] = None  # Columns for temporal features
    categorical_columns: Optional[List[str]] = None  # Columns for interaction features
    numeric_columns: Optional[List[str]] = None

    # === FEATURE SELECTION CONFIGURATION ===

    selection_config: Optional[Dict[str, Any]] = None

    # === NODE 1 OUTPUT: Feature Generation (NO LLM) ===

    X_train_generated: Optional[Any] = None  # DataFrame with generated features
    X_val_generated: Optional[Any] = None
    X_test_generated: Optional[Any] = None
    generated_features: Optional[List[Dict[str, Any]]] = None  # Metadata for generated features
    feature_metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None  # By type
    original_feature_count: Optional[int] = None  # Features before generation
    new_feature_count: Optional[int] = None  # New features added
    new_feature_names: Optional[List[str]] = None  # Names of new features
    feature_generation_time_seconds: Optional[float] = None
    temporal_columns_used: Optional[List[str]] = None
    categorical_columns_used: Optional[List[str]] = None
    numeric_columns_used: Optional[List[str]] = None

    # === NODE 2 OUTPUT: Feature Selection (NO LLM) ===

    X_train_selected: Optional[Any] = None  # DataFrame with selected features
    X_val_selected: Optional[Any] = None
    X_test_selected: Optional[Any] = None
    selected_features: Optional[List[str]] = None  # List of selected numeric feature names
    selected_features_all: Optional[List[str]] = None  # All selected features
    feature_importance: Optional[Dict[str, float]] = None  # Importance scores from selection
    feature_importance_ranked: Optional[List[Tuple[str, float]]] = None  # Ranked features
    removed_features: Optional[Dict[str, List[str]]] = None  # By method: variance, correlation, vif
    selection_history: Optional[List[Dict[str, Any]]] = None  # Step-by-step selection log
    feature_statistics: Optional[Dict[str, Dict[str, Any]]] = None  # Statistics per feature
    selected_feature_count: Optional[int] = None  # Features after selection
    total_selected_count: Optional[int] = None  # Total including non-numeric
    selection_time_seconds: Optional[float] = None

    # === SHAP CONFIGURATION ===

    max_samples: Optional[int] = None  # Max samples for SHAP (default: 1000)
    compute_interactions: Optional[bool] = None  # Whether to compute interactions
    store_in_semantic_memory: Optional[bool] = None  # Whether to store in semantic memory

    # === NODE 3 OUTPUT: SHAP Computation (NO LLM) ===

    # Loaded model
    loaded_model: Optional[Any] = None  # sklearn/xgboost/etc model object
    feature_names: Optional[List[str]] = None  # Feature names from model

    # Training data sample (for SHAP computation)
    X_sample: Optional[Any] = None  # pandas DataFrame or numpy array
    y_sample: Optional[Any] = None  # pandas Series or numpy array
    samples_analyzed: Optional[int] = None  # Number of samples used

    # SHAP values (raw) — np.ndarray; arbitrary_types_allowed (inherited
    # from BaseAgentSchema) covers it. JSON serialization round-trips
    # through .tolist() — handled by pydantic's default for ndarray.
    shap_values: Optional[np.ndarray] = None  # SHAP values (n_samples, n_features)
    base_value: Optional[float] = None  # Base value (expected value)

    # Global importance
    global_importance: Optional[Dict[str, float]] = None  # {"feature_name": score}
    global_importance_ranked: Optional[List[Tuple[str, float]]] = None  # Sorted by importance

    # Directional effects
    feature_directions: Optional[Dict[str, str]] = None  # {"feature": "positive"|"negative"|...}

    # Top features
    top_features: Optional[List[str]] = None  # Top 5 features by importance

    # Computation metadata
    shap_computation_time_seconds: Optional[float] = None
    explainer_type: Optional[str] = None  # "TreeExplainer" | "KernelExplainer" | ...

    # === NODE 2 OUTPUT: Interaction Detection (NO LLM) ===

    # Interaction matrix
    interaction_matrix: Optional[Dict[str, Dict[str, float]]] = None  # {"feat1": {"feat2": ...}}

    # Top interactions
    top_interactions_raw: Optional[List[Tuple[str, str, float]]] = None

    # Interaction computation metadata
    interaction_computation_time_seconds: Optional[float] = None
    interaction_method: Optional[str] = None  # "correlation" | "shap_interaction"

    # === NODE 3 OUTPUT: NL Interpretation (LLM) ===

    # Executive summary
    executive_summary: Optional[str] = None  # High-level summary for stakeholders

    # Feature explanations
    feature_explanations: Optional[Dict[str, str]] = None  # {"feature": "explanation"}

    # Interaction interpretations
    interaction_interpretations: Optional[List[Dict[str, Any]]] = None

    # Key insights
    key_insights: Optional[List[str]] = None  # Bullet points of key findings

    # Actionable recommendations
    recommendations: Optional[List[str]] = None  # Actionable next steps

    # Cautions
    cautions: Optional[List[str]] = None  # Warnings about model behavior

    # LLM metadata
    interpretation_model: Optional[str] = None  # "claude-sonnet-4-20250514"
    interpretation_time_seconds: Optional[float] = None
    interpretation_tokens: Optional[int] = None

    # === SEMANTIC MEMORY ===

    semantic_memory_updated: Optional[bool] = None  # Whether semantic memory was updated
    semantic_memory_entries: Optional[int] = None  # Number of entries added

    # === OUTPUT FIELDS (Final) ===

    # SHAP Analysis ID
    shap_analysis_id: Optional[str] = None  # Unique identifier for this analysis

    # Structured outputs for contracts
    feature_importance_list: Optional[List[Dict[str, Any]]] = None
    interaction_list: Optional[List[Dict[str, Any]]] = None

    # Natural language interpretation (final)
    interpretation: Optional[str] = None  # Complete NL summary

    # Model version
    model_version: Optional[str] = None  # Model version from MLflow

    # Total computation time
    total_computation_time_seconds: Optional[float] = None

    # Status
    status: Optional[str] = None  # "completed" | "failed" | "in_progress"

    # === SKIP HANDLING ===

    shap_skipped: Optional[bool] = None
    skip_reason: Optional[str] = None

    # === ERROR HANDLING ===

    error: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # === AUDIT CHAIN ===
    audit_workflow_id: UUID = Field(default_factory=uuid4)

    _validate_audit_id = audit_workflow_id_validator()

    # === DISCOVERY INTEGRATION (V4.4) ===
    # Configuration for causal discovery integration

    discovery_enabled: Optional[bool] = None  # Enable causal discovery (default: False)
    discovery_config: Optional[Dict[str, Any]] = None  # DiscoveryConfig as dict

    # Discovery results from DiscoveryRunner
    discovery_result: Optional[Dict[str, Any]] = None  # DiscoveryResult as dict

    # Gate evaluation from DiscoveryGate
    discovery_gate_decision: Optional[str] = None  # "accept" | "review" | "reject" | "augment"
    discovery_gate_confidence: Optional[float] = None  # Overall confidence [0, 1]
    discovery_gate_reasons: Optional[List[str]] = None  # Reasons for decision

    # === CAUSAL RANKING (V4.4) ===

    # Target variable for causal analysis
    causal_target_variable: Optional[str] = None  # Target for causal path analysis

    # Rankings from DriverRanker
    causal_rankings: Optional[List[Dict[str, Any]]] = None  # List of FeatureRanking dicts

    # Rank correlation
    rank_correlation: Optional[float] = None  # Spearman correlation

    # Feature categorization
    divergent_features: Optional[List[str]] = None
    causal_only_features: Optional[List[str]] = None
    predictive_only_features: Optional[List[str]] = None
    concordant_features: Optional[List[str]] = None

    # Causal-specific importance
    causal_importance: Optional[Dict[str, float]] = None  # {"feature_name": score}
    causal_importance_ranked: Optional[List[Tuple[str, float]]] = None  # Sorted by causal imp

    # Direct causes (features with direct edge to target)
    direct_cause_features: Optional[List[str]] = None  # Features that are direct causes of target

    # Causal interpretation (from NL node)
    causal_interpretation: Optional[str] = None  # NL explanation
