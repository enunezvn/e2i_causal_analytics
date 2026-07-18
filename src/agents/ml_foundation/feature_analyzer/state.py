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
from uuid import UUID

import numpy as np

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
    # from BaseAgentSchema) covers in-memory construction. JSON
    # serialization is intentionally NOT supported: pydantic v2 has no
    # default np.ndarray serializer, so ``model_dump_json()`` raises
    # ``PydanticSerializationError`` if shap_values is non-None.
    #
    # Sub-shard D5 status: CLOSED — won't-fix (2026-05-05). Persistence
    # for SHAP values fans out into 5 dedicated stores (Postgres
    # ``ml_shap_analyses``, MLflow artifacts, working memory cache,
    # episodic memory, semantic memory) — each storing exactly the
    # aggregated slice that downstream consumers need. The raw ndarray
    # is correctly modeled as transient in-process state and is GC'd
    # at run-end. Zero production callers JSON-serialize this field
    # (verified: ``feature_analyzer/graph.py`` calls ``workflow.compile()``
    # without a checkpointer at lines 96, 131, 170; no ml_foundation
    # agent wires RedisSaver). Implementing a serializer would enable
    # silent multi-MB JSON checkpoints (~10 MB at tier-0 max SHAP shape)
    # — a storage-bloat foot-gun worse than today's loud raise.
    #
    # Re-evaluate D5 if and only if any of these change:
    #   (a) a ``checkpointer=`` arg is added to any of the three
    #       ``workflow.compile()`` calls in ``feature_analyzer/graph.py``;
    #   (b) a new caller invokes ``state.model_dump_json()`` on
    #       FeatureAnalyzerState;
    #   (c) a "resumable feature analysis" or "agent-to-agent SHAP handoff
    #       via channel state" feature appears in the backlog;
    #   (d) a consumer needs SHAP values via JSON instead of the existing
    #       typed Postgres/MLflow stores.
    #
    # The unit test ``test_feature_analyzer_state_shap_values_json_dump_raises``
    # pins this constraint loud (per codex review I4, 2026-05-05). If you
    # are about to delete that test, see sub-shard D5 first.
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
    interpretation_model: Optional[str] = None  # "claude-sonnet-4-6"
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
    # F8: provenance of SHAP importances — "real" | "synthetic" | "unavailable".
    # Declared so the graph state propagates it from compute_shap to the agent output
    # (an undeclared key is dropped on the Pydantic state merge).
    data_provenance: Optional[str] = None

    # === ERROR HANDLING ===

    error: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # === AUDIT CHAIN ===
    # Required: caller MUST provide ``audit_workflow_id`` (backlog #1
    # tightening landed 2026-05-09; PRs #58 / #62 / #65 thread the field).
    audit_workflow_id: UUID

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
