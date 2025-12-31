"""State definition for feature_analyzer agent.

This agent is a HYBRID agent with 5 nodes:
1. Feature Generation (NO LLM) - Generate engineered features
2. Feature Selection (NO LLM) - Select optimal features
3. SHAP Computation (NO LLM) - Compute SHAP values
4. Interaction Detection (NO LLM) - Detect feature interactions
5. NL Interpretation (LLM) - Generate human-readable explanations
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import UUID

import numpy as np
import pandas as pd


class FeatureAnalyzerState(TypedDict, total=False):
    """State for feature_analyzer agent.

    Follows hybrid execution pattern:
    - Feature engineering nodes (1-2): Generate and select features
    - SHAP analysis nodes (3-4): Compute importance and interactions
    - LLM node (5): Add interpretation fields
    """

    # === INPUT FIELDS (from data_preparer) ===

    X_train: Any  # Training features (DataFrame or ndarray)
    X_val: Any  # Validation features (optional)
    X_test: Any  # Test features (optional)
    y_train: Any  # Training target
    y_val: Any  # Validation target (optional)
    y_test: Any  # Test target (optional)
    problem_type: str  # "classification" | "regression"

    # === INPUT FIELDS (for SHAP - from model_trainer) ===

    model_uri: str  # MLflow model URI (e.g., "runs:/abc123/model")
    experiment_id: str  # Experiment identifier
    training_run_id: str  # Training run identifier

    # === FEATURE GENERATION CONFIGURATION ===

    feature_config: Dict[str, Any]  # Feature generation configuration
    temporal_columns: List[str]  # Columns for temporal features
    categorical_columns: List[str]  # Columns for interaction features
    numeric_columns: List[str]  # Numeric columns

    # === FEATURE SELECTION CONFIGURATION ===

    selection_config: Dict[str, Any]  # Feature selection configuration

    # === NODE 1 OUTPUT: Feature Generation (NO LLM) ===

    X_train_generated: Any  # DataFrame with generated features
    X_val_generated: Any  # Validation with generated features
    X_test_generated: Any  # Test with generated features
    generated_features: List[Dict[str, Any]]  # Metadata for generated features
    feature_metadata: Dict[str, List[Dict[str, Any]]]  # By type: temporal, interaction, domain
    original_feature_count: int  # Features before generation
    new_feature_count: int  # New features added
    new_feature_names: List[str]  # Names of new features
    feature_generation_time_seconds: float
    temporal_columns_used: List[str]
    categorical_columns_used: List[str]
    numeric_columns_used: List[str]

    # === NODE 2 OUTPUT: Feature Selection (NO LLM) ===

    X_train_selected: Any  # DataFrame with selected features
    X_val_selected: Any  # Validation with selected features
    X_test_selected: Any  # Test with selected features
    selected_features: List[str]  # List of selected numeric feature names
    selected_features_all: List[str]  # All selected features (including non-numeric)
    feature_importance: Dict[str, float]  # Importance scores from selection
    feature_importance_ranked: List[Tuple[str, float]]  # Ranked features
    removed_features: Dict[str, List[str]]  # By method: variance, correlation, vif
    selection_history: List[Dict[str, Any]]  # Step-by-step selection log
    feature_statistics: Dict[str, Dict[str, Any]]  # Statistics per feature
    selected_feature_count: int  # Features after selection
    total_selected_count: int  # Total including non-numeric
    selection_time_seconds: float

    # === SHAP CONFIGURATION ===

    max_samples: int  # Max samples for SHAP (default: 1000)
    compute_interactions: bool  # Whether to compute interactions (default: True)
    store_in_semantic_memory: bool  # Whether to store in semantic memory (default: True)

    # === NODE 3 OUTPUT: SHAP Computation (NO LLM) ===

    # Loaded model
    loaded_model: Any  # sklearn/xgboost/etc model object
    feature_names: List[str]  # Feature names from model

    # Training data sample (for SHAP computation)
    X_sample: Any  # pandas DataFrame or numpy array
    y_sample: Any  # pandas Series or numpy array
    samples_analyzed: int  # Number of samples used

    # SHAP values (raw)
    shap_values: np.ndarray  # SHAP values array (n_samples, n_features)
    base_value: float  # Base value (expected value)

    # Global importance
    global_importance: Dict[str, float]  # {"feature_name": importance_score}
    global_importance_ranked: List[Tuple[str, float]]  # Sorted by importance

    # Directional effects
    feature_directions: Dict[str, str]  # {"feature_name": "positive"|"negative"|"mixed"}

    # Top features
    top_features: List[str]  # Top 5 features by importance

    # Computation metadata
    shap_computation_time_seconds: float
    explainer_type: str  # "TreeExplainer" | "KernelExplainer" | "LinearExplainer"

    # === NODE 2 OUTPUT: Interaction Detection (NO LLM) ===

    # Interaction matrix
    interaction_matrix: Dict[str, Dict[str, float]]  # {"feat1": {"feat2": strength}}

    # Top interactions
    top_interactions_raw: List[Tuple[str, str, float]]  # [(feat1, feat2, strength), ...]

    # Interaction computation metadata
    interaction_computation_time_seconds: float
    interaction_method: str  # "correlation" | "shap_interaction"

    # === NODE 3 OUTPUT: NL Interpretation (LLM) ===

    # Executive summary
    executive_summary: str  # High-level summary for stakeholders

    # Feature explanations
    feature_explanations: Dict[str, str]  # {"feature": "natural language explanation"}

    # Interaction interpretations
    interaction_interpretations: List[Dict[str, Any]]  # List of interpreted interactions

    # Key insights
    key_insights: List[str]  # Bullet points of key findings

    # Actionable recommendations
    recommendations: List[str]  # Actionable next steps

    # Cautions
    cautions: List[str]  # Warnings about model behavior

    # LLM metadata
    interpretation_model: str  # "claude-sonnet-4-20250514"
    interpretation_time_seconds: float
    interpretation_tokens: int

    # === SEMANTIC MEMORY ===

    semantic_memory_updated: bool  # Whether semantic memory was updated
    semantic_memory_entries: int  # Number of entries added

    # === OUTPUT FIELDS (Final) ===

    # SHAP Analysis ID
    shap_analysis_id: str  # Unique identifier for this analysis

    # Structured outputs for contracts
    feature_importance_list: List[Dict[str, Any]]  # List[FeatureImportance]
    interaction_list: List[Dict[str, Any]]  # List[FeatureInteraction]

    # Natural language interpretation (final)
    interpretation: str  # Complete NL summary combining executive_summary + insights

    # Model version
    model_version: str  # Model version from MLflow

    # Total computation time
    total_computation_time_seconds: float

    # Status
    status: str  # "completed" | "failed" | "in_progress"

    # === ERROR HANDLING ===

    error: Optional[str]  # Error message if failed
    error_type: Optional[str]  # Error classification
    error_details: Optional[Dict[str, Any]]  # Additional error context

    # === AUDIT CHAIN ===
    audit_workflow_id: UUID

    # === DISCOVERY INTEGRATION (V4.4) ===
    # Configuration for causal discovery integration

    discovery_enabled: bool  # Enable causal discovery (default: False)
    discovery_config: Dict[str, Any]  # DiscoveryConfig as dict
    # Keys: algorithms, alpha, min_votes, max_cond_vars, ensemble_method, etc.

    # Discovery results from DiscoveryRunner
    discovery_result: Dict[str, Any]  # DiscoveryResult as dict
    # Keys: ensemble_dag, algorithm_results, edges, n_edges, metadata, success

    # Gate evaluation from DiscoveryGate
    discovery_gate_decision: str  # "accept" | "review" | "reject" | "augment"
    discovery_gate_confidence: float  # Overall confidence [0, 1]
    discovery_gate_reasons: List[str]  # Reasons for decision

    # === CAUSAL RANKING (V4.4) ===
    # Comparison of causal vs predictive feature importance

    # Target variable for causal analysis
    causal_target_variable: str  # Target for causal path analysis

    # Rankings from DriverRanker
    causal_rankings: List[Dict[str, Any]]  # List of FeatureRanking dicts
    # Each dict: feature_name, causal_rank, predictive_rank, causal_score,
    #            predictive_score, rank_difference, is_direct_cause, path_length

    # Rank correlation
    rank_correlation: float  # Spearman correlation between causal & predictive ranks

    # Feature categorization
    divergent_features: List[str]  # Features with |rank_difference| > threshold
    causal_only_features: List[str]  # Features with causal but no predictive signal
    predictive_only_features: List[str]  # Features with predictive but no causal signal
    concordant_features: List[str]  # Features with similar causal & predictive ranks

    # Causal-specific importance
    causal_importance: Dict[str, float]  # {"feature_name": causal_importance_score}
    causal_importance_ranked: List[Tuple[str, float]]  # Sorted by causal importance

    # Direct causes (features with direct edge to target)
    direct_cause_features: List[str]  # Features that are direct causes of target

    # Causal interpretation (from NL node)
    causal_interpretation: str  # NL explanation of causal vs predictive comparison
