"""State definition for feature_analyzer agent.

This agent is a HYBRID agent with 3 nodes:
1. SHAP Computation (NO LLM) - Compute SHAP values
2. Interaction Detection (NO LLM) - Detect feature interactions
3. NL Interpretation (LLM) - Generate human-readable explanations
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np


class FeatureAnalyzerState(TypedDict, total=False):
    """State for feature_analyzer agent.

    Follows hybrid execution pattern:
    - Computation nodes (1-2): Populate SHAP-related fields
    - LLM node (3): Add interpretation fields
    """

    # === INPUT FIELDS (from model_trainer) ===

    model_uri: str  # MLflow model URI (e.g., "runs:/abc123/model")
    experiment_id: str  # Experiment identifier
    training_run_id: str  # Training run identifier

    # === SHAP CONFIGURATION ===

    max_samples: int  # Max samples for SHAP (default: 1000)
    compute_interactions: bool  # Whether to compute interactions (default: True)
    store_in_semantic_memory: bool  # Whether to store in semantic memory (default: True)

    # === NODE 1 OUTPUT: SHAP Computation (NO LLM) ===

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
