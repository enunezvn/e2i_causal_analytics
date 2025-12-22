"""Feature Analyzer Agent - HYBRID.

Performs feature engineering, selection, and SHAP-based interpretability analysis.

Tier: 0 (ML Foundation)
Type: Hybrid (Computation + LLM)
SLA: <120s

Pipeline (5 nodes):
1. Feature Generation (NO LLM) - Create temporal, interaction, domain features
2. Feature Selection (NO LLM) - Variance, correlation, model-based selection
3. SHAP Computation (NO LLM) - Compute SHAP values (optional, requires model)
4. Interaction Detection (NO LLM) - Detect feature interactions
5. NL Interpretation (LLM) - Generate human-readable explanations

Inputs (Feature Engineering):
- X_train, X_val, X_test: Feature DataFrames
- y_train: Target variable
- problem_type: "classification" | "regression"
- feature_config: Configuration for feature generation
- selection_config: Configuration for feature selection

Inputs (SHAP Analysis):
- model_uri: MLflow model URI
- experiment_id: Experiment identifier
- max_samples: Max samples for SHAP (default 1000)

Outputs:
- X_train_selected, X_val_selected, X_test_selected: Selected features
- selected_features: List of selected feature names
- feature_importance: Feature importance scores
- shap_analysis: SHAP values and interactions (if model provided)
- interpretation: Natural language summary

Integration:
- Upstream: data_preparer, model_trainer
- Downstream: model_selector, model_deployer, explainer
- Feature Store: FeatureStoreClient via FeatureAnalyzerAdapter
- Database: ml_shap_analyses, ml_feature_store
"""

from .agent import FeatureAnalyzerAgent
from .graph import (
    create_feature_analyzer_graph,
    create_feature_engineering_graph,
    create_shap_analysis_graph,
)
from .nodes import (
    compute_shap,
    detect_interactions,
    generate_features,
    generate_visualizations,
    get_feature_selection_summary,
    narrate_importance,
    select_features,
)
from .state import FeatureAnalyzerState

__all__ = [
    # Agent
    "FeatureAnalyzerAgent",
    # Graphs
    "create_feature_analyzer_graph",
    "create_feature_engineering_graph",
    "create_shap_analysis_graph",
    # State
    "FeatureAnalyzerState",
    # Nodes
    "generate_features",
    "select_features",
    "get_feature_selection_summary",
    "generate_visualizations",
    "compute_shap",
    "detect_interactions",
    "narrate_importance",
]
