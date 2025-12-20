"""Feature Analyzer Agent - HYBRID.

Analyzes trained models using SHAP for interpretability.

Tier: 0 (ML Foundation)
Type: Hybrid (Computation + LLM)
SLA: <120s

Pipeline:
1. SHAP Computation (NO LLM) - Compute SHAP values
2. Interaction Detection (NO LLM) - Detect feature interactions
3. NL Interpretation (LLM) - Generate human-readable explanations

Inputs:
- model_uri: MLflow model URI
- experiment_id: Experiment identifier
- max_samples: Max samples for SHAP (default 1000)
- compute_interactions: Whether to compute interactions (default True)
- store_in_semantic_memory: Whether to store in semantic memory (default True)

Outputs:
- shap_analysis: SHAPAnalysis with global importance and interactions
- interpretation: Natural language summary
- semantic_memory_updated: Whether semantic memory was updated
- top_features: Top 5 features by importance
- top_interactions: Top 3 feature interactions

Integration:
- Upstream: model_trainer
- Downstream: model_deployer, explainer, causal_impact
- Database: ml_shap_analyses
- Memory: Semantic memory (feature relationships)
"""

from .agent import FeatureAnalyzerAgent

__all__ = ["FeatureAnalyzerAgent"]
