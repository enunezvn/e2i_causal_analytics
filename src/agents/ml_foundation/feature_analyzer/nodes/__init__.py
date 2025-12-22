"""Nodes for feature_analyzer agent.

Extended hybrid pipeline with 5 nodes:
1. Feature Generation (NO LLM) - feature_generator.py
2. Feature Selection (NO LLM) - feature_selector.py
3. SHAP Computation (NO LLM) - shap_computer.py
4. Interaction Detection (NO LLM) - interaction_detector.py
5. NL Interpretation (LLM) - importance_narrator.py

Additional utilities:
- Feature Visualization - feature_visualizer.py (charts, tables)
"""

from .feature_generator import generate_features
from .feature_selector import get_feature_selection_summary, select_features
from .feature_visualizer import generate_visualizations
from .importance_narrator import narrate_importance
from .interaction_detector import detect_interactions
from .shap_computer import compute_shap

__all__ = [
    # Feature engineering
    "generate_features",
    "select_features",
    "get_feature_selection_summary",
    # Visualization
    "generate_visualizations",
    # SHAP analysis
    "compute_shap",
    "detect_interactions",
    "narrate_importance",
]
