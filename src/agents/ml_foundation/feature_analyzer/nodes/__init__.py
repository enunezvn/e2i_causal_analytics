"""Nodes for feature_analyzer agent.

Hybrid pipeline with 3 nodes:
1. SHAP Computation (NO LLM) - shap_computer.py
2. Interaction Detection (NO LLM) - interaction_detector.py
3. NL Interpretation (LLM) - importance_narrator.py
"""

from .shap_computer import compute_shap
from .interaction_detector import detect_interactions
from .importance_narrator import narrate_importance

__all__ = [
    "compute_shap",
    "detect_interactions",
    "narrate_importance",
]
