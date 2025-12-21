"""
E2I Causal Analytics - MLOps Layer
===================================

ML Operations components for model serving, monitoring, and interpretability.

Components:
-----------
- shap_explainer_realtime.py: Real-Time SHAP computation engine
- (future) bentoml_service.py: BentoML model serving
- (future) feast_client.py: Feast feature store client

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

from .opik_connector import (
    LLMSpanContext,
    OpikConfig,
    OpikConnector,
    SpanContext,
)
from .shap_explainer_realtime import (
    ExplainerType,
    RealTimeSHAPExplainer,
    SHAPResult,
    SHAPVisualization,
)

__all__ = [
    # SHAP Explainer
    "RealTimeSHAPExplainer",
    "SHAPResult",
    "ExplainerType",
    "SHAPVisualization",
    # Opik Connector
    "OpikConnector",
    "OpikConfig",
    "SpanContext",
    "LLMSpanContext",
]

__version__ = "4.1.0"
