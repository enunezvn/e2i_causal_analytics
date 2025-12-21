"""
E2I Causal Analytics - MLOps Layer
===================================

ML Operations components for model serving, monitoring, and interpretability.

Components:
-----------
- opik_connector.py: Opik SDK wrapper with circuit breaker (CLOSED → OPEN → HALF_OPEN)
- shap_explainer_realtime.py: Real-Time SHAP computation engine
- (future) bentoml_service.py: BentoML model serving
- (future) feast_client.py: Feast feature store client

Author: E2I Causal Analytics Team
Version: 4.3.0 (Phase 3 - Circuit Breaker Complete)
"""

from .opik_connector import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    LLMSpanContext,
    OpikConfig,
    OpikConnector,
    SpanContext,
    get_opik_connector,
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
    "get_opik_connector",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitState",
]

__version__ = "4.3.0"
