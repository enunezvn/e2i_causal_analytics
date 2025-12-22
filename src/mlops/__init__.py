"""
E2I Causal Analytics - MLOps Layer
===================================

ML Operations components for model serving, monitoring, and interpretability.

Components:
-----------
- opik_connector.py: Opik SDK wrapper with circuit breaker (CLOSED → OPEN → HALF_OPEN)
- shap_explainer_realtime.py: Real-Time SHAP computation engine
- data_quality.py: Great Expectations data quality validation
- (future) bentoml_service.py: BentoML model serving
- (future) feast_client.py: Feast feature store client

Author: E2I Causal Analytics Team
Version: 4.4.0 (Phase 3 - Great Expectations Integration)
"""

from .data_quality import (
    AlertConfig,
    AlertHandler,
    AlertSeverity,
    DataQualityAlerter,
    DataQualityCheckpointError,
    DataQualityResult,
    DataQualityValidator,
    ExpectationSuiteBuilder,
    LogAlertHandler,
    WebhookAlertHandler,
    configure_alerter,
    get_data_quality_validator,
    get_default_alerter,
)
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
    reset_opik_connector,
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
    "reset_opik_connector",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitState",
    # Data Quality (Great Expectations)
    "DataQualityValidator",
    "DataQualityResult",
    "DataQualityCheckpointError",
    "ExpectationSuiteBuilder",
    "get_data_quality_validator",
    # Alerting
    "AlertSeverity",
    "AlertConfig",
    "AlertHandler",
    "LogAlertHandler",
    "WebhookAlertHandler",
    "DataQualityAlerter",
    "get_default_alerter",
    "configure_alerter",
]

__version__ = "4.4.0"
