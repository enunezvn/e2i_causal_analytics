"""
E2I Causal Analytics - MLOps Layer
===================================

ML Operations components for model serving, monitoring, and interpretability.

Components:
-----------
- opik_connector.py: Opik SDK wrapper with circuit breaker (CLOSED → OPEN → HALF_OPEN)
- shap_explainer_realtime.py: Real-Time SHAP computation engine
- data_quality.py: Great Expectations data quality validation
- pandera_schemas.py: Pandera schema validation for E2I data sources
- (future) bentoml_service.py: BentoML model serving
- (future) feast_client.py: Feast feature store client

Author: E2I Causal Analytics Team
Version: 4.5.0 (Pandera Schema Validation Integration)
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
from .pandera_schemas import (
    # Schema classes
    AgentActivitiesSchema,
    BusinessMetricsSchema,
    CausalPathsSchema,
    PatientJourneysSchema,
    PredictionsSchema,
    TriggersSchema,
    # Registry and utilities
    PANDERA_SCHEMA_REGISTRY,
    get_schema,
    list_registered_schemas,
    validate_dataframe,
    # E2I constants
    E2I_BRANDS,
    E2I_REGIONS,
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
    # Pandera Schema Validation
    "BusinessMetricsSchema",
    "PredictionsSchema",
    "TriggersSchema",
    "PatientJourneysSchema",
    "CausalPathsSchema",
    "AgentActivitiesSchema",
    "PANDERA_SCHEMA_REGISTRY",
    "get_schema",
    "validate_dataframe",
    "list_registered_schemas",
    "E2I_BRANDS",
    "E2I_REGIONS",
]

__version__ = "4.5.0"
