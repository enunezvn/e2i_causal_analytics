"""
E2I Causal Analytics - MLOps Layer
===================================

ML Operations components for model serving, monitoring, and interpretability.

Components:
-----------
- opik_connector.py: Opik SDK wrapper with circuit breaker (CLOSED → OPEN → HALF_OPEN)
- opik_feedback.py: Opik feedback loop integration (Phase 4 - G23)
- business_context.py: Business context labels for observability (Phase 4 - G24)
- agent_cost_tracker.py: Per-agent LLM cost tracking (Phase 4 - G25)
- slo_monitor.py: SLO monitoring for agent tiers (Phase 4 - G26)
- shap_explainer_realtime.py: Real-Time SHAP computation engine
- data_quality.py: Great Expectations data quality validation
- pandera_schemas.py: Pandera schema validation for E2I data sources
- (future) bentoml_service.py: BentoML model serving
- (future) feast_client.py: Feast feature store client

Author: E2I Causal Analytics Team
Version: 4.9.0 (SLO Monitoring - G26)
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
from .opik_feedback import (
    AgentFeedbackStats,
    FeedbackRecord,
    FeedbackSignal,
    OpikFeedbackCollector,
    get_feedback_collector,
    get_feedback_signals_for_gepa,
    log_user_feedback,
)
from .business_context import (
    # Enums
    E2IBrand,
    E2IRegion,
    E2ISegmentType,
    # Constants
    VALID_BRANDS,
    VALID_REGIONS,
    # Classes
    BusinessContext,
    BusinessContextModel,
    # Extraction
    get_context_from_request,
    get_context_from_dataframe,
    merge_contexts,
    # Propagation
    apply_context_to_span,
    apply_context_to_mlflow,
    context_to_labels,
    # Response
    enrich_response_with_context,
)
from .agent_cost_tracker import (
    # Classes
    AgentCostTracker,
    CostRecord,
    AgentCostSummary,
    ModelPricing,
    LLMProvider,
    # Pricing
    MODEL_PRICING,
    calculate_cost,
    # Singleton
    get_cost_tracker,
    reset_cost_tracker,
    # Convenience
    record_agent_cost,
    get_agent_cost_summary,
    get_total_cost_summary,
)
from .slo_monitor import (
    # Enums
    AgentTier,
    # Classes
    SLOTarget,
    RequestRecord,
    SLOCompliance,
    SLOMonitor,
    # Configuration
    DEFAULT_SLO_TARGETS,
    AGENT_TIER_MAP,
    get_agent_tier,
    get_slo_target,
    # Singleton
    get_slo_monitor,
    reset_slo_monitor,
    # Convenience
    record_request,
    get_slo_compliance,
    get_all_slo_compliance,
    get_slo_summary,
    get_violated_slos,
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
    # Opik Feedback Loop (Phase 4 - G23)
    "OpikFeedbackCollector",
    "FeedbackRecord",
    "AgentFeedbackStats",
    "FeedbackSignal",
    "get_feedback_collector",
    "log_user_feedback",
    "get_feedback_signals_for_gepa",
    # Business Context (Phase 4 - G24)
    "E2IBrand",
    "E2IRegion",
    "E2ISegmentType",
    "VALID_BRANDS",
    "VALID_REGIONS",
    "BusinessContext",
    "BusinessContextModel",
    "get_context_from_request",
    "get_context_from_dataframe",
    "merge_contexts",
    "apply_context_to_span",
    "apply_context_to_mlflow",
    "context_to_labels",
    "enrich_response_with_context",
    # Agent Cost Tracking (Phase 4 - G25)
    "AgentCostTracker",
    "CostRecord",
    "AgentCostSummary",
    "ModelPricing",
    "LLMProvider",
    "MODEL_PRICING",
    "calculate_cost",
    "get_cost_tracker",
    "reset_cost_tracker",
    "record_agent_cost",
    "get_agent_cost_summary",
    "get_total_cost_summary",
    # SLO Monitoring (Phase 4 - G26)
    "AgentTier",
    "SLOTarget",
    "RequestRecord",
    "SLOCompliance",
    "SLOMonitor",
    "DEFAULT_SLO_TARGETS",
    "AGENT_TIER_MAP",
    "get_agent_tier",
    "get_slo_target",
    "get_slo_monitor",
    "reset_slo_monitor",
    "record_request",
    "get_slo_compliance",
    "get_all_slo_compliance",
    "get_slo_summary",
    "get_violated_slos",
]

__version__ = "4.9.0"
