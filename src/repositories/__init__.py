"""
Data Access Layer for E2I Causal Analytics.

Provides repository pattern implementations for all database tables.
All repositories are split-aware to prevent ML data leakage.

Core Repositories:
- PatientJourneyRepository: Patient journey data with source tracking
- PredictionRepository: ML predictions with rank metrics
- TriggerRepository: HCP triggers with change tracking
- BusinessMetricRepository: KPI snapshots
- CausalPathRepository: Discovered causal relationships
- AgentActivityRepository: Agent analysis outputs
- ConversationRepository: Chat history for RAG

V3 Repositories:
- UserSessionRepository: MAU/WAU/DAU tracking
- AgentRegistryRepository: 11 agents with tier assignments
- DataSourceTrackingRepository: Cross-source match rates

V4.3 Repositories:
- CausalValidationRepository: Refutation test results and gate decisions
- ExpertReviewRepository: Domain expert DAG validation and approval workflow

ML Data Loading (Phase 1):
- MLDataLoader: Load data from Supabase for ML pipelines
- DataSplitter: Train/val/test splitting with multiple strategies
- DataCache: Redis-based caching for repeated experiments
- SampleDataGenerator: Generate realistic test data

Phase 3 (Great Expectations):
- DataQualityReportRepository: Store/retrieve DQ validation results
"""

# Re-export get_supabase_client for convenience
from src.memory.services.factories import get_supabase_client
from src.repositories.agent_activity import AgentActivityRepository
from src.repositories.agent_registry import AgentRegistryRepository
from src.repositories.base import BaseRepository, SplitAwareRepository

# BentoML Service Tracking (Migration 024)
from src.repositories.bentoml_service import (
    BentoMLMetricsRepository,
    BentoMLService,
    BentoMLServiceRepository,
    BentoMLServingMetrics,
    ServiceHealthStatus,
    ServiceStatus,
)
from src.repositories.business_metric import BusinessMetricRepository
from src.repositories.causal_path import CausalPathRepository
from src.repositories.causal_validation import CausalValidationRepository
from src.repositories.chatbot_analytics import (
    ChatbotAnalyticsRepository,
    get_chatbot_analytics_repository,
)
from src.repositories.chatbot_conversation import (
    ChatbotConversationRepository,
    get_chatbot_conversation_repository,
)
from src.repositories.chatbot_feedback import (
    ChatbotFeedbackRepository,
    get_chatbot_feedback_repository,
)
from src.repositories.chatbot_message import (
    ChatbotMessageRepository,
    get_chatbot_message_repository,
)

# Chatbot Repositories (Migration 028-031)
from src.repositories.chatbot_user_profile import (
    ChatbotUserProfileRepository,
    get_chatbot_user_profile_repository,
)
from src.repositories.conversation import ConversationRepository
from src.repositories.data_cache import CacheConfig, DataCache, get_data_cache

# Data Quality (Phase 3)
from src.repositories.data_quality_report import (
    DataQualityReportRepository,
    get_data_quality_report_repository,
)
from src.repositories.data_splitter import (
    DataSplitter,
    SplitConfig,
    SplitResult,
    get_data_splitter,
)

# ML Deployment (Phase 10)
from src.repositories.deployment import (
    DeploymentEnvironment,
    DeploymentStatus,
    MLDeployment,
    MLDeploymentRepository,
)

# Drift Monitoring (Phase 14)
from src.repositories.drift_monitoring import (
    DriftHistoryRecord,
    DriftHistoryRepository,
    MonitoringAlertRecord,
    MonitoringAlertRepository,
    MonitoringRunRecord,
    MonitoringRunRepository,
    PerformanceMetricRecord,
    PerformanceMetricRepository,
    RetrainingHistoryRecord,
    RetrainingHistoryRepository,
)
from src.repositories.expert_review import ExpertReviewRepository

# Feast Feature Store Tracking (Migration 025)
from src.repositories.feast_tracking import (
    FeastFeatureFreshness,
    FeastFeatureView,
    FeastFeatureViewRepository,
    FeastFreshnessRepository,
    FeastMaterializationJob,
    FeastMaterializationRepository,
    FreshnessStatus,
    MaterializationJobType,
    MaterializationStatus,
    SourceType,
)

# ML Data Loading (Phase 1)
from src.repositories.ml_data_loader import MLDataLoader, MLDataset, get_ml_data_loader

# ML Experiment Tracking (Phase 5)
from src.repositories.ml_experiment import (
    MLExperiment,
    MLExperimentRepository,
    MLModelRegistry,
    MLModelRegistryRepository,
    MLTrainingRun,
    MLTrainingRunRepository,
    ModelStage,
    TrainingStatus,
)
from src.repositories.observability_span import ObservabilitySpanRepository
from src.repositories.patient_journey import PatientJourneyRepository
from src.repositories.prediction import PredictionRepository

# Query Logging (G13 - Observability)
from src.repositories.query_logger import (
    QueryLogger,
    QueryMetrics,
    SlowQueryConfig,
    SlowQueryDetector,
    SlowQueryRecord,
    configure_slow_query_thresholds,
    get_query_stats,
    logged_query,
    logged_query_async,
    query_logger,
    query_metrics,
    slow_query_detector,
)
from src.repositories.sample_data import SampleDataGenerator, get_sample_generator

# SHAP Analysis (Tier 0 - feature_analyzer)
from src.repositories.shap_analysis import (
    ShapAnalysisRepository,
    get_shap_analysis_repository,
)
from src.repositories.trigger import TriggerRepository
from src.repositories.user_session import UserSessionRepository

__all__ = [
    # Base classes
    "BaseRepository",
    "SplitAwareRepository",
    # Core repositories
    "PatientJourneyRepository",
    "PredictionRepository",
    "TriggerRepository",
    "BusinessMetricRepository",
    "CausalPathRepository",
    "AgentActivityRepository",
    "ConversationRepository",
    "AgentRegistryRepository",
    "UserSessionRepository",
    "CausalValidationRepository",
    "ExpertReviewRepository",
    "ObservabilitySpanRepository",
    # ML Data Loading
    "MLDataLoader",
    "MLDataset",
    "get_ml_data_loader",
    "DataSplitter",
    "SplitConfig",
    "SplitResult",
    "get_data_splitter",
    "DataCache",
    "CacheConfig",
    "get_data_cache",
    "SampleDataGenerator",
    "get_sample_generator",
    # Data Quality (Phase 3)
    "DataQualityReportRepository",
    "get_data_quality_report_repository",
    # SHAP Analysis (Tier 0 - feature_analyzer)
    "ShapAnalysisRepository",
    "get_shap_analysis_repository",
    # ML Experiment Tracking (Phase 5)
    "MLExperiment",
    "MLExperimentRepository",
    "MLModelRegistry",
    "MLModelRegistryRepository",
    "MLTrainingRun",
    "MLTrainingRunRepository",
    "ModelStage",
    "TrainingStatus",
    # ML Deployment (Phase 10)
    "DeploymentEnvironment",
    "DeploymentStatus",
    "MLDeployment",
    "MLDeploymentRepository",
    # Drift Monitoring (Phase 14)
    "DriftHistoryRecord",
    "DriftHistoryRepository",
    "MonitoringAlertRecord",
    "MonitoringAlertRepository",
    "MonitoringRunRecord",
    "MonitoringRunRepository",
    "PerformanceMetricRecord",
    "PerformanceMetricRepository",
    "RetrainingHistoryRecord",
    "RetrainingHistoryRepository",
    # BentoML Service Tracking (Migration 024)
    "BentoMLMetricsRepository",
    "BentoMLService",
    "BentoMLServiceRepository",
    "BentoMLServingMetrics",
    "ServiceHealthStatus",
    "ServiceStatus",
    # Feast Feature Store Tracking (Migration 025)
    "FeastFeatureFreshness",
    "FeastFeatureView",
    "FeastFeatureViewRepository",
    "FeastFreshnessRepository",
    "FeastMaterializationJob",
    "FeastMaterializationRepository",
    "FreshnessStatus",
    "MaterializationJobType",
    "MaterializationStatus",
    "SourceType",
    # Chatbot Repositories (Migration 028-033)
    "ChatbotUserProfileRepository",
    "get_chatbot_user_profile_repository",
    "ChatbotConversationRepository",
    "get_chatbot_conversation_repository",
    "ChatbotMessageRepository",
    "get_chatbot_message_repository",
    "ChatbotFeedbackRepository",
    "get_chatbot_feedback_repository",
    "ChatbotAnalyticsRepository",
    "get_chatbot_analytics_repository",
    # Utilities
    "get_supabase_client",
    # Query Logging (G13 - Observability)
    "QueryLogger",
    "QueryMetrics",
    "SlowQueryConfig",
    "SlowQueryDetector",
    "SlowQueryRecord",
    "configure_slow_query_thresholds",
    "get_query_stats",
    "logged_query",
    "logged_query_async",
    "query_logger",
    "query_metrics",
    "slow_query_detector",
]
