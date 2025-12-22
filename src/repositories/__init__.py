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

from src.repositories.agent_activity import AgentActivityRepository
from src.repositories.agent_registry import AgentRegistryRepository
from src.repositories.base import BaseRepository, SplitAwareRepository
from src.repositories.business_metric import BusinessMetricRepository
from src.repositories.causal_path import CausalPathRepository
from src.repositories.causal_validation import CausalValidationRepository
from src.repositories.conversation import ConversationRepository
from src.repositories.expert_review import ExpertReviewRepository
from src.repositories.observability_span import ObservabilitySpanRepository
from src.repositories.patient_journey import PatientJourneyRepository
from src.repositories.prediction import PredictionRepository
from src.repositories.trigger import TriggerRepository
from src.repositories.user_session import UserSessionRepository

# ML Data Loading (Phase 1)
from src.repositories.ml_data_loader import MLDataLoader, MLDataset, get_ml_data_loader
from src.repositories.data_splitter import (
    DataSplitter,
    SplitConfig,
    SplitResult,
    get_data_splitter,
)
from src.repositories.data_cache import DataCache, CacheConfig, get_data_cache
from src.repositories.sample_data import SampleDataGenerator, get_sample_generator

# Data Quality (Phase 3)
from src.repositories.data_quality_report import (
    DataQualityReportRepository,
    get_data_quality_report_repository,
)

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

# ML Deployment (Phase 10)
from src.repositories.deployment import (
    DeploymentEnvironment,
    DeploymentStatus,
    MLDeployment,
    MLDeploymentRepository,
)

# Re-export get_supabase_client for convenience
from src.memory.services.factories import get_supabase_client

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
    # Utilities
    "get_supabase_client",
]
