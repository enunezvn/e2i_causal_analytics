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

# Re-export get_supabase_client for convenience
from src.memory.services.factories import get_supabase_client

__all__ = [
    "BaseRepository",
    "SplitAwareRepository",
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
    "get_supabase_client",
]
