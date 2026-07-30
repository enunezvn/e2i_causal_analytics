"""
E2I Synthetic Data Generators

Entity generators for synthetic data:
- HCPGenerator: Generate HCP profiles
- PatientGenerator: Generate patient journeys
- TreatmentGenerator: Generate treatment events
- EngagementGenerator: Generate engagement events
- OutcomeGenerator: Generate business outcomes
- PredictionGenerator: Generate ML predictions
- TriggerGenerator: Generate triggers
- BusinessMetricsGenerator: Generate business metrics time-series
- FeatureStoreSeeder: Seed feature groups and features
- FeatureValueGenerator: Generate feature values time-series
"""

from .agent_activities_generator import AgentActivitiesGenerator
from .base import BaseGenerator, GenerationResult, GeneratorConfig
from .business_metrics_generator import BusinessMetricsGenerator
from .causal_paths_generator import CausalPathsGenerator
from .change_tracking import stamp_change_tracking
from .coverage_tables_generator import CoverageTablesGenerator
from .data_lag import stamp_data_lag_hours, stamp_sequence_number
from .engagement_generator import EngagementGenerator
from .experiment_generator import ABExperimentGenerator, ExperimentGenerator
from .feature_store_seeder import FeatureStoreSeeder
from .feature_value_generator import FeatureValueGenerator
from .feedback_generator import FeedbackGenerator
from .hcp_generator import HCPGenerator
from .mlops_generator import MLOpsGenerator
from .model_metrics import stamp_model_metrics
from .observability_generator import ObservabilityGenerator
from .outcome_generator import OutcomeGenerator
from .patient_generator import PatientGenerator
from .prediction_generator import PredictionGenerator
from .treatment_generator import TreatmentGenerator
from .trigger_generator import TriggerGenerator

__all__ = [
    # Base classes
    "BaseGenerator",
    "GeneratorConfig",
    "GenerationResult",
    # Entity generators
    "HCPGenerator",
    "PatientGenerator",
    "TreatmentGenerator",
    "EngagementGenerator",
    "OutcomeGenerator",
    "PredictionGenerator",
    "TriggerGenerator",
    # Business & Feature generators
    "BusinessMetricsGenerator",
    "FeatureStoreSeeder",
    "FeatureValueGenerator",
    # Shard 09 breadth substrate generators
    "ExperimentGenerator",
    "ABExperimentGenerator",
    "MLOpsGenerator",
    "ObservabilityGenerator",
    "FeedbackGenerator",
    "CoverageTablesGenerator",
    "CausalPathsGenerator",
    "AgentActivitiesGenerator",
    "stamp_data_lag_hours",
    "stamp_sequence_number",
    "stamp_model_metrics",
    "stamp_change_tracking",
]
