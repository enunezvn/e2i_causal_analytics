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

from .base import BaseGenerator, GeneratorConfig, GenerationResult
from .hcp_generator import HCPGenerator
from .patient_generator import PatientGenerator
from .treatment_generator import TreatmentGenerator
from .engagement_generator import EngagementGenerator
from .outcome_generator import OutcomeGenerator
from .prediction_generator import PredictionGenerator
from .trigger_generator import TriggerGenerator
from .business_metrics_generator import BusinessMetricsGenerator
from .feature_store_seeder import FeatureStoreSeeder
from .feature_value_generator import FeatureValueGenerator

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
]
