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
"""

from .base import BaseGenerator, GeneratorConfig, GenerationResult
from .hcp_generator import HCPGenerator
from .patient_generator import PatientGenerator
from .treatment_generator import TreatmentGenerator
from .engagement_generator import EngagementGenerator
from .outcome_generator import OutcomeGenerator
from .prediction_generator import PredictionGenerator
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
]
