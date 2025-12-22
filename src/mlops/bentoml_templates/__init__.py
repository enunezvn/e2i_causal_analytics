"""BentoML Service Templates for E2I Causal Analytics.

This module provides pre-built service templates for common model types:
- ClassificationService: Binary and multiclass classification
- RegressionService: Continuous value prediction
- CausalInferenceService: CATE and treatment effect estimation

Usage:
    from src.mlops.bentoml_templates import ClassificationServiceTemplate

    service = ClassificationServiceTemplate.create(
        model_tag="churn_model:latest",
        service_name="churn_prediction_service",
    )
"""

from .classification_service import ClassificationServiceTemplate
from .regression_service import RegressionServiceTemplate
from .causal_service import CausalInferenceServiceTemplate

__all__ = [
    "ClassificationServiceTemplate",
    "RegressionServiceTemplate",
    "CausalInferenceServiceTemplate",
]
