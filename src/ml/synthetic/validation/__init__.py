"""
E2I Synthetic Data Validation

Schema validation with Pandera and data quality with Great Expectations.

Usage:
    from src.ml.synthetic.validation import quick_validate, validate_and_log

    # Quick validation (no observability)
    is_valid = quick_validate(datasets, dgp_type=DGPType.CONFOUNDED)

    # Full validation with MLflow logging
    is_valid, run_id = validate_and_log(
        datasets,
        dgp_type=DGPType.CONFOUNDED,
        experiment_name="my_experiment",
    )
"""

from .expectations import (
    GX_AVAILABLE,
    ValidationResult,
    get_checkpoint_summary,
    get_expectation_suite,
    run_validation_checkpoint,
)
from .observability import (
    ValidationObserver,
    create_validation_span,
    get_observability_status,
    log_validation_to_mlflow,
)
from .pipeline import (
    PipelineValidationResult,
    get_combined_summary,
    quick_validate,
    validate_and_log,
    validate_dataset,
    validate_pipeline_output,
)
from .schemas import (
    SCHEMA_REGISTRY,
    HCPProfileSchema,
    MLPredictionSchema,
    PatientJourneySchema,
    TreatmentEventSchema,
    TriggerSchema,
    get_validation_summary,
    validate_all_datasets,
    validate_dataframe,
)

__all__ = [
    # Pandera schemas
    "HCPProfileSchema",
    "PatientJourneySchema",
    "TreatmentEventSchema",
    "MLPredictionSchema",
    "TriggerSchema",
    "SCHEMA_REGISTRY",
    "validate_dataframe",
    "validate_all_datasets",
    "get_validation_summary",
    # Great Expectations
    "get_expectation_suite",
    "run_validation_checkpoint",
    "get_checkpoint_summary",
    "ValidationResult",
    "GX_AVAILABLE",
    # Observability
    "log_validation_to_mlflow",
    "create_validation_span",
    "ValidationObserver",
    "get_observability_status",
    # Pipeline integration
    "PipelineValidationResult",
    "validate_dataset",
    "validate_pipeline_output",
    "get_combined_summary",
    "quick_validate",
    "validate_and_log",
]
