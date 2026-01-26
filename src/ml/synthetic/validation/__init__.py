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

from .schemas import (
    HCPProfileSchema,
    PatientJourneySchema,
    TreatmentEventSchema,
    MLPredictionSchema,
    TriggerSchema,
    SCHEMA_REGISTRY,
    validate_dataframe,
    validate_all_datasets,
    get_validation_summary,
)

from .expectations import (
    get_expectation_suite,
    run_validation_checkpoint,
    get_checkpoint_summary,
    ValidationResult,
    GX_AVAILABLE,
)

from .observability import (
    log_validation_to_mlflow,
    create_validation_span,
    ValidationObserver,
    get_observability_status,
)

from .pipeline import (
    PipelineValidationResult,
    validate_dataset,
    validate_pipeline_output,
    get_combined_summary,
    quick_validate,
    validate_and_log,
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
