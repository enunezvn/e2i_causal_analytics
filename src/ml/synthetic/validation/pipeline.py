"""
Validation Pipeline Integration.

Integrates Pandera schemas and Great Expectations into the data generation pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..config import DGPType
from .expectations import (
    GX_AVAILABLE,
    ValidationResult,
    validate_dataframe_with_expectations,
)
from .observability import (
    ValidationObserver,
    create_validation_span,
    get_observability_status,
)
from .schemas import (
    SCHEMA_REGISTRY,
    validate_dataframe,
)


@dataclass
class PipelineValidationResult:
    """Combined validation result from both Pandera and Great Expectations."""

    table_name: str
    pandera_valid: bool
    pandera_errors: Optional[Any] = None
    gx_result: Optional[ValidationResult] = None
    validation_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Overall validation status."""
        gx_valid = self.gx_result.success if self.gx_result else True
        return self.pandera_valid and gx_valid

    @property
    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "table_name": self.table_name,
            "pandera_valid": self.pandera_valid,
            "gx_valid": self.gx_result.success if self.gx_result else None,
            "gx_success_rate": self.gx_result.success_rate if self.gx_result else None,
            "overall_valid": self.is_valid,
            "timestamp": self.validation_timestamp.isoformat(),
        }


def validate_dataset(
    df: pd.DataFrame,
    table_name: str,
    dgp_type: Optional[DGPType] = None,
    run_pandera: bool = True,
    run_gx: bool = True,
) -> PipelineValidationResult:
    """
    Validate a single dataset with both Pandera and Great Expectations.

    Args:
        df: DataFrame to validate
        table_name: Name of the table (must be in SCHEMA_REGISTRY)
        dgp_type: Optional DGP type for causal validation
        run_pandera: Whether to run Pandera schema validation
        run_gx: Whether to run Great Expectations validation

    Returns:
        PipelineValidationResult with combined results
    """
    pandera_valid = True
    pandera_errors = None
    gx_result = None

    # Run Pandera validation
    if run_pandera and table_name in SCHEMA_REGISTRY:
        pandera_valid, pandera_errors = validate_dataframe(df, table_name, lazy=True)

    # Run Great Expectations validation
    if run_gx and GX_AVAILABLE:
        gx_result = validate_dataframe_with_expectations(df, table_name, dgp_type)

    return PipelineValidationResult(
        table_name=table_name,
        pandera_valid=pandera_valid,
        pandera_errors=pandera_errors,
        gx_result=gx_result,
    )


def validate_pipeline_output(
    datasets: Dict[str, pd.DataFrame],
    dgp_type: Optional[DGPType] = None,
    run_pandera: bool = True,
    run_gx: bool = True,
    enable_observability: bool = True,
    experiment_name: str = "synthetic_data_validation",
    tags: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, PipelineValidationResult], Dict[str, Any]]:
    """
    Validate all pipeline outputs with full observability.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        dgp_type: Optional DGP type for causal validation
        run_pandera: Whether to run Pandera schema validation
        run_gx: Whether to run Great Expectations validation
        enable_observability: Whether to log to MLflow/Opik
        experiment_name: MLflow experiment name
        tags: Optional tags for observability

    Returns:
        Tuple of (validation_results, observability_summary)
    """
    results: Dict[str, PipelineValidationResult] = {}
    observer = None

    if enable_observability:
        observer = ValidationObserver(
            experiment_name=experiment_name,
            tags=tags,
        )

    # Validate each dataset
    for table_name, df in datasets.items():
        with create_validation_span(
            operation_name=f"validate_{table_name}",
            table_name=table_name,
            metadata={"dgp_type": dgp_type.value if dgp_type else None},
        ) as span:
            result = validate_dataset(
                df=df,
                table_name=table_name,
                dgp_type=dgp_type,
                run_pandera=run_pandera,
                run_gx=run_gx,
            )
            results[table_name] = result

            # Log to span
            if result.gx_result:
                span.log_result(result.gx_result)

            # Record for MLflow
            if observer and result.gx_result:
                observer.record_result(table_name, result.gx_result)

    # Finalize observability
    obs_summary = {}
    if observer:
        obs_summary = observer.finalize(
            run_name=f"pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    return results, obs_summary


def get_combined_summary(
    results: Dict[str, PipelineValidationResult],
    obs_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a combined validation summary.

    Args:
        results: Output from validate_pipeline_output
        obs_summary: Optional observability summary

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "SYNTHETIC DATA VALIDATION SUMMARY",
        "=" * 70,
        "",
        "PANDERA SCHEMA VALIDATION",
        "-" * 35,
    ]

    all_pandera_valid = True
    for table_name, result in results.items():
        status = "PASS" if result.pandera_valid else "FAIL"
        all_pandera_valid = all_pandera_valid and result.pandera_valid
        lines.append(f"  {table_name}: {status}")
        if not result.pandera_valid and result.pandera_errors:
            error_count = len(result.pandera_errors.failure_cases)
            lines.append(f"    ({error_count} schema errors)")

    lines.extend(
        [
            "",
            "GREAT EXPECTATIONS VALIDATION",
            "-" * 35,
        ]
    )

    if not GX_AVAILABLE:
        lines.append("  (Great Expectations not installed)")
    else:
        all_gx_valid = True
        for table_name, result in results.items():
            if result.gx_result:
                status = "PASS" if result.gx_result.success else "FAIL"
                all_gx_valid = all_gx_valid and result.gx_result.success
                rate = result.gx_result.success_rate * 100
                lines.append(f"  {table_name}: {status} ({rate:.1f}%)")
            else:
                lines.append(f"  {table_name}: SKIPPED")

    lines.extend(
        [
            "",
            "OVERALL RESULTS",
            "-" * 35,
        ]
    )

    all_valid = all(r.is_valid for r in results.values())
    overall_status = "ALL PASSED" if all_valid else "VALIDATION FAILED"
    lines.append(f"  Status: {overall_status}")
    lines.append(f"  Tables Validated: {len(results)}")
    lines.append(
        f"  Pandera Valid: {sum(1 for r in results.values() if r.pandera_valid)}/{len(results)}"
    )

    if GX_AVAILABLE:
        gx_valid_count = sum(1 for r in results.values() if r.gx_result and r.gx_result.success)
        lines.append(f"  GX Valid: {gx_valid_count}/{len(results)}")

    if obs_summary:
        lines.extend(
            [
                "",
                "OBSERVABILITY",
                "-" * 35,
            ]
        )
        if obs_summary.get("mlflow_run_id"):
            lines.append(f"  MLflow Run ID: {obs_summary['mlflow_run_id']}")
        obs_status = get_observability_status()
        lines.append(f"  MLflow Available: {obs_status['mlflow_available']}")
        lines.append(f"  Opik Available: {obs_status['opik_available']}")

    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_validate(
    datasets: Dict[str, pd.DataFrame],
    dgp_type: Optional[DGPType] = None,
    verbose: bool = True,
) -> bool:
    """
    Quick validation without observability.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        dgp_type: Optional DGP type
        verbose: Whether to print summary

    Returns:
        True if all validations pass
    """
    results, _ = validate_pipeline_output(
        datasets=datasets,
        dgp_type=dgp_type,
        enable_observability=False,
    )

    if verbose:
        summary = get_combined_summary(results)
        print(summary)

    return all(r.is_valid for r in results.values())


def validate_and_log(
    datasets: Dict[str, pd.DataFrame],
    dgp_type: Optional[DGPType] = None,
    experiment_name: str = "synthetic_data_validation",
    tags: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate and log results to MLflow.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        dgp_type: Optional DGP type
        experiment_name: MLflow experiment name
        tags: Optional tags
        verbose: Whether to print summary

    Returns:
        Tuple of (all_valid, mlflow_run_id)
    """
    results, obs_summary = validate_pipeline_output(
        datasets=datasets,
        dgp_type=dgp_type,
        enable_observability=True,
        experiment_name=experiment_name,
        tags=tags,
    )

    if verbose:
        summary = get_combined_summary(results, obs_summary)
        print(summary)

    all_valid = all(r.is_valid for r in results.values())
    mlflow_run_id = obs_summary.get("mlflow_run_id")

    return all_valid, mlflow_run_id
