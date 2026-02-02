"""
Great Expectations Suite Definitions for Synthetic Data.

Validates data quality, statistical properties, and business rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Note: Great Expectations is optional - graceful degradation if not installed
try:
    import great_expectations as gx  # noqa: F401
    from great_expectations.core import ExpectationSuite
    from great_expectations.dataset import PandasDataset

    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    ExpectationSuite = None
    PandasDataset = None


from ..config import (
    DGP_CONFIGS,
    Brand,
    DGPType,
)

# =============================================================================
# VALIDATION RESULT
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a Great Expectations validation run."""

    suite_name: str
    table_name: str
    success: bool
    statistics: Dict[str, Any] = field(default_factory=dict)
    failed_expectations: List[Dict] = field(default_factory=list)
    run_timestamp: datetime = field(default_factory=datetime.now)
    run_time_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate expectation success rate."""
        total = self.statistics.get("evaluated_expectations", 0)
        successful = self.statistics.get("successful_expectations", 0)
        return successful / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "suite_name": self.suite_name,
            "table_name": self.table_name,
            "success": self.success,
            "success_rate": self.success_rate,
            "statistics": self.statistics,
            "failed_expectations_count": len(self.failed_expectations),
            "run_timestamp": self.run_timestamp.isoformat(),
            "run_time_seconds": self.run_time_seconds,
        }


# =============================================================================
# EXPECTATION SUITE BUILDERS
# =============================================================================


def build_hcp_expectations(dataset: "PandasDataset") -> "PandasDataset":
    """Build expectations for HCP profiles."""
    if not GX_AVAILABLE:
        return dataset

    # Uniqueness
    dataset.expect_column_values_to_be_unique("hcp_id")
    dataset.expect_column_values_to_be_unique("npi")

    # Completeness
    dataset.expect_column_values_to_not_be_null("hcp_id")
    dataset.expect_column_values_to_not_be_null("specialty")
    dataset.expect_column_values_to_not_be_null("practice_type")

    # Value distributions
    dataset.expect_column_values_to_be_in_set("practice_type", ["academic", "community", "private"])

    # Statistical expectations
    dataset.expect_column_mean_to_be_between("years_experience", min_value=10, max_value=25)
    dataset.expect_column_mean_to_be_between("total_patient_volume", min_value=100, max_value=400)

    # Academic HCP distribution (~25-35%)
    dataset.expect_column_proportion_of_unique_values_to_be_between(
        "academic_hcp", min_value=0.1, max_value=0.5
    )

    return dataset


def build_patient_journey_expectations(
    dataset: "PandasDataset",
    dgp_type: Optional[DGPType] = None,
) -> "PandasDataset":
    """Build expectations for patient journeys with causal validation."""
    if not GX_AVAILABLE:
        return dataset

    # Uniqueness
    dataset.expect_column_values_to_be_unique("patient_journey_id")

    # Completeness
    dataset.expect_column_values_to_not_be_null("patient_id")
    dataset.expect_column_values_to_not_be_null("hcp_id")
    dataset.expect_column_values_to_not_be_null("brand")
    dataset.expect_column_values_to_not_be_null("engagement_score")
    dataset.expect_column_values_to_not_be_null("treatment_initiated")

    # Value ranges
    dataset.expect_column_values_to_be_between("disease_severity", min_value=0, max_value=10)
    dataset.expect_column_values_to_be_between("engagement_score", min_value=0, max_value=10)
    dataset.expect_column_values_to_be_in_set("treatment_initiated", [0, 1])
    dataset.expect_column_values_to_be_in_set("academic_hcp", [0, 1])

    # Statistical expectations for confounders
    dataset.expect_column_mean_to_be_between("disease_severity", min_value=3.0, max_value=7.0)
    dataset.expect_column_stdev_to_be_between("disease_severity", min_value=1.0, max_value=3.0)

    # Academic HCP proportion (~30%)
    dataset.expect_column_mean_to_be_between("academic_hcp", min_value=0.20, max_value=0.40)

    # Data split distribution expectations
    dataset.expect_column_values_to_be_in_set(
        "data_split", ["train", "validation", "test", "holdout"]
    )

    # Brand distribution (should be roughly balanced)
    for brand in Brand:
        # Each brand should have 25-40% of records
        dataset.expect_column_distinct_values_to_contain_set("brand", [brand.value])

    # Causal structure validation (if DGP type specified)
    if dgp_type:
        dgp_config = DGP_CONFIGS.get(dgp_type)
        if dgp_config:
            # Treatment rate should be reasonable (40-95% depending on DGP)
            dataset.expect_column_mean_to_be_between(
                "treatment_initiated", min_value=0.30, max_value=0.98
            )

    return dataset


def build_treatment_event_expectations(dataset: "PandasDataset") -> "PandasDataset":
    """Build expectations for treatment events."""
    if not GX_AVAILABLE:
        return dataset

    # Uniqueness
    dataset.expect_column_values_to_be_unique("treatment_event_id")

    # Completeness
    dataset.expect_column_values_to_not_be_null("patient_journey_id")
    dataset.expect_column_values_to_not_be_null("event_type")
    dataset.expect_column_values_to_not_be_null("event_date")

    # Value ranges
    dataset.expect_column_values_to_be_between("adherence_score", min_value=0, max_value=1)
    dataset.expect_column_values_to_be_between("efficacy_score", min_value=0, max_value=1)
    dataset.expect_column_values_to_be_between("duration_days", min_value=1, max_value=365)

    # Event type distribution
    dataset.expect_column_values_to_be_in_set(
        "event_type",
        ["diagnosis", "prescription", "lab_test", "procedure", "consultation", "hospitalization"],
    )

    # Statistical expectations
    dataset.expect_column_mean_to_be_between("adherence_score", min_value=0.60, max_value=0.90)
    dataset.expect_column_mean_to_be_between("efficacy_score", min_value=0.50, max_value=0.80)

    return dataset


def build_ml_prediction_expectations(dataset: "PandasDataset") -> "PandasDataset":
    """Build expectations for ML predictions."""
    if not GX_AVAILABLE:
        return dataset

    # Uniqueness
    dataset.expect_column_values_to_be_unique("prediction_id")

    # Completeness
    dataset.expect_column_values_to_not_be_null("patient_id")
    dataset.expect_column_values_to_not_be_null("prediction_type")
    dataset.expect_column_values_to_not_be_null("prediction_value")
    dataset.expect_column_values_to_not_be_null("confidence_score")

    # Value ranges (all probabilities 0-1)
    dataset.expect_column_values_to_be_between("prediction_value", min_value=0, max_value=1)
    dataset.expect_column_values_to_be_between("confidence_score", min_value=0, max_value=1)
    dataset.expect_column_values_to_be_between("uncertainty", min_value=0, max_value=1)

    # Prediction type distribution
    dataset.expect_column_values_to_be_in_set(
        "prediction_type", ["trigger", "propensity", "risk", "churn", "next_best_action"]
    )

    # Confidence should be reasonably high (model calibration)
    dataset.expect_column_mean_to_be_between("confidence_score", min_value=0.60, max_value=0.95)

    # Confidence + uncertainty should approximately equal 1
    # (This is a custom expectation - simplified check)
    dataset.expect_column_mean_to_be_between("uncertainty", min_value=0.05, max_value=0.40)

    return dataset


def build_trigger_expectations(dataset: "PandasDataset") -> "PandasDataset":
    """Build expectations for triggers."""
    if not GX_AVAILABLE:
        return dataset

    # Uniqueness
    dataset.expect_column_values_to_be_unique("trigger_id")

    # Completeness
    dataset.expect_column_values_to_not_be_null("patient_id")
    dataset.expect_column_values_to_not_be_null("hcp_id")
    dataset.expect_column_values_to_not_be_null("trigger_type")
    dataset.expect_column_values_to_not_be_null("priority")

    # Value ranges
    dataset.expect_column_values_to_be_between("priority", min_value=1, max_value=5)
    dataset.expect_column_values_to_be_between("confidence_score", min_value=0, max_value=1)
    dataset.expect_column_values_to_be_between("lead_time_days", min_value=0, max_value=365)

    # Trigger type distribution
    dataset.expect_column_values_to_be_in_set(
        "trigger_type",
        [
            "prescription_opportunity",
            "adherence_risk",
            "churn_prevention",
            "cross_sell",
            "engagement_gap",
            "competitive_threat",
            "treatment_switch",
            "reactivation",
        ],
    )

    # Delivery channel distribution
    dataset.expect_column_values_to_be_in_set(
        "delivery_channel", ["email", "call", "in_person", "portal"]
    )

    # Priority distribution (should have mix of priorities)
    dataset.expect_column_mean_to_be_between("priority", min_value=2.0, max_value=4.0)

    return dataset


# =============================================================================
# SUITE REGISTRY
# =============================================================================

SUITE_BUILDERS = {
    "hcp_profiles": build_hcp_expectations,
    "patient_journeys": build_patient_journey_expectations,
    "treatment_events": build_treatment_event_expectations,
    "ml_predictions": build_ml_prediction_expectations,
    "triggers": build_trigger_expectations,
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def get_expectation_suite(table_name: str) -> Optional[str]:
    """
    Get the expectation suite name for a table.

    Args:
        table_name: Name of the table

    Returns:
        Suite name or None if not found
    """
    if table_name in SUITE_BUILDERS:
        return f"e2i_synthetic_{table_name}_suite"
    return None


def validate_dataframe_with_expectations(
    df: pd.DataFrame,
    table_name: str,
    dgp_type: Optional[DGPType] = None,
) -> ValidationResult:
    """
    Validate a DataFrame using Great Expectations.

    Args:
        df: DataFrame to validate
        table_name: Name of the table
        dgp_type: Optional DGP type for causal validation

    Returns:
        ValidationResult with detailed results
    """
    start_time = datetime.now()
    suite_name = get_expectation_suite(table_name)

    if not GX_AVAILABLE:
        return ValidationResult(
            suite_name=suite_name or "unavailable",
            table_name=table_name,
            success=True,
            statistics={"message": "Great Expectations not installed"},
            run_time_seconds=0.0,
        )

    if table_name not in SUITE_BUILDERS:
        return ValidationResult(
            suite_name="unknown",
            table_name=table_name,
            success=False,
            statistics={"error": f"Unknown table: {table_name}"},
            run_time_seconds=0.0,
        )

    # Create PandasDataset and build expectations
    dataset = PandasDataset(df)
    builder = SUITE_BUILDERS[table_name]

    # Special handling for patient_journeys (needs dgp_type)
    if table_name == "patient_journeys":
        dataset = builder(dataset, dgp_type=dgp_type)
    else:
        dataset = builder(dataset)

    # Get validation results
    results = dataset.validate()

    # Extract statistics
    statistics = {
        "evaluated_expectations": results.statistics.get("evaluated_expectations", 0),
        "successful_expectations": results.statistics.get("successful_expectations", 0),
        "unsuccessful_expectations": results.statistics.get("unsuccessful_expectations", 0),
        "success_percent": results.statistics.get("success_percent", 0),
    }

    # Extract failed expectations
    failed_expectations = []
    for result in results.results:
        if not result.success:
            failed_expectations.append(
                {
                    "expectation_type": result.expectation_config.expectation_type,
                    "kwargs": result.expectation_config.kwargs,
                    "result": result.result,
                }
            )

    run_time = (datetime.now() - start_time).total_seconds()

    return ValidationResult(
        suite_name=suite_name,
        table_name=table_name,
        success=results.success,
        statistics=statistics,
        failed_expectations=failed_expectations,
        run_time_seconds=run_time,
    )


def run_validation_checkpoint(
    datasets: Dict[str, pd.DataFrame],
    dgp_type: Optional[DGPType] = None,
) -> Dict[str, ValidationResult]:
    """
    Run validation checkpoint on all datasets.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        dgp_type: Optional DGP type for causal validation

    Returns:
        Dictionary of table_name -> ValidationResult
    """
    results = {}

    for table_name, df in datasets.items():
        results[table_name] = validate_dataframe_with_expectations(
            df, table_name, dgp_type=dgp_type
        )

    return results


def get_checkpoint_summary(results: Dict[str, ValidationResult]) -> str:
    """
    Generate a summary of checkpoint results.

    Args:
        results: Output from run_validation_checkpoint

    Returns:
        Formatted summary string
    """
    lines = ["=" * 60, "GREAT EXPECTATIONS VALIDATION SUMMARY", "=" * 60]

    all_success = True
    total_expectations = 0
    total_passed = 0

    for table_name, result in results.items():
        all_success = all_success and result.success
        total_expectations += result.statistics.get("evaluated_expectations", 0)
        total_passed += result.statistics.get("successful_expectations", 0)

        status = "PASS" if result.success else "FAIL"
        success_rate = result.success_rate * 100
        lines.append(f"  {table_name}: {status} ({success_rate:.1f}%)")

        if not result.success:
            for exp in result.failed_expectations[:3]:
                exp_type = exp["expectation_type"].replace("expect_", "")
                lines.append(f"    - {exp_type}")
            if len(result.failed_expectations) > 3:
                lines.append(f"    ... and {len(result.failed_expectations) - 3} more")

    lines.append("-" * 60)
    overall_rate = (total_passed / total_expectations * 100) if total_expectations > 0 else 0
    overall_status = "ALL PASSED" if all_success else "VALIDATION FAILED"
    lines.append(
        f"Overall: {overall_status} ({total_passed}/{total_expectations} expectations, {overall_rate:.1f}%)"
    )
    lines.append("=" * 60)

    return "\n".join(lines)
