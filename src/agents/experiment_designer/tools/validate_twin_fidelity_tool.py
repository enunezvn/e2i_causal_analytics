"""
Validate Twin Fidelity Tool
============================

LangGraph tool for validating digital twin predictions against
actual experiment outcomes. This closes the loop on the twin
simulation → real experiment → validation cycle.

Usage:
    After a real A/B test completes, use this tool to record
    the actual results and update the twin model's fidelity score.
"""

import logging
from typing import Optional, Dict, Any, Annotated, List
from uuid import UUID
from datetime import datetime, timezone

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.digital_twin.fidelity_tracker import FidelityTracker
from src.digital_twin.twin_repository import TwinRepository
from src.digital_twin.models.simulation_models import FidelityGrade

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Input/Output Schemas
# =============================================================================

class ValidateFidelityInput(BaseModel):
    """Input schema for validate_twin_fidelity tool."""
    
    simulation_id: str = Field(
        description="UUID of the original twin simulation"
    )
    actual_ate: float = Field(
        description="Actual Average Treatment Effect from real experiment"
    )
    actual_ci_lower: Optional[float] = Field(
        default=None,
        description="Lower bound of actual confidence interval"
    )
    actual_ci_upper: Optional[float] = Field(
        default=None,
        description="Upper bound of actual confidence interval"
    )
    actual_sample_size: Optional[int] = Field(
        default=None,
        description="Sample size of actual experiment"
    )
    actual_experiment_id: Optional[str] = Field(
        default=None,
        description="ID of the actual experiment in ml_experiments table"
    )
    validation_notes: Optional[str] = Field(
        default=None,
        description="Optional notes about the validation"
    )
    confounding_factors: Optional[List[str]] = Field(
        default=None,
        description="List of potential confounding factors that may have affected results"
    )


class ValidateFidelityOutput(BaseModel):
    """Output schema for validate_twin_fidelity tool."""
    
    tracking_id: str
    simulation_id: str
    
    # Comparison
    simulated_ate: float
    actual_ate: float
    
    # Fidelity metrics
    prediction_error: float
    absolute_error: float
    ci_coverage: Optional[bool]
    fidelity_grade: str
    
    # Assessment
    assessment_message: str
    model_update_recommended: bool


# =============================================================================
# Tool Implementation
# =============================================================================

# Module-level tracker instance
_fidelity_tracker: Optional[FidelityTracker] = None


def _get_tracker() -> FidelityTracker:
    """Get or create fidelity tracker instance."""
    global _fidelity_tracker
    if _fidelity_tracker is None:
        _fidelity_tracker = FidelityTracker()
    return _fidelity_tracker


@tool
def validate_twin_fidelity(
    simulation_id: Annotated[str, "UUID of the original twin simulation"],
    actual_ate: Annotated[float, "Actual ATE from real experiment"],
    actual_ci_lower: Annotated[Optional[float], "Lower CI bound"] = None,
    actual_ci_upper: Annotated[Optional[float], "Upper CI bound"] = None,
    actual_sample_size: Annotated[Optional[int], "Sample size of real experiment"] = None,
    actual_experiment_id: Annotated[Optional[str], "ID of real experiment"] = None,
    validation_notes: Annotated[Optional[str], "Notes about validation"] = None,
    confounding_factors: Annotated[Optional[List[str]], "Confounding factors"] = None,
) -> Dict[str, Any]:
    """
    Validate a twin simulation prediction against actual experiment results.
    
    This tool should be called AFTER a real A/B test completes to:
    1. Record how well the twin simulation predicted the actual outcome
    2. Update the twin model's fidelity score
    3. Detect if model retraining is needed
    
    The validation process:
    - Calculates prediction error (simulated - actual) / actual
    - Checks if actual result fell within simulated CI
    - Assigns a fidelity grade (excellent, good, fair, poor)
    - Triggers alerts if model accuracy is degrading
    
    Args:
        simulation_id: UUID of the original simulation to validate
        actual_ate: The actual Average Treatment Effect measured
        actual_ci_lower: Lower bound of actual 95% CI (optional)
        actual_ci_upper: Upper bound of actual 95% CI (optional)
        actual_sample_size: Sample size used in real experiment
        actual_experiment_id: Reference to ml_experiments table
        validation_notes: Free-form notes about the validation
        confounding_factors: List of factors that may have affected results
            (e.g., "competitor launch", "COVID impact", "seasonal variation")
    
    Returns:
        Dictionary containing:
        - tracking_id: ID of the fidelity tracking record
        - prediction_error: Percentage error in prediction
        - absolute_error: Absolute difference |simulated - actual|
        - ci_coverage: Whether actual fell within predicted CI
        - fidelity_grade: "excellent", "good", "fair", or "poor"
        - assessment_message: Human-readable assessment
        - model_update_recommended: Whether to retrain model
    
    Example:
        >>> # After A/B test completes with ATE of 0.082
        >>> result = validate_twin_fidelity(
        ...     simulation_id="abc-123-def",
        ...     actual_ate=0.082,
        ...     actual_ci_lower=0.065,
        ...     actual_ci_upper=0.099,
        ...     actual_sample_size=2500,
        ...     validation_notes="Test completed successfully"
        ... )
        >>> print(f"Prediction error: {result['prediction_error']:.1%}")
        >>> print(f"Fidelity grade: {result['fidelity_grade']}")
    """
    logger.info(
        f"validate_twin_fidelity called: simulation={simulation_id}, "
        f"actual_ate={actual_ate}"
    )
    
    try:
        tracker = _get_tracker()
        
        # Parse simulation ID
        try:
            sim_uuid = UUID(simulation_id)
        except ValueError:
            return _create_error_response(
                f"Invalid simulation_id format: {simulation_id}"
            )
        
        # Parse experiment ID if provided
        exp_uuid = None
        if actual_experiment_id:
            try:
                exp_uuid = UUID(actual_experiment_id)
            except ValueError:
                logger.warning(f"Invalid experiment_id format: {actual_experiment_id}")
        
        # Validate
        try:
            record = tracker.validate(
                simulation_id=sim_uuid,
                actual_ate=actual_ate,
                actual_ci=(actual_ci_lower, actual_ci_upper) if actual_ci_lower else None,
                actual_sample_size=actual_sample_size,
                actual_experiment_id=exp_uuid,
                notes=validation_notes,
                confounding_factors=confounding_factors or [],
            )
        except ValueError as e:
            # Simulation not found - create new record with provided info
            logger.warning(f"Simulation not found, creating new record: {e}")
            return _create_new_validation(
                simulation_id=simulation_id,
                actual_ate=actual_ate,
                actual_ci_lower=actual_ci_lower,
                actual_ci_upper=actual_ci_upper,
                validation_notes=validation_notes,
            )
        
        # Generate assessment
        assessment = _generate_assessment(record)
        
        return {
            "tracking_id": str(record.tracking_id),
            "simulation_id": str(record.simulation_id),
            "simulated_ate": round(record.simulated_ate, 4),
            "actual_ate": round(record.actual_ate, 4),
            "prediction_error": round(record.prediction_error, 4) if record.prediction_error else 0,
            "absolute_error": round(record.absolute_error, 4) if record.absolute_error else 0,
            "ci_coverage": record.ci_coverage,
            "fidelity_grade": record.fidelity_grade.value,
            "assessment_message": assessment["message"],
            "model_update_recommended": assessment["update_recommended"],
        }
    
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return _create_error_response(str(e))


def _generate_assessment(record) -> Dict[str, Any]:
    """Generate human-readable assessment from fidelity record."""
    grade = record.fidelity_grade
    error = abs(record.prediction_error) if record.prediction_error else 0
    
    if grade == FidelityGrade.EXCELLENT:
        message = (
            f"Excellent prediction! The simulation predicted {record.simulated_ate:.4f} "
            f"and actual was {record.actual_ate:.4f} (error: {error:.1%}). "
            "Twin model is performing very well."
        )
        update_recommended = False
    
    elif grade == FidelityGrade.GOOD:
        message = (
            f"Good prediction. Simulated: {record.simulated_ate:.4f}, "
            f"Actual: {record.actual_ate:.4f} (error: {error:.1%}). "
            "Twin model is performing adequately."
        )
        update_recommended = False
    
    elif grade == FidelityGrade.FAIR:
        message = (
            f"Fair prediction with moderate error. Simulated: {record.simulated_ate:.4f}, "
            f"Actual: {record.actual_ate:.4f} (error: {error:.1%}). "
            "Consider reviewing model features or retraining."
        )
        update_recommended = True
    
    else:  # POOR
        message = (
            f"Poor prediction with significant error. Simulated: {record.simulated_ate:.4f}, "
            f"Actual: {record.actual_ate:.4f} (error: {error:.1%}). "
            "Model retraining strongly recommended."
        )
        update_recommended = True
    
    # Check CI coverage
    if record.ci_coverage is not None:
        if record.ci_coverage:
            message += " Actual result fell within predicted confidence interval."
        else:
            message += " WARNING: Actual result fell outside predicted CI."
            update_recommended = True
    
    # Note confounding factors
    if record.confounding_factors:
        factors = ", ".join(record.confounding_factors)
        message += f" Note: Confounding factors reported: {factors}."
    
    return {
        "message": message,
        "update_recommended": update_recommended,
    }


def _create_error_response(error_message: str) -> Dict[str, Any]:
    """Create error response."""
    return {
        "tracking_id": "error",
        "simulation_id": "unknown",
        "simulated_ate": 0.0,
        "actual_ate": 0.0,
        "prediction_error": 0.0,
        "absolute_error": 0.0,
        "ci_coverage": None,
        "fidelity_grade": FidelityGrade.UNVALIDATED.value,
        "assessment_message": f"Validation failed: {error_message}",
        "model_update_recommended": False,
    }


def _create_new_validation(
    simulation_id: str,
    actual_ate: float,
    actual_ci_lower: Optional[float],
    actual_ci_upper: Optional[float],
    validation_notes: Optional[str],
) -> Dict[str, Any]:
    """Create validation for simulation not in tracker (manual entry)."""
    from uuid import uuid4
    
    # Create a minimal record for manual validation
    # This handles cases where simulation wasn't tracked
    return {
        "tracking_id": str(uuid4()),
        "simulation_id": simulation_id,
        "simulated_ate": 0.0,  # Unknown
        "actual_ate": round(actual_ate, 4),
        "prediction_error": 0.0,
        "absolute_error": 0.0,
        "ci_coverage": None,
        "fidelity_grade": FidelityGrade.UNVALIDATED.value,
        "assessment_message": (
            f"Recorded actual results (ATE={actual_ate:.4f}) but original "
            "simulation not found. Cannot calculate prediction error."
        ),
        "model_update_recommended": False,
    }


# =============================================================================
# Additional Utility Functions
# =============================================================================

@tool
def get_model_fidelity_report(
    model_id: Annotated[str, "UUID of the twin model"],
    lookback_days: Annotated[int, "Days to look back for validations"] = 90,
) -> Dict[str, Any]:
    """
    Get fidelity report for a digital twin model.
    
    This provides an aggregate view of how well a twin model has been
    predicting actual experiment outcomes over time.
    
    Args:
        model_id: UUID of the twin generator model
        lookback_days: Number of days of history to include
    
    Returns:
        Report containing:
        - validation_count: Number of validated predictions
        - mean_absolute_error: Average prediction error
        - ci_coverage_rate: How often actuals fell within predicted CI
        - fidelity_score: Overall model fidelity (0-1)
        - grade_distribution: Count by fidelity grade
        - degradation_alert: Whether model is degrading
    """
    logger.info(f"get_model_fidelity_report called: model={model_id}")
    
    try:
        tracker = _get_tracker()
        model_uuid = UUID(model_id)
        
        report = tracker.get_model_fidelity_report(
            model_id=model_uuid,
            lookback_days=lookback_days,
        )
        
        return {
            "model_id": str(model_uuid),
            "lookback_days": lookback_days,
            "validation_count": report.get("validation_count", 0),
            "metrics": report.get("metrics", {}),
            "fidelity_score": report.get("fidelity_score", 0.5),
            "grade_distribution": report.get("grade_distribution", {}),
            "degradation_alert": report.get("degradation_alert", False),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Failed to get fidelity report: {e}", exc_info=True)
        return {
            "model_id": model_id,
            "error": str(e),
            "validation_count": 0,
        }
