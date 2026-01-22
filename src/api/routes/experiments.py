"""
E2I A/B Testing & Experiment Execution API
===========================================

FastAPI endpoints for A/B testing execution, monitoring, and analysis.

Phase 15: A/B Testing Infrastructure

Endpoints:
- /experiments/{id}/randomize: Randomize units to variants
- /experiments/{id}/enroll: Enroll units in experiments
- /experiments/{id}/assignments: Get experiment assignments
- /experiments/{id}/enrollments: Get enrollment stats
- /experiments/{id}/interim-analysis: Trigger interim analysis
- /experiments/{id}/results: Get experiment results
- /experiments/{id}/srm-checks: Get SRM detection history
- /experiments/{id}/fidelity: Get Digital Twin fidelity comparison
- /experiments/monitor: Trigger experiment monitoring sweep

Integration Points:
- ExperimentMonitorAgent (Tier 3)
- RandomizationService
- EnrollmentService
- InterimAnalysisService
- ResultsAnalysisService
- Digital Twin fidelity tracking
- Celery tasks for scheduled monitoring

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_auth, require_operator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["A/B Testing"])


# =============================================================================
# ENUMS
# =============================================================================


class RandomizationMethod(str, Enum):
    """Randomization methods."""

    SIMPLE = "simple"
    STRATIFIED = "stratified"
    BLOCK = "block"


class EnrollmentStatus(str, Enum):
    """Enrollment status values."""

    ACTIVE = "active"
    WITHDRAWN = "withdrawn"
    COMPLETED = "completed"


class AnalysisType(str, Enum):
    """Analysis types."""

    INTERIM = "interim"
    FINAL = "final"


class AnalysisMethod(str, Enum):
    """Analysis methods."""

    ITT = "itt"
    PER_PROTOCOL = "per_protocol"


class StoppingDecision(str, Enum):
    """Interim analysis stopping decisions."""

    CONTINUE = "continue"
    STOP_EFFICACY = "stop_efficacy"
    STOP_FUTILITY = "stop_futility"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class HealthStatus(str, Enum):
    """Experiment health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class RandomizeRequest(BaseModel):
    """Request to randomize units to experiment variants."""

    units: List[Dict[str, Any]] = Field(
        ..., description="List of units to randomize (each with unit_id, unit_type, and optional strata)"
    )
    method: RandomizationMethod = Field(
        default=RandomizationMethod.STRATIFIED, description="Randomization method"
    )
    strata_columns: Optional[List[str]] = Field(
        None, description="Columns to use for stratification"
    )
    allocation_ratio: Optional[Dict[str, float]] = Field(
        default={"control": 0.5, "treatment": 0.5}, description="Allocation ratio by variant"
    )
    block_size: Optional[int] = Field(
        default=4, description="Block size for block randomization"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "units": [
                    {"unit_id": "hcp_001", "unit_type": "hcp", "region": "northeast"},
                    {"unit_id": "hcp_002", "unit_type": "hcp", "region": "southwest"},
                ],
                "method": "stratified",
                "strata_columns": ["region"],
                "allocation_ratio": {"control": 0.5, "treatment": 0.5},
            }
        }
    )


class EnrollUnitRequest(BaseModel):
    """Request to enroll a unit in an experiment."""

    unit_id: str = Field(..., description="Unit identifier (HCP, patient, etc.)")
    unit_type: str = Field(..., description="Type of unit (hcp, patient, territory)")
    consent_timestamp: Optional[datetime] = Field(
        None, description="When consent was obtained"
    )
    eligibility_criteria_met: Optional[Dict[str, Any]] = Field(
        None, description="Eligibility criteria evaluation results"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "unit_id": "hcp_001",
                "unit_type": "hcp",
                "consent_timestamp": "2024-12-20T10:30:00Z",
                "eligibility_criteria_met": {"specialty": True, "experience_years": True},
            }
        }
    )


class WithdrawRequest(BaseModel):
    """Request to withdraw a unit from an experiment."""

    reason: str = Field(..., description="Reason for withdrawal")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reason": "Subject requested withdrawal from study"
            }
        }
    )


class TriggerInterimAnalysisRequest(BaseModel):
    """Request to trigger an interim analysis."""

    analysis_number: Optional[int] = Field(
        None, description="Specific analysis number (auto-detected if not provided)"
    )
    force: bool = Field(
        default=False, description="Force analysis even if milestone not reached"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_number": 2,
                "force": False,
            }
        }
    )


class TriggerMonitorRequest(BaseModel):
    """Request to trigger experiment monitoring."""

    experiment_ids: Optional[List[str]] = Field(
        None, description="Specific experiments to check (all active if not provided)"
    )
    check_srm: bool = Field(default=True, description="Check for SRM")
    check_enrollment: bool = Field(default=True, description="Check enrollment rates")
    check_fidelity: bool = Field(default=True, description="Check Digital Twin fidelity")
    srm_threshold: float = Field(default=0.001, description="SRM p-value threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_ids": None,
                "check_srm": True,
                "check_enrollment": True,
                "check_fidelity": True,
                "srm_threshold": 0.001,
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AssignmentResult(BaseModel):
    """Result of a single assignment."""

    assignment_id: str
    experiment_id: str
    unit_id: str
    unit_type: str
    variant: str
    assigned_at: datetime
    randomization_method: str
    stratification_key: Optional[Dict[str, Any]] = None
    block_id: Optional[str] = None


class RandomizeResponse(BaseModel):
    """Response from randomization."""

    experiment_id: str
    total_units: int
    assignments: List[AssignmentResult]
    variant_counts: Dict[str, int]
    randomization_method: str
    timestamp: datetime


class EnrollmentResult(BaseModel):
    """Result of enrollment."""

    enrollment_id: str
    assignment_id: str
    experiment_id: str
    unit_id: str
    variant: str
    enrolled_at: datetime
    enrollment_status: EnrollmentStatus
    consent_timestamp: Optional[datetime] = None


class EnrollmentStatsResponse(BaseModel):
    """Enrollment statistics for an experiment."""

    experiment_id: str
    total_enrolled: int
    active_count: int
    withdrawn_count: int
    completed_count: int
    enrollment_rate_per_day: float
    variant_breakdown: Dict[str, int]
    enrollment_trend: List[Dict[str, Any]]


class InterimAnalysisResult(BaseModel):
    """Result of interim analysis."""

    analysis_id: str
    experiment_id: str
    analysis_number: int
    performed_at: datetime
    information_fraction: float
    alpha_spent: float
    adjusted_alpha: float
    test_statistic: float
    p_value: float
    conditional_power: float
    decision: StoppingDecision
    metrics_snapshot: Dict[str, Any]


class ExperimentResults(BaseModel):
    """Experiment analysis results."""

    result_id: str
    experiment_id: str
    analysis_type: AnalysisType
    analysis_method: AnalysisMethod
    computed_at: datetime
    primary_metric: str
    control_mean: float
    treatment_mean: float
    effect_estimate: float
    effect_ci_lower: float
    effect_ci_upper: float
    p_value: float
    sample_size_control: int
    sample_size_treatment: int
    statistical_power: float
    is_significant: bool
    secondary_metrics: Optional[Dict[str, Any]] = None
    segment_results: Optional[Dict[str, Any]] = None


class SRMCheckResult(BaseModel):
    """SRM check result."""

    check_id: str
    experiment_id: str
    checked_at: datetime
    expected_ratio: Dict[str, float]
    actual_counts: Dict[str, int]
    chi_squared_statistic: float
    p_value: float
    is_srm_detected: bool
    investigation_notes: Optional[str] = None


class FidelityComparison(BaseModel):
    """Digital Twin fidelity comparison."""

    comparison_id: str
    experiment_id: str
    twin_simulation_id: str
    comparison_timestamp: datetime
    predicted_effect: float
    actual_effect: float
    prediction_error: float
    confidence_interval_coverage: bool
    fidelity_score: float
    calibration_adjustment: Optional[Dict[str, Any]] = None


class MonitorAlert(BaseModel):
    """Experiment monitoring alert."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    experiment_id: str
    experiment_name: str
    message: str
    details: Dict[str, Any]
    recommended_action: str
    timestamp: datetime


class ExperimentHealthSummary(BaseModel):
    """Experiment health summary."""

    experiment_id: str
    experiment_name: str
    health_status: HealthStatus
    total_enrolled: int
    enrollment_rate_per_day: float
    current_information_fraction: float
    has_srm: bool
    active_alerts: int
    last_checked: datetime


class MonitorResponse(BaseModel):
    """Response from experiment monitoring."""

    experiments_checked: int
    healthy_count: int
    warning_count: int
    critical_count: int
    experiments: List[ExperimentHealthSummary]
    alerts: List[MonitorAlert]
    monitor_summary: str
    recommended_actions: List[str]
    check_latency_ms: int
    timestamp: datetime


# =============================================================================
# RANDOMIZATION ENDPOINTS
# =============================================================================


@router.post("/{experiment_id}/randomize", response_model=RandomizeResponse)
async def randomize_units(
    experiment_id: str,
    request: RandomizeRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> RandomizeResponse:
    """
    Randomize units to experiment variants.

    Supports simple, stratified, and block randomization methods.

    Args:
        experiment_id: Experiment ID
        request: Randomization parameters and units

    Returns:
        Assignment results for all units
    """
    from src.services.randomization import RandomizationService

    logger.info(f"Randomization requested for experiment: {experiment_id}")

    try:
        service = RandomizationService()

        if request.method == RandomizationMethod.STRATIFIED:
            assignments = await service.stratified_randomize(
                experiment_id=UUID(experiment_id),
                units=request.units,
                strata_columns=request.strata_columns or [],
                allocation_ratio=request.allocation_ratio or {"control": 0.5, "treatment": 0.5},
            )
        elif request.method == RandomizationMethod.BLOCK:
            assignments = await service.block_randomize(
                experiment_id=UUID(experiment_id),
                units=request.units,
                block_size=request.block_size or 4,
                allocation_ratio=request.allocation_ratio or {"control": 0.5, "treatment": 0.5},
            )
        else:
            # Simple randomization
            assignments = await service.simple_randomize(
                experiment_id=UUID(experiment_id),
                units=request.units,
                allocation_ratio=request.allocation_ratio or {"control": 0.5, "treatment": 0.5},
            )

        # Count by variant
        variant_counts: Dict[str, int] = {}
        for assignment in assignments:
            variant = assignment.variant
            variant_counts[variant] = variant_counts.get(variant, 0) + 1

        results = [
            AssignmentResult(
                assignment_id=str(a.id),
                experiment_id=str(a.experiment_id),
                unit_id=a.unit_id,
                unit_type=a.unit_type,
                variant=a.variant,
                assigned_at=a.assigned_at,
                randomization_method=a.randomization_method,
                stratification_key=a.stratification_key,
                block_id=a.block_id,
            )
            for a in assignments
        ]

        return RandomizeResponse(
            experiment_id=experiment_id,
            total_units=len(assignments),
            assignments=results,
            variant_counts=variant_counts,
            randomization_method=request.method.value,
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:
        logger.error(f"Randomization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/assignments")
async def get_assignments(
    experiment_id: str,
    variant: Optional[str] = Query(None, description="Filter by variant"),
    unit_type: Optional[str] = Query(None, description="Filter by unit type"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max assignments"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """
    Get experiment assignments.

    Args:
        experiment_id: Experiment ID
        variant: Optional variant filter
        unit_type: Optional unit type filter
        limit: Maximum assignments to return
        offset: Pagination offset

    Returns:
        List of assignments
    """
    from src.repositories.ab_experiment import ABExperimentRepository

    try:
        repo = ABExperimentRepository()
        assignments = await repo.get_assignments(
            experiment_id=UUID(experiment_id),
            variant=variant,
            unit_type=unit_type,
            limit=limit,
            offset=offset,
        )

        return {
            "experiment_id": experiment_id,
            "total_count": len(assignments),
            "assignments": [
                {
                    "assignment_id": str(a.id),
                    "unit_id": a.unit_id,
                    "unit_type": a.unit_type,
                    "variant": a.variant,
                    "assigned_at": a.assigned_at.isoformat(),
                    "randomization_method": a.randomization_method,
                }
                for a in assignments
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get assignments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENROLLMENT ENDPOINTS
# =============================================================================


@router.post("/{experiment_id}/enroll", response_model=EnrollmentResult)
async def enroll_unit(
    experiment_id: str,
    request: EnrollUnitRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> EnrollmentResult:
    """
    Enroll a unit in an experiment.

    The unit must already be assigned to a variant.

    Args:
        experiment_id: Experiment ID
        request: Enrollment details

    Returns:
        Enrollment result
    """
    from src.services.enrollment import EnrollmentService

    logger.info(f"Enrollment requested for unit {request.unit_id} in experiment {experiment_id}")

    try:
        service = EnrollmentService()

        # Check eligibility and enroll
        enrollment = await service.enroll_unit(
            experiment_id=UUID(experiment_id),
            unit_id=request.unit_id,
            unit_type=request.unit_type,
            consent_timestamp=request.consent_timestamp,
            eligibility_criteria_met=request.eligibility_criteria_met,
        )

        return EnrollmentResult(
            enrollment_id=str(enrollment.id),
            assignment_id=str(enrollment.assignment_id),
            experiment_id=experiment_id,
            unit_id=request.unit_id,
            variant=enrollment.variant,
            enrolled_at=enrollment.enrolled_at,
            enrollment_status=EnrollmentStatus(enrollment.enrollment_status),
            consent_timestamp=enrollment.consent_timestamp,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}/enrollments/{enrollment_id}")
async def withdraw_unit(
    experiment_id: str,
    enrollment_id: str,
    request: WithdrawRequest,
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Withdraw a unit from an experiment.

    Args:
        experiment_id: Experiment ID
        enrollment_id: Enrollment ID
        request: Withdrawal details

    Returns:
        Confirmation of withdrawal
    """
    from src.services.enrollment import EnrollmentService

    logger.info(f"Withdrawal requested for enrollment {enrollment_id}")

    try:
        service = EnrollmentService()
        await service.withdraw_unit(
            enrollment_id=UUID(enrollment_id),
            reason=request.reason,
        )

        return {
            "status": "withdrawn",
            "enrollment_id": enrollment_id,
            "experiment_id": experiment_id,
            "reason": request.reason,
            "withdrawn_at": datetime.now(timezone.utc).isoformat(),
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Withdrawal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/enrollments", response_model=EnrollmentStatsResponse)
async def get_enrollment_stats(
    experiment_id: str,
) -> EnrollmentStatsResponse:
    """
    Get enrollment statistics for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        Enrollment statistics
    """
    from src.services.enrollment import EnrollmentService

    try:
        service = EnrollmentService()
        stats = await service.get_enrollment_stats(UUID(experiment_id))

        return EnrollmentStatsResponse(
            experiment_id=experiment_id,
            total_enrolled=stats.total_enrolled,
            active_count=stats.active_count,
            withdrawn_count=stats.withdrawn_count,
            completed_count=stats.completed_count,
            enrollment_rate_per_day=stats.enrollment_rate_per_day,
            variant_breakdown=stats.variant_breakdown,
            enrollment_trend=stats.enrollment_trend,
        )

    except Exception as e:
        logger.error(f"Failed to get enrollment stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INTERIM ANALYSIS ENDPOINTS
# =============================================================================


@router.post("/{experiment_id}/interim-analysis", response_model=InterimAnalysisResult)
async def trigger_interim_analysis(
    experiment_id: str,
    request: TriggerInterimAnalysisRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=False, description="Run asynchronously"),
    user: Dict[str, Any] = Depends(require_operator),
) -> InterimAnalysisResult:
    """
    Trigger an interim analysis for an experiment.

    Uses O'Brien-Fleming alpha spending for multiple comparisons.

    Args:
        experiment_id: Experiment ID
        request: Analysis parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, runs asynchronously

    Returns:
        Interim analysis results
    """
    from src.services.interim_analysis import InterimAnalysisService

    logger.info(f"Interim analysis requested for experiment: {experiment_id}")

    if async_mode:
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        task = scheduled_interim_analysis.delay(experiment_id)
        return InterimAnalysisResult(
            analysis_id=task.id,
            experiment_id=experiment_id,
            analysis_number=request.analysis_number or 0,
            performed_at=datetime.now(timezone.utc),
            information_fraction=0.0,
            alpha_spent=0.0,
            adjusted_alpha=0.0,
            test_statistic=0.0,
            p_value=1.0,
            conditional_power=0.0,
            decision=StoppingDecision.CONTINUE,
            metrics_snapshot={"status": "queued", "task_id": task.id},
        )

    try:
        service = InterimAnalysisService()
        result = await service.perform_interim_analysis(
            experiment_id=UUID(experiment_id),
            analysis_number=request.analysis_number,
            force=request.force,
        )

        return InterimAnalysisResult(
            analysis_id=str(result.id),
            experiment_id=experiment_id,
            analysis_number=result.analysis_number,
            performed_at=result.performed_at,
            information_fraction=result.information_fraction,
            alpha_spent=result.alpha_spent,
            adjusted_alpha=result.adjusted_alpha,
            test_statistic=result.test_statistic,
            p_value=result.p_value,
            conditional_power=result.conditional_power,
            decision=StoppingDecision(result.decision),
            metrics_snapshot=result.metrics_snapshot or {},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Interim analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/interim-analyses")
async def list_interim_analyses(
    experiment_id: str,
) -> Dict[str, Any]:
    """
    List all interim analyses for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        List of interim analysis results
    """
    from src.repositories.ab_experiment import ABExperimentRepository

    try:
        repo = ABExperimentRepository()
        analyses = await repo.get_interim_analyses(UUID(experiment_id))

        return {
            "experiment_id": experiment_id,
            "total_analyses": len(analyses),
            "analyses": [
                {
                    "analysis_id": str(a.id),
                    "analysis_number": a.analysis_number,
                    "performed_at": a.performed_at.isoformat(),
                    "information_fraction": a.information_fraction,
                    "p_value": a.p_value,
                    "decision": a.decision,
                }
                for a in analyses
            ],
        }

    except Exception as e:
        logger.error(f"Failed to list interim analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RESULTS ENDPOINTS
# =============================================================================


@router.get("/{experiment_id}/results", response_model=ExperimentResults)
async def get_experiment_results(
    experiment_id: str,
    analysis_type: AnalysisType = Query(default=AnalysisType.FINAL),
    analysis_method: AnalysisMethod = Query(default=AnalysisMethod.ITT),
    recompute: bool = Query(default=False, description="Force recomputation"),
    background_tasks: BackgroundTasks = None,
) -> ExperimentResults:
    """
    Get experiment results.

    Args:
        experiment_id: Experiment ID
        analysis_type: Type of analysis (interim or final)
        analysis_method: Analysis method (ITT or per-protocol)
        recompute: If True, recomputes results

    Returns:
        Experiment analysis results
    """
    from src.services.results_analysis import ResultsAnalysisService

    logger.info(f"Results requested for experiment: {experiment_id}")

    try:
        service = ResultsAnalysisService()

        if recompute:
            if analysis_method == AnalysisMethod.ITT:
                result = await service.compute_itt_results(UUID(experiment_id))
            else:
                result = await service.compute_per_protocol_results(UUID(experiment_id))
        else:
            # Get cached results
            from src.repositories.ab_results import ABResultsRepository

            repo = ABResultsRepository()
            results = await repo.get_results(UUID(experiment_id))

            if not results:
                # Compute if no cached results
                if analysis_method == AnalysisMethod.ITT:
                    result = await service.compute_itt_results(UUID(experiment_id))
                else:
                    result = await service.compute_per_protocol_results(UUID(experiment_id))
            else:
                result = results[0]

        return ExperimentResults(
            result_id=str(result.id),
            experiment_id=experiment_id,
            analysis_type=AnalysisType(result.analysis_type),
            analysis_method=AnalysisMethod(result.analysis_method),
            computed_at=result.computed_at,
            primary_metric=result.primary_metric,
            control_mean=result.control_mean,
            treatment_mean=result.treatment_mean,
            effect_estimate=result.effect_estimate,
            effect_ci_lower=result.effect_ci_lower,
            effect_ci_upper=result.effect_ci_upper,
            p_value=result.p_value,
            sample_size_control=result.sample_size_control,
            sample_size_treatment=result.sample_size_treatment,
            statistical_power=result.statistical_power,
            is_significant=result.is_significant,
            secondary_metrics=result.secondary_metrics,
            segment_results=result.segment_results,
        )

    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/results/segments")
async def get_segment_results(
    experiment_id: str,
    segments: List[str] = Query(default=["region", "specialty"], description="Segments to analyze"),
) -> Dict[str, Any]:
    """
    Get heterogeneous treatment effects by segment.

    Args:
        experiment_id: Experiment ID
        segments: List of segment dimensions to analyze

    Returns:
        Treatment effects by segment
    """
    from src.services.results_analysis import ResultsAnalysisService

    try:
        service = ResultsAnalysisService()
        hte_results = await service.compute_heterogeneous_effects(
            UUID(experiment_id),
            segments=segments,
        )

        return {
            "experiment_id": experiment_id,
            "segments_analyzed": segments,
            "segment_results": hte_results,
        }

    except Exception as e:
        logger.error(f"Failed to get segment results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SRM ENDPOINTS
# =============================================================================


@router.get("/{experiment_id}/srm-checks")
async def get_srm_checks(
    experiment_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Max checks"),
) -> Dict[str, Any]:
    """
    Get SRM check history for an experiment.

    Args:
        experiment_id: Experiment ID
        limit: Maximum checks to return

    Returns:
        SRM check history
    """
    from src.repositories.ab_results import ABResultsRepository

    try:
        repo = ABResultsRepository()
        checks = await repo.get_srm_history(UUID(experiment_id), limit=limit)

        return {
            "experiment_id": experiment_id,
            "total_checks": len(checks),
            "srm_detected_count": sum(1 for c in checks if c.is_srm_detected),
            "checks": [
                {
                    "check_id": str(c.id),
                    "checked_at": c.checked_at.isoformat(),
                    "expected_ratio": c.expected_ratio,
                    "actual_counts": c.actual_counts,
                    "chi_squared": c.chi_squared_statistic,
                    "p_value": c.p_value,
                    "is_srm_detected": c.is_srm_detected,
                }
                for c in checks
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get SRM checks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/srm-check", response_model=SRMCheckResult)
async def run_srm_check(
    experiment_id: str,
    user: Dict[str, Any] = Depends(require_auth),
) -> SRMCheckResult:
    """
    Run an SRM check for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        SRM check result
    """
    from src.services.results_analysis import ResultsAnalysisService

    try:
        service = ResultsAnalysisService()
        result = await service.check_sample_ratio_mismatch(UUID(experiment_id))

        return SRMCheckResult(
            check_id=str(result.id),
            experiment_id=experiment_id,
            checked_at=result.checked_at,
            expected_ratio=result.expected_ratio,
            actual_counts=result.actual_counts,
            chi_squared_statistic=result.chi_squared_statistic,
            p_value=result.p_value,
            is_srm_detected=result.is_srm_detected,
            investigation_notes=result.investigation_notes,
        )

    except Exception as e:
        logger.error(f"SRM check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FIDELITY ENDPOINTS
# =============================================================================


@router.get("/{experiment_id}/fidelity")
async def get_fidelity_comparisons(
    experiment_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Max comparisons"),
) -> Dict[str, Any]:
    """
    Get Digital Twin fidelity comparisons for an experiment.

    Args:
        experiment_id: Experiment ID
        limit: Maximum comparisons to return

    Returns:
        Fidelity comparison history
    """
    from src.repositories.ab_results import ABResultsRepository

    try:
        repo = ABResultsRepository()
        comparisons = await repo.get_fidelity_comparisons(UUID(experiment_id), limit=limit)

        return {
            "experiment_id": experiment_id,
            "total_comparisons": len(comparisons),
            "average_fidelity_score": (
                sum(c.fidelity_score for c in comparisons) / len(comparisons)
                if comparisons
                else 0.0
            ),
            "comparisons": [
                {
                    "comparison_id": str(c.id),
                    "twin_simulation_id": str(c.twin_simulation_id),
                    "timestamp": c.comparison_timestamp.isoformat(),
                    "predicted_effect": c.predicted_effect,
                    "actual_effect": c.actual_effect,
                    "prediction_error": c.prediction_error,
                    "fidelity_score": c.fidelity_score,
                }
                for c in comparisons
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get fidelity comparisons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/fidelity/{twin_simulation_id}", response_model=FidelityComparison)
async def update_fidelity_comparison(
    experiment_id: str,
    twin_simulation_id: str,
    user: Dict[str, Any] = Depends(require_auth),
) -> FidelityComparison:
    """
    Update fidelity comparison with latest experiment results.

    Args:
        experiment_id: Experiment ID
        twin_simulation_id: Digital Twin simulation ID

    Returns:
        Updated fidelity comparison
    """
    from src.services.results_analysis import ResultsAnalysisService

    try:
        service = ResultsAnalysisService()
        result = await service.compare_with_twin_prediction(
            UUID(experiment_id),
            UUID(twin_simulation_id),
        )

        return FidelityComparison(
            comparison_id=str(result.id),
            experiment_id=experiment_id,
            twin_simulation_id=twin_simulation_id,
            comparison_timestamp=result.comparison_timestamp,
            predicted_effect=result.predicted_effect,
            actual_effect=result.actual_effect,
            prediction_error=result.prediction_error,
            confidence_interval_coverage=result.confidence_interval_coverage,
            fidelity_score=result.fidelity_score,
            calibration_adjustment=result.calibration_adjustment,
        )

    except Exception as e:
        logger.error(f"Fidelity comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MONITORING ENDPOINTS
# =============================================================================


@router.post("/monitor", response_model=MonitorResponse)
async def trigger_experiment_monitoring(
    request: TriggerMonitorRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=False, description="Run asynchronously"),
    user: Dict[str, Any] = Depends(require_auth),
) -> MonitorResponse:
    """
    Trigger experiment monitoring sweep.

    Checks all active experiments (or specified ones) for health issues.

    Args:
        request: Monitoring parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, runs asynchronously

    Returns:
        Monitoring results with alerts
    """
    from src.agents.experiment_monitor import ExperimentMonitorAgent, ExperimentMonitorInput

    logger.info("Experiment monitoring requested")

    if async_mode:
        from src.tasks.ab_testing_tasks import check_all_active_experiments

        task = check_all_active_experiments.delay(
            srm_threshold=request.srm_threshold,
        )
        return MonitorResponse(
            experiments_checked=0,
            healthy_count=0,
            warning_count=0,
            critical_count=0,
            experiments=[],
            alerts=[],
            monitor_summary=f"Monitoring task queued. Task ID: {task.id}",
            recommended_actions=["Poll task status for results"],
            check_latency_ms=0,
            timestamp=datetime.now(timezone.utc),
        )

    try:
        agent = ExperimentMonitorAgent()
        result = await agent.run_async(
            ExperimentMonitorInput(
                experiment_ids=request.experiment_ids,
                check_all_active=request.experiment_ids is None,
                srm_threshold=request.srm_threshold,
                check_interim=True,
            )
        )

        experiments = [
            ExperimentHealthSummary(
                experiment_id=exp.get("experiment_id", ""),
                experiment_name=exp.get("name", ""),
                health_status=HealthStatus(exp.get("health_status", "unknown")),
                total_enrolled=exp.get("total_enrolled", 0),
                enrollment_rate_per_day=exp.get("enrollment_rate_per_day", 0.0),
                current_information_fraction=exp.get("current_information_fraction", 0.0),
                has_srm=exp.get("has_srm", False),
                active_alerts=exp.get("active_alerts", 0),
                last_checked=datetime.now(timezone.utc),
            )
            for exp in result.experiments
        ]

        alerts = [
            MonitorAlert(
                alert_id=alert.get("alert_id", ""),
                alert_type=alert.get("alert_type", ""),
                severity=AlertSeverity(alert.get("severity", "info")),
                experiment_id=alert.get("experiment_id", ""),
                experiment_name=alert.get("experiment_name", ""),
                message=alert.get("message", ""),
                details=alert.get("details", {}),
                recommended_action=alert.get("recommended_action", ""),
                timestamp=datetime.fromisoformat(alert.get("timestamp", datetime.now(timezone.utc).isoformat())),
            )
            for alert in result.alerts
        ]

        return MonitorResponse(
            experiments_checked=result.experiments_checked,
            healthy_count=result.healthy_count,
            warning_count=result.warning_count,
            critical_count=result.critical_count,
            experiments=experiments,
            alerts=alerts,
            monitor_summary=result.monitor_summary,
            recommended_actions=result.recommended_actions,
            check_latency_ms=result.check_latency_ms,
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:
        logger.error(f"Experiment monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/health", response_model=ExperimentHealthSummary)
async def get_experiment_health(
    experiment_id: str,
) -> ExperimentHealthSummary:
    """
    Get health status for a single experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        Experiment health summary
    """
    from src.agents.experiment_monitor import ExperimentMonitorAgent, ExperimentMonitorInput

    try:
        agent = ExperimentMonitorAgent()
        result = await agent.run_async(
            ExperimentMonitorInput(
                experiment_ids=[experiment_id],
                check_all_active=False,
            )
        )

        if not result.experiments:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

        exp = result.experiments[0]
        return ExperimentHealthSummary(
            experiment_id=exp.get("experiment_id", experiment_id),
            experiment_name=exp.get("name", ""),
            health_status=HealthStatus(exp.get("health_status", "unknown")),
            total_enrolled=exp.get("total_enrolled", 0),
            enrollment_rate_per_day=exp.get("enrollment_rate_per_day", 0.0),
            current_information_fraction=exp.get("current_information_fraction", 0.0),
            has_srm=exp.get("has_srm", False),
            active_alerts=len([a for a in result.alerts if a.get("experiment_id") == experiment_id]),
            last_checked=datetime.now(timezone.utc),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/alerts")
async def get_experiment_alerts(
    experiment_id: str,
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    limit: int = Query(default=50, ge=1, le=200, description="Max alerts"),
) -> Dict[str, Any]:
    """
    Get alerts for an experiment.

    Args:
        experiment_id: Experiment ID
        severity: Optional severity filter
        limit: Maximum alerts to return

    Returns:
        List of alerts
    """
    from src.repositories.ab_results import ABResultsRepository

    try:
        repo = ABResultsRepository()
        alerts = await repo.get_experiment_alerts(
            UUID(experiment_id),
            severity=severity.value if severity else None,
            limit=limit,
        )

        return {
            "experiment_id": experiment_id,
            "total_alerts": len(alerts),
            "critical_count": sum(1 for a in alerts if a.severity == "critical"),
            "warning_count": sum(1 for a in alerts if a.severity == "warning"),
            "alerts": [
                {
                    "alert_id": str(a.id),
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "details": a.details,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in alerts
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get experiment alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
