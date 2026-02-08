"""
E2I Model Monitoring & Drift Detection API
===========================================

FastAPI endpoints for drift monitoring, alerting, and model health tracking.

Phase 14: Model Monitoring & Drift Detection

Endpoints:
- /monitoring/drift: Trigger and query drift detection
- /monitoring/alerts: Manage drift alerts
- /monitoring/runs: View monitoring runs
- /monitoring/performance: Track model performance

Integration Points:
- Drift Monitor Agent (Tier 3)
- Celery tasks for scheduled monitoring
- Supabase for persistence
- Alert routing (email/Slack)

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_admin
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/monitoring",
    tags=["Model Monitoring"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS & MODELS
# =============================================================================


class DriftType(str, Enum):
    """Types of drift detection."""

    DATA = "data"
    MODEL = "model"
    CONCEPT = "concept"
    ALL = "all"


class DriftSeverity(str, Enum):
    """Drift severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status values."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SNOOZED = "snoozed"


class AlertAction(str, Enum):
    """Actions that can be taken on alerts."""

    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"
    SNOOZE = "snooze"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class TriggerDriftDetectionRequest(BaseModel):
    """Request to trigger drift detection for a model."""

    model_id: str = Field(..., description="Model version/ID to check for drift")
    time_window: str = Field(
        default="7d", description="Time window for comparison (e.g., '7d', '14d', '30d')"
    )
    features: Optional[List[str]] = Field(
        None, description="Specific features to check (None = all available)"
    )
    check_data_drift: bool = Field(default=True, description="Enable data drift detection")
    check_model_drift: bool = Field(default=True, description="Enable model drift detection")
    check_concept_drift: bool = Field(default=True, description="Enable concept drift detection")
    brand: Optional[str] = Field(None, description="Optional brand filter")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "propensity_v2.1.0",
                "time_window": "7d",
                "check_data_drift": True,
                "check_model_drift": True,
                "check_concept_drift": True,
            }
        }
    )


class AlertActionRequest(BaseModel):
    """Request to update an alert."""

    action: AlertAction = Field(..., description="Action to take on the alert")
    user_id: Optional[str] = Field(None, description="User performing the action")
    notes: Optional[str] = Field(None, description="Optional notes about the action")
    snooze_until: Optional[datetime] = Field(
        None, description="Snooze until this time (for snooze action)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "acknowledge",
                "user_id": "user_123",
                "notes": "Investigating the drift issue",
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class DriftResult(BaseModel):
    """Single drift detection result."""

    feature: str = Field(..., description="Feature or metric name")
    drift_type: DriftType = Field(..., description="Type of drift detected")
    test_statistic: float = Field(..., description="Statistical test value")
    p_value: float = Field(..., description="P-value from statistical test")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    severity: DriftSeverity = Field(..., description="Severity level")
    baseline_period: str = Field(..., description="Baseline time period label")
    current_period: str = Field(..., description="Current time period label")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature": "days_since_last_visit",
                "drift_type": "data",
                "test_statistic": 0.156,
                "p_value": 0.023,
                "drift_detected": True,
                "severity": "medium",
                "baseline_period": "2024-12-01 to 2024-12-08",
                "current_period": "2024-12-08 to 2024-12-15",
            }
        }
    )


class DriftDetectionResponse(BaseModel):
    """Response from drift detection."""

    task_id: str = Field(..., description="Celery task ID (if async)")
    model_id: str = Field(..., description="Model that was checked")
    status: str = Field(..., description="Detection status")
    overall_drift_score: float = Field(default=0.0, description="Overall drift severity (0-1)")
    features_checked: int = Field(default=0, description="Number of features checked")
    features_with_drift: List[str] = Field(
        default_factory=list, description="Features with detected drift"
    )
    results: List[DriftResult] = Field(default_factory=list, description="Detailed drift results")
    drift_summary: str = Field(default="", description="Human-readable summary")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    detection_latency_ms: int = Field(default=0, description="Detection time in ms")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "abc123",
                "model_id": "propensity_v2.1.0",
                "status": "completed",
                "overall_drift_score": 0.35,
                "features_checked": 25,
                "features_with_drift": ["days_since_last_visit", "rx_count"],
                "drift_summary": "Moderate drift detected in 2 features",
                "recommended_actions": ["Investigate feature drift", "Consider retraining"],
                "detection_latency_ms": 1250,
            }
        }
    )


class DriftHistoryItem(BaseModel):
    """Historical drift record."""

    id: str
    model_version: str
    feature_name: str
    drift_type: str
    drift_score: float
    severity: str
    detected_at: datetime
    baseline_start: datetime
    baseline_end: datetime
    current_start: datetime
    current_end: datetime


class DriftHistoryResponse(BaseModel):
    """Response for drift history query."""

    model_id: str
    total_records: int
    records: List[DriftHistoryItem]


class AlertItem(BaseModel):
    """Alert record."""

    id: str
    model_version: str
    alert_type: str
    severity: str
    title: str
    description: str
    status: AlertStatus
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class AlertListResponse(BaseModel):
    """Response for alert listing."""

    total_count: int
    active_count: int
    alerts: List[AlertItem]


class MonitoringRunItem(BaseModel):
    """Monitoring run record."""

    id: str
    model_version: str
    run_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    features_checked: int
    drift_detected_count: int
    alerts_generated: int
    duration_ms: int
    error_message: Optional[str] = None


class MonitoringRunsResponse(BaseModel):
    """Response for monitoring runs query."""

    model_id: Optional[str]
    total_runs: int
    runs: List[MonitoringRunItem]


class ModelHealthSummary(BaseModel):
    """Summary of model health status."""

    model_id: str
    overall_health: str  # healthy, warning, critical
    last_check: Optional[datetime] = None
    drift_score: float
    active_alerts: int
    last_retrained: Optional[datetime] = None
    performance_trend: str  # stable, improving, degrading
    recommendations: List[str]


# =============================================================================
# DRIFT DETECTION ENDPOINTS
# =============================================================================


@router.post(
    "/drift/detect",
    response_model=DriftDetectionResponse,
    summary="Trigger drift detection",
    operation_id="trigger_drift_detection",
)
async def trigger_drift_detection(
    request: TriggerDriftDetectionRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=True, description="Run detection asynchronously"),
) -> DriftDetectionResponse:
    """
    Trigger drift detection for a model.

    Runs data, model, and concept drift detection based on request parameters.
    By default runs asynchronously via Celery.

    Args:
        request: Drift detection parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, returns immediately with task ID

    Returns:
        Detection results or task ID for async polling
    """
    from src.tasks.drift_monitoring_tasks import run_drift_detection

    logger.info(f"Drift detection requested for model: {request.model_id}")

    if async_mode:
        # Queue Celery task
        task = run_drift_detection.delay(
            model_id=request.model_id,
            time_window=request.time_window,
            features=request.features,
            check_data_drift=request.check_data_drift,
            check_model_drift=request.check_model_drift,
            check_concept_drift=request.check_concept_drift,
            brand=request.brand,
        )

        return DriftDetectionResponse(
            task_id=task.id,
            model_id=request.model_id,
            status="queued",
            drift_summary="Detection task queued. Poll /monitoring/drift/status/{task_id} for results.",
        )
    else:
        # Run synchronously (for testing or small jobs)
        try:
            result = run_drift_detection(
                model_id=request.model_id,
                time_window=request.time_window,
                features=request.features,
                check_data_drift=request.check_data_drift,
                check_model_drift=request.check_model_drift,
                check_concept_drift=request.check_concept_drift,
                brand=request.brand,
            )

            return DriftDetectionResponse(
                task_id=result.get("run_id", "sync"),
                model_id=request.model_id,
                status=result.get("status", "completed"),
                overall_drift_score=result.get("overall_drift_score", 0.0),
                features_checked=result.get("features_checked", 0),
                features_with_drift=result.get("features_with_drift", []),
                drift_summary=result.get("drift_summary", ""),
                recommended_actions=result.get("recommended_actions", []),
                detection_latency_ms=result.get("detection_latency_ms", 0),
            )
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/drift/status/{task_id}",
    summary="Get drift detection status",
    operation_id="get_drift_detection_status",
)
async def get_drift_detection_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of an async drift detection task.

    Args:
        task_id: Celery task ID

    Returns:
        Task status and results if complete
    """
    from celery.result import AsyncResult

    from src.workers.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
    }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)

    return response


@router.get(
    "/drift/latest/{model_id}",
    response_model=DriftDetectionResponse,
    summary="Get latest drift status",
    operation_id="get_latest_drift_status",
)
async def get_latest_drift_status(
    model_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Max results per type"),
) -> DriftDetectionResponse:
    """
    Get the latest drift status for a model.

    Args:
        model_id: Model version/ID
        limit: Maximum drift results to return

    Returns:
        Latest drift detection results
    """
    from src.repositories.drift_monitoring import DriftHistoryRepository

    try:
        repo = DriftHistoryRepository()
        records = await repo.get_latest_drift_status(model_id, limit=limit)

        # Aggregate results
        features_with_drift = []
        results = []
        max_score = 0.0

        for record in records:
            if record.severity in ("high", "critical"):
                features_with_drift.append(record.feature_name)
            # Map severity to drift score
            severity_to_score = {
                "none": 0.0,
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "critical": 1.0,
            }
            score = severity_to_score.get(record.severity, 0.0)
            max_score = max(max_score, score)

            results.append(
                DriftResult(
                    feature=record.feature_name,
                    drift_type=DriftType(record.drift_type),
                    test_statistic=record.test_statistic or 0.0,
                    p_value=record.p_value or 0.0,
                    drift_detected=record.severity != "none",
                    severity=DriftSeverity(record.severity),
                    baseline_period=f"{record.baseline_start} to {record.baseline_end}",
                    current_period=f"{record.current_start} to {record.current_end}",
                )
            )

        return DriftDetectionResponse(
            task_id="history",
            model_id=model_id,
            status="retrieved",
            overall_drift_score=max_score,
            features_checked=len(records),
            features_with_drift=features_with_drift,
            results=results,
            drift_summary=f"Retrieved {len(records)} drift records",
        )

    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/drift/history/{model_id}",
    response_model=DriftHistoryResponse,
    summary="Get drift history",
    operation_id="get_drift_history",
)
async def get_drift_history(
    model_id: str,
    feature_name: Optional[str] = Query(None, description="Filter by feature"),
    days: int = Query(default=30, ge=1, le=365, description="Days of history"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max records"),
) -> DriftHistoryResponse:
    """
    Get drift detection history for a model.

    Args:
        model_id: Model version/ID
        feature_name: Optional feature filter
        days: Number of days to look back
        limit: Maximum records to return

    Returns:
        Historical drift records
    """
    from src.repositories.drift_monitoring import DriftHistoryRepository

    try:
        repo = DriftHistoryRepository()

        if feature_name:
            records = await repo.get_drift_trend(model_id, feature_name, days=days)
        else:
            records = await repo.get_latest_drift_status(model_id, limit=limit)

        # Map severity to drift score for response
        severity_to_score = {
            "none": 0.0,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0,
        }

        items = [
            DriftHistoryItem(
                id=str(r.id),
                model_version=r.model_version,
                feature_name=r.feature_name,
                drift_type=r.drift_type,
                drift_score=severity_to_score.get(r.severity, 0.0),
                severity=r.severity,
                detected_at=r.detected_at,
                baseline_start=r.baseline_start,
                baseline_end=r.baseline_end,
                current_start=r.current_start,
                current_end=r.current_end,
            )
            for r in records[:limit]
        ]

        return DriftHistoryResponse(
            model_id=model_id,
            total_records=len(items),
            records=items,
        )

    except Exception as e:
        logger.error(f"Failed to get drift history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ALERT ENDPOINTS
# =============================================================================


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List drift alerts",
    operation_id="list_drift_alerts",
)
async def list_alerts(
    model_id: Optional[str] = Query(None, description="Filter by model"),
    status: Optional[AlertStatus] = Query(None, description="Filter by status"),
    severity: Optional[DriftSeverity] = Query(None, description="Filter by severity"),
    limit: int = Query(default=50, ge=1, le=200, description="Max alerts"),
) -> AlertListResponse:
    """
    List drift alerts.

    Args:
        model_id: Optional model filter
        status: Optional status filter
        severity: Optional severity filter
        limit: Maximum alerts to return

    Returns:
        List of alerts matching criteria
    """
    from src.repositories.drift_monitoring import MonitoringAlertRepository

    try:
        repo = MonitoringAlertRepository()

        if model_id:
            records = await repo.get_active_alerts(model_id, limit=limit)
        else:
            # Get all active alerts
            records = await repo.get_active_alerts(None, limit=limit)

        # Apply filters
        filtered = []
        for r in records:
            if status and r.status != status.value:
                continue
            if severity and r.severity != severity.value:
                continue
            filtered.append(r)

        items = [
            AlertItem(
                id=str(r.id),
                model_version=r.model_version,
                alert_type=r.alert_type,
                severity=r.severity,
                title=r.message,  # MonitoringAlertRecord uses 'message' not 'title'
                description=r.recommended_action or "",  # Use recommended_action as description
                status=AlertStatus(r.status),
                triggered_at=r.triggered_at,
                acknowledged_at=None,  # MonitoringAlertRecord doesn't have acknowledged_at
                acknowledged_by=None,  # MonitoringAlertRecord doesn't have acknowledged_by
                resolved_at=r.resolved_at,
                resolved_by=r.resolved_by,
            )
            for r in filtered[:limit]
        ]

        active_count = sum(1 for item in items if item.status == AlertStatus.ACTIVE)

        return AlertListResponse(
            total_count=len(items),
            active_count=active_count,
            alerts=items,
        )

    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertItem,
    summary="Get alert details",
    operation_id="get_drift_alert",
)
async def get_alert(alert_id: str) -> AlertItem:
    """
    Get a specific alert by ID.

    Args:
        alert_id: Alert UUID

    Returns:
        Alert details
    """
    from src.repositories.drift_monitoring import MonitoringAlertRepository

    try:
        repo = MonitoringAlertRepository()
        record = await repo.get_by_id(alert_id)

        if not record:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return AlertItem(
            id=str(record.id),
            model_version=record.model_version,
            alert_type=record.alert_type,
            severity=record.severity,
            title=record.message,  # MonitoringAlertRecord uses 'message' not 'title'
            description=record.recommended_action or "",
            status=AlertStatus(record.status),
            triggered_at=record.triggered_at,
            acknowledged_at=None,  # MonitoringAlertRecord doesn't have acknowledged_at
            acknowledged_by=None,  # MonitoringAlertRecord doesn't have acknowledged_by
            resolved_at=record.resolved_at,
            resolved_by=record.resolved_by,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/alerts/{alert_id}/action",
    response_model=AlertItem,
    summary="Update alert status",
    operation_id="update_drift_alert",
)
async def update_alert(alert_id: str, request: AlertActionRequest) -> AlertItem:
    """
    Perform an action on an alert (acknowledge, resolve, snooze).

    Args:
        alert_id: Alert UUID
        request: Action to perform

    Returns:
        Updated alert
    """
    from src.repositories.drift_monitoring import MonitoringAlertRepository

    try:
        repo = MonitoringAlertRepository()

        if request.action == AlertAction.ACKNOWLEDGE:
            record = await repo.acknowledge_alert(
                alert_id, acknowledged_by=request.user_id or "api_user"
            )
        elif request.action == AlertAction.RESOLVE:
            record = await repo.resolve_alert(alert_id, resolved_by=request.user_id or "api_user")
        elif request.action == AlertAction.SNOOZE:
            # Snooze is acknowledge with a note
            record = await repo.acknowledge_alert(
                alert_id, acknowledged_by=f"snoozed_by_{request.user_id or 'api_user'}"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

        if not record:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return AlertItem(
            id=str(record.id),
            model_version=record.model_version,
            alert_type=record.alert_type,
            severity=record.severity,
            title=record.message,  # MonitoringAlertRecord uses 'message' not 'title'
            description=record.recommended_action or "",
            status=AlertStatus(record.status),
            triggered_at=record.triggered_at,
            acknowledged_at=None,  # MonitoringAlertRecord doesn't have acknowledged_at
            acknowledged_by=None,  # MonitoringAlertRecord doesn't have acknowledged_by
            resolved_at=record.resolved_at,
            resolved_by=record.resolved_by,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MONITORING RUNS ENDPOINTS
# =============================================================================


@router.get(
    "/runs",
    response_model=MonitoringRunsResponse,
    summary="List monitoring runs",
    operation_id="list_monitoring_runs",
)
async def list_monitoring_runs(
    model_id: Optional[str] = Query(None, description="Filter by model"),
    days: int = Query(default=7, ge=1, le=90, description="Days of history"),
    limit: int = Query(default=50, ge=1, le=200, description="Max runs"),
) -> MonitoringRunsResponse:
    """
    List monitoring runs.

    Args:
        model_id: Optional model filter
        days: Number of days to look back
        limit: Maximum runs to return

    Returns:
        List of monitoring runs
    """
    from datetime import timedelta

    from src.repositories.drift_monitoring import MonitoringRunRepository

    try:
        repo = MonitoringRunRepository()
        datetime.now(timezone.utc) - timedelta(days=days)

        # Get runs for model or all (get_recent_runs handles both cases)
        all_records = await repo.get_recent_runs(model_version=model_id, limit=limit)

        items = [
            MonitoringRunItem(
                id=str(r.id),
                model_version=r.model_version,
                run_type=r.run_type,
                started_at=r.started_at,
                completed_at=r.completed_at,
                features_checked=r.features_checked or 0,
                drift_detected_count=r.drift_detected_count or 0,
                alerts_generated=r.alerts_generated or 0,
                duration_ms=r.duration_ms or 0,
                error_message=r.error_message,
            )
            for r in all_records[:limit]
        ]

        return MonitoringRunsResponse(
            model_id=model_id,
            total_runs=len(items),
            runs=items,
        )

    except Exception as e:
        logger.error(f"Failed to list monitoring runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MODEL HEALTH ENDPOINTS
# =============================================================================


@router.get(
    "/health/{model_id}",
    response_model=ModelHealthSummary,
    summary="Get model health summary",
    operation_id="get_model_health_summary",
)
async def get_model_health(model_id: str) -> ModelHealthSummary:
    """
    Get overall health summary for a model.

    Aggregates drift status, alerts, and performance metrics.

    Args:
        model_id: Model version/ID

    Returns:
        Health summary with recommendations
    """
    from src.repositories.drift_monitoring import (
        DriftHistoryRepository,
        MonitoringAlertRepository,
        MonitoringRunRepository,
    )

    try:
        drift_repo = DriftHistoryRepository()
        alert_repo = MonitoringAlertRepository()
        run_repo = MonitoringRunRepository()

        # Get latest drift status
        drift_records = await drift_repo.get_latest_drift_status(model_id, limit=20)
        severity_to_score = {
            "none": 0.0,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "critical": 1.0,
        }
        max_drift_score = max(
            (severity_to_score.get(r.severity, 0.0) for r in drift_records),
            default=0.0,
        )

        # Get active alerts
        alerts = await alert_repo.get_active_alerts(model_id, limit=100)
        active_count = len([a for a in alerts if a.status == "active"])

        # Get recent runs
        runs = await run_repo.get_recent_runs(model_version=model_id, limit=10)
        last_check = runs[0].completed_at if runs else None

        # Determine overall health
        if max_drift_score >= 0.7 or active_count >= 3:
            overall_health = "critical"
        elif max_drift_score >= 0.4 or active_count >= 1:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        # Generate recommendations
        recommendations = []
        if max_drift_score >= 0.5:
            recommendations.append("Consider retraining model due to significant drift")
        if active_count > 0:
            recommendations.append(f"Review {active_count} active alert(s)")
        if not last_check:
            recommendations.append("No recent monitoring runs - schedule drift detection")

        # Determine performance trend (placeholder)
        performance_trend = "stable"

        return ModelHealthSummary(
            model_id=model_id,
            overall_health=overall_health,
            last_check=last_check,
            drift_score=max_drift_score,
            active_alerts=active_count,
            last_retrained=None,  # Would come from model registry
            performance_trend=performance_trend,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PERFORMANCE TRACKING ENDPOINTS
# =============================================================================


class RecordPerformanceRequest(BaseModel):
    """Request to record model performance metrics."""

    model_id: str = Field(..., description="Model version/ID")
    predictions: List[int] = Field(..., description="Predicted labels")
    actuals: List[int] = Field(..., description="Actual labels")
    prediction_scores: Optional[List[float]] = Field(
        None, description="Prediction probability scores (for AUC)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "propensity_v2.1.0",
                "predictions": [1, 0, 1, 1, 0],
                "actuals": [1, 0, 1, 0, 0],
                "prediction_scores": [0.85, 0.23, 0.91, 0.67, 0.12],
            }
        }
    )


class PerformanceMetricItem(BaseModel):
    """Single performance metric."""

    metric_name: str
    metric_value: float
    recorded_at: datetime


class PerformanceTrendResponse(BaseModel):
    """Response for performance trend query."""

    model_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    trend: str  # improving, stable, degrading
    is_significant: bool
    alert_threshold_breached: bool
    history: List[PerformanceMetricItem] = []


class PerformanceRecordResponse(BaseModel):
    """Response from recording performance."""

    model_id: str
    recorded_at: datetime
    sample_size: int
    metrics: Dict[str, float]
    alerts_generated: int = 0


class PerformanceAlertItem(BaseModel):
    """Performance alert."""

    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    trend: str
    severity: str
    message: str


class PerformanceAlertsResponse(BaseModel):
    """Response for performance alerts query."""

    model_id: str
    alert_count: int
    alerts: List[PerformanceAlertItem]


@router.post(
    "/performance/record",
    response_model=PerformanceRecordResponse,
    summary="Record model performance",
    operation_id="record_model_performance",
)
async def record_performance(
    request: RecordPerformanceRequest,
    async_mode: bool = Query(default=True, description="Run asynchronously"),
) -> PerformanceRecordResponse:
    """
    Record model performance metrics.

    Calculates and persists standard ML metrics (accuracy, precision, recall, F1, AUC).

    Args:
        request: Performance data including predictions and actuals
        async_mode: If True, processes asynchronously via Celery

    Returns:
        Recorded performance metrics
    """
    from src.services.performance_tracking import record_model_performance
    from src.tasks.drift_monitoring_tasks import track_model_performance

    logger.info(f"Recording performance for model: {request.model_id}")

    if async_mode:
        # Queue Celery task
        track_model_performance.delay(
            model_id=request.model_id,
            predictions=request.predictions,
            actuals=request.actuals,
            prediction_scores=request.prediction_scores,
        )

        return PerformanceRecordResponse(
            model_id=request.model_id,
            recorded_at=datetime.now(timezone.utc),
            sample_size=len(request.predictions),
            metrics={},
            alerts_generated=0,
        )
    else:
        try:
            result = await record_model_performance(
                model_version=request.model_id,
                predictions=request.predictions,
                actuals=request.actuals,
                prediction_scores=request.prediction_scores,
            )

            return PerformanceRecordResponse(
                model_id=request.model_id,
                recorded_at=datetime.fromisoformat(result["recorded_at"]),
                sample_size=result["sample_size"],
                metrics=result["metrics"],
                alerts_generated=0,
            )
        except Exception as e:
            logger.error(f"Failed to record performance: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/performance/{model_id}/trend",
    response_model=PerformanceTrendResponse,
    summary="Get performance trend",
    operation_id="get_model_performance_trend",
)
async def get_performance_trend(
    model_id: str,
    metric_name: str = Query(default="accuracy", description="Metric to analyze"),
    days: int = Query(default=30, ge=1, le=90, description="Days of history"),
) -> PerformanceTrendResponse:
    """
    Get performance trend for a model.

    Analyzes metric trends over time and detects degradation.

    Args:
        model_id: Model version/ID
        metric_name: Metric to analyze (accuracy, precision, recall, f1_score, auc_roc)
        days: Number of days to look back

    Returns:
        Performance trend analysis
    """
    from src.repositories.drift_monitoring import PerformanceMetricRepository
    from src.services.performance_tracking import get_performance_tracker

    try:
        tracker = get_performance_tracker()
        trend = await tracker.get_performance_trend(model_id, metric_name)

        # Get historical values
        repo = PerformanceMetricRepository()
        records = await repo.get_metric_trend(model_id, metric_name, days=days)

        history = [
            PerformanceMetricItem(
                metric_name=r.metric_name,
                metric_value=r.metric_value,
                recorded_at=r.recorded_at,
            )
            for r in records
        ]

        return PerformanceTrendResponse(
            model_id=model_id,
            metric_name=metric_name,
            current_value=trend.current_value,
            baseline_value=trend.baseline_value,
            change_percent=trend.change_percent,
            trend=trend.trend,
            is_significant=trend.is_significant,
            alert_threshold_breached=trend.alert_threshold_breached,
            history=history,
        )

    except Exception as e:
        logger.error(f"Failed to get performance trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/performance/{model_id}/alerts",
    response_model=PerformanceAlertsResponse,
    summary="Get performance alerts",
    operation_id="get_model_performance_alerts",
)
async def get_performance_alerts(model_id: str) -> PerformanceAlertsResponse:
    """
    Check for performance-related alerts.

    Analyzes all tracked metrics for degradation.

    Args:
        model_id: Model version/ID

    Returns:
        List of performance alerts
    """
    from src.services.performance_tracking import get_performance_tracker

    try:
        tracker = get_performance_tracker()
        alerts = await tracker.check_performance_alerts(model_id)

        items = [
            PerformanceAlertItem(
                metric_name=a["metric_name"],
                current_value=a["current_value"],
                baseline_value=a["baseline_value"],
                change_percent=a["change_percent"],
                trend=a["trend"],
                severity=a["severity"],
                message=a["message"],
            )
            for a in alerts
        ]

        return PerformanceAlertsResponse(
            model_id=model_id,
            alert_count=len(items),
            alerts=items,
        )

    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/performance/{model_id}/compare/{other_model_id}",
    summary="Compare model performance",
    operation_id="compare_model_performance",
)
async def compare_model_performance(
    model_id: str,
    other_model_id: str,
    metric_name: str = Query(default="accuracy", description="Metric to compare"),
) -> Dict[str, Any]:
    """
    Compare performance between two model versions.

    Args:
        model_id: First model version
        other_model_id: Second model version
        metric_name: Metric to compare

    Returns:
        Comparison results
    """
    from src.services.performance_tracking import get_performance_tracker

    try:
        tracker = get_performance_tracker()
        result = await tracker.compare_model_versions(model_id, other_model_id, metric_name)

        return result

    except Exception as e:
        logger.error(f"Failed to compare model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PRODUCTION MODEL SWEEP
# =============================================================================


@router.post(
    "/sweep/production",
    summary="Trigger production model sweep",
    operation_id="trigger_production_model_sweep",
)
async def trigger_production_sweep(
    time_window: str = Query(default="7d", description="Time window for comparison"),
) -> Dict[str, Any]:
    """
    Trigger drift detection sweep for all production models.

    Queues Celery tasks for each production model.

    Args:
        time_window: Time window for drift comparison

    Returns:
        Summary of queued tasks
    """
    from src.tasks.drift_monitoring_tasks import check_all_production_models

    logger.info("Production model sweep requested")

    task = check_all_production_models.delay(time_window=time_window)

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Production model sweep queued",
        "time_window": time_window,
    }


# =============================================================================
# RETRAINING TRIGGER ENDPOINTS
# =============================================================================


class TriggerReasonEnum(str, Enum):
    """Reasons for triggering retraining."""

    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class RetrainingStatusEnum(str, Enum):
    """Status of retraining job."""

    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class TriggerRetrainingRequest(BaseModel):
    """Request to trigger model retraining."""

    reason: TriggerReasonEnum = Field(..., description="Reason for retraining")
    notes: Optional[str] = Field(None, description="Additional notes")
    auto_approve: bool = Field(default=False, description="Auto-approve retraining")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reason": "data_drift",
                "notes": "Significant drift detected in key features",
                "auto_approve": False,
            }
        }
    )


class CompleteRetrainingRequest(BaseModel):
    """Request to mark retraining as complete."""

    performance_after: float = Field(..., description="Performance metric after retraining")
    success: bool = Field(default=True, description="Whether retraining was successful")
    notes: Optional[str] = Field(None, description="Additional notes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "performance_after": 0.92,
                "success": True,
                "notes": "Model retrained with expanded dataset",
            }
        }
    )


class RollbackRetrainingRequest(BaseModel):
    """Request to rollback a retraining."""

    reason: str = Field(..., description="Reason for rollback")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reason": "Performance degradation on validation set",
            }
        }
    )


class RetrainingDecisionResponse(BaseModel):
    """Response for retraining evaluation."""

    model_id: str
    should_retrain: bool
    confidence: float
    reasons: List[str]
    trigger_factors: Dict[str, Any]
    cooldown_active: bool
    cooldown_ends_at: Optional[datetime] = None
    recommended_action: str


class RetrainingJobResponse(BaseModel):
    """Response for retraining job."""

    job_id: str
    model_version: str
    status: RetrainingStatusEnum
    trigger_reason: str
    triggered_at: datetime
    triggered_by: str
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    performance_before: Optional[float] = None
    performance_after: Optional[float] = None
    notes: Optional[str] = None


@router.post(
    "/retraining/evaluate/{model_id}",
    response_model=RetrainingDecisionResponse,
    summary="Evaluate retraining need",
    operation_id="evaluate_model_retraining_need",
)
async def evaluate_retraining_need(model_id: str) -> RetrainingDecisionResponse:
    """
    Evaluate whether a model needs retraining.

    Analyzes drift scores, performance trends, and other factors.

    Args:
        model_id: Model version/ID

    Returns:
        Retraining decision with reasoning
    """
    from src.services.retraining_trigger import get_retraining_trigger_service

    try:
        service = get_retraining_trigger_service()
        decision = await service.evaluate_retraining_need(model_id)

        return RetrainingDecisionResponse(
            model_id=model_id,
            should_retrain=decision.should_retrain,
            confidence=decision.confidence,
            reasons=decision.reasons,
            trigger_factors=decision.trigger_factors,
            cooldown_active=decision.cooldown_active,
            cooldown_ends_at=decision.cooldown_ends_at,
            recommended_action=decision.recommended_action,
        )

    except Exception as e:
        logger.error(f"Failed to evaluate retraining need: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retraining/trigger/{model_id}",
    response_model=RetrainingJobResponse,
    summary="Trigger model retraining",
    operation_id="trigger_model_retraining",
)
async def trigger_retraining(
    model_id: str,
    request: TriggerRetrainingRequest,
    triggered_by: str = Query(default="api_user", description="User triggering retraining"),
    _admin: dict = Depends(require_admin),
) -> RetrainingJobResponse:
    """
    Trigger model retraining.

    Creates a retraining job and optionally auto-approves it.

    Args:
        model_id: Model version/ID
        request: Retraining parameters
        triggered_by: User or system triggering retraining

    Returns:
        Created retraining job
    """
    from src.services.retraining_trigger import (
        TriggerReason,
        get_retraining_trigger_service,
    )

    try:
        service = get_retraining_trigger_service()

        # Map enum
        reason = TriggerReason(request.reason.value)

        job = await service.trigger_retraining(
            model_version=model_id,
            reason=reason,
            triggered_by=triggered_by,
            notes=request.notes,
            auto_approve=request.auto_approve,
        )

        return RetrainingJobResponse(
            job_id=job.job_id,
            model_version=job.model_version,
            status=RetrainingStatusEnum(job.status.value),
            trigger_reason=job.trigger_reason.value,
            triggered_at=job.triggered_at,
            triggered_by=job.triggered_by,
            approved_at=job.approved_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            performance_before=job.performance_before,
            performance_after=job.performance_after,
            notes=job.notes,
        )

    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/retraining/status/{job_id}",
    response_model=RetrainingJobResponse,
    summary="Get retraining job status",
    operation_id="get_retraining_job_status",
)
async def get_retraining_status(job_id: str) -> RetrainingJobResponse:
    """
    Get status of a retraining job.

    Args:
        job_id: Retraining job ID

    Returns:
        Retraining job details
    """
    from src.services.retraining_trigger import get_retraining_trigger_service

    try:
        service = get_retraining_trigger_service()
        job = await service.get_retraining_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Retraining job {job_id} not found")

        return RetrainingJobResponse(
            job_id=job.job_id,
            model_version=job.model_version,
            status=RetrainingStatusEnum(job.status.value),
            trigger_reason=job.trigger_reason.value,
            triggered_at=job.triggered_at,
            triggered_by=job.triggered_by,
            approved_at=job.approved_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            performance_before=job.performance_before,
            performance_after=job.performance_after,
            notes=job.notes,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get retraining status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retraining/{job_id}/complete",
    response_model=RetrainingJobResponse,
    summary="Complete retraining job",
    operation_id="complete_retraining_job",
)
async def complete_retraining(
    job_id: str,
    request: CompleteRetrainingRequest,
) -> RetrainingJobResponse:
    """
    Mark a retraining job as complete.

    Args:
        job_id: Retraining job ID
        request: Completion details

    Returns:
        Updated retraining job
    """
    from src.services.retraining_trigger import get_retraining_trigger_service

    try:
        service = get_retraining_trigger_service()
        job = await service.complete_retraining(
            job_id=job_id,
            performance_after=request.performance_after,
            success=request.success,
        )

        if not job:
            raise HTTPException(status_code=404, detail=f"Retraining job {job_id} not found")

        return RetrainingJobResponse(
            job_id=job.job_id,
            model_version=job.model_version,
            status=RetrainingStatusEnum(job.status.value),
            trigger_reason=job.trigger_reason.value,
            triggered_at=job.triggered_at,
            triggered_by=job.triggered_by,
            approved_at=job.approved_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            performance_before=job.performance_before,
            performance_after=job.performance_after,
            notes=job.notes,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retraining/{job_id}/rollback",
    response_model=RetrainingJobResponse,
    summary="Rollback retraining job",
    operation_id="rollback_retraining_job",
)
async def rollback_retraining(
    job_id: str,
    request: RollbackRetrainingRequest,
    _admin: dict = Depends(require_admin),
) -> RetrainingJobResponse:
    """
    Rollback a completed retraining.

    Reverts to previous model version.

    Args:
        job_id: Retraining job ID
        request: Rollback reason

    Returns:
        Updated retraining job
    """
    from src.services.retraining_trigger import get_retraining_trigger_service

    try:
        service = get_retraining_trigger_service()
        job = await service.rollback_retraining(
            job_id=job_id,
            reason=request.reason,
        )

        if not job:
            raise HTTPException(status_code=404, detail=f"Retraining job {job_id} not found")

        return RetrainingJobResponse(
            job_id=job.job_id,
            model_version=job.model_version,
            status=RetrainingStatusEnum(job.status.value),
            trigger_reason=job.trigger_reason.value,
            triggered_at=job.triggered_at,
            triggered_by=job.triggered_by,
            approved_at=job.approved_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            performance_before=job.performance_before,
            performance_after=job.performance_after,
            notes=job.notes,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rollback retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/retraining/sweep",
    summary="Evaluate all models for retraining",
    operation_id="trigger_retraining_sweep",
)
async def trigger_retraining_sweep(
    _admin: dict = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Evaluate retraining need for all production models.

    Queues evaluation tasks for each production model.

    Returns:
        Summary of queued tasks
    """
    from src.tasks.drift_monitoring_tasks import check_retraining_for_all_models

    logger.info("Retraining sweep requested")

    task = check_retraining_for_all_models.delay()

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Retraining evaluation sweep queued for all production models",
    }
