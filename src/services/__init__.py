"""
E2I Services Package.

Provides business logic and calculation services.

Services:
- ROICalculationService: ROI calculations for gap analysis
- AlertRouter: Alert routing for drift monitoring (Phase 14)
- PerformanceTracker: Model performance tracking (Phase 14)
"""

# Alert Routing (Phase 14)
from src.services.alert_routing import (
    AlertPayload,
    AlertRouter,
    AlertRoutingConfig,
    NotificationChannel,
    get_alert_router,
    route_drift_alerts,
)

# Performance Tracking (Phase 14)
from src.services.performance_tracking import (
    PerformanceSnapshot,
    PerformanceTracker,
    PerformanceTrackingConfig,
    PerformanceTrend,
    get_performance_tracker,
    record_model_performance,
)

# Retraining Trigger (Phase 14)
from src.services.retraining_trigger import (
    RetrainingDecision,
    RetrainingJob,
    RetrainingStatus,
    RetrainingTriggerConfig,
    RetrainingTriggerService,
    TriggerReason,
    evaluate_and_trigger_retraining,
    get_retraining_trigger_service,
)
from src.services.roi_calculation import (
    AttributionLevel,
    ConfidenceInterval,
    CostInput,
    RiskAssessment,
    RiskLevel,
    ROICalculationService,
    ROIResult,
    ValueDriverInput,
    ValueDriverType,
)

__all__ = [
    # ROI Calculation
    "ROICalculationService",
    "ValueDriverInput",
    "CostInput",
    "RiskAssessment",
    "ConfidenceInterval",
    "ROIResult",
    "ValueDriverType",
    "AttributionLevel",
    "RiskLevel",
    # Alert Routing
    "AlertPayload",
    "AlertRouter",
    "AlertRoutingConfig",
    "NotificationChannel",
    "get_alert_router",
    "route_drift_alerts",
    # Performance Tracking
    "PerformanceSnapshot",
    "PerformanceTrend",
    "PerformanceTracker",
    "PerformanceTrackingConfig",
    "get_performance_tracker",
    "record_model_performance",
    # Retraining Trigger
    "TriggerReason",
    "RetrainingStatus",
    "RetrainingTriggerConfig",
    "RetrainingDecision",
    "RetrainingJob",
    "RetrainingTriggerService",
    "get_retraining_trigger_service",
    "evaluate_and_trigger_retraining",
]
