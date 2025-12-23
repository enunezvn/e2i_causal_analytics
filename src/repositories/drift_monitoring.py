"""
Drift monitoring repository for persisting drift detection results.

This repository handles CRUD operations for:
- ml_drift_history: Individual drift detection results
- ml_monitoring_alerts: Alerts generated from drift detection
- ml_monitoring_runs: Monitoring run metadata
- ml_performance_metrics: Model performance metrics over time
- ml_retraining_history: Model retraining events

Tables: database/ml/017_model_monitoring_tables.sql
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.repositories.base import BaseRepository


# Pydantic models for type safety
class DriftHistoryRecord(BaseModel):
    """Record for ml_drift_history table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str
    feature_name: str
    drift_type: str  # data, model, concept
    test_statistic: float
    p_value: float
    drift_detected: bool
    severity: str  # none, low, medium, high, critical
    baseline_start: datetime
    baseline_end: datetime
    current_start: datetime
    current_end: datetime
    sample_size_baseline: int = 0
    sample_size_current: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MonitoringAlertRecord(BaseModel):
    """Record for ml_monitoring_alerts table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str
    alert_type: str  # data_drift, model_drift, concept_drift, performance_degradation
    severity: str  # warning, critical
    message: str
    affected_features: List[str] = Field(default_factory=list)
    recommended_action: str
    status: str = "active"  # active, acknowledged, investigating, resolved, dismissed
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class MonitoringRunRecord(BaseModel):
    """Record for ml_monitoring_runs table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str
    run_type: str = "scheduled"  # scheduled, manual, triggered
    status: str = "running"  # running, completed, failed
    features_checked: int = 0
    drift_detected_count: int = 0
    alerts_generated: int = 0
    duration_ms: int = 0
    config: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class PerformanceMetricRecord(BaseModel):
    """Record for ml_performance_metrics table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str
    metric_name: str
    metric_value: float
    sample_size: int
    window_start: datetime
    window_end: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrainingHistoryRecord(BaseModel):
    """Record for ml_retraining_history table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    old_model_version: str
    new_model_version: str
    trigger_reason: str
    drift_score_before: float
    performance_before: float
    performance_after: Optional[float] = None
    training_config: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # pending, training, completed, failed, rolled_back
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class DriftHistoryRepository(BaseRepository[DriftHistoryRecord]):
    """Repository for drift detection history."""

    table_name = "ml_drift_history"
    model_class = DriftHistoryRecord

    async def record_drift_results(
        self,
        model_version: str,
        drift_results: List[Dict[str, Any]],
        baseline_window: Dict[str, datetime],
        current_window: Dict[str, datetime],
    ) -> List[DriftHistoryRecord]:
        """
        Record multiple drift detection results.

        Args:
            model_version: Model version being monitored
            drift_results: List of drift detection results
            baseline_window: Baseline period timestamps
            current_window: Current period timestamps

        Returns:
            List of created records
        """
        if not self.client or not drift_results:
            return []

        records = []
        for result in drift_results:
            record = DriftHistoryRecord(
                model_version=model_version,
                feature_name=result.get("feature", "unknown"),
                drift_type=result.get("drift_type", "data"),
                test_statistic=result.get("test_statistic", 0.0),
                p_value=result.get("p_value", 1.0),
                drift_detected=result.get("drift_detected", False),
                severity=result.get("severity", "none"),
                baseline_start=baseline_window.get("start", datetime.now(timezone.utc)),
                baseline_end=baseline_window.get("end", datetime.now(timezone.utc)),
                current_start=current_window.get("start", datetime.now(timezone.utc)),
                current_end=current_window.get("end", datetime.now(timezone.utc)),
                metadata=result.get("metadata", {}),
            )
            records.append(record)

        # Batch insert
        data = [r.model_dump() for r in records]
        for item in data:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()

        await self.client.table(self.table_name).insert(data).execute()

        return records

    async def get_latest_drift_status(
        self,
        model_version: str,
        limit: int = 50,
    ) -> List[DriftHistoryRecord]:
        """
        Get latest drift status for a model.

        Uses the ml_drift_status_latest view for efficient retrieval.

        Args:
            model_version: Model version to check
            limit: Maximum records to return

        Returns:
            List of latest drift records per feature
        """
        if not self.client:
            return []

        result = await (
            self.client.table("ml_drift_status_latest")
            .select("*")
            .eq("model_version", model_version)
            .limit(limit)
            .execute()
        )

        return [self._to_model(row) for row in result.data]

    async def get_drift_trend(
        self,
        model_version: str,
        feature_name: str,
        days: int = 7,
    ) -> List[DriftHistoryRecord]:
        """
        Get drift trend for a specific feature.

        Args:
            model_version: Model version
            feature_name: Feature to analyze
            days: Number of days to look back

        Returns:
            List of drift records for trend analysis
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("model_version", model_version)
            .eq("feature_name", feature_name)
            .gte("detected_at", f"now() - interval '{days} days'")
            .order("detected_at", desc=True)
            .execute()
        )

        return [self._to_model(row) for row in result.data]


class MonitoringAlertRepository(BaseRepository[MonitoringAlertRecord]):
    """Repository for monitoring alerts."""

    table_name = "ml_monitoring_alerts"
    model_class = MonitoringAlertRecord

    async def create_alerts_from_drift(
        self,
        model_version: str,
        drift_results: List[Dict[str, Any]],
    ) -> List[MonitoringAlertRecord]:
        """
        Create alerts from drift detection results.

        Generates alerts for critical and high severity drifts.

        Args:
            model_version: Model version
            drift_results: Drift detection results

        Returns:
            List of created alerts
        """
        if not self.client:
            return []

        # Group by drift type and severity
        critical_by_type: Dict[str, List[str]] = {}
        high_by_type: Dict[str, List[str]] = {}

        for result in drift_results:
            drift_type = result.get("drift_type", "data")
            severity = result.get("severity", "none")
            feature = result.get("feature", "unknown")

            if severity == "critical":
                if drift_type not in critical_by_type:
                    critical_by_type[drift_type] = []
                critical_by_type[drift_type].append(feature)
            elif severity == "high":
                if drift_type not in high_by_type:
                    high_by_type[drift_type] = []
                high_by_type[drift_type].append(feature)

        alerts = []

        # Create critical alerts
        for drift_type, features in critical_by_type.items():
            alert = MonitoringAlertRecord(
                model_version=model_version,
                alert_type=f"{drift_type}_drift",
                severity="critical",
                message=f"CRITICAL {drift_type} drift detected in: {', '.join(features[:5])}",
                affected_features=features,
                recommended_action=self._get_recommendation(drift_type, "critical"),
            )
            alerts.append(alert)

        # Create warning alerts
        for drift_type, features in high_by_type.items():
            alert = MonitoringAlertRecord(
                model_version=model_version,
                alert_type=f"{drift_type}_drift",
                severity="warning",
                message=f"HIGH {drift_type} drift detected in: {', '.join(features[:5])}",
                affected_features=features,
                recommended_action=self._get_recommendation(drift_type, "warning"),
            )
            alerts.append(alert)

        if alerts:
            data = [a.model_dump() for a in alerts]
            for item in data:
                for key, value in item.items():
                    if isinstance(value, datetime):
                        item[key] = value.isoformat()
                    elif value is None:
                        item[key] = None

            await self.client.table(self.table_name).insert(data).execute()

        return alerts

    async def get_active_alerts(
        self,
        model_version: Optional[str] = None,
        limit: int = 100,
    ) -> List[MonitoringAlertRecord]:
        """
        Get all active (unresolved) alerts.

        Args:
            model_version: Optional filter by model
            limit: Maximum records

        Returns:
            List of active alerts
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("status", "active")
            .order("triggered_at", desc=True)
            .limit(limit)
        )

        if model_version:
            query = query.eq("model_version", model_version)

        result = await query.execute()

        return [self._to_model(row) for row in result.data]

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> Optional[MonitoringAlertRecord]:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert UUID
            acknowledged_by: User who acknowledged

        Returns:
            Updated alert or None
        """
        return await self.update(
            alert_id,
            {
                "status": "acknowledged",
            },
        )

    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
    ) -> Optional[MonitoringAlertRecord]:
        """
        Resolve an alert.

        Args:
            alert_id: Alert UUID
            resolved_by: User who resolved

        Returns:
            Updated alert or None
        """
        return await self.update(
            alert_id,
            {
                "status": "resolved",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "resolved_by": resolved_by,
            },
        )

    def _get_recommendation(self, drift_type: str, severity: str) -> str:
        """Get recommended action based on drift type and severity."""
        recommendations = {
            ("data", "critical"): "Immediate action required: Retrain model with recent data",
            ("data", "warning"): "Monitor closely: Schedule retraining if drift persists",
            ("model", "critical"): "Immediate action required: Investigate model degradation",
            ("model", "warning"): "Monitor closely: Check prediction accuracy",
            ("concept", "critical"): "Immediate action required: Review feature-target relationships",
            ("concept", "warning"): "Monitor closely: Validate model on current data",
        }
        return recommendations.get((drift_type, severity), "Review drift detection results")


class MonitoringRunRepository(BaseRepository[MonitoringRunRecord]):
    """Repository for monitoring run metadata."""

    table_name = "ml_monitoring_runs"
    model_class = MonitoringRunRecord

    async def start_run(
        self,
        model_version: str,
        run_type: str = "scheduled",
        config: Optional[Dict[str, Any]] = None,
    ) -> MonitoringRunRecord:
        """
        Start a new monitoring run.

        Args:
            model_version: Model being monitored
            run_type: Type of run (scheduled, manual, triggered)
            config: Run configuration

        Returns:
            Created run record
        """
        record = MonitoringRunRecord(
            model_version=model_version,
            run_type=run_type,
            config=config or {},
        )

        if self.client:
            data = record.model_dump()
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                elif value is None:
                    data[key] = None

            await self.client.table(self.table_name).insert(data).execute()

        return record

    async def complete_run(
        self,
        run_id: str,
        features_checked: int,
        drift_detected_count: int,
        alerts_generated: int,
        duration_ms: int,
        error_message: Optional[str] = None,
    ) -> Optional[MonitoringRunRecord]:
        """
        Complete a monitoring run.

        Args:
            run_id: Run UUID
            features_checked: Number of features checked
            drift_detected_count: Number of features with drift
            alerts_generated: Number of alerts created
            duration_ms: Run duration in milliseconds
            error_message: Optional error message if failed

        Returns:
            Updated run record
        """
        status = "completed" if error_message is None else "failed"

        return await self.update(
            run_id,
            {
                "status": status,
                "features_checked": features_checked,
                "drift_detected_count": drift_detected_count,
                "alerts_generated": alerts_generated,
                "duration_ms": duration_ms,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error_message": error_message,
            },
        )

    async def get_recent_runs(
        self,
        model_version: Optional[str] = None,
        limit: int = 10,
    ) -> List[MonitoringRunRecord]:
        """
        Get recent monitoring runs.

        Args:
            model_version: Optional filter by model
            limit: Maximum records

        Returns:
            List of recent runs
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .order("started_at", desc=True)
            .limit(limit)
        )

        if model_version:
            query = query.eq("model_version", model_version)

        result = await query.execute()

        return [self._to_model(row) for row in result.data]


class PerformanceMetricRepository(BaseRepository[PerformanceMetricRecord]):
    """Repository for model performance metrics."""

    table_name = "ml_performance_metrics"
    model_class = PerformanceMetricRecord

    async def record_metrics(
        self,
        model_version: str,
        metrics: Dict[str, float],
        sample_size: int,
        window_start: datetime,
        window_end: datetime,
    ) -> List[PerformanceMetricRecord]:
        """
        Record performance metrics for a model.

        Args:
            model_version: Model version
            metrics: Metric name -> value mapping
            sample_size: Number of samples used
            window_start: Evaluation window start
            window_end: Evaluation window end

        Returns:
            List of created records
        """
        if not self.client or not metrics:
            return []

        records = []
        for metric_name, metric_value in metrics.items():
            record = PerformanceMetricRecord(
                model_version=model_version,
                metric_name=metric_name,
                metric_value=metric_value,
                sample_size=sample_size,
                window_start=window_start,
                window_end=window_end,
            )
            records.append(record)

        data = [r.model_dump() for r in records]
        for item in data:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()

        await self.client.table(self.table_name).insert(data).execute()

        return records

    async def get_metric_trend(
        self,
        model_version: str,
        metric_name: str,
        days: int = 30,
    ) -> List[PerformanceMetricRecord]:
        """
        Get metric trend over time.

        Args:
            model_version: Model version
            metric_name: Metric to retrieve
            days: Number of days to look back

        Returns:
            List of metric records
        """
        if not self.client:
            return []

        result = await (
            self.client.table(self.table_name)
            .select("*")
            .eq("model_version", model_version)
            .eq("metric_name", metric_name)
            .gte("recorded_at", f"now() - interval '{days} days'")
            .order("recorded_at", desc=True)
            .execute()
        )

        return [self._to_model(row) for row in result.data]


class RetrainingHistoryRepository(BaseRepository[RetrainingHistoryRecord]):
    """Repository for model retraining history."""

    table_name = "ml_retraining_history"
    model_class = RetrainingHistoryRecord

    async def trigger_retraining(
        self,
        old_model_version: str,
        new_model_version: str,
        trigger_reason: str,
        drift_score_before: float,
        performance_before: float,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> RetrainingHistoryRecord:
        """
        Record a retraining trigger.

        Args:
            old_model_version: Current model version
            new_model_version: New model version being trained
            trigger_reason: Why retraining was triggered
            drift_score_before: Drift score before retraining
            performance_before: Performance metric before retraining
            training_config: Training configuration

        Returns:
            Created retraining record
        """
        record = RetrainingHistoryRecord(
            old_model_version=old_model_version,
            new_model_version=new_model_version,
            trigger_reason=trigger_reason,
            drift_score_before=drift_score_before,
            performance_before=performance_before,
            training_config=training_config or {},
            status="pending",
        )

        if self.client:
            data = record.model_dump()
            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                elif value is None:
                    data[key] = None

            await self.client.table(self.table_name).insert(data).execute()

        return record

    async def complete_retraining(
        self,
        record_id: str,
        performance_after: float,
        success: bool = True,
    ) -> Optional[RetrainingHistoryRecord]:
        """
        Complete a retraining run.

        Args:
            record_id: Retraining record UUID
            performance_after: Performance after retraining
            success: Whether retraining was successful

        Returns:
            Updated record
        """
        status = "completed" if success else "failed"

        return await self.update(
            record_id,
            {
                "status": status,
                "performance_after": performance_after,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def rollback_retraining(
        self,
        record_id: str,
    ) -> Optional[RetrainingHistoryRecord]:
        """
        Mark retraining as rolled back.

        Args:
            record_id: Retraining record UUID

        Returns:
            Updated record
        """
        return await self.update(
            record_id,
            {
                "status": "rolled_back",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
