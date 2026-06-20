"""Model Performance Tracking Service.

Phase 14: Model Monitoring & Drift Detection

Tracks model performance metrics over time:
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Custom business metrics
- Performance degradation detection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""

    model_version: str
    recorded_at: datetime
    sample_size: int
    window_start: datetime
    window_end: datetime
    metrics: Dict[str, float]
    segments: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""

    model_version: str
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    trend: str  # improving, stable, degrading
    is_significant: bool
    alert_threshold_breached: bool
    # The metric level below which an alert fires — the higher of the relative
    # floor (baseline * (1 - degradation_threshold)) and the absolute floor
    # (absolute_min_accuracy). 0.0 when there is no data. Surfaced so the UI can
    # plot it as the trend chart's alert-threshold line.
    alert_threshold: float = 0.0


@dataclass
class PerformanceTrackingConfig:
    """Configuration for performance tracking."""

    # Metrics to track
    tracked_metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
        ]
    )

    # Thresholds
    degradation_threshold: float = 0.1  # 10% relative drop
    absolute_min_accuracy: float = 0.5  # Minimum acceptable accuracy
    trend_window_days: int = 365
    min_samples: int = 100

    # Comparison settings
    baseline_window_days: int = 7
    current_window_days: int = 1


class PerformanceTracker:
    """Service for tracking model performance over time."""

    def __init__(self, config: Optional[PerformanceTrackingConfig] = None):
        self.config: PerformanceTrackingConfig = config or PerformanceTrackingConfig()

    async def record_performance(
        self,
        model_version: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prediction_scores: Optional[np.ndarray] = None,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
        segments: Optional[Dict[str, np.ndarray]] = None,
    ) -> PerformanceSnapshot:
        """
        Record performance metrics for a model.

        Args:
            model_version: Model version/ID
            predictions: Predicted labels
            actuals: Actual labels
            prediction_scores: Predicted probabilities (for AUC)
            window_start: Evaluation window start
            window_end: Evaluation window end
            segments: Optional segment masks for segmented metrics

        Returns:
            Performance snapshot
        """
        from src.repositories.drift_monitoring import (
            PerformanceMetricRepository,
            get_drift_monitoring_client,
        )

        now = datetime.now(timezone.utc)
        window_end = window_end or now
        window_start = window_start or (now - timedelta(days=1))

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, actuals, prediction_scores)

        # Calculate segmented metrics if provided
        segment_metrics = None
        if segments:
            segment_metrics = {}
            for segment_name, segment_mask in segments.items():
                seg_preds = predictions[segment_mask]
                seg_actuals = actuals[segment_mask]
                seg_scores = (
                    prediction_scores[segment_mask] if prediction_scores is not None else None
                )
                if len(seg_preds) >= self.config.min_samples:
                    segment_metrics[segment_name] = self._calculate_metrics(
                        seg_preds, seg_actuals, seg_scores
                    )

        # Persist metrics
        repo = PerformanceMetricRepository(await get_drift_monitoring_client())
        await repo.record_metrics(
            model_version=model_version,
            metrics=metrics,
            sample_size=len(predictions),
            window_start=window_start,
            window_end=window_end,
        )

        return PerformanceSnapshot(
            model_version=model_version,
            recorded_at=now,
            sample_size=len(predictions),
            window_start=window_start,
            window_end=window_end,
            metrics=metrics,
            segments=segment_metrics,
        )

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics = {}

        # Basic metrics
        try:
            metrics["accuracy"] = float(accuracy_score(actuals, predictions))
        except Exception:
            metrics["accuracy"] = 0.0

        try:
            metrics["precision"] = float(
                precision_score(actuals, predictions, average="weighted", zero_division=0)
            )
        except Exception:
            metrics["precision"] = 0.0

        try:
            metrics["recall"] = float(
                recall_score(actuals, predictions, average="weighted", zero_division=0)
            )
        except Exception:
            metrics["recall"] = 0.0

        try:
            metrics["f1"] = float(
                f1_score(actuals, predictions, average="weighted", zero_division=0)
            )
        except Exception:
            metrics["f1"] = 0.0

        # AUC-ROC (requires probability scores)
        if scores is not None:
            try:
                # Handle binary and multi-class
                unique_classes = np.unique(actuals)
                if len(unique_classes) == 2:
                    metrics["auc_roc"] = float(roc_auc_score(actuals, scores))
                else:
                    metrics["auc_roc"] = float(
                        roc_auc_score(actuals, scores, multi_class="ovr", average="weighted")
                    )
            except Exception:
                metrics["auc_roc"] = 0.0

        return metrics

    async def get_performance_trend(
        self,
        model_version: str,
        metric_name: str = "accuracy",
    ) -> PerformanceTrend:
        """
        Analyze performance trend for a model.

        Args:
            model_version: Model version/ID
            metric_name: Metric to analyze

        Returns:
            Performance trend analysis
        """
        from src.repositories.drift_monitoring import (
            PerformanceMetricRepository,
            get_drift_monitoring_client,
        )

        repo = PerformanceMetricRepository(await get_drift_monitoring_client())

        # Get historical metrics
        records = await repo.get_metric_trend(
            model_version=model_version,
            metric_name=metric_name,
            days=self.config.trend_window_days,
        )

        if not records:
            return PerformanceTrend(
                model_version=model_version,
                metric_name=metric_name,
                current_value=0.0,
                baseline_value=0.0,
                change_percent=0.0,
                trend="unknown",
                is_significant=False,
                alert_threshold_breached=False,
                alert_threshold=0.0,
            )

        # Get current and baseline values
        values = [r.metric_value for r in records]
        current_value = values[0] if values else 0.0

        # Baseline is average of older records
        baseline_values = values[self.config.current_window_days :]
        baseline_value = float(np.mean(baseline_values)) if baseline_values else current_value

        # Calculate change
        if baseline_value > 0:
            change_percent = float((current_value - baseline_value) / baseline_value * 100)
        else:
            change_percent = 0.0

        # Determine trend
        if change_percent > 5:
            trend = "improving"
        elif change_percent < -5:
            trend = "degrading"
        else:
            trend = "stable"

        # Check significance and thresholds
        threshold = float(self.config.degradation_threshold) * 100
        is_significant = abs(change_percent) > threshold
        alert_threshold_breached = change_percent < -threshold or float(current_value) < float(
            self.config.absolute_min_accuracy
        )
        # The accuracy level below which the breach above triggers: the stricter
        # (higher) of the relative floor and the absolute floor. Mirrors the two
        # OR'd conditions so a value at/below this line == breached.
        alert_threshold = max(
            float(baseline_value) * (1.0 - float(self.config.degradation_threshold)),
            float(self.config.absolute_min_accuracy),
        )

        return PerformanceTrend(
            model_version=model_version,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=float(baseline_value),
            change_percent=float(change_percent),
            trend=trend,
            is_significant=is_significant,
            alert_threshold_breached=bool(alert_threshold_breached),
            alert_threshold=float(alert_threshold),
        )

    async def check_performance_alerts(
        self,
        model_version: str,
    ) -> List[Dict[str, Any]]:
        """
        Check for performance-related alerts.

        Args:
            model_version: Model version/ID

        Returns:
            List of alert dictionaries
        """
        from src.memory.services.factories import ServiceConnectionError

        alerts = []

        for metric_name in self.config.tracked_metrics:
            try:
                trend = await self.get_performance_trend(model_version, metric_name)

                if trend.alert_threshold_breached:
                    alerts.append(
                        {
                            "model_version": model_version,
                            "metric_name": metric_name,
                            "current_value": trend.current_value,
                            "baseline_value": trend.baseline_value,
                            "change_percent": trend.change_percent,
                            "trend": trend.trend,
                            "severity": "high" if trend.change_percent < -20 else "medium",
                            "message": f"{metric_name} degraded by {abs(trend.change_percent):.1f}%",
                        }
                    )
            except ServiceConnectionError:
                # Fail-closed (#845): get_performance_trend resolves the Supabase
                # client; an unconfigured/unreachable backend must SURFACE here,
                # not be swallowed into an empty "no alerts" that reads as healthy.
                raise
            except Exception as e:
                logger.warning(f"Failed to check performance for {metric_name}: {e}")

        return alerts

    async def compare_model_versions(
        self,
        model_a: str,
        model_b: str,
        metric_name: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Compare performance between two model versions.

        Args:
            model_a: First model version
            model_b: Second model version
            metric_name: Metric to compare

        Returns:
            Comparison results
        """
        trend_a = await self.get_performance_trend(model_a, metric_name)
        trend_b = await self.get_performance_trend(model_b, metric_name)

        diff = trend_b.current_value - trend_a.current_value
        relative_diff = (diff / trend_a.current_value * 100) if trend_a.current_value > 0 else 0

        return {
            "model_a": {
                "version": model_a,
                "value": trend_a.current_value,
                "trend": trend_a.trend,
            },
            "model_b": {
                "version": model_b,
                "value": trend_b.current_value,
                "trend": trend_b.trend,
            },
            "metric": metric_name,
            "difference": diff,
            "relative_difference_percent": relative_diff,
            "better_model": model_b if diff > 0 else model_a,
            "is_significant": abs(relative_diff) > 5,
        }

    async def get_confusion_matrix(
        self,
        model_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Latest holdout confusion matrix for a model, or None if not recorded.

        Reads the structured artifact the gold-standard eval persists
        (``metric_name='confusion_matrix'``); the counts live in the row's
        ``metadata``. None means "not recorded yet" — an honest empty state, not
        an error (the eval re-run populates it).
        """
        from src.repositories.drift_monitoring import (
            PerformanceMetricRepository,
            get_drift_monitoring_client,
        )

        repo = PerformanceMetricRepository(await get_drift_monitoring_client())
        rec = await repo.get_latest_curve(model_version, "confusion_matrix")
        if rec is None:
            return None
        meta = rec.metadata or {}
        return {
            "tn": int(meta.get("tn", 0)),
            "fp": int(meta.get("fp", 0)),
            "fn": int(meta.get("fn", 0)),
            "tp": int(meta.get("tp", 0)),
            "threshold": float(meta.get("threshold", 0.5)),
            "sample_size": rec.sample_size,
            "measured_at": rec.measured_at,
        }

    async def get_roc_curve(
        self,
        model_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Latest holdout ROC curve for a model, or None if not recorded.

        Reads the structured artifact the gold-standard eval persists
        (``metric_name='roc_curve'``); the (fpr, tpr, threshold) points live in
        the row's ``metadata`` and the AUC is the row's ``metric_value``.
        """
        from src.repositories.drift_monitoring import (
            PerformanceMetricRepository,
            get_drift_monitoring_client,
        )

        repo = PerformanceMetricRepository(await get_drift_monitoring_client())
        rec = await repo.get_latest_curve(model_version, "roc_curve")
        if rec is None:
            return None
        meta = rec.metadata or {}
        return {
            "points": meta.get("points", []),
            "auc": float(rec.metric_value),
            "sample_size": rec.sample_size,
            "measured_at": rec.measured_at,
        }


# =============================================================================
# FACTORY
# =============================================================================


def get_performance_tracker(
    config: Optional[PerformanceTrackingConfig] = None,
) -> PerformanceTracker:
    """Get performance tracker instance."""
    return PerformanceTracker(config)


# =============================================================================
# CELERY TASK INTEGRATION
# =============================================================================


async def record_model_performance(
    model_version: str,
    predictions: List[int],
    actuals: List[int],
    prediction_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Record model performance from a Celery task.

    Args:
        model_version: Model version/ID
        predictions: Predicted labels
        actuals: Actual labels
        prediction_scores: Predicted probabilities

    Returns:
        Performance snapshot as dict
    """
    tracker = get_performance_tracker()

    snapshot = await tracker.record_performance(
        model_version=model_version,
        predictions=np.array(predictions),
        actuals=np.array(actuals),
        prediction_scores=np.array(prediction_scores) if prediction_scores else None,
    )

    return {
        "model_version": snapshot.model_version,
        "recorded_at": snapshot.recorded_at.isoformat(),
        "sample_size": snapshot.sample_size,
        "metrics": snapshot.metrics,
    }
