"""
E2I Health Score Agent - Model Health Node
Version: 4.2
Purpose: Check health of deployed models by aggregating performance metrics
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Protocol, cast

from ..metrics import DEFAULT_THRESHOLDS
from ..state import HealthScoreState, ModelMetrics

logger = logging.getLogger(__name__)


class MetricsStore(Protocol):
    """Protocol for metrics storage"""

    async def get_active_models(self) -> List[str]:
        """Get list of active model IDs"""
        ...

    async def get_model_metrics(self, model_id: str, time_window: str) -> Dict[str, Any]:
        """Get metrics for a specific model"""
        ...


class ModelHealthNode:
    """
    Check health of deployed models.
    Aggregates performance metrics and determines status.
    """

    def __init__(
        self,
        metrics_store: Optional[MetricsStore] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model health node.

        Args:
            metrics_store: Store for model metrics
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.metrics_store = metrics_store
        self.thresholds = thresholds or {
            "min_accuracy": DEFAULT_THRESHOLDS.min_accuracy,
            "min_auc": DEFAULT_THRESHOLDS.min_auc,
            "max_error_rate": DEFAULT_THRESHOLDS.max_error_rate,
            "max_latency_p99_ms": DEFAULT_THRESHOLDS.max_latency_p99_ms,
            "min_predictions_24h": DEFAULT_THRESHOLDS.min_predictions_24h,
        }

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Execute model health checks."""
        start_time = time.time()

        # Skip if scope doesn't include models
        if state.get("check_scope") not in ["full", "models"]:
            logger.debug("Skipping model health for non-model scope")
            return {
                **state,
                "model_metrics": [],
                "model_health_score": 1.0,
            }

        try:
            if self.metrics_store:
                # Fetch all active models
                active_models = await self.metrics_store.get_active_models()

                # Fetch metrics for each model in parallel
                if active_models:
                    tasks = [self._get_model_metrics(model_id) for model_id in active_models]
                    metrics_list = await asyncio.gather(*tasks)
                else:
                    metrics_list = []
            else:
                # No store - return empty for testing
                metrics_list = []

            # Calculate overall model health
            if metrics_list:
                healthy = sum(1 for m in metrics_list if m["status"] == "healthy")
                degraded = sum(1 for m in metrics_list if m["status"] == "degraded")
                total_score = healthy + (degraded * 0.5)
                health_score = total_score / len(metrics_list)
            else:
                health_score = 1.0  # No models = healthy by default

            check_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Model health check complete: {len(metrics_list)} models, "
                f"score={health_score:.2f}, duration={check_time}ms"
            )

            return {
                **state,
                "model_metrics": metrics_list,
                "model_health_score": health_score,
                "total_latency_ms": state.get("total_latency_ms", 0) + check_time,
            }

        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                **state,
                "errors": [{"node": "model_health", "error": str(e)}],
                "model_health_score": 0.5,  # Unknown = degraded
                "model_metrics": [],
            }

    async def _get_model_metrics(self, model_id: str) -> ModelMetrics:
        """Get metrics for a single model."""
        try:
            assert self.metrics_store is not None
            metrics = await self.metrics_store.get_model_metrics(
                model_id=model_id, time_window="24h"
            )

            # Determine health status
            status = self._determine_status(metrics)

            return ModelMetrics(
                model_id=model_id,
                accuracy=metrics.get("accuracy"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1"),
                auc_roc=metrics.get("auc_roc"),
                prediction_latency_p50_ms=metrics.get("latency_p50"),
                prediction_latency_p99_ms=metrics.get("latency_p99"),
                predictions_last_24h=metrics.get("prediction_count", 0),
                error_rate=metrics.get("error_rate", 0),
                status=cast(Literal["healthy", "degraded", "unhealthy"], status),
            )

        except Exception as e:
            logger.warning(f"Failed to get metrics for model {model_id}: {e}")
            return ModelMetrics(
                model_id=model_id,
                accuracy=None,
                precision=None,
                recall=None,
                f1_score=None,
                auc_roc=None,
                prediction_latency_p50_ms=None,
                prediction_latency_p99_ms=None,
                predictions_last_24h=0,
                error_rate=1.0,
                status="unhealthy",
            )

    def _determine_status(self, metrics: Dict[str, Any]) -> str:
        """Determine model health status from metrics."""
        issues = []

        # Check accuracy
        accuracy = metrics.get("accuracy")
        if accuracy is not None and accuracy < self.thresholds["min_accuracy"]:
            issues.append("low_accuracy")

        # Check AUC
        auc = metrics.get("auc_roc")
        if auc is not None and auc < self.thresholds["min_auc"]:
            issues.append("low_auc")

        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.thresholds["max_error_rate"]:
            issues.append("high_error_rate")

        # Check latency
        latency_p99 = metrics.get("latency_p99", 0)
        if latency_p99 > self.thresholds["max_latency_p99_ms"]:
            issues.append("high_latency")

        # Check prediction volume
        pred_count = metrics.get("prediction_count", 0)
        if pred_count < self.thresholds["min_predictions_24h"]:
            issues.append("low_volume")

        # Determine overall status
        if len(issues) >= 2:
            return "unhealthy"
        elif len(issues) == 1:
            return "degraded"
        return "healthy"
