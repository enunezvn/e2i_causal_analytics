"""
E2I Health Score Agent - Pipeline Health Node
Version: 4.2
Purpose: Check health of data pipelines
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from ..state import HealthScoreState, PipelineStatus
from ..metrics import DEFAULT_THRESHOLDS

logger = logging.getLogger(__name__)


class PipelineStore(Protocol):
    """Protocol for pipeline status storage"""

    async def get_all_pipelines(self) -> List[str]:
        """Get list of all pipeline names"""
        ...

    async def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get status for a specific pipeline"""
        ...


class PipelineHealthNode:
    """
    Check health of data pipelines.
    Monitors freshness, success rates, and processing metrics.
    """

    def __init__(
        self,
        pipeline_store: Optional[PipelineStore] = None,
        max_freshness_hours: float = DEFAULT_THRESHOLDS.max_freshness_hours,
        stale_threshold_hours: float = DEFAULT_THRESHOLDS.stale_threshold_hours,
    ):
        """
        Initialize pipeline health node.

        Args:
            pipeline_store: Store for pipeline status
            max_freshness_hours: Hours after which data is considered failed
            stale_threshold_hours: Hours after which data is considered stale
        """
        self.pipeline_store = pipeline_store
        self.max_freshness_hours = max_freshness_hours
        self.stale_threshold_hours = stale_threshold_hours

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Execute pipeline health checks."""
        start_time = time.time()

        # Skip if scope doesn't include pipelines
        if state.get("check_scope") not in ["full", "pipelines"]:
            logger.debug("Skipping pipeline health for non-pipeline scope")
            return {
                **state,
                "pipeline_statuses": [],
                "pipeline_health_score": 1.0,
            }

        try:
            if self.pipeline_store:
                # Fetch all pipelines
                pipelines = await self.pipeline_store.get_all_pipelines()

                # Fetch status for each pipeline in parallel
                if pipelines:
                    tasks = [
                        self._get_pipeline_status(name) for name in pipelines
                    ]
                    statuses = await asyncio.gather(*tasks)
                else:
                    statuses = []
            else:
                # No store - return empty for testing
                statuses = []

            # Calculate overall pipeline health
            if statuses:
                healthy = sum(1 for s in statuses if s["status"] == "healthy")
                stale = sum(1 for s in statuses if s["status"] == "stale")
                # Healthy = 1.0, Stale = 0.5, Failed = 0.0
                total_score = healthy + (stale * 0.5)
                health_score = total_score / len(statuses)
            else:
                health_score = 1.0  # No pipelines = healthy by default

            check_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Pipeline health check complete: {len(statuses)} pipelines, "
                f"score={health_score:.2f}, duration={check_time}ms"
            )

            return {
                **state,
                "pipeline_statuses": statuses,
                "pipeline_health_score": health_score,
                "check_latency_ms": state.get("check_latency_ms", 0) + check_time,
            }

        except Exception as e:
            logger.error(f"Pipeline health check failed: {e}")
            return {
                **state,
                "errors": [{"node": "pipeline_health", "error": str(e)}],
                "pipeline_health_score": 0.5,  # Unknown = degraded
                "pipeline_statuses": [],
            }

    async def _get_pipeline_status(self, pipeline_name: str) -> PipelineStatus:
        """Get status for a single pipeline."""
        try:
            status = await self.pipeline_store.get_pipeline_status(pipeline_name)

            # Calculate freshness
            last_success = status.get("last_success")
            if last_success:
                try:
                    last_success_dt = datetime.fromisoformat(
                        last_success.replace("Z", "+00:00")
                    )
                    # Ensure timezone-aware comparison
                    if last_success_dt.tzinfo is None:
                        last_success_dt = last_success_dt.replace(tzinfo=timezone.utc)
                    freshness_hours = (
                        datetime.now(timezone.utc) - last_success_dt
                    ).total_seconds() / 3600
                except (ValueError, AttributeError):
                    freshness_hours = float("inf")
            else:
                freshness_hours = float("inf")

            # Determine status
            if status.get("failed", False) or freshness_hours > self.max_freshness_hours:
                pipeline_status = "failed"
            elif freshness_hours > self.stale_threshold_hours:
                pipeline_status = "stale"
            else:
                pipeline_status = "healthy"

            return PipelineStatus(
                pipeline_name=pipeline_name,
                last_run=status.get("last_run", ""),
                last_success=status.get("last_success", ""),
                rows_processed=status.get("rows_processed", 0),
                freshness_hours=freshness_hours
                if freshness_hours != float("inf")
                else -1,
                status=pipeline_status,
            )

        except Exception as e:
            logger.warning(f"Failed to get status for pipeline {pipeline_name}: {e}")
            return PipelineStatus(
                pipeline_name=pipeline_name,
                last_run="",
                last_success="",
                rows_processed=0,
                freshness_hours=-1,
                status="failed",
            )
