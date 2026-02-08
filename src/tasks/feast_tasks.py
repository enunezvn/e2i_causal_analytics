"""Feast Feature Store Tasks.

Celery tasks for feature materialization and freshness monitoring.
These tasks integrate with the Feast feature store for scheduled operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml  # type: ignore[import-untyped]

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "feast_materialization.yaml"


def load_config() -> Dict[str, Any]:
    """Load Feast materialization configuration."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load Feast config: {e}")
    return {}


def run_async(coro):
    """Helper to run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new loop if current is running
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)


@celery_app.task(bind=True, name="src.tasks.materialize_features")
def materialize_features(
    self,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    feature_views: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Full feature materialization task.

    Args:
        start_date: Start date (ISO format). Defaults to 7 days ago.
        end_date: End date (ISO format). Defaults to now.
        feature_views: Specific feature views to materialize.
        dry_run: If True, validate only without materializing.

    Returns:
        Materialization result dict.
    """
    from scripts.feast_materialize import MaterializationJob

    logger.info(f"Starting full materialization task: {self.request.id}")

    config = load_config()
    default_days_back = config.get("full_materialization", {}).get("default_days_back", 7)

    # Parse dates
    now = datetime.now(timezone.utc)
    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00")) if end_date else now
    start_dt = (
        datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if start_date
        else end_dt - timedelta(days=default_days_back)
    )

    async def run_job():
        job = MaterializationJob()
        try:
            result = await job.run_full_materialization(
                start_date=start_dt,
                end_date=end_dt,
                feature_views=feature_views,
                dry_run=dry_run,
            )
            return result
        finally:
            await job.close()

    result = cast(Dict[str, Any], run_async(run_job()))

    # Log result
    status = result.get("status", "unknown")
    if status == "completed":
        duration = result.get("duration_seconds", 0)
        logger.info(
            f"Full materialization complete: duration={duration:.2f}s, "
            f"views={feature_views or 'all'}"
        )
    elif status == "failed":
        logger.error(f"Full materialization failed: {result.get('error')}")
    else:
        logger.info(f"Full materialization: status={status}")

    return result


@celery_app.task(bind=True, name="src.tasks.materialize_incremental_features")
def materialize_incremental_features(
    self,
    end_date: Optional[str] = None,
    feature_views: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Incremental feature materialization task.

    This task materializes features since the last successful run.

    Args:
        end_date: End date (ISO format). Defaults to now.
        feature_views: Specific feature views to materialize.
        dry_run: If True, validate only without materializing.

    Returns:
        Materialization result dict.
    """
    from scripts.feast_materialize import MaterializationJob

    logger.info(f"Starting incremental materialization task: {self.request.id}")

    # Parse dates
    now = datetime.now(timezone.utc)
    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00")) if end_date else now

    async def run_job():
        job = MaterializationJob()
        try:
            result = await job.run_incremental_materialization(
                end_date=end_dt,
                feature_views=feature_views,
                dry_run=dry_run,
            )
            return result
        finally:
            await job.close()

    result = cast(Dict[str, Any], run_async(run_job()))

    # Log result
    status = result.get("status", "unknown")
    if status == "completed":
        duration = result.get("duration_seconds", 0)
        logger.info(f"Incremental materialization complete: duration={duration:.2f}s")
    elif status == "failed":
        logger.error(f"Incremental materialization failed: {result.get('error')}")

        # Auto-recovery: try full materialization if configured
        config = load_config()
        recovery = config.get("recovery", {})
        if recovery.get("auto_recover", True):
            logger.info("Attempting recovery with full materialization")
            recovery_days = recovery.get("recovery_days_back", 3)
            start_dt = now - timedelta(days=recovery_days)

            async def run_recovery():
                recovery_job = MaterializationJob()
                try:
                    return await recovery_job.run_full_materialization(
                        start_date=start_dt,
                        end_date=end_dt,
                        feature_views=feature_views,
                        dry_run=False,
                    )
                finally:
                    await recovery_job.close()

            result["recovery_attempt"] = run_async(run_recovery())

    return result


@celery_app.task(bind=True, name="src.tasks.check_feature_freshness")
def check_feature_freshness(
    self,
    feature_views: Optional[List[str]] = None,
    max_staleness_hours: Optional[float] = None,
    alert_on_stale: bool = True,
) -> Dict[str, Any]:
    """Check feature freshness and optionally alert.

    Args:
        feature_views: Specific feature views to check.
        max_staleness_hours: Override staleness threshold.
        alert_on_stale: Whether to send alerts for stale features.

    Returns:
        Freshness check result dict.
    """
    from scripts.feast_materialize import MaterializationJob

    logger.info(f"Starting feature freshness check: {self.request.id}")

    config = load_config()
    staleness_hours = max_staleness_hours or config.get("materialization", {}).get(
        "max_staleness_hours", 24.0
    )

    async def run_check():
        job = MaterializationJob()
        try:
            result = await job.check_feature_freshness(
                feature_views=feature_views,
                max_staleness_hours=staleness_hours,
            )
            return result
        finally:
            await job.close()

    result = cast(Dict[str, Any], run_async(run_check()))

    # Log and alert
    if result.get("status") == "completed":
        fresh = result.get("fresh", True)
        stale_count = len(result.get("stale_features", []))
        fresh_count = len(result.get("fresh_features", []))

        if fresh:
            logger.info(f"All {fresh_count} feature views are fresh")
        else:
            stale_names = [f["feature_view"] for f in result.get("stale_features", [])]
            logger.warning(f"Found {stale_count} stale feature view(s): {stale_names}")

            if alert_on_stale:
                # Could integrate with alerting system here
                _send_staleness_alert(result)

    return result


def _send_staleness_alert(freshness_result: Dict[str, Any]) -> None:
    """Send alert for stale features.

    This is a placeholder for integration with alerting systems.
    Configure in feast_materialization.yaml -> alerting.
    """
    config = load_config()
    alerting = config.get("alerting", {})

    if not alerting.get("enabled", True):
        return

    stale = freshness_result.get("stale_features", [])
    if not stale:
        return

    channels = alerting.get("channels", [])
    for channel in channels:
        channel_type = channel.get("type")

        if channel_type == "log":
            level = channel.get("level", "warning")
            log_func = getattr(logger, level, logger.warning)
            log_func(
                f"ALERT: {len(stale)} stale feature view(s) detected. "
                f"Views: {[s['feature_view'] for s in stale]}"
            )

        # Placeholder for other channel types
        # elif channel_type == "slack":
        #     _send_slack_alert(channel, stale)
        # elif channel_type == "pagerduty":
        #     _send_pagerduty_alert(channel, stale)


@celery_app.task(bind=True, name="src.tasks.materialize_feature_view")
def materialize_feature_view(
    self,
    feature_view: str,
    end_date: Optional[str] = None,
    mode: str = "incremental",
) -> Dict[str, Any]:
    """Materialize a single feature view.

    Args:
        feature_view: Name of the feature view to materialize.
        end_date: End date (ISO format). Defaults to now.
        mode: "incremental" or "full".

    Returns:
        Materialization result dict.
    """
    logger.info(f"Materializing feature view: {feature_view}")

    if mode == "incremental":
        return cast(
            Dict[str, Any],
            materialize_incremental_features.delay(
                end_date=end_date,
                feature_views=[feature_view],
            ).get(),
        )
    else:
        return cast(
            Dict[str, Any],
            materialize_features.delay(
                end_date=end_date,
                feature_views=[feature_view],
            ).get(),
        )
