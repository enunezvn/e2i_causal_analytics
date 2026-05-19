"""
Celery tasks for the insight-lifecycle subsystem.

- ``consolidate_insights``  : daily at 04:00 UTC. Promotes causal_paths to
                              semantic and procedural_memories to procedural.
- ``sentinel_dispatcher``   : every 5 minutes. Evaluates all enabled
                              sentinels and fires matching actions.
- ``reanalyze_finding``     : per-finding reanalysis hand-off (#378). Enqueued
                              by the ``notify_and_queue_reanalysis`` sentinel
                              action handler; publishes a brand-scoped
                              ``reanalysis:e2i:{brand}`` signal carrying the
                              finding metadata so downstream consumers can
                              re-run analysis.

All three tasks are idempotent — re-running them within their schedule
produces no extra side effects beyond a few SELECTs / a duplicate pub/sub
notification (subscribers are expected to dedupe by finding_id).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
)

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# Brand-namespaced reanalysis-request channel. The orchestrator (or a future
# Tier-3 single-finding re-evaluation worker) subscribes to
# ``reanalysis:e2i:{brand}`` and picks up requests as JSON payloads. The
# channel name mirrors the ``invalidation:e2i:{brand}`` pattern used by
# ``src.memory.lifecycle.invalidator._publish_invalidation_signal`` so
# subscribers can fan-in both signal types per brand.
REANALYSIS_CHANNEL_PREFIX = "reanalysis:e2i:"


@celery_app.task(bind=True, name="src.tasks.consolidate_insights")
def consolidate_insights(self, brand: Optional[str] = None) -> Dict[str, Any]:
    """
    Daily consolidator pass. Returns a JSON-serializable summary.

    Args:
        brand: optional brand to scope the run (default: all brands)
    """
    from src.memory.lifecycle.consolidator import consolidate_insights as run_consolidator

    try:
        result = asyncio.run(run_consolidator(brand=brand))
        return {
            "promoted_to_semantic": result.promoted_to_semantic,
            "promoted_to_procedural": result.promoted_to_procedural,
            "causal_paths_examined": result.causal_paths_examined,
            "procedural_examined": result.procedural_examined,
            "errors": result.errors,
            "by_brand": result.by_brand,
        }
    except Exception:
        logger.exception("consolidate_insights task failed")
        raise


@celery_app.task(bind=True, name="src.tasks.sentinel_dispatcher")
def sentinel_dispatcher(self) -> Dict[str, Any]:
    """5-minute sentinel evaluation pass."""
    from src.memory.sentinels.registry import dispatch_sentinels

    try:
        result = asyncio.run(dispatch_sentinels())
        return {
            "examined": result.examined,
            "fired": result.fired,
            "actions_taken": result.actions_taken,
            "errors": result.errors,
            "by_sentinel": result.by_sentinel,
        }
    except Exception:
        logger.exception("sentinel_dispatcher task failed")
        raise


async def _publish_reanalysis_signal(
    *,
    finding_id: str,
    brand: str,
    triggered_by: str,
) -> bool:
    """
    Publish a per-brand reanalysis request on ``reanalysis:e2i:{brand}``.

    Returns ``True`` if the publish succeeded, ``False`` if Redis was
    unreachable (in which case the failure is logged at WARNING but the
    Celery task does NOT crash — best-effort mirrors the
    ``cascade_invalidate`` and ``publish_alert`` patterns).
    """
    # Lazy import to avoid hardening the task module's import-time
    # dependency on the Redis client factory.
    from src.memory.services.factories import get_redis_client

    channel = f"{REANALYSIS_CHANNEL_PREFIX}{brand}"
    payload = json.dumps(
        {
            "type": "reanalysis_requested",
            "finding_id": finding_id,
            "brand": brand,
            "triggered_by": triggered_by,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    try:
        redis = get_redis_client()
        await redis.publish(channel, payload)
        return True
    except (
        ConnectionError,
        RedisConnectionError,
        TimeoutError,
        RedisTimeoutError,
    ) as exc:
        # Narrow: only redis-py transport-error classes.
        #
        # Catch surface (codex iter-1 H2 + M3, iter-2 M4):
        # * builtin ``ConnectionError`` — defensive coverage for lower
        #   socket-layer errors that can escape redis-py's normalization.
        # * ``redis.exceptions.ConnectionError`` (aliased as
        #   ``RedisConnectionError``) — redis-py's canonical transport
        #   failure class. Does NOT inherit from builtin ``ConnectionError``
        #   (inherits from ``redis.exceptions.RedisError -> Exception``),
        #   so the explicit alias is load-bearing. (H2)
        # * builtin ``TimeoutError`` — defensive coverage for socket-layer
        #   timeouts. Symmetric with the ConnectionError pair. (M4)
        # * ``redis.exceptions.TimeoutError`` (aliased as
        #   ``RedisTimeoutError``) — redis-py's timeout class. Inherits
        #   from ``redis.exceptions.ConnectionError -> RedisError ->
        #   Exception``, so it does NOT match builtin ``TimeoutError``.
        #   Same root-cause shape as H2. (M4)
        #
        # ``RuntimeError`` was dropped (M3): a source-grep of redis-py
        # shows the only ``raise RuntimeError`` sites are on the
        # PubSub-CONSUMER side (subscribe/psubscribe lifecycle gates), not
        # on publish. Keeping it would have masked real programming bugs.
        #
        # Programming errors (TypeError, AttributeError, etc.) propagate
        # so we don't silently mask shape mismatches. No broad
        # ``except Exception`` fallback (L2 codex iter-0) — such a fallback
        # would contradict the contract documented above. If an unexpected
        # exception class escapes here, that's a real bug we want the
        # Celery task wrapper to record (via the ``task_failure`` signal
        # handler in celery_app.py).
        logger.warning(
            f"reanalysis-signal publish failed for finding={finding_id} brand={brand}: {exc}"
        )
        return False


@celery_app.task(
    bind=True,
    name="src.tasks.insight_lifecycle_tasks.reanalyze_finding",
)
def reanalyze_finding(
    self: Any,
    finding_id: str,
    brand: str,
    *,
    triggered_by: str = "manual",
) -> Dict[str, Any]:
    """
    Per-finding reanalysis hand-off (#378).

    Enqueued by ``notify_and_queue_reanalysis`` (sentinel_actions.py) for
    each of the top-5 most-stale findings when a ``staleness_threshold``
    sentinel fires. The task is the durable Celery boundary the sentinel
    action handler sends to; the actual single-finding re-evaluation
    pipeline is still moving under #237 / #373 follow-ups, so this task
    intentionally restricts its scope to:

    1. Validate the dispatch shape (raises ``ValueError`` on empty inputs
       so a malformed enqueue does not silently no-op).
    2. Publish a ``reanalysis_requested`` event on the brand-scoped
       ``reanalysis:e2i:{brand}`` Redis pub/sub channel. Downstream
       orchestrator consumers subscribe here.
    3. Return a JSON-serializable summary so Celery result-backend
       observers see what was attempted.

    Args:
        finding_id: row pk of the finding (causal_path / trigger /
            ml_prediction / executive_insight, per the invalidation-aware
            table set).
        brand: per-finding brand carried in the sentinel match dict.
        triggered_by: free-text origin tag; the sentinel action handler
            uses ``"sentinel:staleness"``. Other entry points (manual
            re-run, ops tooling) supply their own.

    Returns:
        Dict with ``finding_id``, ``brand``, ``triggered_by``,
        ``signal_published`` (bool). ``signal_published=False`` means the
        Redis publish itself failed — observers should treat that as a
        degraded but non-crashing run.

    Raises:
        ValueError: if ``finding_id`` or ``brand`` is empty / None.
    """
    if not finding_id:
        raise ValueError("reanalyze_finding: finding_id is required (got empty/None)")
    if not brand:
        raise ValueError("reanalyze_finding: brand is required (got empty/None)")

    signal_published = asyncio.run(
        _publish_reanalysis_signal(
            finding_id=finding_id,
            brand=brand,
            triggered_by=triggered_by,
        )
    )

    logger.info(
        f"reanalyze_finding finding={finding_id} brand={brand} "
        f"triggered_by={triggered_by} signal_published={signal_published}"
    )
    return {
        "finding_id": finding_id,
        "brand": brand,
        "triggered_by": triggered_by,
        "signal_published": signal_published,
    }
