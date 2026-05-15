"""
Celery tasks for the insight-lifecycle subsystem.

- ``consolidate_insights``  : daily at 04:00 UTC. Promotes causal_paths to
                              semantic and procedural_memories to procedural.
- ``sentinel_dispatcher``   : every 5 minutes. Evaluates all enabled
                              sentinels and fires matching actions.

Both tasks are idempotent — re-running them within their schedule produces
no extra side effects beyond a few SELECTs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


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
