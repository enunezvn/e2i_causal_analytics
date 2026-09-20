"""The forecast worker's one task: a batched TimesFM 2.5 forward pass (#2115, Lane B).

One task for a WHOLE rolling-origin backtest, not one per origin. The 200M weights load
once per worker process and the batch is a single forward pass, so a 24-origin backtest
costs one dispatch and ~9 s (measured 2026-09-20, prod api image, 2 CPU) instead of 24
dispatches and 24 weight loads.

The task body is deliberately a thin shell over ``src.kpi.forecast.timesfm.forecast_batch``
so the code that runs in the worker is the SAME code the tests exercise directly — a
task that reimplemented the forward pass could pass its tests and still serve something
else in production.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from src.kpi.forecast.timesfm import TIMESFM_TASK_NAME
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name=TIMESFM_TASK_NAME,
    queue="forecast",
    # No retry: the caller's fallback is Holt-Winters, which it can run immediately.
    # Retrying here would hold a chat turn open waiting for a model it does not need.
    max_retries=0,
    time_limit=300,
    soft_time_limit=270,
)
def forecast_timesfm_batch(
    contexts: Sequence[Sequence[float]], horizon: int
) -> List[Optional[List[Any]]]:
    """Forecast ``horizon`` steps for each context. Result k belongs to context k."""
    from src.kpi.forecast.timesfm import forecast_batch

    logger.info("TimesFM batch: %d contexts, horizon %d", len(contexts), horizon)
    return forecast_batch(contexts, horizon)
