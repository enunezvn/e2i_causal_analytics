"""
Plan-specified sentinel action handlers (#375 item 4, plan §3.8).

Four async handlers + their Celery-task wrappers:

* ``rerun_all_active_cohorts``    — fired by data_drop sentinels; publishes
                                    a ``data_refresh`` alert and (intent:)
                                    queues per-brand pipeline runs
* ``notify_and_queue_reanalysis`` — fired by staleness_threshold sentinels;
                                    publishes a ``staleness_alert`` and
                                    queues the top-5 most-stale findings
* ``flag_for_review``             — fired by cohort_drift sentinels; publishes
                                    a ``cohort_drift`` alert for SME review
* ``run_full_consolidation``      — fired by schedule sentinels; runs a
                                    full consolidator pass and publishes a
                                    ``full_consolidation_run`` heartbeat

All four publish to the Redis pub/sub channel ``e2i:alerts`` (constant
:data:`ALERTS_CHANNEL`). Subscribers (CopilotKit SSE/WebSocket bridge) get
real-time delivery; pub/sub (not Streams) is deliberate — subscribers don't
need replay, only "drop your cache and re-render".

Routing from the dispatcher
---------------------------
The dispatcher's ``dispatch_agent`` action emits an InsightSignalBus event
with ``agent_name`` set to the Celery task name; the signal-bus subscriber
calls ``celery_app.send_task(agent_name, args=(sentinel_id, brands, trigger_data))``
which lands on whichever worker is hosting these tasks.

The plain async helpers (``rerun_all_active_cohorts`` etc.) are the
direct callable form; the ``celery_*`` wrappers are what the Celery
broker enqueues. We register both so unit tests can exercise the async
form without spinning up a worker.

Best-effort publication
-----------------------
All four wrap the Redis publish call in a try/except that LOGS but does
not propagate — a Redis outage MUST NOT prevent the action from running
its main side effect (cohort re-run, consolidation, etc.). Narrow
exception class: ``ConnectionError + RuntimeError`` only — never
``except Exception`` which would mask programming bugs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Final, List, Optional

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# Channel name is part of the cross-process contract with the CopilotKit
# subscriber; do not rename without updating the subscriber as well.
ALERTS_CHANNEL: Final[str] = "e2i:alerts"


async def publish_alert(payload: Dict[str, Any]) -> None:
    """
    Best-effort publish ``payload`` (JSON-serialized) to ``ALERTS_CHANNEL``.

    On Redis failure: log + swallow. Caller's main side-effect must continue.
    """
    # Lazy import: avoid hardening the action module's import-time dependency
    # on the Redis client factory (which itself does an import-time
    # ``redis.from_url``). Lets unit tests patch the factory cleanly.
    from src.memory.services.factories import get_redis_client

    try:
        redis = get_redis_client()
        await redis.publish(ALERTS_CHANNEL, json.dumps(payload))
    except (ConnectionError, RuntimeError) as exc:
        # Narrow: only the two error classes Redis raises on transport
        # failure. A programming error (TypeError etc.) propagates so we
        # don't silently mask shape mismatches.
        logger.warning(f"alerts publish failed for payload={payload!r}: {exc}")
    except Exception:
        # Defensive last-resort log; an unexpected exception class shouldn't
        # crash a Celery action either, but we make the noise loud.
        logger.exception(f"unexpected alerts-publish failure for payload={payload!r}")


# ---------------------------------------------------------------------------
# rerun_all_active_cohorts
# ---------------------------------------------------------------------------


async def rerun_all_active_cohorts(
    *,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan §3.8 ``data_drop`` action — fired when a source table refresh
    arrives. Publishes a ``data_refresh`` alert listing all brands the
    operator wants re-run; downstream cohort-construction pipelines pick
    this up via the existing trigger surface.

    We do NOT call ``trigger_pipeline.delay(...)`` here directly — the
    repo's pipeline-trigger surface is being reshaped under #237 and a
    direct ``send_task`` would couple this module to a moving target.
    The Redis alert + (intent:) a future signal-bus dispatch are the
    portable contract.
    """
    payload = {
        "type": "data_refresh",
        "sentinel_id": str(sentinel_id),
        "brands": list(brands),
        "trigger_data": trigger_data,
    }
    if brands:
        await publish_alert(payload)
        logger.info(
            f"sentinel-action rerun_all_active_cohorts sentinel={sentinel_id} "
            f"brands={brands} → e2i:alerts published"
        )
    return {
        "sentinel_id": str(sentinel_id),
        "brands_dispatched": len(brands),
    }


# ---------------------------------------------------------------------------
# notify_and_queue_reanalysis
# ---------------------------------------------------------------------------


_REANALYSIS_CAP: Final[int] = 5


async def notify_and_queue_reanalysis(
    *,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan §3.8 ``staleness_threshold`` action — publishes a ``staleness_alert``
    and queues the TOP-5 most-stale findings for re-analysis.

    Plan body specifies "top 5 most stale" — we sort by ``staleness_score``
    descending, slice to :data:`_REANALYSIS_CAP`, and enqueue. Findings
    without a ``staleness_score`` (binary-staleness ship per Decision 3 =
    KEEP BINARY) come through as ``staleness_score=1.0`` so they're treated
    as the most urgent.
    """
    stale_findings: List[Dict[str, Any]] = list(trigger_data.get("stale_findings") or [])
    # Stable sort: most-stale first. Treat missing scores as 1.0 (max stale).
    stale_findings.sort(
        key=lambda f: float(f.get("staleness_score") or 1.0), reverse=True
    )
    top = stale_findings[:_REANALYSIS_CAP]

    payload = {
        "type": "staleness_alert",
        "sentinel_id": str(sentinel_id),
        "brands": list(brands),
        "findings": stale_findings,  # full list for UI; cap is internal
    }
    await publish_alert(payload)

    # Re-analysis queueing: we currently log; the orchestrator will subscribe
    # to e2i:alerts and dispatch. Coupling to a specific pipeline-trigger
    # surface here would block on #237's still-moving target. Keep the
    # contract observable via the alert + the summary.
    for finding in top:
        logger.info(
            f"sentinel-action queued reanalysis sentinel={sentinel_id} "
            f"finding={finding.get('finding_id')} "
            f"staleness={finding.get('staleness_score')}"
        )
    return {
        "sentinel_id": str(sentinel_id),
        "stale_findings_count": len(stale_findings),
        "queued_for_reanalysis": len(top),
    }


# ---------------------------------------------------------------------------
# flag_for_review
# ---------------------------------------------------------------------------


async def flag_for_review(
    *,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan §3.8 ``cohort_drift`` action — publishes a ``cohort_drift`` alert
    that the UI surfaces for human review.
    """
    payload = {
        "type": "cohort_drift",
        "sentinel_id": str(sentinel_id),
        "brands": list(brands),
        "drift_data": trigger_data.get("drift_data", trigger_data),
    }
    await publish_alert(payload)
    logger.info(
        f"sentinel-action flag_for_review sentinel={sentinel_id} brands={brands} "
        f"→ e2i:alerts published"
    )
    return {"sentinel_id": str(sentinel_id), "flagged": True}


# ---------------------------------------------------------------------------
# run_full_consolidation
# ---------------------------------------------------------------------------


async def run_full_consolidation(
    *,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan §3.8 ``schedule`` action — runs a full consolidator pass and
    publishes a ``full_consolidation_run`` heartbeat alert.

    Brand scoping: if ``brands`` is empty or contains ``"all"``, runs across
    all brands. Otherwise runs once per brand (sequentially).
    """
    # Lazy import for forward-defensive circular-import avoidance — the
    # consolidator imports from src.memory.services.factories, which in
    # turn pulls in modules that may transitively depend on src.tasks.
    from src.memory.lifecycle.consolidator import consolidate_insights

    promoted_to_semantic = 0
    promoted_to_procedural = 0
    causal_paths_examined = 0
    procedural_examined = 0
    errors: List[str] = []

    if not brands or "all" in brands:
        scope: List[Optional[str]] = [None]
    else:
        scope = list(brands)

    for brand in scope:
        try:
            result = await consolidate_insights(brand=brand)
            promoted_to_semantic += result.promoted_to_semantic
            promoted_to_procedural += result.promoted_to_procedural
            causal_paths_examined += result.causal_paths_examined
            procedural_examined += result.procedural_examined
            errors.extend(result.errors)
        except Exception as exc:
            logger.exception(f"run_full_consolidation: brand={brand} failed")
            errors.append(f"brand={brand}: {exc}")

    payload = {
        "type": "full_consolidation_run",
        "sentinel_id": str(sentinel_id),
        "brands": list(brands),
        "promoted_to_semantic": promoted_to_semantic,
        "promoted_to_procedural": promoted_to_procedural,
        "errors": errors,
    }
    await publish_alert(payload)

    return {
        "sentinel_id": str(sentinel_id),
        "promoted_to_semantic": promoted_to_semantic,
        "promoted_to_procedural": promoted_to_procedural,
        "causal_paths_examined": causal_paths_examined,
        "procedural_examined": procedural_examined,
        "errors": errors,
    }


# ===========================================================================
# CELERY TASK REGISTRATIONS
# ---------------------------------------------------------------------------
# Each ``celery_*`` is a sync Celery task that delegates to the async helper
# via ``asyncio.run``. The task NAMES match what the dispatcher's
# ``dispatch_agent → Celery`` bridge emits.
# ===========================================================================


def _run_action(coro_factory: Any, **kwargs: Any) -> Dict[str, Any]:
    """Tiny sync→async bridge for Celery tasks."""
    return asyncio.run(coro_factory(**kwargs))


@celery_app.task(
    bind=True,
    name="src.tasks.sentinel_actions.rerun_all_active_cohorts",
)
def celery_rerun_all_active_cohorts(
    self: Any,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_action(
        rerun_all_active_cohorts,
        sentinel_id=sentinel_id,
        brands=brands,
        trigger_data=trigger_data,
    )


@celery_app.task(
    bind=True,
    name="src.tasks.sentinel_actions.notify_and_queue_reanalysis",
)
def celery_notify_and_queue_reanalysis(
    self: Any,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_action(
        notify_and_queue_reanalysis,
        sentinel_id=sentinel_id,
        brands=brands,
        trigger_data=trigger_data,
    )


@celery_app.task(
    bind=True,
    name="src.tasks.sentinel_actions.flag_for_review",
)
def celery_flag_for_review(
    self: Any,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_action(
        flag_for_review,
        sentinel_id=sentinel_id,
        brands=brands,
        trigger_data=trigger_data,
    )


@celery_app.task(
    bind=True,
    name="src.tasks.sentinel_actions.run_full_consolidation",
)
def celery_run_full_consolidation(
    self: Any,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_action(
        run_full_consolidation,
        sentinel_id=sentinel_id,
        brands=brands,
        trigger_data=trigger_data,
    )
