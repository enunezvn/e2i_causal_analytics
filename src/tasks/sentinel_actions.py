"""
Plan-specified sentinel action handlers (#375 item 4, plan §3.8).

Four async handlers + their Celery-task wrappers:

* ``rerun_all_active_cohorts``    — fired by data_drop sentinels; publishes
                                    a ``data_refresh`` alert and (intent:)
                                    queues per-brand pipeline runs
* ``notify_and_queue_reanalysis`` — fired by staleness_threshold sentinels;
                                    publishes a ``staleness_alert`` for the
                                    top-5 most-stale findings AND enqueues
                                    a ``reanalyze_finding`` Celery task per
                                    finding (#378). Broker outage on the
                                    per-finding enqueue is best-effort: the
                                    Redis alert publication still goes out.
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
import time
from typing import Any, Dict, Final, List, Optional

from kombu.exceptions import OperationalError as KombuOperationalError
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import (
    TimeoutError as RedisTimeoutError,
)

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# Channel name is part of the cross-process contract with the CopilotKit
# subscriber; do not rename without updating the subscriber as well.
ALERTS_CHANNEL: Final[str] = "e2i:alerts"


async def publish_alert(payload: Dict[str, Any]) -> None:
    """
    Best-effort publish ``payload`` (JSON-serialized) to ``ALERTS_CHANNEL``.

    On Redis failure: log + swallow. Caller's main side-effect must continue.

    Stamps ``payload['publish_at']`` with an integer epoch-ms timestamp
    BEFORE serialization. The consumer side (the SSE bridge in
    :mod:`src.api.routes.staleness_alerts`) reads this field to compute
    the publish→receive delivery latency via
    :func:`src.mlops.lifecycle_monitoring.record_alert_latency` (#391
    monitoring slice, box 3). The field is added in-place so the
    serialized JSON on Redis carries it; back-compat with payloads
    that already include ``publish_at`` is preserved (only stamps when
    absent).
    """
    # Lazy import: avoid hardening the action module's import-time dependency
    # on the Redis client factory (which itself does an import-time
    # ``redis.from_url``). Lets unit tests patch the factory cleanly.
    from src.memory.services.factories import get_redis_client

    # Stamp publish_at if not already present. Idempotent — re-publication
    # of an already-stamped payload preserves the original timestamp so
    # downstream latency math reflects the FIRST publish, not the retry.
    if "publish_at" not in payload:
        # int(time.time() * 1000) yields milliseconds-since-epoch matching
        # the consumer's ``time.time() * 1000`` in
        # ``lifecycle_monitoring.record_alert_latency`` so delta math is
        # symmetric.
        payload["publish_at"] = int(time.time() * 1000)

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


_REANALYSIS_TASK_NAME: Final[str] = "src.tasks.insight_lifecycle_tasks.reanalyze_finding"


async def notify_and_queue_reanalysis(
    *,
    sentinel_id: str,
    brands: List[str],
    trigger_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan §3.8 ``staleness_threshold`` action — publishes a ``staleness_alert``
    for the top-5 most-stale findings AND enqueues a ``reanalyze_finding``
    Celery task per finding (#378).

    Plan body specifies "top 5 most stale" — we sort by ``staleness_score``
    descending, slice to :data:`_REANALYSIS_CAP`, and notify. Findings
    without a ``staleness_score`` (binary-staleness ship per Decision 3 =
    KEEP BINARY) come through as ``staleness_score=1.0`` so they're treated
    as the most urgent.

    Return contract
    ---------------
    * ``notified_for_reanalysis`` — count of findings logged + included in
                                    the Redis ``staleness_alert`` payload
                                    (== ``len(top)`` for the cap).
    * ``queued_for_reanalysis``   — count of ACTUAL successful Celery
                                    ``send_task`` calls. May be less than
                                    ``notified_for_reanalysis`` if the
                                    broker is degraded — the field
                                    separation lets observers detect a
                                    partial degraded run.

    Both fields kept for back-compat with #375's honesty fix contract:
    ``notified_for_reanalysis`` was the iter-1 honesty rename, kept here
    even though it numerically equals ``queued_for_reanalysis`` in the
    happy path. They diverge under broker outage.

    Best-effort enqueue
    -------------------
    A ``send_task`` failure (broker down, transient ConnectionError) is
    logged at WARNING but does NOT propagate — the Redis alert publish is
    the cross-process audit trail and the queued counter reflects only
    successful enqueues. This mirrors the dispatcher's per-match send_task
    pattern in :mod:`src.memory.sentinels.registry`.
    """
    stale_findings: List[Dict[str, Any]] = list(trigger_data.get("stale_findings") or [])
    # Stable sort: most-stale first. Treat missing scores as 1.0 (max stale).
    stale_findings.sort(key=lambda f: float(f.get("staleness_score") or 1.0), reverse=True)
    top = stale_findings[:_REANALYSIS_CAP]

    payload = {
        "type": "staleness_alert",
        "sentinel_id": str(sentinel_id),
        "brands": list(brands),
        "findings": stale_findings,  # full list for UI; cap is internal
    }
    await publish_alert(payload)

    notified_count = 0
    queued_count = 0
    # Default brand for findings missing one: the first sentinel-scope brand,
    # else "all". The `reanalyze_finding` task ValueErrors on empty brand,
    # so we always resolve to a non-empty string here.
    default_brand = brands[0] if brands else "all"
    for finding in top:
        finding_id = finding.get("finding_id")
        finding_brand = finding.get("brand") or default_brand
        logger.info(
            f"sentinel-action notified-for-reanalysis sentinel={sentinel_id} "
            f"finding={finding_id} "
            f"staleness={finding.get('staleness_score')}"
        )
        notified_count += 1
        if not finding_id:
            # Skip enqueue if the match is malformed (no pk). The notify
            # count still goes up because the finding made it into the
            # alert payload — operators can see the malformed shape via
            # the alert subscriber.
            #
            # L1 (codex iter-0): log only safe-shape metadata, never the
            # full finding dict. Findings can carry PHI / patient-level
            # fields and per-HIPAA we MUST NOT page that through general
            # logging. Brand + key list is enough for ops to triage.
            #
            # L3 (codex iter-1) — intentional schema-key exposure: the
            # warning logs ``sorted(finding.keys())``, which exposes
            # schema-level column NAMES (e.g. ``patient_mrn``,
            # ``patient_dob``). The trade-off is deliberate: operators
            # need to see what KEYS were present on the malformed match
            # to triage the upstream evaluator bug (missing finding_id),
            # but the corresponding VALUES are NEVER interpolated. The
            # boundary is pinned by
            # ``test_notify_and_queue_reanalysis_malformed_finding_log_omits_sensitive_payload``
            # which puts MRN/DOB/clinical-notes VALUES in the finding
            # and asserts the key NAMES appear in the log but the VALUES
            # do not.
            logger.warning(
                f"sentinel-action notify_and_queue_reanalysis: skipping enqueue "
                f"for finding without finding_id sentinel={sentinel_id} "
                f"brand={finding.get('brand')} keys={sorted(finding.keys())}"
            )
            continue
        try:
            celery_app.send_task(
                _REANALYSIS_TASK_NAME,
                args=[finding_id, finding_brand],
                kwargs={"triggered_by": "sentinel:staleness"},
            )
            queued_count += 1
        except (
            KombuOperationalError,
            ConnectionError,
            RedisConnectionError,
            TimeoutError,
            RedisTimeoutError,
        ) as exc:
            # Narrow: only broker/transport failures. Programming errors
            # (TypeError, AttributeError, KeyError from bad finding shapes)
            # propagate so they surface in error tracking instead of being
            # silently indistinguishable from broker outage.
            #
            # Catch surface:
            # * ``kombu.exceptions.OperationalError`` — Celery's canonical
            #   broker connection failure (re-exported as
            #   ``celery.exceptions.OperationalError``).
            # * builtin ``ConnectionError`` — lower-level transport errors
            #   that can escape kombu's normalization in some broker
            #   configurations.
            # * ``redis.exceptions.ConnectionError`` — redis-py's own
            #   transport-error class, which does NOT inherit from builtin
            #   ``ConnectionError`` (it inherits from
            #   ``redis.exceptions.RedisError -> Exception``). Without this
            #   alias, a bare redis-py call in the celery transport path
            #   would escape this catch. (Codex iter-1 H1.)
            # * builtin ``TimeoutError`` — broker timeout.
            # * ``redis.exceptions.TimeoutError`` (aliased as
            #   ``RedisTimeoutError``) — redis-py's timeout class. Inherits
            #   from ``redis.exceptions.ConnectionError -> RedisError ->
            #   Exception``, so it does NOT match builtin ``TimeoutError``.
            #   Same root-cause shape as the H1 ConnectionError gap. (Codex
            #   iter-2 M4.)
            #
            # The Redis alert already published, so subscribers still see
            # the staleness signal. Mirrors registry.py:680 pattern.
            logger.warning(
                f"sentinel-action notify_and_queue_reanalysis: send_task "
                f"failed (broker/transport) for finding={finding_id} "
                f"sentinel={sentinel_id}: {exc}"
            )

    return {
        "sentinel_id": str(sentinel_id),
        "stale_findings_count": len(stale_findings),
        "notified_for_reanalysis": notified_count,
        "queued_for_reanalysis": queued_count,
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
