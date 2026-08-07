"""Chatbot DSPy optimization queue drainer (#1515).

Closes the producer->queue->executor loop that migration
``database/chat/035_chatbot_optimization_requests.sql`` scaffolded:

- **Producer**: ``submit_signals_for_optimization`` (src/api/routes/
  chatbot_dspy.py) enqueues per-module optimization requests when enough
  high-quality training signals exist. Before #1515 nothing routed to it; it
  is now routed HERE (once per cycle, only while the queue is idle, so a
  4-module burst drains before the next burst can enqueue duplicates).
- **Queue substrate**: the 035 table + lifecycle functions. This module is
  their first Python caller.
- **Executor**: ``ChatbotOptimizer.optimize_module`` — the real GEPA path
  whose save step #1507 fixed. The drainer calls it through
  :func:`_execute_request`; nothing in the production path is mocked.

Claim semantics (why the claim is NOT a 035 function)
-----------------------------------------------------
035's ``get_next_optimization_request`` is a pure peek — plain ``SELECT ...
LIMIT 1`` with no ``FOR UPDATE SKIP LOCKED`` — and its
``update_optimization_request_status`` updates unconditionally (no prior-status
guard), so two overlapping drain runs could both "successfully" mark the same
row processing and double-spend a GEPA run. The claim is therefore a single
compare-and-set UPDATE (``status='processing' WHERE request_id=? AND
status='pending'``) issued through the service-role client: atomic at the
statement level, loser sees zero updated rows and re-peeks — the same effective
semantics SKIP LOCKED would give, without a schema change the live DB does not
yet have. Peek and close-out DO go through the 035 functions per #1515's
acceptance.

Cost gate (#1513 precedent)
---------------------------
The executor runs LLM-expensive GEPA (dspy 3.1.0: ``auto="light"`` alone is
~390 metric calls regardless of trainset size). ``CHATBOT_OPT_DRAIN_ENABLED``
is the single opt-in switch, parsed fail-closed exactly like
``DSPY_RAG_DB_FEEDSTOCK_ENABLED``: unset/garbage means the beat tick is a
logged no-op costing zero API calls and zero DB writes. Enabling it is an
operator's cost decision and must not arrive as a side effect of merging this
module. ``force=True`` (manual ``celery call``) bypasses ONLY this gate, never
the per-cycle execution bound.

Durability (the #1515 core defect)
----------------------------------
Requests live in the table, so they survive worker restarts. A row a dead
worker left in 'processing' is returned to 'pending' once it is older than
``CHATBOT_OPT_ZOMBIE_HOURS``; a row nothing drained for
``CHATBOT_OPT_STALE_HOURS`` is cancelled via the 035
``cancel_stale_optimization_requests`` (the drainer passes its own 168 h
default explicitly — the SQL default of 24 h would cancel a queued 4-module
burst faster than the 1-execution-per-cycle default can serve it).

All env knobs are forwarded via docker-compose ``x-common-env`` (the #1489
deferral-2 lesson: without that, host-.env values never reach the worker
containers and the in-code defaults govern unconditionally).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

from src.tasks.dspy_optimization_tasks import run_async
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

QUEUE_TABLE = "chatbot_optimization_requests"

# Opt-in switch for the whole drain cycle. Fail-closed: anything but a truthy
# value means "skip". See the module docstring for why this defaults OFF.
DRAIN_ENABLED_ENV = "CHATBOT_OPT_DRAIN_ENABLED"

# GEPA executions per beat tick. 1 bounds worst-case LLM spend to one
# optimization run per cycle; 0 turns the tick into bookkeeping-only
# (zombie/stale management + producer) with no executions.
DRAIN_MAX_PER_CYCLE_ENV = "CHATBOT_OPT_DRAIN_MAX_PER_CYCLE"
_DEFAULT_MAX_PER_CYCLE = 1

# Pending rows older than this are cancelled via the 035 stale-cancel function.
STALE_HOURS_ENV = "CHATBOT_OPT_STALE_HOURS"
_DEFAULT_STALE_HOURS = 168  # 7 days: a 4-module burst at 1 execution/day fits

# 'processing' rows older than this are treated as orphaned by a dead worker
# and returned to 'pending'. Must comfortably exceed the longest plausible GEPA
# run so a live run is never yanked from under its worker.
ZOMBIE_HOURS_ENV = "CHATBOT_OPT_ZOMBIE_HOURS"
_DEFAULT_ZOMBIE_HOURS = 12

# Producer threshold, passed through to submit_signals_for_optimization.
MIN_SIGNALS_ENV = "CHATBOT_OPT_MIN_SIGNALS"
_DEFAULT_MIN_SIGNALS = 50

_TRUTHY = ("1", "true", "yes")


def _drain_enabled() -> bool:
    """Opt-in cost gate, parsed fail-closed (mirrors rag_example_sources)."""
    return os.environ.get(DRAIN_ENABLED_ENV, "").strip().lower() in _TRUTHY


def _int_env(name: str, default: int, floor: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %d", name, raw, default)
        return default
    return max(floor, value)


def _max_per_cycle() -> int:
    return _int_env(DRAIN_MAX_PER_CYCLE_ENV, _DEFAULT_MAX_PER_CYCLE, floor=0)


def _stale_hours() -> int:
    return _int_env(STALE_HOURS_ENV, _DEFAULT_STALE_HOURS, floor=1)


def _zombie_hours() -> int:
    return _int_env(ZOMBIE_HOURS_ENV, _DEFAULT_ZOMBIE_HOURS, floor=1)


def _min_signals() -> int:
    return _int_env(MIN_SIGNALS_ENV, _DEFAULT_MIN_SIGNALS, floor=1)


async def _get_client() -> Optional[Any]:
    """Service-role async client. The queue writes need it: 035's RLS grants
    write access to service_role only (env key is the ANON key; the workers get
    SUPABASE_SERVICE_KEY via compose x-common-env)."""
    try:
        from src.memory.services.factories import get_async_supabase_service_client

        return await get_async_supabase_service_client()
    except Exception as e:  # noqa: BLE001 - a keyless box is a failed cycle, not a crash
        logger.error("chatbot optimization drainer: no database client: %s", e)
        return None


async def _recover_zombies(client: Any) -> int:
    """Return orphaned 'processing' rows to 'pending' (worker died mid-run).

    Compare-and-set on (status='processing', started_at < cutoff): a live GEPA
    run younger than the cutoff is left alone. A crash-looping request is still
    bounded: stale-cancel keys on created_at, so it terminates as 'cancelled'
    once it exceeds CHATBOT_OPT_STALE_HOURS.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=_zombie_hours())).isoformat()
    result = await (
        client.table(QUEUE_TABLE)
        .update(
            {
                "status": "pending",
                "started_at": None,
                "error_message": (
                    "requeued by drainer: 'processing' row orphaned "
                    "(worker restart or crash mid-run)"
                ),
            }
        )
        .eq("status", "processing")
        .lt("started_at", cutoff)
        .execute()
    )
    recovered = len(result.data or [])
    if recovered:
        logger.warning("chatbot optimization drainer: requeued %d orphaned request(s)", recovered)
    return recovered


async def _cancel_stale(client: Any) -> int:
    """Terminal backstop via the 035 function, with the drainer's own default."""
    result = await client.rpc(
        "cancel_stale_optimization_requests",
        {"p_max_age_hours": _stale_hours()},
    ).execute()
    data = result.data
    if isinstance(data, (int, float)):
        return int(data)
    if isinstance(data, list) and data:
        return int(data[0])
    return 0


async def _queue_is_idle(client: Any) -> bool:
    result = await (
        client.table(QUEUE_TABLE)
        .select("id")
        .in_("status", ["pending", "processing"])
        .limit(1)
        .execute()
    )
    return not result.data


async def _produce_requests(client: Any) -> Optional[Dict[str, Any]]:
    """Producer leg: route submit_signals_for_optimization (#1515 acceptance).

    Runs only while the queue is idle — submit enqueues a fresh request per
    module whenever signals suffice, so running it every tick would pile
    duplicate pending rows for work already in flight.
    """
    if not await _queue_is_idle(client):
        logger.info("chatbot optimization drainer: queue busy, producer deferred")
        return None
    # Function-local: the routes module imports dspy at module top.
    from src.api.routes.chatbot_dspy import submit_signals_for_optimization

    produced = await submit_signals_for_optimization(min_signals=_min_signals())
    logger.info("chatbot optimization drainer: producer result %s", produced)
    return produced


async def _peek_next(client: Any) -> Optional[Dict[str, Any]]:
    """Next pending request via the 035 peek (priority DESC, created_at ASC)."""
    result = await client.rpc("get_next_optimization_request", {}).execute()
    rows = result.data or []
    return dict(rows[0]) if rows else None


async def _claim(client: Any, request_id: str) -> bool:
    """Compare-and-set claim: pending -> processing, atomically.

    See the module docstring — 035 ships no SKIP LOCKED claim, so this single
    conditional UPDATE is the guard. Zero updated rows == lost the race.
    """
    result = await (
        client.table(QUEUE_TABLE)
        .update(
            {
                "status": "processing",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("request_id", request_id)
        .eq("status", "pending")
        .execute()
    )
    return bool(result.data)


async def _close_out(
    client: Any,
    request_id: str,
    status: str,
    *,
    baseline_score: Optional[float] = None,
    optimized_score: Optional[float] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Terminal transition via the 035 status function (#1515 acceptance)."""
    params: Dict[str, Any] = {"p_request_id": request_id, "p_status": status}
    if baseline_score is not None:
        params["p_baseline_score"] = baseline_score
    if optimized_score is not None:
        params["p_optimized_score"] = optimized_score
    if error_message is not None:
        params["p_error_message"] = error_message
    result = await client.rpc("update_optimization_request_status", params).execute()
    return bool(result.data)


async def _execute_request(row: Dict[str, Any]) -> Dict[str, Any]:
    """Run the REAL executor seam for one claimed request. LLM-expensive:
    everything upstream (gate, per-cycle bound, claim) exists to bound how
    often this is reached. Tests may substitute THIS seam only; a unit test
    pins that it calls ChatbotOptimizer.optimize_module."""
    from src.api.routes.chatbot_dspy import get_chatbot_optimizer

    optimizer = get_chatbot_optimizer()
    return cast(
        Dict[str, Any],
        await optimizer.optimize_module(
            row["module_name"],
            budget=row["budget"],
            min_reward=row["min_reward"],
        ),
    )


async def _drain_cycle(force: bool = False) -> Dict[str, Any]:
    """One full drain cycle: gate -> bookkeeping -> produce -> claim/execute/close."""
    if not (force or _drain_enabled()):
        reason = (
            f"{DRAIN_ENABLED_ENV} not enabled (fail-closed default). Enabling runs "
            "LLM-expensive GEPA optimization on the analytics worker — an operator "
            f"cost decision; set {DRAIN_ENABLED_ENV}=1 in the host .env to opt in."
        )
        logger.info("chatbot optimization drain skipped: %s", reason)
        return {"status": "skipped", "reason": reason}

    client = await _get_client()
    if client is None:
        return {"status": "failed", "reason": "database client unavailable"}

    zombies_recovered = 0
    try:
        zombies_recovered = await _recover_zombies(client)
    except Exception as e:  # noqa: BLE001 - bookkeeping must not abort the drain
        logger.warning("zombie recovery failed: %s", e)

    stale_cancelled = 0
    try:
        stale_cancelled = await _cancel_stale(client)
    except Exception as e:  # noqa: BLE001 - bookkeeping must not abort the drain
        logger.warning("stale cancellation failed: %s", e)

    produced: Optional[Dict[str, Any]] = None
    try:
        produced = await _produce_requests(client)
    except Exception as e:  # noqa: BLE001 - the producer leg must not abort the drain
        logger.warning("producer leg failed: %s", e)
        produced = {"error": str(e)}

    executed: List[Dict[str, Any]] = []
    for _ in range(_max_per_cycle()):
        row = await _peek_next(client)
        if row is None:
            break
        request_id = row["request_id"]
        if not await _claim(client, request_id):
            # A competing claimer won between peek and claim; the row is no
            # longer pending, so the next peek serves the next request. The
            # lost slot still counts against the per-cycle bound (spend cap).
            logger.info("lost claim race for %s; re-peeking", request_id)
            continue

        try:
            result = await _execute_request(row)
        except Exception as e:  # noqa: BLE001 - one request must not abort the cycle
            logger.error("optimization execution failed for %s: %s", request_id, e, exc_info=True)
            await _close_out(client, request_id, "failed", error_message=str(e))
            executed.append(
                {
                    "request_id": request_id,
                    "module_name": row["module_name"],
                    "status": "failed",
                }
            )
            continue

        if result.get("success"):
            await _close_out(
                client,
                request_id,
                "completed",
                optimized_score=result.get("best_score"),
            )
            outcome = "completed"
        else:
            await _close_out(
                client,
                request_id,
                "failed",
                error_message=str(result.get("error") or "optimization returned success=False"),
            )
            outcome = "failed"
        executed.append(
            {
                "request_id": request_id,
                "module_name": row["module_name"],
                "status": outcome,
            }
        )

    return {
        "status": "completed",
        "zombies_recovered": zombies_recovered,
        "stale_cancelled": stale_cancelled,
        "produced": produced,
        "executed": executed,
    }


@celery_app.task(bind=True, name="src.tasks.drain_chatbot_optimization_queue")
def drain_chatbot_optimization_queue(self, force: bool = False) -> Dict[str, Any]:
    """Beat entry point (see celery_app.py 'chatbot-optimization-drain').

    ``force=True`` (manual ``celery call``) bypasses the enable gate only —
    the per-cycle execution bound still applies.
    """
    logger.info(
        "Starting chatbot optimization queue drain: task %s (force=%s)",
        self.request.id,
        force,
    )
    try:
        result = run_async(_drain_cycle(force=force))
        result["task_id"] = self.request.id
        return cast(Dict[str, Any], result)
    except Exception as exc:  # noqa: BLE001 - best-effort, never raise out of the beat
        logger.error("chatbot optimization queue drain failed: %s", exc, exc_info=True)
        return {"status": "failed", "reason": str(exc), "task_id": self.request.id}
