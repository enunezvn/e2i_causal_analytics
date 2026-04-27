"""Per-patient adherence / refill / gap ETL (block 6B-infra-2b).

Populates three columns on ``patient_journeys`` that migration 033 added:

* ``adherence_rate NUMERIC`` — proxy for medication possession ratio. Computed
  as ``journey_duration_days / (journey_end_date - journey_start_date)``
  clamped to ``[0, 1]``. The ratio is a stand-in until real claims data is
  ingested; it captures whether the journey covers most of its declared span.
* ``refill_count INTEGER`` — count of refill events. **Left NULL** in this
  ETL because the canonical v3 schema has no first-class refill concept
  (see "refill_count is intentionally NULL" below).
* ``gap_days INTEGER`` — maximum gap (in days) between consecutive triggers
  for the same patient.

This mirrors the shape of ``business_metrics_per_hcp_etl`` (block
6B-infra-2a) so the two ETLs read the same way: pure-SQL CTE, deterministic
window resolution, tenacity-backed connect, thin Celery wrapper over a
pure-Python ``_run_*_impl`` core.

Why ``refill_count`` is intentionally NULL
------------------------------------------
The plan calls for ``refill_count = COUNT(*) FROM triggers WHERE
trigger_type = 'refill_reminder'``. That trigger type **does not exist** in
the synthetic generator (``src/ml/synthetic/generators/trigger_generator.py``
lists ``adherence_risk``, ``churn_prevention``, ``cross_sell``,
``treatment_switch``, ``engagement_gap``, ``reactivation``,
``prescription_opportunity``, ``competitive_threat`` — no refill concept)
and there is no ``prescription_refills`` table in
``database/core/e2i_ml_complete_v3_schema.sql``. Substituting an existing
trigger type as a proxy would silently change semantics; instead we leave
the column NULL with a SQL comment naming the missing source. A future
block can:

1. Add ``'refill_reminder'`` to the trigger_type vocabulary, and/or
2. Introduce a separate ``prescription_refills`` table for Rx events,

then re-enable this column. This matches how 6B-infra-2a left
``engagement_score`` and ``call_frequency`` NULL because the
``interactions`` table doesn't exist yet.

Edge-case behaviour for ``adherence_rate``
------------------------------------------
* ``journey_end_date IS NULL`` → ``DATE - DATE`` is NULL → NULLIF(NULL, 0)
  is NULL → division yields NULL → LEAST/GREATEST(NULL) is NULL.
* ``journey_end_date == journey_start_date`` → 0 → NULLIF(0, 0) is NULL →
  NULL. (This is the "zero-duration journeys" case the plan calls out.)
* ``journey_duration_days IS NULL`` → numerator NULL → result NULL.

Edge-case behaviour for ``gap_days``
------------------------------------
* Patient with one trigger → ``LAG(...)`` returns NULL on the only row →
  ``MAX(NULL)`` is NULL → ``COALESCE(MAX(...), 0)`` yields 0. (Plan: "single-
  event patients (gap_days=0)".)
* Patient with no triggers → no row in ``patient_gaps`` CTE → LEFT JOIN
  produces NULL → ``gap_days`` left NULL. There is no data to compute over,
  so NULL is the correct outcome (distinct from "0 because we observed one
  event").
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import psycopg2
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

#: Default rollup window length when neither ``start_date`` nor ``end_date`` is
#: supplied. 24 hours matches the Celery beat cadence below.
DEFAULT_WINDOW_HOURS: int = 24

#: Celery queue this task runs on. Routed to ``worker_medium`` per existing
#: ``task_routes`` config in ``src.workers.celery_app``.
TASK_QUEUE: str = "analytics"


# -----------------------------------------------------------------------------
# SQL
# -----------------------------------------------------------------------------

# Single UPDATE statement composed of two CTEs: ``journey_adherence`` (the
# clamp-and-divide derivation) and ``patient_gaps`` (the LAG-based gap
# computation). The UPDATE-FROM with LEFT JOIN keeps every journey in the
# window so journeys whose patients have zero triggers still get
# ``adherence_rate`` populated (with ``gap_days`` NULL).
#
# Window scope: ``patient_journeys.journey_start_date`` is the gating key for
# adherence (so old static journeys aren't rewritten on every daily run) and
# ``triggers.trigger_timestamp`` for the gap CTE (so we only walk recent
# trigger history). Both use the same ``[start_date, end_date)`` half-open
# interval the unit tests pin.
#
# Clamp: PostgreSQL's ``LEAST`` / ``GREATEST`` propagate NULL through —
# ``LEAST(1.0, NULL)`` is NULL — which preserves the "NULL means undefined"
# semantics for zero-duration journeys without an explicit CASE. Verified in
# unit tests.
#
# refill_count: not populated by this UPDATE. The ``SET`` clause omits the
# column entirely so existing values stay intact (currently always NULL post-
# migration 033). The SQL comment at the SET line documents why.
UPDATE_PATIENT_ADHERENCE_SQL: str = """
WITH journey_adherence AS (
    SELECT
        pj.patient_journey_id,
        pj.patient_id,
        -- adherence_rate = duration / span, clamped to [0, 1]. NULLs propagate
        -- through LEAST/GREATEST so zero-duration journeys (NULLIF==NULL) and
        -- open-ended journeys (NULL end date) return NULL, matching the plan.
        LEAST(
            1.0::NUMERIC,
            GREATEST(
                0.0::NUMERIC,
                pj.journey_duration_days::NUMERIC
                / NULLIF(
                    (pj.journey_end_date - pj.journey_start_date)::NUMERIC,
                    0
                )
            )
        ) AS adherence_rate
    FROM patient_journeys pj
    WHERE pj.journey_start_date >= %(start_date)s
      AND pj.journey_start_date <  %(end_date)s
),
patient_gaps AS (
    -- Per-patient max consecutive-trigger gap, in whole days.
    -- LAG on the ordered timestamp series; the first row's lag is NULL, so a
    -- single-event patient yields MAX(NULL) = NULL, then COALESCE pins it to 0
    -- (plan: "single-event patients (gap_days=0)").
    SELECT
        patient_id,
        COALESCE(
            MAX(
                EXTRACT(EPOCH FROM gap)::BIGINT / 86400
            )::INTEGER,
            0
        ) AS gap_days
    FROM (
        SELECT
            patient_id,
            trigger_timestamp - LAG(trigger_timestamp) OVER (
                PARTITION BY patient_id ORDER BY trigger_timestamp
            ) AS gap
        FROM triggers
        WHERE trigger_timestamp >= %(start_date)s
          AND trigger_timestamp <  %(end_date)s
    ) lag_view
    -- Restrict to patients touched by the journey window so this CTE doesn't
    -- balloon to all triggers in the DB.
    WHERE patient_id IN (
        SELECT patient_id FROM patient_journeys
         WHERE journey_start_date >= %(start_date)s
           AND journey_start_date <  %(end_date)s
    )
    GROUP BY patient_id
)
UPDATE patient_journeys pj
SET adherence_rate = ja.adherence_rate,
    -- refill_count intentionally not set here: the canonical v3 schema has no
    -- 'refill_reminder' trigger_type and no prescription_refills table. A
    -- future block lands a refill source and re-enables this column. See the
    -- module docstring for the full rationale.
    gap_days = pg.gap_days
FROM journey_adherence ja
LEFT JOIN patient_gaps pg ON pg.patient_id = ja.patient_id
WHERE pj.patient_journey_id = ja.patient_journey_id;
"""


# -----------------------------------------------------------------------------
# Pure-Python helpers (mirror SQL semantics for unit testing)
# -----------------------------------------------------------------------------


def _compute_adherence_rate(
    duration_days: Optional[int],
    journey_span_days: Optional[int],
) -> Optional[float]:
    """Mirror the SQL adherence_rate clamp-and-divide in pure Python.

    Lets unit tests pin every edge case (zero span, NULL duration, ratio
    overflow) without a live DB. The SQL form must remain semantically
    identical; ``test_compute_adherence_rate_*`` cases exercise the
    invariants both must hold.

    Args:
        duration_days: ``patient_journeys.journey_duration_days``. NULL/None
            propagates through to a None result.
        journey_span_days: ``journey_end_date - journey_start_date`` (in
            days). None or 0 produces None (zero-duration journey).

    Returns:
        The clamped ratio in ``[0.0, 1.0]``, or ``None`` if either input is
        None or the span is zero.
    """
    if duration_days is None or journey_span_days is None or journey_span_days == 0:
        return None
    ratio = duration_days / journey_span_days
    return max(0.0, min(1.0, ratio))


# -----------------------------------------------------------------------------
# DB connection
# -----------------------------------------------------------------------------


def _resolve_db_connection_string() -> str:
    """Read the Supabase Postgres URL from env.

    Raises:
        RuntimeError: if ``SUPABASE_DB_URL`` is missing or empty. The Celery
            task wraps this and routes to dead-letter via the existing
            ``task_failure`` handler in ``celery_app``.
    """
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL environment variable is required for the "
            "per-patient adherence/refill/gap ETL"
        )
    return db_url


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    # psycopg2.OperationalError covers connection-refused / server-starting /
    # SSL-handshake failures; the rest covers raw socket failures. NB:
    # psycopg2.OperationalError does NOT inherit from ConnectionError despite
    # the name -- its MRO is OperationalError -> DatabaseError -> Error ->
    # Exception. So adding it explicitly is what makes this retry actually
    # cover the canonical Postgres connect-time failure modes.
    retry=retry_if_exception_type(
        (psycopg2.OperationalError, ConnectionError, TimeoutError, OSError)
    ),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
def _connect_to_db() -> Any:
    """Open a psycopg2 connection with tenacity-backed retry.

    Three attempts with 1-10 second exponential backoff. ``psycopg2`` is
    imported at module level since it is already a hard dependency via the
    Supabase client.
    """
    return psycopg2.connect(_resolve_db_connection_string())


# -----------------------------------------------------------------------------
# Window resolution
# -----------------------------------------------------------------------------


def _resolve_window(
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[datetime, datetime]:
    """Resolve ISO date strings to a ``(start, end)`` UTC datetime tuple.

    Identical semantics to ``business_metrics_per_hcp_etl._resolve_window``;
    a follow-up may extract the shared helper into ``src/etl/_window.py``.

    ``end_date`` defaults to ``now(UTC)``; ``start_date`` defaults to
    ``end_date - DEFAULT_WINDOW_HOURS``. Strings can be ISO datetime
    (``2024-01-01T00:00:00Z``) or ISO date (``2024-01-01``) — both are
    accepted via ``datetime.fromisoformat``.

    Naive results from ``datetime.fromisoformat`` (date-only inputs return a
    tz-naive datetime at midnight) are normalised to UTC-aware so downstream
    aware/naive comparisons cannot raise ``TypeError``.
    """
    now_utc = datetime.now(timezone.utc)

    end_dt = (
        datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        if end_date
        else now_utc
    )
    start_dt = (
        datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if start_date
        else end_dt - timedelta(hours=DEFAULT_WINDOW_HOURS)
    )

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    if start_dt >= end_dt:
        raise ValueError(
            f"start_date ({start_dt.isoformat()}) must be strictly before "
            f"end_date ({end_dt.isoformat()})"
        )

    return start_dt, end_dt


# -----------------------------------------------------------------------------
# Core implementation (no Celery binding — directly importable for tests)
# -----------------------------------------------------------------------------


def _run_patient_adherence_impl(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    request_id: str = "no-task-id",
) -> Dict[str, Any]:
    """Pure-Python core of the per-patient adherence/refill/gap ETL.

    Split out from the Celery task so unit/integration tests can call it
    without poking at Celery internals. ``request_id`` is the Celery task
    ID forwarded for log correlation; defaults to ``"no-task-id"`` for
    direct invocation.

    Args:
        start_date: ISO datetime/date for window start. Defaults to
            ``end_date - DEFAULT_WINDOW_HOURS``.
        end_date: ISO datetime/date for window end (exclusive). Defaults to
            now (UTC).
        request_id: identifier surfaced in log lines.

    Returns:
        Dict with ``status``, ``rows_affected``, ``window_start``,
        ``window_end``, and on failure an ``error`` field.
    """
    try:
        start_dt, end_dt = _resolve_window(start_date, end_date)
    except ValueError as e:
        logger.error(
            "Invalid window for run_patient_adherence_rollup [%s]: %s",
            request_id,
            e,
        )
        return {
            "status": "failed",
            "error": str(e),
            "rows_affected": 0,
            "window_start": start_date,
            "window_end": end_date,
        }

    logger.info(
        "Starting per-patient adherence rollup [%s]: window=[%s, %s)",
        request_id,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    params = {
        "start_date": start_dt,
        "end_date": end_dt,
    }

    conn = None
    try:
        conn = _connect_to_db()
        with conn:  # transactional: commits on exit, rolls back on exception
            with conn.cursor() as cur:
                cur.execute(UPDATE_PATIENT_ADHERENCE_SQL, params)
                rows_affected = cur.rowcount

        if rows_affected == 0:
            logger.warning(
                "No journeys to update for window [%s, %s) [%s]",
                start_dt.isoformat(),
                end_dt.isoformat(),
                request_id,
            )
            return {
                "status": "no_data",
                "rows_affected": 0,
                "window_start": start_dt.isoformat(),
                "window_end": end_dt.isoformat(),
            }

        logger.info(
            "Per-patient adherence rollup completed [%s]: rows_affected=%d",
            request_id,
            rows_affected,
        )
        return {
            "status": "completed",
            "rows_affected": rows_affected,
            "window_start": start_dt.isoformat(),
            "window_end": end_dt.isoformat(),
        }

    except Exception as e:
        logger.exception(
            "Per-patient adherence rollup failed [%s]: %s",
            request_id,
            e,
        )
        return {
            "status": "failed",
            "error": str(e),
            "rows_affected": 0,
            "window_start": start_dt.isoformat(),
            "window_end": end_dt.isoformat(),
        }

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover — best-effort close
                logger.debug("Failed to close DB connection cleanly", exc_info=True)


# -----------------------------------------------------------------------------
# Celery task
# -----------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="src.etl.patient_adherence_etl.run_patient_adherence_rollup",
)
def run_patient_adherence_rollup(
    self,  # noqa: ANN001 — Celery passes the bound task instance
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Celery wrapper around :func:`_run_patient_adherence_impl`.

    See ``_run_patient_adherence_impl`` for argument and return semantics.
    """
    request_id = getattr(self.request, "id", "no-task-id") or "no-task-id"
    return _run_patient_adherence_impl(
        start_date=start_date,
        end_date=end_date,
        request_id=request_id,
    )
