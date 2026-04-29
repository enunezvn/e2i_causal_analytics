"""Per-HCP business_metrics rollup ETL (block 6B-infra-2a).

Aggregates ``triggers`` joined to ``patient_journeys`` (for brand derivation)
and ``hcp_profiles`` (for territory + region) into per-(hcp_id, brand,
metric_date) rollup rows on the ``business_metrics`` table.

Why brand comes from ``patient_journeys``
-----------------------------------------
Migration 033 added ``triggers.brand_id`` with the sentinel ``'UNKNOWN'`` and
made no attempt to back-fill — there is no clean trigger->brand join key in
the canonical v3 schema. This ETL therefore derives brand from
``patient_journeys.brand`` (the canonical ``brand_type`` enum), picking the
most recent journey for each patient whose ``journey_start_date`` is at or
before the trigger's timestamp. The right join shape is a LATERAL subquery
with ``ORDER BY journey_start_date DESC LIMIT 1``.

Idempotency strategy
--------------------
The plan asks for ``ON CONFLICT (hcp_id, brand, metric_date) DO UPDATE`` but
``business_metrics`` PRIMARY KEY is ``metric_id VARCHAR(50)`` and migration
033 did NOT add a UNIQUE constraint on the natural key. Adding such a
constraint from inside an ETL would be poor hygiene (mixes schema migration
with rollup logic). Instead we synthesise a deterministic ``metric_id`` by
md5-hashing the natural key ``(hcp_id, brand, metric_date)`` and rely on the
existing PK for ``ON CONFLICT (metric_id)``. This achieves the idempotency
intent of the plan without DDL drift.

The hash is required because the column is ``VARCHAR(50)`` and a naive
concat of ``hcp_rollup_<hcp_id>_<brand>_<YYYY-MM-DD>`` can exceed 50
characters in the worst case (max-length hcp_id + ``Remibrutinib`` + ISO
date). The format is now ``hcp_rollup_<md5_hex_32>`` — a constant 43 chars
that fits comfortably and is still deterministic (md5 of the same input
always yields the same digest, so re-runs map to the same row).

Both the SQL ``md5(...)`` call and the Python helper ``_build_metric_id``
must produce byte-identical strings; see ``test_sql_metric_id_uses_md5``
which pins the SQL component-order against drift.

Why ``engagement_score`` and ``call_frequency`` are NULL
--------------------------------------------------------
The plan calls for both fields to be sourced from an ``interactions`` table.
That table does not exist in
``database/core/e2i_ml_complete_v3_schema.sql`` (verified). They are left
NULL in this ETL with a SQL comment naming the missing source; a follow-up
block lands the ``interactions`` table and this ETL evolves then.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any, Dict, Optional

# Re-exported from _common so existing test imports
# (`from src.etl.business_metrics_per_hcp_etl import _resolve_window`, etc.)
# stay valid post-extraction. The helpers themselves moved to ``_common.py``
# in fix-up for 6B-infra-2b so a third ETL (6B-infra-2c) can import the same
# code without creating a third duplicate copy.
from src.etl._common import (  # noqa: F401 — re-exported for backward compatibility
    _connect_to_db,
    _resolve_db_connection_string,
    _resolve_window,
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

#: Deterministic metric_id prefix; encoded into the natural key for idempotency.
METRIC_ID_PREFIX: str = "hcp_rollup"

#: Marker stamped on per-HCP rollup rows in ``business_metrics.metric_type``.
METRIC_TYPE: str = "per_hcp_rollup"


# -----------------------------------------------------------------------------
# SQL
# -----------------------------------------------------------------------------

# Pure-SQL CTE chain. Aggregation stays in PostgreSQL — no row shuffling
# through Python.
#
# Pipeline:
#   1. triggers_with_brand: triggers x most-recent-prior patient_journey.brand,
#      filtered to [start_date, end_date).
#   2. hcp_brand_daily: collapse to per-(hcp_id, brand, metric_date) counts +
#      conversion_rate (NRx / TRx with NULLIF guard).
#   3. territory_totals: sum total_rx_count per (territory_id, brand,
#      metric_date) so the SELECT can compute market_share = HCP /
#      territory_total within each territory window.
#   4. INSERT with deterministic metric_id and ON CONFLICT DO UPDATE for
#      idempotency.
#
# Columns NOT populated:
#   * engagement_score, call_frequency: source table ``interactions`` does
#     not exist in canonical schema -- left NULL until a future block lands
#     the table.
#   * value, target, achievement_rate, year_over_year_change,
#     month_over_month_change, roi, statistical_significance, CI bounds,
#     sample_size: these belong to the pre-existing per-(brand, region)
#     aggregate rows and are out of scope for the per-HCP rollup.
#
# data_split: defaults to 'unassigned' per column default; the rollup row is
# fed into the ML splitter elsewhere.
INSERT_PER_HCP_ROLLUP_SQL: str = """
WITH triggers_with_brand AS (
    SELECT
        t.hcp_id,
        pj.brand,
        DATE(t.trigger_timestamp)         AS metric_date,
        t.delivery_status,
        t.acceptance_status
    FROM triggers t
    JOIN LATERAL (
        SELECT pj_inner.brand
          FROM patient_journeys pj_inner
         WHERE pj_inner.patient_id = t.patient_id
           AND pj_inner.brand IS NOT NULL
           AND pj_inner.journey_start_date <= t.trigger_timestamp
         ORDER BY pj_inner.journey_start_date DESC
         LIMIT 1
    ) pj ON TRUE
    WHERE t.trigger_timestamp >= %(start_date)s
      AND t.trigger_timestamp <  %(end_date)s
      AND t.hcp_id IS NOT NULL
),
hcp_brand_daily AS (
    SELECT
        hcp_id,
        brand,
        metric_date,
        COUNT(*) FILTER (WHERE delivery_status = 'delivered')                       AS trx_count,
        COUNT(*) FILTER (WHERE acceptance_status IN ('accepted', 'responded'))      AS nrx_count,
        COUNT(*)                                                                    AS total_rx_count,
        COALESCE(
            COUNT(*) FILTER (WHERE acceptance_status IN ('accepted', 'responded'))::NUMERIC
            / NULLIF(COUNT(*) FILTER (WHERE delivery_status = 'delivered'), 0),
            0
        )                                                                           AS conversion_rate
    FROM triggers_with_brand
    GROUP BY hcp_id, brand, metric_date
),
territory_totals AS (
    SELECT
        hp.territory_id,
        hbd.brand,
        hbd.metric_date,
        SUM(hbd.total_rx_count) AS territory_total
    FROM hcp_brand_daily hbd
    JOIN hcp_profiles  hp ON hbd.hcp_id = hp.hcp_id
    GROUP BY hp.territory_id, hbd.brand, hbd.metric_date
)
INSERT INTO business_metrics (
    metric_id,
    metric_date,
    metric_type,
    brand,
    region,
    hcp_id,
    trx_count,
    nrx_count,
    total_rx_count,
    market_share,
    conversion_rate,
    -- engagement_score and call_frequency intentionally NULL: the
    -- canonical `interactions` table does not exist in v3 schema. A
    -- future ETL block will populate these once the table lands.
    created_at
)
SELECT
    -- metric_id = '<prefix>_' || md5(hcp_id ':' brand ':' metric_date).
    -- Hashed because business_metrics.metric_id is VARCHAR(50); the
    -- natural-key concat could overflow it in the worst case (max
    -- hcp_id + 'Remibrutinib' + ISO date). md5 is deterministic so
    -- idempotency holds; ':' separators avoid ambiguity if any
    -- component ever contains underscores. The Python helper
    -- ``_build_metric_id`` mirrors this construction byte-for-byte.
    %(metric_id_prefix)s || '_' || md5(
        hbd.hcp_id || ':' || hbd.brand::TEXT || ':' || hbd.metric_date::TEXT
    )                                           AS metric_id,
    hbd.metric_date,
    %(metric_type)s                             AS metric_type,
    hbd.brand,
    hp.geographic_region                        AS region,
    hbd.hcp_id,
    hbd.trx_count,
    hbd.nrx_count,
    hbd.total_rx_count,
    CASE
        WHEN tt.territory_total > 0
            THEN hbd.total_rx_count::NUMERIC / tt.territory_total
        ELSE 0
    END                                         AS market_share,
    hbd.conversion_rate,
    NOW()                                       AS created_at
FROM hcp_brand_daily hbd
JOIN hcp_profiles      hp ON hbd.hcp_id = hp.hcp_id
JOIN territory_totals  tt
  ON tt.territory_id = hp.territory_id
 AND tt.brand        = hbd.brand
 AND tt.metric_date  = hbd.metric_date
ON CONFLICT (metric_id) DO UPDATE SET
    trx_count       = EXCLUDED.trx_count,
    nrx_count       = EXCLUDED.nrx_count,
    total_rx_count  = EXCLUDED.total_rx_count,
    market_share    = EXCLUDED.market_share,
    conversion_rate = EXCLUDED.conversion_rate,
    region          = EXCLUDED.region,
    metric_type     = EXCLUDED.metric_type;
"""


# -----------------------------------------------------------------------------
# metric_id helpers
# -----------------------------------------------------------------------------


def _build_metric_id(hcp_id: str, brand: str, metric_date: date) -> str:
    """Mirror the SQL ``metric_id`` construction in pure Python.

    The SQL builds the same string with ``%(metric_id_prefix)s || '_' ||
    md5(hbd.hcp_id || ':' || hbd.brand::TEXT || ':' || hbd.metric_date::TEXT)``;
    this helper exists so unit tests can pin the length property against
    ``business_metrics.metric_id VARCHAR(50)`` without a live DB. Any change
    here must also change the SQL (and vice versa) — the SQL-shape test
    pins the natural-key component order so the two cannot drift silently.

    Args:
        hcp_id: The HCP identifier as it appears in ``hcp_profiles.hcp_id``.
        brand: A ``brand_type`` enum value (e.g. ``"Remibrutinib"``).
        metric_date: The rollup day. Serialised via
            :meth:`datetime.date.isoformat` to match Postgres' default
            ``DATE::TEXT`` cast (``YYYY-MM-DD``).

    Returns:
        The deterministic ``metric_id`` value, always 43 characters long
        (``"hcp_rollup_"`` is 11 chars + 32-hex md5 digest).
    """
    natural_key = f"{hcp_id}:{brand}:{metric_date.isoformat()}"
    digest = hashlib.md5(natural_key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{METRIC_ID_PREFIX}_{digest}"


# -----------------------------------------------------------------------------
# DB connection + window resolution: see ``src.etl._common``. The names
# ``_resolve_db_connection_string``, ``_connect_to_db`` and ``_resolve_window``
# are re-exported at the top of this module.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Core implementation (no Celery binding — directly importable for tests)
# -----------------------------------------------------------------------------


def _run_per_hcp_rollup_impl(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    request_id: str = "no-task-id",
) -> Dict[str, Any]:
    """Pure-Python core of the per-HCP rollup ETL.

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
        logger.error("Invalid window for run_per_hcp_rollup [%s]: %s", request_id, e)
        return {
            "status": "failed",
            "error": str(e),
            "rows_affected": 0,
            "window_start": start_date,
            "window_end": end_date,
        }

    logger.info(
        "Starting per-HCP business_metrics rollup [%s]: window=[%s, %s)",
        request_id,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    params = {
        "start_date": start_dt,
        "end_date": end_dt,
        "metric_id_prefix": METRIC_ID_PREFIX,
        "metric_type": METRIC_TYPE,
    }

    conn = None
    try:
        conn = _connect_to_db()
        with conn:  # transactional: commits on exit, rolls back on exception
            with conn.cursor() as cur:
                cur.execute(INSERT_PER_HCP_ROLLUP_SQL, params)
                rows_affected = cur.rowcount

        if rows_affected == 0:
            logger.warning(
                "No rows to roll up for window [%s, %s) [%s]",
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
            "Per-HCP business_metrics rollup completed [%s]: rows_affected=%d",
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
            "Per-HCP business_metrics rollup failed [%s]: %s",
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
    name="src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup",
)
def run_per_hcp_rollup(
    self,  # noqa: ANN001 — Celery passes the bound task instance
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Celery wrapper around :func:`_run_per_hcp_rollup_impl`.

    See ``_run_per_hcp_rollup_impl`` for argument and return semantics.
    """
    request_id = getattr(self.request, "id", "no-task-id") or "no-task-id"
    return _run_per_hcp_rollup_impl(
        start_date=start_date,
        end_date=end_date,
        request_id=request_id,
    )
