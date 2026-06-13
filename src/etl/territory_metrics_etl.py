"""Territory rollup ETL (block 6B-infra-2c).

Aggregates per-HCP ``business_metrics`` rows (produced by 6B-infra-2a) and
``triggers`` rows (joined to ``hcp_profiles.territory_id``) into per-
``(territory_id, metric_date)`` rollup rows on the ``territory_metrics``
table created by migration 031 (with ``event_timestamp`` added by migration
033).

Aggregations
------------

* ``total_trx`` — SUM of ``business_metrics.trx_count`` across the
  territory's per-HCP rollup rows for the metric_date.
* ``total_nrx`` — SUM of ``business_metrics.nrx_count`` likewise.
* ``active_hcp_count`` — DISTINCT ``hcp_id`` count from ``triggers`` whose
  ``trigger_timestamp`` falls in the **30-day** window ending on (and
  inclusive of) ``metric_date``. The 30-day lookback is independent of the
  ETL run window — it is a sliding count *as of metric_date*. See "Window
  semantics" below.
* ``covered_lives`` — SUM of ``hcp_profiles.total_patient_volume`` across
  the territory's HCPs. Time-invariant in the synthetic schema; replicated
  across each ``(territory_id, metric_date)`` row.

Why ``market_potential`` and ``resource_allocation_score`` stay NULL
--------------------------------------------------------------------
The plan calls for both columns to be populated from real Reltio/Veeva
sources when available, NULL otherwise. Migration 031's seed values were
``random()`` placeholders — the scaffolding this ETL replaces. Real Reltio/
Veeva integration is out of scope for this block; we leave the columns NULL
on **new** rows and **untouched** on existing rows so the random seeds
created by 031 (if any) persist until a real source lands. A future ETL
block can:

1. Add a Reltio / Veeva mirror table for these two metrics.
2. Re-enable the corresponding SET clauses here.

INSERT writes NULL explicitly because migration 033 dropped the NOT NULL
constraint that 031 had set; ON CONFLICT DO UPDATE intentionally omits
these so the random seed from 031 survives until real Reltio/Veeva
integration.

This mirrors how 6B-infra-2a left ``engagement_score`` and
``call_frequency`` NULL because the ``interactions`` table doesn't exist
yet, and how 6B-infra-2b left ``refill_count`` NULL because the canonical
v3 schema has no first-class refill concept.

Provenance inheritance (issue #895)
-----------------------------------
``territory_metrics`` had NO ``is_synthetic`` column until migration 074
(031 predates the 063/069 provenance family), so this rollup used to write
provenance-less rows derived from tagged inputs — second-order laundering.
Post-074 the SQL inherits ``is_synthetic = bool(any synthetic input)``:

* ``per_hcp_in_territory`` BOOL_ORs the (already-inherited, post-#895)
  ``business_metrics.is_synthetic`` plus ``hcp_profiles.is_synthetic`` —
  inheritance composes across the two-stage rollup;
* ``active_hcp_per_territory_date`` taints only on triggers that actually
  contribute to the DISTINCT count (non-NULL after the LEFT JOIN);
* ``territory_hcp_volume`` BOOL_ORs profile provenance (every profile row
  contributes to the covered_lives SUM).

Mixed-substrate semantics match 6B-infra-2a: an aggregate mixing real and
synthetic inputs is tagged synthetic (fail-closed, #872 direction). The ON
CONFLICT update arm recomputes the tag with the values.

DEPLOY ORDERING: migration 074 must be applied before this code runs in
production — the INSERT names ``is_synthetic`` and fails closed (42703
undefined_column, no rows written) on a pre-074 schema.

Window semantics
----------------
Two windows operate in the SQL:

1. The ETL "run window" ``[start_date, end_date)`` — gates which
   metric_dates are computed for the rollup. Defaults to the last 24h
   (matching the daily beat cadence). At daily cadence this collapses to a
   single metric_date.
2. The 30-day "active HCP" window — backward-looking from each
   metric_date. ``trigger_timestamp >= metric_date - INTERVAL '30 days'``
   AND ``trigger_timestamp < metric_date + INTERVAL '1 day'`` (inclusive of
   the metric_date itself).

Order dependency
----------------
This ETL aggregates ``business_metrics`` rows produced by 6B-infra-2a
(``run_per_hcp_rollup``). In production the per-HCP rollup must run **before**
the territory rollup for the day. Celery beat does not enforce ordering;
the daily 24h schedules will fire concurrently. If 2a hasn't run yet for
the target metric_date the ETL will simply produce zeros for ``total_trx``
/ ``total_nrx`` (and zero ``active_hcp_count`` if no triggers landed), and
log a warning when no rows materialise. A future block can sequence them
explicitly via Celery chords or a chain.

Idempotency strategy
--------------------
``territory_metrics`` PRIMARY KEY is ``(territory_id, metric_date)``,
matching the natural rollup key exactly, so ``ON CONFLICT (territory_id,
metric_date) DO UPDATE`` works directly — no md5-hashing trick like
6B-infra-2a needed. ``market_potential`` and ``resource_allocation_score``
are deliberately omitted from the SET clause so existing non-NULL values
(from migration 031's random seed, if present) stay intact. They are
included in the INSERT column list with explicit NULL so new rows reflect
the spec's "NULL when no real source" semantics rather than picking up
the table default that migration 031 originally declared.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# Re-exported from _common so existing test imports
# (`from src.etl.territory_metrics_etl import _resolve_window`, etc.)
# stay valid even if a future cleanup drops the shim. Mirrors the shape of
# 6B-infra-2a / 2b post-extraction.
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

#: Marker for the per-HCP rollup row source. Must match
#: ``src.etl.business_metrics_per_hcp_etl.METRIC_TYPE``; redeclared here
#: rather than imported to keep the two ETLs cleanly separable (a future
#: split into different services would not need to import each other).
PER_HCP_METRIC_TYPE: str = "per_hcp_rollup"


# -----------------------------------------------------------------------------
# SQL
# -----------------------------------------------------------------------------

# Pure-SQL CTE chain. Aggregation stays in PostgreSQL — no row shuffling
# through Python.
#
# Pipeline:
#   1. metric_dates: distinct metric_dates within the run window. Sourced
#      from `business_metrics.metric_date` for per-HCP rollup rows so we
#      only roll up days where 6B-infra-2a has produced output (avoids
#      writing all-zeros rows for days with nothing to aggregate).
#   2. territory_dates: cross-product of every territory_id with each
#      metric_date in the window. Anchors LEFT JOINs so a territory with
#      no business_metrics for the day still gets a row (with zeros).
#   3. per_hcp_in_territory: SUM(trx_count) / SUM(nrx_count) per
#      (territory_id, metric_date) from per-HCP rollup rows.
#   4. active_hcp_per_territory_date: DISTINCT hcp_id count for the 30-day
#      backward-looking window ending on each metric_date.
#   5. territory_hcp_volume: time-invariant SUM(total_patient_volume) per
#      territory. JOINed (no date dimension).
#   6. INSERT INTO territory_metrics with ON CONFLICT (territory_id,
#      metric_date) DO UPDATE on the four real aggregates only.
#
# Columns NOT set on conflict (preserved):
#   * market_potential: random seed from migration 031 if present, else
#     NULL on new rows. Real Reltio source not yet integrated.
#   * resource_allocation_score: same reasoning.
#   These are INCLUDED in the INSERT column list with explicit NULL on
#   new rows (the spec calls for "NULL otherwise (NOT random)"). They are
#   OMITTED from the ON CONFLICT SET clause so existing values (e.g.
#   migration 031's random seed) survive re-runs unchanged.
#
# NOTE: migration 033 (033.5) drops the NOT NULL + DEFAULT 0 that 031 had
#   declared on these two columns, so writing NULL explicitly is now
#   well-defined. Pre-existing 031 rows keep their random values until
#   real Reltio/Veeva integration replaces this preservation behaviour.
INSERT_TERRITORY_ROLLUP_SQL: str = """
WITH metric_dates AS (
    SELECT DISTINCT bm.metric_date
      FROM business_metrics bm
     WHERE bm.metric_type = %(per_hcp_metric_type)s
       AND bm.hcp_id IS NOT NULL
       AND bm.metric_date >= %(start_date)s::DATE
       AND bm.metric_date <  %(end_date)s::DATE
),
territories AS (
    SELECT DISTINCT territory_id
      FROM hcp_profiles
     WHERE territory_id IS NOT NULL
),
territory_dates AS (
    -- Cross-product so every (territory, date) cell exists, even when the
    -- territory has no business_metrics rows for the day. LEFT JOINs below
    -- coalesce missing aggregates to zero.
    SELECT t.territory_id, md.metric_date
      FROM territories t
      CROSS JOIN metric_dates md
),
per_hcp_in_territory AS (
    -- total_trx / total_nrx per (territory, date) from 6B-infra-2a output.
    -- Filter to per-HCP rollup rows so per-(brand, region) aggregate rows
    -- (which keep hcp_id IS NULL) are excluded.
    SELECT
        hp.territory_id,
        bm.metric_date,
        SUM(COALESCE(bm.trx_count, 0))::BIGINT AS total_trx,
        SUM(COALESCE(bm.nrx_count, 0))::BIGINT AS total_nrx,
        -- Provenance inheritance (issue #895): the per-HCP rollup rows are
        -- themselves provenance-tagged (6B-infra-2a post-#895), so this
        -- composes -- any synthetic input row (or synthetic HCP profile)
        -- taints the territory aggregate.
        BOOL_OR(bm.is_synthetic OR hp.is_synthetic) AS any_synthetic
    FROM business_metrics bm
    JOIN hcp_profiles      hp ON bm.hcp_id = hp.hcp_id
    WHERE bm.metric_type = %(per_hcp_metric_type)s
      AND bm.hcp_id IS NOT NULL
      AND bm.metric_date >= %(start_date)s::DATE
      AND bm.metric_date <  %(end_date)s::DATE
    GROUP BY hp.territory_id, bm.metric_date
),
active_hcp_per_territory_date AS (
    -- DISTINCT hcp_id with at least one trigger in the 30-day window
    -- ending on metric_date (inclusive). The 30-day lookback is INDEPENDENT
    -- of the run window -- it is the spec's "active HCP" definition,
    -- evaluated as-of metric_date.
    --
    -- Inclusive of metric_date is enforced by `< td.metric_date + INTERVAL
    -- '1 day'`; the lower bound `>= td.metric_date - INTERVAL '30 days'`
    -- yields a 31-day inclusive interval (30 days back through metric_date
    -- itself), matching the plan's "in last 30 days" wording.
    SELECT
        td.territory_id,
        td.metric_date,
        COUNT(DISTINCT t.hcp_id)::BIGINT AS active_hcp_count,
        -- Provenance inheritance (issue #895): only triggers that actually
        -- contribute to the DISTINCT count (t.hcp_id IS NOT NULL after the
        -- LEFT JOIN) can taint it. The guard also keeps the BOOL_OR input
        -- non-NULL on anchor rows with no matching trigger.
        BOOL_OR(t.hcp_id IS NOT NULL AND (t.is_synthetic OR hp.is_synthetic))
            AS any_synthetic
    FROM territory_dates td
    LEFT JOIN hcp_profiles hp ON hp.territory_id = td.territory_id
    LEFT JOIN triggers     t  ON t.hcp_id = hp.hcp_id
                              AND t.trigger_timestamp >= td.metric_date - INTERVAL '30 days'
                              AND t.trigger_timestamp <  td.metric_date + INTERVAL '1 day'
    GROUP BY td.territory_id, td.metric_date
),
territory_hcp_volume AS (
    -- covered_lives = SUM(total_patient_volume) per territory. Time-
    -- invariant at the schema level (hcp_profiles.total_patient_volume
    -- doesn't change daily in synthetic data). Replicated across each
    -- (territory, date) row of the rollup.
    SELECT
        territory_id,
        SUM(COALESCE(total_patient_volume, 0))::BIGINT AS covered_lives,
        -- Provenance inheritance (issue #895): every profile row in the
        -- territory contributes to the covered_lives SUM, so any synthetic
        -- profile taints it.
        BOOL_OR(is_synthetic) AS any_synthetic
    FROM hcp_profiles
    WHERE territory_id IS NOT NULL
    GROUP BY territory_id
)
INSERT INTO territory_metrics (
    territory_id,
    metric_date,
    total_trx,
    total_nrx,
    active_hcp_count,
    covered_lives,
    -- market_potential / resource_allocation_score: NULL on new rows;
    -- existing seeds preserved on UPDATE. Migration 033 (033.5) dropped
    -- the NOT NULL + DEFAULT 0 that 031 had declared, so writing NULL
    -- explicitly is the spec-faithful "no real Reltio/Veeva source"
    -- behaviour. The ON CONFLICT SET clause below intentionally omits
    -- both columns so existing values (e.g. 031's random seed) survive
    -- re-runs untouched until a real source ETL replaces this.
    market_potential,
    resource_allocation_score,
    is_synthetic,
    created_at
)
SELECT
    td.territory_id,
    td.metric_date,
    COALESCE(pht.total_trx, 0)               AS total_trx,
    COALESCE(pht.total_nrx, 0)               AS total_nrx,
    COALESCE(ahd.active_hcp_count, 0)        AS active_hcp_count,
    COALESCE(thv.covered_lives, 0)           AS covered_lives,
    CAST(NULL AS DOUBLE PRECISION)           AS market_potential,
    CAST(NULL AS DOUBLE PRECISION)           AS resource_allocation_score,
    -- Inherited provenance (issue #895, column added by migration 074):
    -- synthetic if ANY of the three aggregate sources mixed in a synthetic
    -- input row. COALESCE because the LEFT JOINs yield NULL for territory/
    -- date cells with no matching aggregate (those contribute zeros, not
    -- contamination).
    (COALESCE(pht.any_synthetic, false)
     OR COALESCE(ahd.any_synthetic, false)
     OR COALESCE(thv.any_synthetic, false))  AS is_synthetic,
    NOW()                                    AS created_at
FROM territory_dates td
LEFT JOIN per_hcp_in_territory          pht ON pht.territory_id = td.territory_id
                                            AND pht.metric_date  = td.metric_date
LEFT JOIN active_hcp_per_territory_date ahd ON ahd.territory_id = td.territory_id
                                            AND ahd.metric_date  = td.metric_date
LEFT JOIN territory_hcp_volume          thv ON thv.territory_id = td.territory_id
ON CONFLICT (territory_id, metric_date) DO UPDATE SET
    total_trx        = EXCLUDED.total_trx,
    total_nrx        = EXCLUDED.total_nrx,
    active_hcp_count = EXCLUDED.active_hcp_count,
    covered_lives    = EXCLUDED.covered_lives,
    -- Re-runs recompute every aggregate from the base tables; the
    -- provenance tag tracks the same recomputation (issue #895). Keeping a
    -- stale tag here would let laundered semantics survive via the update
    -- arm.
    is_synthetic     = EXCLUDED.is_synthetic;
"""


# -----------------------------------------------------------------------------
# DB connection + window resolution: see ``src.etl._common``. The names
# ``_resolve_db_connection_string``, ``_connect_to_db`` and ``_resolve_window``
# are re-exported at the top of this module.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Core implementation (no Celery binding — directly importable for tests)
# -----------------------------------------------------------------------------


def _run_territory_rollup_impl(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    request_id: str = "no-task-id",
) -> Dict[str, Any]:
    """Pure-Python core of the territory rollup ETL.

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
            "Invalid window for run_territory_rollup [%s]: %s",
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
        "Starting territory_metrics rollup [%s]: window=[%s, %s)",
        request_id,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    params = {
        "start_date": start_dt,
        "end_date": end_dt,
        "per_hcp_metric_type": PER_HCP_METRIC_TYPE,
    }

    conn = None
    try:
        conn = _connect_to_db()
        with conn:  # transactional: commits on exit, rolls back on exception
            with conn.cursor() as cur:
                cur.execute(INSERT_TERRITORY_ROLLUP_SQL, params)
                rows_affected = cur.rowcount

        if rows_affected == 0:
            # Most likely: 6B-infra-2a hasn't produced per-HCP rollup rows
            # yet for the run window. The CTE filters on metric_type =
            # 'per_hcp_rollup' so an empty per-HCP set yields an empty
            # metric_dates set yields an empty cross-product yields zero
            # INSERT rows. Document the order dependency and warn loudly so
            # operators notice if 2a is misbehaving.
            logger.warning(
                "No territory rollup rows for window [%s, %s) [%s] -- "
                "check that per-HCP business_metrics rollup (6B-infra-2a) "
                "has run for the target metric_dates",
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
            "Territory_metrics rollup completed [%s]: rows_affected=%d",
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
            "Territory_metrics rollup failed [%s]: %s",
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
    name="src.etl.territory_metrics_etl.run_territory_rollup",
)
def run_territory_rollup(
    self,  # noqa: ANN001 — Celery passes the bound task instance
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Celery wrapper around :func:`_run_territory_rollup_impl`.

    See ``_run_territory_rollup_impl`` for argument and return semantics.
    """
    request_id = getattr(self.request, "id", "no-task-id") or "no-task-id"
    return _run_territory_rollup_impl(
        start_date=start_date,
        end_date=end_date,
        request_id=request_id,
    )
