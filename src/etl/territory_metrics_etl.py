"""Territory rollup ETL (block 6B-infra-2c).

Aggregates per-HCP ``business_metrics`` rows (produced by 6B-infra-2a) and
``triggers`` rows (joined to ``hcp_profiles.territory_id``) into per-
``(territory_id, metric_date)`` rollup rows on the ``territory_metrics``
table created by migration 031 (with ``event_timestamp`` added by migration
033).

Aggregations
------------

* ``total_trx`` — SUM of ``business_metrics.triggers_delivered_count`` across the
  territory's per-HCP rollup rows for the metric_date.
* ``total_nrx`` — SUM of ``business_metrics.triggers_accepted_count`` likewise.

  (Canonical TRx lane: the two territory OUTPUT columns ``total_trx`` and
  ``total_nrx`` are still named for prescriptions although they sum trigger
  counts. This lane renames only the SOURCE columns, so the mislabel now stops
  at the ``territory_metrics`` table instead of running through both. No issue
  tracks the output-column names as of 2026-09-17 — searched open and closed;
  the nearest hits are #1640, which only mentions ``total_trx`` in passing, and
  #895, which is about provenance on this same file rather than naming.)
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

#: Scheduled runs select territory dates from ARRIVALS over one weekly batch cycle plus
#: margin. Redeclared (not imported) like PER_HCP_METRIC_TYPE; a unit test pins it equal to
#: ``business_metrics_per_hcp_etl.ARRIVAL_WINDOW_HOURS`` (canonical TRx lane, owner decision #3).
ARRIVAL_MARGIN_HOURS: int = 6
ARRIVAL_WINDOW_HOURS: int = 7 * 24 + ARRIVAL_MARGIN_HOURS

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
#      Scheduled runs instead select the dates touched by ARRIVALS
#      (_TERRITORY_METRIC_DATES_BY_ARRIVAL); either way each date is rebuilt whole.
#   2. territory_dates: cross-product of every territory_id with each
#      metric_date in the window. Anchors LEFT JOINs so a territory with
#      no business_metrics for the day still gets a row (with zeros).
#   3. per_hcp_in_territory: SUM(triggers_delivered_count) / SUM(triggers_accepted_count) per
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
_TERRITORY_METRIC_DATES_BY_WINDOW: str = """
metric_dates AS (
    SELECT DISTINCT bm.metric_date
      FROM business_metrics bm
     WHERE bm.metric_type = %(per_hcp_metric_type)s
       AND bm.hcp_id IS NOT NULL
       AND bm.metric_date >= %(start_date)s::DATE
       AND bm.metric_date <  %(end_date)s::DATE
)"""

_TERRITORY_METRIC_DATES_BY_ARRIVAL: str = """
metric_dates AS (
    -- Late-arrival fix (canonical TRx lane, owner decision #3). A territory date is affected
    -- when (a) a per-HCP row for it was WRITTEN in the run window, or (b) a trigger that
    -- ARRIVED in the run window falls inside the date's 30-day active-HCP lookback. (b) is
    -- required: business_metrics has no updated_at and the per-HCP ON CONFLICT arm keeps
    -- created_at, so a date re-rolled for late triggers keeps its old created_at; and a late
    -- trigger on day D changes active_hcp_count for every date in [D, D + 30].
    SELECT per_hcp.metric_date
      FROM (
            SELECT bm.metric_date,
                   BOOL_OR(bm.created_at >= %(start_date)s AND bm.created_at < %(end_date)s)
                       AS written_in_window
              FROM business_metrics bm
             WHERE bm.metric_type = %(per_hcp_metric_type)s
               AND bm.hcp_id IS NOT NULL
             GROUP BY bm.metric_date
           ) per_hcp
     WHERE per_hcp.written_in_window
        OR EXISTS (
            SELECT 1
              FROM triggers t
             WHERE t.created_at >= %(start_date)s
               AND t.created_at <  %(end_date)s
               AND t.hcp_id IS NOT NULL
               AND t.trigger_timestamp >= per_hcp.metric_date - INTERVAL '30 days'
               AND t.trigger_timestamp <  per_hcp.metric_date + INTERVAL '1 day'
        )
)"""

_TERRITORY_ROLLUP_CTES_TEMPLATE: str = """
WITH __METRIC_DATES_CTE__,
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
        SUM(COALESCE(bm.triggers_delivered_count, 0))::BIGINT AS total_trx,
        SUM(COALESCE(bm.triggers_accepted_count, 0))::BIGINT AS total_nrx,
        -- Provenance inheritance (issue #895): the per-HCP rollup rows are
        -- themselves provenance-tagged (6B-infra-2a post-#895), so this
        -- composes -- any synthetic input row (or synthetic HCP profile)
        -- taints the territory aggregate.
        BOOL_OR(bm.is_synthetic OR hp.is_synthetic) AS any_synthetic
    FROM business_metrics bm
    JOIN hcp_profiles      hp ON bm.hcp_id = hp.hcp_id
    WHERE bm.metric_type = %(per_hcp_metric_type)s
      AND bm.hcp_id IS NOT NULL
      AND bm.metric_date IN (SELECT metric_date FROM metric_dates)
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
"""

_TERRITORY_ROLLUP_INSERT_HEAD: str = """
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
"""

_TERRITORY_ROLLUP_ROWS_SELECT: str = """
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
"""

_TERRITORY_ROLLUP_ON_CONFLICT: str = """
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


def _compose_territory_rollup(metric_dates_cte: str) -> str:
    return (
        _TERRITORY_ROLLUP_CTES_TEMPLATE.replace("__METRIC_DATES_CTE__", metric_dates_cte)
        + _TERRITORY_ROLLUP_INSERT_HEAD
        + _TERRITORY_ROLLUP_ROWS_SELECT
        + _TERRITORY_ROLLUP_ON_CONFLICT
    )


#: Explicit ``start_date``/``end_date`` (manual backfills): per-HCP metric_dates in the window.
INSERT_TERRITORY_ROLLUP_SQL: str = _compose_territory_rollup(_TERRITORY_METRIC_DATES_BY_WINDOW)

#: Scheduled run (no dates): the per-HCP metric_dates touched by arrivals in the window.
INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL: str = _compose_territory_rollup(
    _TERRITORY_METRIC_DATES_BY_ARRIVAL
)

#: codex r13-08: a date whose per-HCP rows all disappeared is no longer selectable, so its
#: stale territory rows must be deleted by calendar range in an explicit run.
#: codex r14-04: ...but only rows THIS ETL owns. market_potential and resource_allocation_score
#: are the two columns the ON CONFLICT SET arm refuses to overwrite; a row on which neither has
#: ever been set is one nothing but this rollup produced. An obsolete row that carries either is
#: left alone and reported as rows_obsolete_foreign — a stale rollup is a reporting error, while
#: deleting another writer's only copy of a budget figure is data loss.
_TERRITORY_OWNED_BY_THIS_ETL: str = (
    "m.market_potential IS NULL AND m.resource_allocation_score IS NULL"
)

#: The exact complement of the predicate above, declared ONCE and substituted into the preview
#: rather than restated inline. The two must PARTITION the obsolete set — every stale row
#: counted exactly once — and a unit test EXTRACTS both from the composed SQL and evaluates
#: them over every null combination, so a divergence here cannot pass as agreement.
_TERRITORY_NOT_OWNED_BY_THIS_ETL: str = (
    "o.market_potential IS NOT NULL OR o.resource_allocation_score IS NOT NULL"
)

_TERRITORY_PREVIEW_COUNTS_SQL: str = """
SELECT
    COUNT(DISTINCT r.metric_date)                        AS metric_dates,
    COUNT(*) FILTER (WHERE m.territory_id IS NULL)       AS rows_new,
    COUNT(*) FILTER (
        WHERE m.territory_id IS NOT NULL
          AND (m.total_trx, m.total_nrx, m.active_hcp_count, m.covered_lives, m.is_synthetic)
              IS DISTINCT FROM
              (r.total_trx, r.total_nrx, r.active_hcp_count, r.covered_lives, r.is_synthetic)
    )                                                    AS rows_changed,
    COUNT(*) FILTER (WHERE m.territory_id IS NOT NULL)   AS rows_existing,
    -- codex r14-04: rows_obsolete counts exactly what the reconcile would DELETE (same
    -- ownership predicate, substituted from the same constant), and rows_obsolete_foreign
    -- counts the stale rows it deliberately leaves because another writer put a value in them.
    (SELECT count(*) FROM territory_metrics o
      WHERE o.metric_date >= %(start_date)s::DATE AND o.metric_date < %(end_date)s::DATE
        AND __OWNED_O__
        AND NOT EXISTS (SELECT 1 FROM rollup r2 WHERE r2.territory_id = o.territory_id AND r2.metric_date = o.metric_date))
                                                         AS rows_obsolete,
    (SELECT count(*) FROM territory_metrics o
      WHERE o.metric_date >= %(start_date)s::DATE AND o.metric_date < %(end_date)s::DATE
        AND (__NOT_OWNED_O__)
        AND NOT EXISTS (SELECT 1 FROM rollup r2 WHERE r2.territory_id = o.territory_id AND r2.metric_date = o.metric_date))
                                                         AS rows_obsolete_foreign,
    MIN(r.metric_date)                                   AS first_date,
    MAX(r.metric_date)                                   AS last_date
FROM rollup r
LEFT JOIN territory_metrics m
       ON m.territory_id = r.territory_id
      AND m.metric_date  = r.metric_date
""".replace("__OWNED_O__", _TERRITORY_OWNED_BY_THIS_ETL.replace("m.", "o.")).replace(
    "__NOT_OWNED_O__", _TERRITORY_NOT_OWNED_BY_THIS_ETL
)

_TERRITORY_RECONCILE_SCOPE_BY_ARRIVAL: str = (
    "m.metric_date IN (SELECT metric_date FROM metric_dates)"
)
_TERRITORY_RECONCILE_SCOPE_BY_WINDOW: str = (
    "m.metric_date >= %(start_date)s::DATE AND m.metric_date <  %(end_date)s::DATE"
)

_TERRITORY_RECONCILE_TAIL: str = """
, rollup AS (__ROWS_SELECT__
)
DELETE FROM territory_metrics m
 WHERE __SCOPE__
   AND __OWNED__
   AND NOT EXISTS (
        SELECT 1 FROM rollup r
         WHERE r.territory_id = m.territory_id AND r.metric_date = m.metric_date
   );
"""


def _compose_territory_reconcile(metric_dates_cte: str, scope: str) -> str:
    return _TERRITORY_ROLLUP_CTES_TEMPLATE.replace("__METRIC_DATES_CTE__", metric_dates_cte) + (
        _TERRITORY_RECONCILE_TAIL.replace("__ROWS_SELECT__", _TERRITORY_ROLLUP_ROWS_SELECT)
        .replace("__SCOPE__", scope)
        .replace("__OWNED__", _TERRITORY_OWNED_BY_THIS_ETL)
    )


RECONCILE_TERRITORY_ROLLUP_SQL: str = _compose_territory_reconcile(
    _TERRITORY_METRIC_DATES_BY_WINDOW, _TERRITORY_RECONCILE_SCOPE_BY_WINDOW
)

RECONCILE_TERRITORY_ROLLUP_BY_ARRIVAL_SQL: str = _compose_territory_reconcile(
    _TERRITORY_METRIC_DATES_BY_ARRIVAL, _TERRITORY_RECONCILE_SCOPE_BY_ARRIVAL
)

#: Read-only readout of an explicit-window run. It compares only the columns the SET arm
#: writes; market_potential / resource_allocation_score are preserved on update by design.
PREVIEW_TERRITORY_ROLLUP_SQL: str = (
    _TERRITORY_ROLLUP_CTES_TEMPLATE.replace(
        "__METRIC_DATES_CTE__", _TERRITORY_METRIC_DATES_BY_WINDOW
    )
    + ",\nrollup AS ("
    + _TERRITORY_ROLLUP_ROWS_SELECT
    + ")"
    + _TERRITORY_PREVIEW_COUNTS_SQL
)

#: codex r14-04: how many obsolete keys a readout lists before it truncates. The counts in
#: PREVIEW_TERRITORY_ROLLUP_SQL are never truncated, so len(list) < count is the tell.
PREVIEW_KEY_LIMIT: int = 200

_TERRITORY_PREVIEW_OBSOLETE_TAIL: str = """
SELECT
    o.territory_id,
    o.metric_date,
    (__OWNED_O__) AS owned_by_this_etl
FROM territory_metrics o
WHERE o.metric_date >= %(start_date)s::DATE
  AND o.metric_date <  %(end_date)s::DATE
  AND NOT EXISTS (
        SELECT 1 FROM rollup r
         WHERE r.territory_id = o.territory_id AND r.metric_date = o.metric_date
  )
ORDER BY owned_by_this_etl, o.metric_date, o.territory_id
LIMIT %(limit)s
""".replace("__OWNED_O__", _TERRITORY_OWNED_BY_THIS_ETL.replace("m.", "o."))

#: codex r14-04: the exact rows the reconcile would delete (owned_by_this_etl = true) and the
#: stale ones it would keep (false), so the owner reviews a deletion SET and not a number.
#: Ordered false-first, because a foreign row is the one a reviewer must see even under LIMIT.
PREVIEW_TERRITORY_OBSOLETE_SQL: str = (
    _TERRITORY_ROLLUP_CTES_TEMPLATE.replace(
        "__METRIC_DATES_CTE__", _TERRITORY_METRIC_DATES_BY_WINDOW
    )
    + ",\nrollup AS ("
    + _TERRITORY_ROLLUP_ROWS_SELECT
    + ")"
    + _TERRITORY_PREVIEW_OBSOLETE_TAIL
)


# -----------------------------------------------------------------------------
# DB connection + window resolution: see ``src.etl._common``. The names
# ``_resolve_db_connection_string``, ``_connect_to_db`` and ``_resolve_window``
# are re-exported at the top of this module.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Core implementation (no Celery binding — directly importable for tests)
# -----------------------------------------------------------------------------


def preview_territory_rollup(start_date: str, end_date: str) -> Dict[str, Any]:
    """Read-only readout of what an explicit-window territory rollup would write.

    Runs ``PREVIEW_TERRITORY_ROLLUP_SQL`` inside a ``READ ONLY`` transaction (Postgres refuses
    any write in it) and counts the touched dates and the new / changed / existing rows.
    It is the dry-run before an owner-gated backfill.

    codex r14-04: a count is not reviewable. The readout also carries ``obsolete_keys`` — the
    exact ``(territory_id, metric_date)`` rows the reconcile would delete — and
    ``obsolete_foreign_keys``, the stale rows it would leave because another writer has put a
    value in ``market_potential`` / ``resource_allocation_score``. Both lists are capped at
    ``PREVIEW_KEY_LIMIT`` rows; the counts are not capped, so a truncated list is visible as
    ``len(obsolete_keys) < rows_obsolete``.
    """
    start_dt, end_dt = _resolve_window(start_date, end_date)
    params = {
        "start_date": start_dt,
        "end_date": end_dt,
        "per_hcp_metric_type": PER_HCP_METRIC_TYPE,
    }
    conn = _connect_to_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(PREVIEW_TERRITORY_ROLLUP_SQL, params)
                row = cur.fetchone()
                cur.execute(PREVIEW_TERRITORY_OBSOLETE_SQL, {**params, "limit": PREVIEW_KEY_LIMIT})
                obsolete = cur.fetchall()
    finally:
        conn.close()
    if row is None:  # an aggregate always returns one row
        raise RuntimeError("territory rollup preview returned no row")
    keys = (
        "metric_dates",
        "rows_new",
        "rows_changed",
        "rows_existing",
        "rows_obsolete",
        "rows_obsolete_foreign",
        "first_date",
        "last_date",
    )
    # NB: one line, deliberately. Step 3(j) rewrites every line that reads
    # `"window_end": end_dt.isoformat(),` inside the impl; this function has no `selected_by`,
    # so its window must not present that shape.
    window = {"window_start": start_dt.isoformat(), "window_end": end_dt.isoformat()}
    return {
        **window,
        **dict(zip(keys, row, strict=True)),
        "obsolete_keys": [(t, d.isoformat()) for t, d, owned in obsolete if owned],
        "obsolete_foreign_keys": [(t, d.isoformat()) for t, d, owned in obsolete if not owned],
    }


def _run_territory_rollup_impl(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    request_id: str = "no-task-id",
    arrived_before: Optional[str] = None,
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
        arrived_before: end of the ARRIVAL window when no dates are given
            (defaults to now, UTC); refused together with start_date/end_date.

    Returns:
        Dict with ``status``, ``rows_affected``, ``rows_deleted`` (obsolete rows
        the reconcile removed in the same transaction, restricted to rows this
        ETL owns), ``selected_by`` (``"arrival"`` for a scheduled run,
        ``"metric_date"`` for an explicit window), ``window_start``,
        ``window_end``, and on failure an ``error`` field.
    """
    by_arrival = start_date is None and end_date is None
    selected_by = "arrival" if by_arrival else "metric_date"
    try:
        if arrived_before is not None and not by_arrival:
            raise ValueError("arrived_before cannot be combined with start_date/end_date")
        if by_arrival:
            start_dt, end_dt = _resolve_window(
                None, arrived_before, default_lookback_seconds=ARRIVAL_WINDOW_HOURS * 3600
            )
        else:
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
            "selected_by": selected_by,
        }
    sql = INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL if by_arrival else INSERT_TERRITORY_ROLLUP_SQL
    reconcile_sql = (
        RECONCILE_TERRITORY_ROLLUP_BY_ARRIVAL_SQL if by_arrival else RECONCILE_TERRITORY_ROLLUP_SQL
    )

    logger.info(
        "Starting territory_metrics rollup [%s]: selected_by=%s window=[%s, %s)",
        request_id,
        selected_by,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    params = {
        "start_date": start_dt,
        "end_date": end_dt,
        "per_hcp_metric_type": PER_HCP_METRIC_TYPE,
    }

    conn = None
    rows_deleted = 0
    try:
        conn = _connect_to_db()
        with conn:  # transactional: commits on exit, rolls back on exception
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows_affected = cur.rowcount
                # codex r13-08: delete territory rows whose (territory, date) is no longer
                # produced, in the same transaction as the upsert.
                cur.execute(reconcile_sql, params)
                rows_deleted = cur.rowcount

        if rows_affected == 0 and rows_deleted == 0:
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
                "selected_by": selected_by,
                "rows_deleted": rows_deleted,
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
            "selected_by": selected_by,
            "rows_deleted": rows_deleted,
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
            "selected_by": selected_by,
            "rows_deleted": rows_deleted,
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
