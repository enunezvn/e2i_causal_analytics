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

Provenance inheritance (issue #895)
-----------------------------------
``business_metrics.is_synthetic`` (migration 063) defaults to ``false``, so
omitting it from the INSERT column list would stamp every derived row
"real" even when ALL aggregated inputs are synthetic — write-side
provenance laundering. The rollup therefore computes
``is_synthetic = bool(any synthetic input)`` inside the SQL itself:

* each (trigger, lateral-journey) input pair is synthetic if either row is;
* ``hcp_brand_daily`` collapses the cell with ``BOOL_OR``;
* ``territory_totals`` carries a cell-level ``any_synthetic`` because the
  ``market_share`` denominator mixes counts across HCPs — a "real" HCP's
  share computed against a denominator containing synthetic counts is a
  synthetic-contaminated number;
* the final row is synthetic if its own inputs, its HCP profile, or its
  territory denominator cell are.

Mixed-substrate semantics: an aggregate that mixes real and synthetic
inputs is tagged synthetic (fail-closed, same direction as the #872
real-mode default-exclude precedent). Real-mode reads lose mixed cells
rather than consuming numbers partially derived from synthetic rows; a
provenance-split computation (separate real/synthetic rollup rows) would
need provenance in the natural key and is deliberately out of scope here.
The ON CONFLICT update arm recomputes the tag alongside the value columns
so re-runs track current provenance instead of freezing a stale tag.

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
from src.data.per_hcp_cohort_columns import PLANTED_COLUMNS
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

#: Scheduled runs select dates by trigger ARRIVAL (``triggers.created_at``) over one weekly
#: batch cycle plus margin (the host reseed lands one batch per week). Every daily run
#: re-touches the last week's arrivals, which is idempotent, so a missed or failed beat
#: self-heals on any later run within the week (canonical TRx lane, owner decision #2).
ARRIVAL_MARGIN_HOURS: int = 6
ARRIVAL_WINDOW_HOURS: int = 7 * 24 + ARRIVAL_MARGIN_HOURS

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
#   0. affected_dates: the metric_dates touched by triggers whose trigger_timestamp
#      (explicit window) or created_at (scheduled run) falls in [start_date, end_date).
#   1. triggers_with_brand: ALL triggers of those dates x most-recent-prior
#      patient_journey.brand (a date is always recomputed whole).
#   2. hcp_brand_daily: collapse to per-(hcp_id, brand, metric_date) counts +
#      conversion_rate. That ratio is accepted triggers / delivered triggers
#      (with a NULLIF guard), NOT NRx / TRx as this line used to say: the three
#      counts below are trigger funnel stages, which is why migration 144
#      renamed them to triggers_{delivered,accepted,total}_count.
#   3. territory_totals: sum triggers_total_count per (territory_id, brand,
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
_PER_HCP_ROLLUP_CTES_TEMPLATE: str = """
WITH affected_dates AS (
    -- Late-arrival fix (canonical TRx lane, owner decision #2): the run window chooses WHICH
    -- dates to touch; every later CTE recomputes ALL triggers of those dates. market_share
    -- divides by the per-(territory, brand, date) total, so aggregating only the triggers
    -- selected by the window would overwrite a complete row with partial counts.
    SELECT DISTINCT DATE(t.trigger_timestamp) AS metric_date
      FROM triggers t
     WHERE t.__WINDOW_COLUMN__ >= %(start_date)s
       AND t.__WINDOW_COLUMN__ <  %(end_date)s
       AND t.hcp_id IS NOT NULL
),
triggers_with_brand AS (
    SELECT
        t.hcp_id,
        pj.brand,
        DATE(t.trigger_timestamp)         AS metric_date,
        t.delivery_status,
        t.acceptance_status,
        -- Provenance of this input pair (issue #895): a trigger row OR the
        -- journey row that supplied its brand being synthetic makes the
        -- pair synthetic. Both columns exist via migration 063.
        (t.is_synthetic OR pj.is_synthetic)  AS is_synthetic
    FROM triggers t
    JOIN LATERAL (
        SELECT pj_inner.brand, pj_inner.is_synthetic
          FROM patient_journeys pj_inner
         WHERE pj_inner.patient_id = t.patient_id
           AND pj_inner.brand IS NOT NULL
           AND pj_inner.journey_start_date <= t.trigger_timestamp
         ORDER BY pj_inner.journey_start_date DESC
         LIMIT 1
    ) pj ON TRUE
    WHERE DATE(t.trigger_timestamp) IN (SELECT metric_date FROM affected_dates)
      AND t.hcp_id IS NOT NULL
),
hcp_brand_daily AS (
    SELECT
        hcp_id,
        brand,
        metric_date,
        -- #1387: "delivered" = delivery_status IN ('delivered','viewed') — the
        -- ruled denominator (migrations 090/092, #1119/#1124): 'viewed' is a
        -- further progression of 'delivered', and once accepted implies viewed
        -- a delivered-exclusive count keeps only the never-viewed remainder
        -- (conversion_rate could then exceed 1).
        COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed'))          AS triggers_delivered_count,
        COUNT(*) FILTER (WHERE acceptance_status IN ('accepted', 'responded'))      AS triggers_accepted_count,
        COUNT(*)                                                                    AS triggers_total_count,
        COALESCE(
            COUNT(*) FILTER (WHERE acceptance_status IN ('accepted', 'responded'))::NUMERIC
            / NULLIF(COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed')), 0),
            0
        )                                                                           AS conversion_rate,
        -- Provenance inheritance (issue #895): any synthetic input row in
        -- the cell taints the derived aggregate.
        BOOL_OR(is_synthetic)                                                       AS any_synthetic
    FROM triggers_with_brand
    GROUP BY hcp_id, brand, metric_date
),
territory_totals AS (
    SELECT
        hp.territory_id,
        hbd.brand,
        hbd.metric_date,
        SUM(hbd.triggers_total_count) AS territory_total,
        -- Provenance of the market_share DENOMINATOR (issue #895): if any
        -- HCP cell feeding this territory total is synthetic (or the HCP
        -- profile itself is), every market_share computed against it is a
        -- synthetic-contaminated number.
        BOOL_OR(hbd.any_synthetic OR hp.is_synthetic) AS any_synthetic
    FROM hcp_brand_daily hbd
    JOIN hcp_profiles  hp ON hbd.hcp_id = hp.hcp_id
    GROUP BY hp.territory_id, hbd.brand, hbd.metric_date
)
"""

_PER_HCP_ROLLUP_INSERT_HEAD: str = """
INSERT INTO business_metrics (
    metric_id,
    metric_date,
    metric_type,
    brand,
    region,
    hcp_id,
    triggers_delivered_count,
    triggers_accepted_count,
    triggers_total_count,
    market_share,
    conversion_rate,
    -- engagement_score and call_frequency intentionally NULL: the
    -- canonical `interactions` table does not exist in v3 schema. A
    -- future ETL block will populate these once the table lands.
    is_synthetic,
    created_at
)
"""

_PER_HCP_ROLLUP_ROWS_SELECT: str = """
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
    hbd.triggers_delivered_count,
    hbd.triggers_accepted_count,
    hbd.triggers_total_count,
    CASE
        WHEN tt.territory_total > 0
            THEN hbd.triggers_total_count::NUMERIC / tt.territory_total
        ELSE 0
    END                                         AS market_share,
    hbd.conversion_rate,
    -- Inherited provenance (issue #895). tt.any_synthetic already subsumes
    -- this row's own cell (the territory total includes it), but the row-
    -- local terms are kept explicit so the semantics survive a refactor of
    -- territory_totals: synthetic if (a) any aggregated trigger/journey
    -- pair was synthetic, (b) the HCP profile is synthetic, or (c) the
    -- market_share denominator mixed in synthetic counts.
    (hbd.any_synthetic OR hp.is_synthetic OR tt.any_synthetic) AS is_synthetic,
    NOW()                                       AS created_at
FROM hcp_brand_daily hbd
JOIN hcp_profiles      hp ON hbd.hcp_id = hp.hcp_id
JOIN territory_totals  tt
  ON tt.territory_id = hp.territory_id
 AND tt.brand        = hbd.brand
 AND tt.metric_date  = hbd.metric_date
"""

_PER_HCP_ROLLUP_ON_CONFLICT: str = """
ON CONFLICT (metric_id) DO UPDATE SET
    triggers_delivered_count       = EXCLUDED.triggers_delivered_count,
    triggers_accepted_count       = EXCLUDED.triggers_accepted_count,
    triggers_total_count  = EXCLUDED.triggers_total_count,
    market_share    = EXCLUDED.market_share,
    conversion_rate = EXCLUDED.conversion_rate,
    region          = EXCLUDED.region,
    metric_type     = EXCLUDED.metric_type,
    -- Re-runs recompute every value column from the base tables; the
    -- provenance tag tracks the same recomputation (issue #895). Keeping a
    -- stale tag here would let laundered semantics survive via the update
    -- arm.
    is_synthetic    = EXCLUDED.is_synthetic;
"""


def _compose_rollup_insert(window_column: str) -> str:
    return (
        _PER_HCP_ROLLUP_CTES_TEMPLATE.replace("__WINDOW_COLUMN__", window_column)
        + _PER_HCP_ROLLUP_INSERT_HEAD
        + _PER_HCP_ROLLUP_ROWS_SELECT
        + _PER_HCP_ROLLUP_ON_CONFLICT
    )


#: Explicit ``start_date``/``end_date`` (manual backfills): the dates of triggers that
#: HAPPENED in the window, each recomputed whole.
INSERT_PER_HCP_ROLLUP_SQL: str = _compose_rollup_insert("trigger_timestamp")

#: Scheduled run (no dates): the dates of triggers that ARRIVED in the window, each
#: recomputed whole.
INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL: str = _compose_rollup_insert("created_at")

#: The reconcile's WHERE clause, alias-parameterised: ``__A__`` is the business_metrics
#: alias, ``__R__`` the rollup CTE alias, ``__SCOPE__`` the date scope on ``__A__``. The
#: explicit-window reconcile, the arrival reconcile and BOTH preview obsolete counts are
#: composed from this one template (codex r1/r2, 2026-09-22), so "a stored row on a scoped
#: date that no recomputed row reproduces" has exactly one definition.
_PER_HCP_OBSOLETE_WHERE_TEMPLATE: str = """__A__.metric_type = %(metric_type)s
   AND __A__.hcp_id IS NOT NULL
   AND __SCOPE__
   AND NOT EXISTS (SELECT 1 FROM rollup __R__ WHERE __R__.metric_id = __A__.metric_id)"""
_PER_HCP_SCOPE_BY_ARRIVAL_TEMPLATE: str = (
    "__A__.metric_date IN (SELECT metric_date FROM affected_dates)"
)
_PER_HCP_SCOPE_BY_WINDOW_TEMPLATE: str = (
    "__A__.metric_date >= %(start_date)s::DATE AND __A__.metric_date <  %(end_date)s::DATE"
)


def _obsolete_where(alias: str, rollup_alias: str, scope_template: str) -> str:
    """The obsolete-row predicate on the given aliases. ``scope_template`` is one of the
    two ``_PER_HCP_SCOPE_BY_*_TEMPLATE`` strings."""
    return (
        _PER_HCP_OBSOLETE_WHERE_TEMPLATE.replace("__SCOPE__", scope_template)
        .replace("__A__", alias)
        .replace("__R__", rollup_alias)
    )


#: Named forms of the two scopes on the reconcile's alias (referenced by the integration
#: prod-write guard's docs).
_PER_HCP_RECONCILE_SCOPE_BY_ARRIVAL: str = _PER_HCP_SCOPE_BY_ARRIVAL_TEMPLATE.replace("__A__", "b")
_PER_HCP_RECONCILE_SCOPE_BY_WINDOW: str = _PER_HCP_SCOPE_BY_WINDOW_TEMPLATE.replace("__A__", "b")

#: Both preview obsolete counts use the explicit-window predicate on the preview's alias.
_PREVIEW_OBSOLETE_WHERE: str = _obsolete_where("o", "r2", _PER_HCP_SCOPE_BY_WINDOW_TEMPLATE)

_PREVIEW_COUNTS_SQL: str = """
SELECT
    COUNT(DISTINCT r.metric_date)                        AS metric_dates,
    COUNT(*) FILTER (WHERE b.metric_id IS NULL)          AS rows_new,
    COUNT(*) FILTER (
        WHERE b.metric_id IS NOT NULL
          AND (b.triggers_delivered_count, b.triggers_accepted_count, b.triggers_total_count,
               b.market_share, b.conversion_rate, b.region, b.metric_type, b.is_synthetic)
              IS DISTINCT FROM
              (r.triggers_delivered_count, r.triggers_accepted_count, r.triggers_total_count,
               r.market_share, r.conversion_rate, r.region, r.metric_type, r.is_synthetic)
    )                                                    AS rows_changed,
    COUNT(*) FILTER (WHERE b.metric_id IS NOT NULL)      AS rows_existing,
    (SELECT count(*) FROM business_metrics o
      WHERE __OBSOLETE_WHERE__)
                                                         AS rows_obsolete,
    -- Of the obsolete rows, those still carrying the Digital Twin's planted channels or
    -- outcome (columns this ETL never writes; 2026-09-21 a full-window reconcile deleted
    -- them wholesale and the twin went dark). The operator sees this before --execute.
    (SELECT count(*) FROM business_metrics o
      WHERE __OBSOLETE_WHERE__
        AND (__COHORT_DATA_PREDICATE__))
                                                         AS rows_obsolete_with_cohort_data,
    MIN(r.metric_date)                                   AS first_date,
    MAX(r.metric_date)                                   AS last_date
FROM rollup r
LEFT JOIN business_metrics b ON b.metric_id = r.metric_id
"""

#: Every column the twin's plant writes and this ETL never does (one shared list; the
#: single-writer tests pin it against the plant script). Named in the preview only.
COHORT_DATA_COLUMNS: tuple[str, ...] = PLANTED_COLUMNS
_PREVIEW_COUNTS_SQL = _PREVIEW_COUNTS_SQL.replace(
    "__OBSOLETE_WHERE__", _PREVIEW_OBSOLETE_WHERE
).replace(
    "__COHORT_DATA_PREDICATE__", " OR ".join(f"o.{col} IS NOT NULL" for col in COHORT_DATA_COLUMNS)
)


#: codex r13-08: an upsert cannot delete. A row whose (hcp, brand, date) lost its last
#: trigger must go, or that date's market shares exceed 1.
_PER_HCP_RECONCILE_TAIL: str = """
, rollup AS (__ROWS_SELECT__
)
DELETE FROM business_metrics b
 WHERE __WHERE__;
"""


def _compose_rollup_reconcile(window_column: str, scope_template: str) -> str:
    return _PER_HCP_ROLLUP_CTES_TEMPLATE.replace("__WINDOW_COLUMN__", window_column) + (
        _PER_HCP_RECONCILE_TAIL.replace("__ROWS_SELECT__", _PER_HCP_ROLLUP_ROWS_SELECT).replace(
            "__WHERE__", _obsolete_where("b", "r", scope_template)
        )
    )


#: Explicit window: reconcile the WHOLE calendar range, so a date that lost every trigger
#: (and is therefore unselectable by arrival) is cleaned too.
RECONCILE_PER_HCP_ROLLUP_SQL: str = _compose_rollup_reconcile(
    "trigger_timestamp", _PER_HCP_SCOPE_BY_WINDOW_TEMPLATE
)

#: Scheduled run: reconcile the dates this run touched.
RECONCILE_PER_HCP_ROLLUP_BY_ARRIVAL_SQL: str = _compose_rollup_reconcile(
    "created_at", _PER_HCP_SCOPE_BY_ARRIVAL_TEMPLATE
)

#: Read-only readout of an explicit-window run: the same CTE text and row SELECT as
#: ``INSERT_PER_HCP_ROLLUP_SQL``, counted instead of written.
PREVIEW_PER_HCP_ROLLUP_SQL: str = (
    _PER_HCP_ROLLUP_CTES_TEMPLATE.replace("__WINDOW_COLUMN__", "trigger_timestamp")
    + ",\nrollup AS ("
    + _PER_HCP_ROLLUP_ROWS_SELECT
    + ")"
    + _PREVIEW_COUNTS_SQL
)

#: #2210: the explicit-window run's preflight -- the preview's ``rows_obsolete_with_cohort_data``
#: aggregate on its own, composed from the same CTEs, the same obsolete predicate and the same
#: planted-column list, so what the run refuses on is exactly what the preview reported. Runs
#: inside the run's own transaction, before the INSERT, so the count is race-free with the DELETE.
PREFLIGHT_COHORT_DATA_SQL: str = (
    _PER_HCP_ROLLUP_CTES_TEMPLATE.replace("__WINDOW_COLUMN__", "trigger_timestamp")
    + ",\nrollup AS ("
    + _PER_HCP_ROLLUP_ROWS_SELECT
    + ")\nSELECT count(*), (array_agg(o.metric_id ORDER BY o.metric_id))[1:5]\n  FROM business_metrics o\n WHERE "
    + _PREVIEW_OBSOLETE_WHERE
    + "\n   AND ("
    + " OR ".join(f"o.{col} IS NOT NULL" for col in COHORT_DATA_COLUMNS)
    + ")"
)


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


def preview_per_hcp_rollup(start_date: str, end_date: str) -> Dict[str, Any]:
    """Read-only readout of what an explicit-window rollup would write.

    Runs ``PREVIEW_PER_HCP_ROLLUP_SQL`` inside a ``READ ONLY`` transaction (Postgres refuses
    any write in it) and counts the touched dates, the new / changed / existing rows, the
    obsolete rows the reconcile would delete and, of those, the ones still carrying the
    Digital Twin's planted cohort data (``rows_obsolete_with_cohort_data``). It is the
    dry-run before an owner-gated backfill; a non-zero cohort count means the backfill must
    be followed by the plant (``scripts/backfill_segment_engagement.py --execute``).
    """
    start_dt, end_dt = _resolve_window(start_date, end_date)
    params = {
        "start_date": start_dt,
        "end_date": end_dt,
        "metric_id_prefix": METRIC_ID_PREFIX,
        "metric_type": METRIC_TYPE,
    }
    conn = _connect_to_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(PREVIEW_PER_HCP_ROLLUP_SQL, params)
                row = cur.fetchone()
    finally:
        conn.close()
    if row is None:  # an aggregate always returns one row
        raise RuntimeError("per-HCP rollup preview returned no row")
    keys = (
        "metric_dates",
        "rows_new",
        "rows_changed",
        "rows_existing",
        "rows_obsolete",
        "rows_obsolete_with_cohort_data",
        "first_date",
        "last_date",
    )
    return {
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        # strict=True is a guard, not lint appeasement: if _PREVIEW_COUNTS_SQL ever
        # returns a different number of columns than `keys` names, zip would
        # silently drop the surplus and the readout would lose a count without
        # anyone noticing. Raise instead.
        **dict(zip(keys, row, strict=True)),
    }


def _run_per_hcp_rollup_impl(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    request_id: str = "no-task-id",
    arrived_before: Optional[str] = None,
    allow_cohort_data_loss: bool = False,
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
        arrived_before: end of the ARRIVAL window when no dates are given
            (defaults to now, UTC); refused together with start_date/end_date.

    Returns:
        Dict with ``status``, ``rows_affected``, ``rows_deleted`` (obsolete rows
        the reconcile removed in the same transaction), ``selected_by``
        (``"arrival"`` for a scheduled run, ``"trigger_timestamp"`` for an
        explicit window), ``window_start``, ``window_end``, and on failure an
        ``error`` field. An explicit-window run that found cohort data at stake also
        reports ``rows_obsolete_with_cohort_data``, its ``_sample`` of ids and
        ``cohort_data_loss_acknowledged``.

    #2210 -- fail closed. An explicit-window run's reconcile deletes every stored row on the
    window's dates that the recompute no longer produces. On 2026-09-21 those rows carried the
    Digital Twin's planted cohort data (columns this ETL never writes) and the twin went dark.
    So the run first counts, under REPEATABLE READ in the same transaction as the DELETE,
    the obsolete rows still carrying that data (``PREFLIGHT_COHORT_DATA_SQL`` -- the
    preview's own aggregate, plus the first five ids); a positive count is refused (status
    ``refused``, nothing written) unless ``allow_cohort_data_loss=True``, in which case the
    loss and the replant are logged by name and the caller owns the replant. A zero count
    leaves the result payload exactly as before. The scheduled arrival run is untouched: its
    ``affected_dates`` scope is the beat's own contract.
    """
    by_arrival = start_date is None and end_date is None
    selected_by = "arrival" if by_arrival else "trigger_timestamp"
    try:
        if arrived_before is not None and not by_arrival:
            raise ValueError("arrived_before cannot be combined with start_date/end_date")
        if allow_cohort_data_loss and by_arrival:
            raise ValueError("allow_cohort_data_loss applies to explicit windows only")
        if by_arrival:
            start_dt, end_dt = _resolve_window(
                None, arrived_before, default_lookback_seconds=ARRIVAL_WINDOW_HOURS * 3600
            )
        else:
            start_dt, end_dt = _resolve_window(start_date, end_date)
    except ValueError as e:
        logger.error("Invalid window for run_per_hcp_rollup [%s]: %s", request_id, e)
        return {
            "status": "failed",
            "error": str(e),
            "rows_affected": 0,
            "window_start": start_date,
            "window_end": end_date,
            "selected_by": selected_by,
        }
    sql = INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL if by_arrival else INSERT_PER_HCP_ROLLUP_SQL
    reconcile_sql = (
        RECONCILE_PER_HCP_ROLLUP_BY_ARRIVAL_SQL if by_arrival else RECONCILE_PER_HCP_ROLLUP_SQL
    )

    logger.info(
        "Starting per-HCP business_metrics rollup [%s]: selected_by=%s window=[%s, %s)",
        request_id,
        selected_by,
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
    rows_deleted = 0
    cohort_extra: Dict[str, Any] = {}
    try:
        conn = _connect_to_db()
        with conn:  # transactional: commits on exit, rolls back on exception
            with conn.cursor() as cur:
                if not by_arrival:
                    # #2210: one snapshot for the preflight, the INSERT and the DELETE. Under
                    # READ COMMITTED (the connection default) each statement would see its
                    # own snapshot and a concurrent plant could land a cohort-bearing row
                    # between the count and the DELETE. Under REPEATABLE READ a row written
                    # after the snapshot is invisible to the DELETE, and a row the DELETE
                    # touches that another transaction changed fails the run (serialization
                    # failure -> status "failed", nothing committed): closed either way.
                    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                    cur.execute(PREFLIGHT_COHORT_DATA_SQL, params)
                    row = cur.fetchone()
                    if row is None or row[0] is None:
                        raise RuntimeError(
                            "cohort-data preflight returned no aggregate row; refusing to write"
                        )
                    at_stake = int(row[0])
                    sample = [str(m) for m in (row[1] or [])]
                    if at_stake > 0:
                        cohort_extra = {
                            "rows_obsolete_with_cohort_data": at_stake,
                            "rows_obsolete_with_cohort_data_sample": sample,
                            "cohort_data_loss_acknowledged": bool(allow_cohort_data_loss),
                        }
                    if at_stake > 0 and not allow_cohort_data_loss:
                        message = (
                            f"refused: {at_stake} obsolete per_hcp_rollup rows in "
                            f"[{start_dt.isoformat()}, {end_dt.isoformat()}) still carry the "
                            "Digital Twin's planted cohort data and the reconcile would delete "
                            f"them (first ids: {', '.join(sample)}); run preview_per_hcp_rollup, "
                            "plan the replant (scripts/backfill_segment_engagement.py --execute), "
                            "then re-run with allow_cohort_data_loss=True"
                        )
                        logger.error("Per-HCP business_metrics rollup %s [%s]", message, request_id)
                        return {
                            "status": "refused",
                            "error": message,
                            "rows_affected": 0,
                            "rows_deleted": 0,
                            "window_start": start_dt.isoformat(),
                            "window_end": end_dt.isoformat(),
                            "selected_by": selected_by,
                            **cohort_extra,
                        }
                    if at_stake > 0:
                        logger.warning(
                            "Per-HCP business_metrics rollup [%s]: allow_cohort_data_loss=True "
                            "-- %d obsolete rows carrying planted cohort data will be deleted "
                            "(first ids: %s); the replant is "
                            "scripts/backfill_segment_engagement.py --execute and the caller "
                            "owns running it",
                            request_id,
                            at_stake,
                            ", ".join(sample),
                        )
                cur.execute(sql, params)
                rows_affected = cur.rowcount
                # codex r13-08: the upsert cannot delete a row whose group lost its last
                # trigger. Same transaction, same scope, so the date is never left mixed.
                cur.execute(reconcile_sql, params)
                rows_deleted = cur.rowcount

        if rows_affected == 0 and rows_deleted == 0:
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
                "selected_by": selected_by,
                "rows_deleted": rows_deleted,
                **cohort_extra,
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
            "selected_by": selected_by,
            "rows_deleted": rows_deleted,
            **cohort_extra,
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
    name="src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup",
)
def run_per_hcp_rollup(
    self,  # noqa: ANN001 — Celery passes the bound task instance
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    allow_cohort_data_loss: bool = False,
) -> Dict[str, Any]:
    """Celery wrapper around :func:`_run_per_hcp_rollup_impl`.

    See ``_run_per_hcp_rollup_impl`` for argument and return semantics.
    """
    request_id = getattr(self.request, "id", "no-task-id") or "no-task-id"
    return _run_per_hcp_rollup_impl(
        start_date=start_date,
        end_date=end_date,
        request_id=request_id,
        allow_cohort_data_loss=allow_cohort_data_loss,
    )
