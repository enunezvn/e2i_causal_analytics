#!/usr/bin/env python
"""Generate database/migrations/143_canonical_volume_kpis.sql (canonical TRx lane).

Owner decision 2026-09-15: TRx / NRx / NBRx / TRx Share (WS3-BI-005..008) read
the canonical business_metrics monthly series. This generator is the single
source of that SQL; tests/unit/test_kpi/test_mig143_canonical_volume_registry.py
asserts the committed migration equals ``render()``.

Statement contract (every statement):
* reads only ``business_metrics``: base ids exclude ``is_synthetic`` rows (the
  M4 idiom), ``*_include_synthetic`` twins read every row;
* NEVER serves the in-progress calendar month
  (``metric_date < date_trunc('month', CURRENT_DATE)``);
* headline statements answer the latest COMPLETE month, anchored to the GLOBAL
  trx frontier (migration 125 precedent: never narrowed per scope), and
  disclose ``data_month`` and ``data_through`` (= that month's last day);
* windowed statements sum only months FULLY inside [start, end];
* national = sum over regions; brand and region are nullable text filters,
  region case-insensitive (the 125 idiom).

Usage (from the checkout root):
    PYTHONPATH=. python -m scripts.gen_canonical_volume_registry          # write
    PYTHONPATH=. python -m scripts.gen_canonical_volume_registry --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from src.kpi.volume_family import DIMENSIONED_ROW_SQL

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "database" / "migrations" / "143_canonical_volume_kpis.sql"
ROLLBACK_OUT = REPO / "database" / "migrations" / "rollback_143_canonical_volume_kpis.sql"

IN_PROGRESS_MONTH = "date_trunc('month', CURRENT_DATE)::date"
VOLUME_METRICS: Tuple[str, ...] = ("trx", "nrx", "nbrx")
SYNTHETIC_SUFFIX = "_include_synthetic"
VARIANTS: Tuple[Tuple[str, bool, bool], ...] = (
    ("", False, False),
    ("_region", True, False),
    ("_windowed", False, True),
    ("_windowed_region", True, True),
)
KPI_HISTORY_REKEY: Tuple[Tuple[str, str], ...] = (
    ("WS3-BI-005", "WS3-BI-011"),
    ("WS3-BI-006", "WS3-BI-012"),
    ("WS3-BI-007", "WS3-BI-013"),
    ("WS3-BI-008", "WS3-BI-014"),
)
PANEL_SOURCE = "treatment_events.event_date"

Row = Tuple[str, str, int, str]


def _src(synthetic: bool) -> str:
    """The eligible rows. Every read of the table (sum, frontier, share denominator,
    series) goes through here, so the shared NULL-dimension rule
    (volume_family.DIMENSIONED_ROW_SQL, codex r2) cannot be skipped by one variant."""
    where = DIMENSIONED_ROW_SQL if synthetic else f"{DIMENSIONED_ROW_SQL} AND is_synthetic = false"
    return f"(SELECT * FROM business_metrics WHERE {where})"


def _frontier(synthetic: bool) -> str:
    return (
        f"(SELECT MAX(f.metric_date) FROM {_src(synthetic)} f "
        f"WHERE f.metric_type = 'trx' AND f.metric_date < {IN_PROGRESS_MONTH})"
    )


def _month_end(expr: str) -> str:
    return f"({expr} + INTERVAL '1 month' - INTERVAL '1 day')::date"


def _params(region: bool, windowed: bool) -> Tuple[str, str, int]:
    """(start placeholder, end placeholder, max_params) for a variant."""
    if windowed:
        return ("$3", "$4", 4) if region else ("$2", "$3", 3)
    return ("", "", 2 if region else 1)


def volume_sql(metric: str, *, synthetic: bool, region: bool, windowed: bool) -> Tuple[str, int]:
    src = _src(synthetic)
    brand = "($1::text IS NULL OR bm.brand::text = $1)"
    region_clause = " AND LOWER(bm.region::text) = LOWER($2)" if region else ""
    start, end, n = _params(region, windowed)
    if not windowed:
        frontier = _frontier(synthetic)
        return (
            f"SELECT SUM(bm.value) AS {metric}, {frontier} AS data_month, "
            f"{_month_end(frontier)} AS data_through "
            f"FROM {src} bm WHERE bm.metric_type = '{metric}' AND bm.metric_date = {frontier} "
            f"AND {brand}{region_clause}",
            n,
        )
    return (
        f"SELECT SUM(bm.value) AS {metric}, MAX(bm.metric_date) AS data_month, "
        f"{_month_end('MAX(bm.metric_date)')} AS data_through, "
        f"COUNT(DISTINCT bm.metric_date) AS months_in_window "
        f"FROM {src} bm WHERE bm.metric_type = '{metric}' "
        f"AND bm.metric_date >= {start}::date AND {_month_end('bm.metric_date')} <= {end}::date "
        f"AND bm.metric_date < {IN_PROGRESS_MONTH} AND {brand}{region_clause}",
        n,
    )


def share_sql(*, synthetic: bool, region: bool, windowed: bool) -> Tuple[str, int]:
    src = _src(synthetic)
    share = "SUM(bm.value) FILTER (WHERE bm.brand::text = $1) / NULLIF(SUM(bm.value), 0) AS share"
    region_clause = " AND LOWER(bm.region::text) = LOWER($2)" if region else ""
    start, end, n = _params(region, windowed)
    if not windowed:
        frontier = _frontier(synthetic)
        return (
            f"SELECT {share}, {frontier} AS data_month, {_month_end(frontier)} AS data_through "
            f"FROM {src} bm WHERE bm.metric_type = 'trx' AND bm.metric_date = {frontier}"
            f"{region_clause}",
            n,
        )
    return (
        f"SELECT {share}, MAX(bm.metric_date) AS data_month, "
        f"{_month_end('MAX(bm.metric_date)')} AS data_through, "
        f"COUNT(DISTINCT bm.metric_date) AS months_in_window "
        f"FROM {src} bm WHERE bm.metric_type = 'trx' "
        f"AND bm.metric_date >= {start}::date AND {_month_end('bm.metric_date')} <= {end}::date "
        f"AND bm.metric_date < {IN_PROGRESS_MONTH}{region_clause}",
        n,
    )


def series_sql(*, synthetic: bool) -> Tuple[str, int]:
    return (
        f"SELECT bm.metric_date, SUM(bm.value) AS value, COUNT(*) AS n_rows "
        f"FROM {_src(synthetic)} bm WHERE $1::text IN ('trx', 'nrx', 'nbrx') "
        f"AND bm.metric_type = $1 AND bm.metric_date < {IN_PROGRESS_MONTH} "
        f"AND ($2::text IS NULL OR bm.brand::text = $2) "
        f"AND ($3::text IS NULL OR LOWER(bm.region::text) = LOWER($3)) "
        f"GROUP BY bm.metric_date ORDER BY bm.metric_date",
        3,
    )


_PARAM_DOC = {
    ("", False): "$1 brand|NULL",
    ("_region", False): "$1 brand|NULL, $2 region",
    ("_windowed", False): "$1 brand|NULL, $2 start, $3 end",
    ("_windowed_region", False): "$1 brand|NULL, $2 region, $3 start, $4 end",
    ("", True): "$1 brand (required)",
    ("_region", True): "$1 brand (required), $2 region",
    ("_windowed", True): "$1 brand (required), $2 start, $3 end",
    ("_windowed_region", True): "$1 brand (required), $2 region, $3 start, $4 end",
}


def registry_rows() -> List[Row]:
    rows: List[Row] = []
    for synthetic in (False, True):
        sfx = SYNTHETIC_SUFFIX if synthetic else ""
        scope = "includes synthetic (showcase)" if synthetic else "M4 default-exclude synthetic"
        for metric in VOLUME_METRICS:
            for variant, region, windowed in VARIANTS:
                sql, n = volume_sql(metric, synthetic=synthetic, region=region, windowed=windowed)
                note = (
                    f"canonical TRx lane: {metric} latest complete month over business_metrics; "
                    f"{_PARAM_DOC[(variant, False)]}; {scope}"
                )
                rows.append((f"canonical_volume_{metric}{variant}{sfx}", sql, n, note))
        for variant, region, windowed in VARIANTS:
            sql, n = share_sql(synthetic=synthetic, region=region, windowed=windowed)
            note = (
                "canonical TRx lane: brand share of portfolio trx over business_metrics; "
                f"{_PARAM_DOC[(variant, True)]}; {scope}"
            )
            rows.append((f"canonical_volume_trx_share{variant}{sfx}", sql, n, note))
        sql, n = series_sql(synthetic=synthetic)
        note = (
            "canonical TRx lane: complete-month series; $1 metric trx|nrx|nbrx, "
            f"$2 brand|NULL, $3 region|NULL; {scope}"
        )
        rows.append((f"canonical_volume_monthly_series{sfx}", sql, n, note))
    return rows


HEADER = """-- ============================================================================
-- Migration 143: canonical Rx-volume KPIs on business_metrics (canonical TRx lane)
-- ============================================================================
-- GENERATED by scripts/gen_canonical_volume_registry.py -- edit the generator,
-- then run it; tests/unit/test_kpi/test_mig143_canonical_volume_registry.py
-- fails on any hand edit.
--
-- Owner decision 2026-09-15: TRx / NRx / NBRx / TRx Share (WS3-BI-005..008)
-- read the canonical monthly business_metrics series. The treatment_events
-- prescription counts they used to compute become the patient-panel KPIs
-- WS3-BI-011..014, which keep every existing business_impact_* statement.
--
-- Every statement below reads only business_metrics, never serves the
-- in-progress calendar month, and (headline) answers the latest COMPLETE month
-- against the GLOBAL trx frontier with data_month / data_through disclosed.
-- ADDITIVE registry rows (no existing id changes); twins follow the 066 idiom.
-- DB application follows the droplet recipe (psql --single-transaction owns the
-- outer txn - no COMMIT here).
-- ----------------------------------------------------------------------------

"""

#: Where migration 143 records what it moved (the rollback's provenance, codex r2).
REKEY_AUDIT_TABLE = "public.kpi_history_rekey_143"
#: Payload equivalence for a kpi_history point (079_kpi_history.sql:18-24). computed_at
#: is when a point was computed, not what it says, so it is not compared.
PAYLOAD_COLUMNS: Tuple[str, ...] = ("value", "status", "source", "is_synthetic")
_AUDIT_COLUMNS = (
    "source_history_id, dest_history_id, disposition, source_kpi_id, dest_kpi_id, "
    "brand, region, metric_date, value, status, source, is_synthetic, computed_at"
)

REKEY_COMMENT = """-- kpi_history: the event-backed history of WS3-BI-005..008 now belongs to the
-- panel KPIs. UNIQUE (kpi_id, brand, region, metric_date) can already hold a
-- destination row (kpi_history.py upserts on that key). Payload equivalence is
-- value, status, source and is_synthetic (IS NOT DISTINCT FROM). For each pair:
--   (1) a destination row whose payload differs from its source row ABORTS the
--       migration and reports the count (codex r1);
--   (2) every source row is recorded in public.kpi_history_rekey_143 before it
--       moves: 'absorbed' when an equivalent destination already existed (that
--       destination is pre-existing data), 'moved' when this migration created
--       the destination copy (its new id is recorded);
--   (3) only source rows recorded in (2) are DELETED.
-- Rollback (codex r2) re-inserts every recorded source row under its original id
-- and deletes ONLY the destination rows this migration created, so pre-existing
-- and later destination rows are never deleted or re-keyed.
-- Canonical-source rows (business_metrics.value) are never touched. A rerun is a no-op.
"""


def _same_payload(a: str, b: str) -> str:
    return " AND ".join(f"{a}.{c} IS NOT DISTINCT FROM {b}.{c}" for c in PAYLOAD_COLUMNS)


def audit_table_sql(audit: str = REKEY_AUDIT_TABLE) -> str:
    """The re-key provenance table (the live tests pass a pg_temp name)."""
    rls = (
        "" if audit.startswith("pg_temp.") else f"\nALTER TABLE {audit} ENABLE ROW LEVEL SECURITY;"
    )
    return f"""CREATE TABLE IF NOT EXISTS {audit} (
    source_history_id uuid PRIMARY KEY,
    dest_history_id   uuid NOT NULL,
    disposition       text NOT NULL CHECK (disposition IN ('moved', 'absorbed')),
    source_kpi_id     text NOT NULL,
    dest_kpi_id       text NOT NULL,
    brand             text NOT NULL,
    region            text NOT NULL,
    metric_date       date NOT NULL,
    value             double precision NOT NULL,
    status            text,
    source            text NOT NULL,
    is_synthetic      boolean NOT NULL,
    computed_at       timestamptz NOT NULL,
    recorded_at       timestamptz NOT NULL DEFAULT now()
);{rls}"""


def rekey_sql(
    table: str = "public.kpi_history",
    pairs: Tuple[Tuple[str, str], ...] = KPI_HISTORY_REKEY,
    audit: str = REKEY_AUDIT_TABLE,
) -> str:
    """The guarded, provenance-recording re-key DO block (live tests use TEMP tables)."""
    array = ", ".join(f"['{old}', '{new}']" for old, new in pairs)
    key_join = (
        f"JOIN {table} d\n"
        "            ON d.kpi_id = pair[2] AND d.brand = s.brand\n"
        "           AND d.region = s.region AND d.metric_date = s.metric_date"
    )
    source_rows = f"s.kpi_id = pair[1] AND s.source = '{PANEL_SOURCE}'"
    return f"""DO $rekey$
DECLARE
    pair text[];
    conflicts integer;
BEGIN
    FOREACH pair SLICE 1 IN ARRAY ARRAY[{array}]::text[] LOOP
        SELECT count(*) INTO conflicts
          FROM {table} s
          {key_join}
         WHERE {source_rows}
           AND NOT ({_same_payload("d", "s")});
        IF conflicts > 0 THEN
            RAISE EXCEPTION 'migration 143: % conflicting rows already under % for % event rows; re-key refused', conflicts, pair[2], pair[1];
        END IF;
        INSERT INTO {audit} ({_AUDIT_COLUMNS})
        SELECT s.id, d.id, 'absorbed', s.kpi_id, d.kpi_id, s.brand, s.region, s.metric_date,
               s.value, s.status, s.source, s.is_synthetic, s.computed_at
          FROM {table} s
          {key_join}
         WHERE {source_rows};
        WITH moved AS (
            INSERT INTO {table} (kpi_id, brand, region, metric_date, value, status, source, is_synthetic, computed_at)
            SELECT pair[2], s.brand, s.region, s.metric_date, s.value, s.status, s.source, s.is_synthetic, s.computed_at
              FROM {table} s
             WHERE {source_rows}
               AND NOT EXISTS (SELECT 1 FROM {audit} a WHERE a.source_history_id = s.id)
            RETURNING id, kpi_id, brand, region, metric_date
        )
        INSERT INTO {audit} ({_AUDIT_COLUMNS})
        SELECT s.id, m.id, 'moved', s.kpi_id, m.kpi_id, s.brand, s.region, s.metric_date,
               s.value, s.status, s.source, s.is_synthetic, s.computed_at
          FROM moved m
          JOIN {table} s
            ON {source_rows}
           AND s.brand = m.brand AND s.region = m.region AND s.metric_date = m.metric_date;
        DELETE FROM {table} s
         USING {audit} a
         WHERE a.source_history_id = s.id AND a.source_kpi_id = pair[1];
    END LOOP;
END
$rekey$;"""


def restore_sql(table: str = "public.kpi_history", audit: str = REKEY_AUDIT_TABLE) -> str:
    """Rollback of the re-key from its provenance: restores ONLY what 143 moved."""
    return f"""DO $restore$
DECLARE
    conflicts integer;
BEGIN
    IF to_regclass('{audit}') IS NULL THEN
        RAISE EXCEPTION 'rollback 143: % is missing, so what the migration moved is unknown; restore refused', '{audit}';
    END IF;
    SELECT count(*) INTO conflicts
      FROM {audit} a
      JOIN {table} t
        ON t.kpi_id = a.source_kpi_id AND t.brand = a.brand
       AND t.region = a.region AND t.metric_date = a.metric_date
     WHERE NOT ({_same_payload("t", "a")});
    IF conflicts > 0 THEN
        RAISE EXCEPTION 'rollback 143: % rows under the original keys differ from the recorded payload; restore refused', conflicts;
    END IF;
    INSERT INTO {table} (id, kpi_id, brand, region, metric_date, value, status, source, is_synthetic, computed_at)
    SELECT a.source_history_id, a.source_kpi_id, a.brand, a.region, a.metric_date,
           a.value, a.status, a.source, a.is_synthetic, a.computed_at
      FROM {audit} a
    ON CONFLICT (kpi_id, brand, region, metric_date) DO NOTHING;
    DELETE FROM {table} t
     USING {audit} a
     WHERE a.disposition = 'moved' AND t.id = a.dest_history_id;
    DROP TABLE {audit};
END
$restore$;"""


def rollback_render() -> str:
    """Recovery-only reversal of migration 143 (plan Task 29 Step 6b), from its provenance."""
    ids = ", ".join(f"'{qid}'" for qid, *_ in registry_rows())
    return (
        "-- ROLLBACK for migration 143 (canonical TRx lane). NOT a forward migration:\n"
        "-- scripts/run_migrations.sh excludes rollback_* files. Apply by hand only in a\n"
        "-- recovery procedure (plan Task 29 Step 6b), from the checksummed copy staged\n"
        "-- outside the deploy's reset (Task 29 Step 5b).\n"
        "-- GENERATED by scripts/gen_canonical_volume_registry.py.\n\n"
        "-- Canonical history first, so a restored event row cannot collide with it.\n"
        "DELETE FROM public.kpi_history WHERE kpi_id IN ('WS3-BI-005', 'WS3-BI-006', "
        "'WS3-BI-007', 'WS3-BI-008') AND source = 'business_metrics.value';\n\n"
        + restore_sql()
        + f"\n\nDELETE FROM public.kpi_query_registry WHERE query_id IN ({ids});\n\n"
        + "NOTIFY pgrst, 'reload schema';\n"
    )


def render() -> str:
    values = ",\n".join(
        f"    ('{qid}', $kpi${sql}$kpi$, {n}, $note${note}$note$)"
        for qid, sql, n, note in registry_rows()
    )
    return (
        HEADER
        + "INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES\n"
        + values
        + "\nON CONFLICT (query_id) DO UPDATE SET sql = EXCLUDED.sql, "
        + "max_params = EXCLUDED.max_params, note = EXCLUDED.note;\n\n"
        + REKEY_COMMENT
        + audit_table_sql()
        + "\n\n"
        + rekey_sql()
        + "\n\n-- PostgREST caches the schema; reload so the new ids are visible.\n"
        + "NOTIFY pgrst, 'reload schema';\n\n"
        + "-- (No COMMIT; psql --single-transaction owns the outer txn.)\n"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a file is stale")
    args = parser.parse_args(argv)
    outputs = {OUT: render(), ROLLBACK_OUT: rollback_render()}
    if args.check:
        stale = [p for p, text in outputs.items() if (p.read_text() if p.exists() else "") != text]
        for path in stale:
            print(f"{path} is stale -- run python -m scripts.gen_canonical_volume_registry")
        if not stale:
            print("migration 143 and its rollback are current")
        return 1 if stale else 0
    for path, text in outputs.items():
        path.write_text(text)
        print(f"wrote {path}")
    print(f"{len(registry_rows())} registry rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
