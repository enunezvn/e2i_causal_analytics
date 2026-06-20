#!/usr/bin/env python3
"""Codegen for the windowed-allowlist KPI variants (arbitrary KPI time window).

The kpi_query allowlist RPC (migration 044) runs ONLY pre-registered, vetted
read-only SQL with positionally-bound params (capped at 4, param_count must
equal max_params). The KPI engine now routes a windowable KPI to a
``*_windowed`` query id (see :func:`src.kpi.synthetic_mode.windowed_query_id`
and ``BusinessImpactCalculator._resolve_windowed_call``) with param order::

    no region:  [brand, start, end]                  -> max_params = 3
    region:     [brand, region, start, end]          -> max_params = 4

This script generates the matching registry rows. Each ``*_windowed*`` variant
is the corresponding base / region / synthetic variant with its rolling
``<col> >= NOW() - INTERVAL '<n> days'`` predicate replaced by an EXPLICIT,
positionally-bound window::

    <col> >= $K::timestamptz AND <col> < $(K+1)::timestamptz

where K is the next positional slot after brand ($1) and region ($2 when
present). Everything else -- the synthetic-excluding ``(SELECT * FROM <t> WHERE
is_synthetic = false) <t>`` wrappers, the patient_journeys region join with
``LOWER(geographic_region::text) = LOWER($N)``, and the optional brand filter
``($1::text IS NULL OR brand::text = $1)`` -- is preserved byte-for-byte from
the variant being transformed.

The base statements embedded below are the ground truth, transcribed verbatim
from the live registry (migrations 044 -> 066 synthetic split -> 077 region
split). The window transform is the ONLY change applied; nothing else in a
vetted statement is rewritten.

The validated reference (NRx ``_windowed_include_synthetic``) returned 3394 for
a 90-day Kisqali read against the live RPC.

Generate the migration with::

    python scripts/gen_kpi_windowed_variants.py > database/migrations/084_kpi_windowed_variants.sql
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# KPIs that get windowed variants: the 3 clean event_date volume metrics.
# (ROI is intentionally absent -- it is a two-source probe with divergent
# timestamp columns and no single clean windowed SQL; see module note / report.)
WINDOWED_KPIS: tuple[str, ...] = (
    "business_impact_nrx",
    "business_impact_trx",
    "business_impact_nbrx",
)


@dataclass(frozen=True)
class Variant:
    """A single windowed kpi_query_registry row."""

    query_id: str
    sql: str
    max_params: int
    note: str


# ---------------------------------------------------------------------------
# Base statements (verbatim from the live registry). For each KPI we hold the
# four shapes the four windowed variants derive from:
#   ("noregion", "exclude")          -> 066 synthetic-EXCLUDING base
#   ("noregion", "include")          -> 066 *_include_synthetic base
#   ("region",   "exclude")          -> 077 *_region base
#   ("region",   "include")          -> 077 *_region_include_synthetic base
# Each value is a tuple: (base_sql, window_col, n_static_params).
#   n_static_params = the count of positional params BEFORE the window slots:
#       noregion -> 1 (brand=$1);  region -> 2 (brand=$1, region=$2).
# The window column's ``>= NOW() - INTERVAL '<n> days'`` predicate is the only
# thing replaced; for KPIs whose 30-day filter lives on a derived column
# (NBRx: outer ``first_date``), window_col names that derived column.
# ---------------------------------------------------------------------------

_BASES: dict[str, dict[tuple[str, str], tuple[str, str]]] = {
    "business_impact_nrx": {
        ("noregion", "exclude"): (
            "SELECT COUNT(*) AS nrx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription' AND sequence_number = 1 "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1)",
            "event_date",
        ),
        ("noregion", "include"): (
            "SELECT COUNT(*) AS nrx FROM treatment_events "
            "WHERE event_type::text = 'prescription' AND sequence_number = 1 "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1)",
            "event_date",
        ),
        ("region", "exclude"): (
            "SELECT COUNT(*) AS nrx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription' AND sequence_number = 1 "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1) "
            "AND patient_journey_id IN (SELECT patient_journey_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))",
            "event_date",
        ),
        ("region", "include"): (
            "SELECT COUNT(*) AS nrx FROM treatment_events "
            "WHERE event_type::text = 'prescription' AND sequence_number = 1 "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1) "
            "AND patient_journey_id IN (SELECT patient_journey_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))",
            "event_date",
        ),
    },
    "business_impact_trx": {
        ("noregion", "exclude"): (
            "SELECT COUNT(*) AS trx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription' "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1)",
            "event_date",
        ),
        ("noregion", "include"): (
            "SELECT COUNT(*) AS trx FROM treatment_events "
            "WHERE event_type::text = 'prescription' "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1)",
            "event_date",
        ),
        ("region", "exclude"): (
            "SELECT COUNT(*) AS trx FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription' "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1) "
            "AND patient_journey_id IN (SELECT patient_journey_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))",
            "event_date",
        ),
        ("region", "include"): (
            "SELECT COUNT(*) AS trx FROM treatment_events "
            "WHERE event_type::text = 'prescription' "
            "AND event_date >= NOW() - INTERVAL '30 days' "
            "AND ($1::text IS NULL OR brand::text = $1) "
            "AND patient_journey_id IN (SELECT patient_journey_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2))",
            "event_date",
        ),
    },
    # NBRx: the 30-day filter lives on the OUTER derived column ``first_date``
    # (MIN(event_date) per patient), NOT on the raw event_date inside the CTE.
    # The window therefore binds to ``first_date`` -- the SAME predicate the
    # base used for its 30-day filter -- so "new-to-brand within the window"
    # keeps its meaning (a patient's first-ever brand Rx fell in the window).
    "business_impact_nbrx": {
        ("noregion", "exclude"): (
            "WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) "
            "SELECT COUNT(*) AS nbrx FROM first_brand WHERE first_date >= NOW() - INTERVAL '30 days'",
            "first_date",
        ),
        ("noregion", "include"): (
            "WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM treatment_events "
            "WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) GROUP BY patient_id) "
            "SELECT COUNT(*) AS nbrx FROM first_brand WHERE first_date >= NOW() - INTERVAL '30 days'",
            "first_date",
        ),
        ("region", "exclude"): (
            "WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM (SELECT * FROM treatment_events WHERE is_synthetic = false) treatment_events "
            "WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) "
            "AND patient_journey_id IN (SELECT patient_journey_id FROM (SELECT * FROM patient_journeys WHERE is_synthetic = false) patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2)) GROUP BY patient_id) "
            "SELECT COUNT(*) AS nbrx FROM first_brand WHERE first_date >= NOW() - INTERVAL '30 days'",
            "first_date",
        ),
        ("region", "include"): (
            "WITH first_brand AS (SELECT patient_id, MIN(event_date) AS first_date FROM treatment_events "
            "WHERE event_type::text = 'prescription' AND ($1::text IS NULL OR brand::text = $1) "
            "AND patient_journey_id IN (SELECT patient_journey_id FROM patient_journeys WHERE LOWER(geographic_region::text) = LOWER($2)) GROUP BY patient_id) "
            "SELECT COUNT(*) AS nbrx FROM first_brand WHERE first_date >= NOW() - INTERVAL '30 days'",
            "first_date",
        ),
    },
}


def _apply_window(base_sql: str, window_col: str, *, region: bool) -> str:
    """Replace the rolling NOW()-INTERVAL filter with a positional window.

    The window binds to the next positional slot(s) after brand ($1) and, when
    present, region ($2): start=$K, end=$(K+1) with K = 3 (region) or 2 (no
    region). The match is anchored on the EXACT predicate text the base uses
    (``<col> >= NOW() - INTERVAL '<n> days'``) so any drift fails loud rather
    than silently mis-binding -- vetted SQL must transform exactly or not at all.
    """
    start_slot = 3 if region else 2
    end_slot = start_slot + 1
    needle = f"{window_col} >= NOW() - INTERVAL '30 days'"
    if needle not in base_sql:
        raise ValueError(
            f"window predicate {needle!r} not found in base SQL; refusing to "
            f"emit an un-transformed (still-rolling) windowed variant"
        )
    if base_sql.count(needle) != 1:
        raise ValueError(
            f"window predicate {needle!r} appears {base_sql.count(needle)} times; "
            f"expected exactly 1 -- ambiguous transform"
        )
    replacement = (
        f"{window_col} >= ${start_slot}::timestamptz AND {window_col} < ${end_slot}::timestamptz"
    )
    return base_sql.replace(needle, replacement)


def generate_variant(base_query_id: str, *, region: bool, include_synthetic: bool) -> Variant:
    """Build the windowed ``Variant`` for one (kpi, region, synthetic) combo.

    query_id: ``{base}_windowed[_region][_include_synthetic]`` (canonical suffix
    order, matching :func:`src.kpi.synthetic_mode.windowed_query_id`).
    max_params: 3 (no region) or 4 (region).
    """
    if base_query_id not in _BASES:
        raise KeyError(f"no windowed base registered for {base_query_id!r}")

    region_key = "region" if region else "noregion"
    synth_key = "include" if include_synthetic else "exclude"
    base_sql, window_col = _BASES[base_query_id][(region_key, synth_key)]

    sql = _apply_window(base_sql, window_col, region=region)

    query_id = f"{base_query_id}_windowed"
    if region:
        query_id += "_region"
    if include_synthetic:
        query_id += "_include_synthetic"

    max_params = 4 if region else 3

    note_bits = ["windowed [brand"]
    note_bits.append(", region" if region else "")
    note_bits.append(", start, end]")
    synth_note = " (includes synthetic)" if include_synthetic else ""
    note = f"{''.join(note_bits)}{synth_note}"

    return Variant(query_id=query_id, sql=sql, max_params=max_params, note=note)


def _all_variants() -> list[Variant]:
    """All 12 variants: 3 KPIs x {no-region, region} x {exclude, include}."""
    variants: list[Variant] = []
    for base in WINDOWED_KPIS:
        for region in (False, True):
            for include_synthetic in (False, True):
                variants.append(
                    generate_variant(base, region=region, include_synthetic=include_synthetic)
                )
    return variants


def render_migration() -> str:
    """Render the idempotent INSERT ... ON CONFLICT migration (style of 077)."""
    header = """\
-- ============================================================================
-- 084_kpi_windowed_variants.sql
-- Windowed variants of the clean event_date VOLUME KPI queries (NRx/TRx/NBRx),
-- for the "arbitrary KPI time window" feature.
--
-- WHY ADDITIVE (not in-place): the kpi_query RPC strictly enforces
-- param_count == max_params, and the base queries feed the certified KPI gates.
-- These ADD parallel `*_windowed[_region][_include_synthetic]` query ids that
-- the calculator routes to ONLY when a window is selected (see
-- BusinessImpactCalculator._resolve_windowed_call / windowed_query_id). The
-- original queries stay byte-for-byte unchanged, so the un-windowed path and
-- the gates are unaffected.
--
-- TRANSFORM: each variant is the corresponding base / region / synthetic
-- statement with its rolling `<col> >= NOW() - INTERVAL '30 days'` filter
-- replaced by an explicit positional window
-- `<col> >= $K::timestamptz AND <col> < $(K+1)::timestamptz`
-- (K = 2 no-region, 3 region). Param order:
--   no region:  [brand=$1, start=$2, end=$3]            (max_params 3)
--   region:     [brand=$1, region=$2, start=$3, end=$4] (max_params 4)
-- NBRx windows on the OUTER derived `first_date` (MIN(event_date) per patient),
-- the SAME column the base 30-day filter used, preserving "new-to-brand in
-- window" semantics. Region uses the patient_journeys.geographic_region join
-- (LOWER-case-insensitive), identical to migration 077. Synthetic-EXCLUDING
-- variants keep the `(SELECT * ... WHERE is_synthetic=false)` wrappers; the
-- `_include_synthetic` variants do not.
--
-- AUTO-GENERATED by scripts/gen_kpi_windowed_variants.py (do not hand-edit;
-- regenerate). The NRx `_windowed_include_synthetic` form was validated against
-- the live RPC (returned 3394 for a 90-day Kisqali read).
--
-- ROI (WS3-BI-010) is intentionally NOT windowed here: it is a two-source
-- fallback probe (business_metrics.metric_date / agent_activities
-- .activity_timestamp) with divergent timestamp columns and no single clean
-- windowed SQL -- deferred to a follow-up.
--
-- Idempotent (ON CONFLICT DO UPDATE). Depends on: 044 (registry+RPC), 066
-- (synthetic twins), 077 (region join pattern).
-- ============================================================================

INSERT INTO public.kpi_query_registry (query_id, sql, max_params, note) VALUES
"""
    rows = []
    for v in _all_variants():
        rows.append(
            f"    ('{v.query_id}', $kpi${v.sql}$kpi$, {v.max_params}, $note${v.note}$note$)"
        )
    body = ",\n".join(rows)
    footer = (
        "\nON CONFLICT (query_id) DO UPDATE SET "
        "sql = EXCLUDED.sql, max_params = EXCLUDED.max_params, note = EXCLUDED.note;\n"
        "\n-- PostgREST caches the schema; reload so the new ids are visible.\n"
        "NOTIFY pgrst, 'reload schema';\n"
    )
    return header + body + footer


def main(argv: list[str] | None = None) -> int:
    """Write the migration SQL to stdout (redirect into the migration file)."""
    sys.stdout.write(render_migration())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
